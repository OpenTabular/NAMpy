import torch

from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch.nn as nn

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
nl = "\n"


class BayesianNN(PyroModule):
    def __init__(
        self,
        in_dim: int=1,
        out_dim: int=1,
        config: DefaultBayesianNNConfig = DefaultBayesianNNConfig()
    ):
        self._logger = logging.getLogger(__name__)

        super().__init__()

        self._config = config

        self._activation = self._config.activation

        # Add all linear layers to the NN.
        self._layers = PyroModule[torch.nn.ModuleList](
            [
                PyroModule[nn.Linear](in_dim, self._config.hidden_layer_sizes[0])
            ]
        )
        for i in range(1, len(self._config.hidden_layer_sizes)):
            self._layers.append(
                PyroModule[nn.Linear](
                    self._config.hidden_layer_sizes[i-1], self._config.hidden_layer_sizes[i]
                )
            )
        self._layers.append(PyroModule[nn.Linear](self._config.hidden_layer_sizes[-1], out_dim))

        # Initialize layer parameters as samples (random variables ~ prior).
        # Note:
        # Sampling weights from the Gaussian prior yields a tensor of shape (hidden_dim, in_dim).
        # This tensor is a single random event from a Multivariate Gaussian.
        self._layers[0].weight = PyroSample(
            dist.Normal(
                loc=self._config.gaussian_prior_location,
                scale=self._config.gaussian_prior_scale
            ).expand(
                [self._config.hidden_layer_sizes[0], in_dim]
            ).to_event(reinterpreted_batch_ndims=2)
        )
        for i in range(1, len(self._config.hidden_layer_sizes)):
            self._layers[i-1].bias = PyroSample(
                dist.Normal(
                    loc=self._config.gaussian_prior_location,
                    scale=self._config.gaussian_prior_scale
                ).expand(
                    [self._config.hidden_layer_sizes[i]]
                ).to_event(reinterpreted_batch_ndims=1)
            )
            self._layers[i].weight = PyroSample(
                dist.Normal(
                    loc=self._config.gaussian_prior_location,
                    scale=self._config.gaussian_prior_scale
                ).expand(
                    [self._config.hidden_layer_sizes[i], self._config.hidden_layer_sizes[i-1]]
                ).to_event(reinterpreted_batch_ndims=2)
            )
        self._layers[-1].bias = PyroSample(
            dist.Normal(
                loc=self._config.gaussian_prior_location,
                scale=self._config.gaussian_prior_scale
            ).expand(
                [out_dim]
            ).to_event(reinterpreted_batch_ndims=1)
        )
        # Note:
        # In contrast to pytorch tensors, which have one shape attribute
        # (with a single .shape attribute),
        # pyro distributions have two additive shape attributes
        # (.shape = .batch_shape + .event_shape).
        # The dimensions corresponding to batch_shape denote independent RVs,
        # whereas those corresponding to event_shape denote dependent RVs.
        # By calling .to_event(n), the n right-most dimensions of a tensor are declared dependent.

        self._logger.info(f"Bayesian NN successfully initialized.")


    def forward(self, x, y=None):
        x = x.reshape(-1, 1)

        for i in range(len(self._layers) - 1):  # Exclude final layer.
            x = self._activation(self._layers[i](x))
        mu = self._layers[-1](x).squeeze()
        sigma = pyro.sample(
            name="sigma",
            fn=dist.Gamma(
                self._config.gamma_prior_shape,
                self._config.gamma_prior_scale
            )
        )  # Sample the response variable noise from the specified prior.

        # Sample from Gaussian with mean estimated by NN and variance sampled from variance prior.
        # i.e., the likelihood function is a P(y_i|x_i;\theta) = N(NN_{\theta}(x_i); sigma))
        # with \sigma ~ Gamma(loc;scale)
        with pyro.plate(name="data", size=x.shape[0]):
            pyro.sample(
                name="obs",
                fn=dist.Normal(loc=mu, scale=sigma*sigma),
                obs=y  # True response observation.
            )  # .expand(x.shape[0]) is automatically executed.

        return mu






