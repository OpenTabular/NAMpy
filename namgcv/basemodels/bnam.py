from typing import Tuple, Any

import numpy as np
import torch

from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange

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
        assert len(self._config.hidden_layer_sizes) > 0, (
            "Please ensure that there is at least one hidden layer size defined in the "
            "model configuration file."
        )
        self._layer_sizes = [in_dim] + list(self._config.hidden_layer_sizes) + [out_dim]
        for layer in self._layer_sizes:
            assert layer > 0, (
                "Please ensure that all layers defined in the model configuration file "
                "have size > 0."
            )

        # noinspection PyTypeChecker
        # self._layers = PyroModule[torch.nn.ModuleList]([
        #     PyroModule[nn.Linear](
        #         self._layer_sizes[idx - 1],
        #         self._layer_sizes[idx]
        #     ) for idx in range(1, len(self._layer_sizes))
        # ])

        # noinspection PyTypeChecker
        self._layers = PyroModule[torch.nn.ModuleList](
            [
                module
                for i in range(1, len(self._layer_sizes))
                for module in (
                    [PyroModule[nn.Linear](self._layer_sizes[i - 1], self._layer_sizes[i])]
                    +
                    (
                      [PyroModule[nn.BatchNorm1d](self._layer_sizes[i])]
                      if self._config.batch_norm else []
                    )
                    +
                    (
                      [PyroModule[nn.LayerNorm](self._layer_sizes[i])]
                      if self._config.layer_norm else []
                    )
                    +
                    (
                      [getattr(nn, self._config.activation)()]  # Handle activation as a callable
                      if isinstance(self._config.activation, str)
                      else [self._config.activation] # Directly add the activation if pre-instantiated
                    ) if not self._config.use_glu else [PyroModule[nn.GLU]()])
                    +
                    (
                      [PyroModule[nn.Dropout](self._config.dropout)]
                      if self._config.dropout > 0.0 else []
                    )
            ]
        )

        layer_idx = 0
        for layer in self._layers:
            # Check if the layer is a linear layer.
            if isinstance(layer, nn.Linear):
                layer.weight = PyroSample(
                    dist.Normal(
                        loc=self._config.gaussian_prior_location,
                        scale=self._config.gaussian_prior_scale * np.sqrt(
                            2 / self._layer_sizes[layer_idx]
                        )  # He initialization to control for vanishing/exploding gradients.
                    ).expand([
                        self._layer_sizes[layer_idx + 1],
                        self._layer_sizes[layer_idx]
                    ]).to_event(2)
                )
                layer.bias = PyroSample(
                    dist.Normal(
                        loc=self._config.gaussian_prior_location,
                        scale=self._config.gaussian_prior_scale
                    ).expand([
                        self._layer_sizes[layer_idx + 1]
                    ]).to_event(1))
                layer_idx += 1


        # Initialize layer parameters as samples (random variables ~ prior).
        # Note:
        # Sampling weights from the Gaussian prior yields a tensor of shape (hidden_dim, in_dim).
        # This tensor is a single random event from a Multivariate Gaussian.

        # Note:
        # In contrast to pytorch tensors, which have one shape attribute
        # (with a single .shape attribute),
        # pyro distributions have two additive shape attributes
        # (.shape = .batch_shape + .event_shape).
        # The dimensions corresponding to batch_shape denote independent RVs,
        # whereas those corresponding to event_shape denote dependent RVs.
        # By calling .to_event(n), the n right-most dimensions of a tensor are declared dependent.

        self._generate_predictive_distribution = {
            "variational": self._get_variational_predictive_distribution,
            "mcmc": self._get_mcmc_predictive_distribution
        }

        self._logger.info(f"Bayesian NN successfully initialized.")


    def forward(self, x, y=None):
        x = x.reshape(-1, 1)

        for i in range(len(self._layers) - 1):  # Exclude final layer.
            x = self._layers[i](x)
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


    def _get_variational_predictive_distribution(self, x_train, y_train, num_samples: int):
        pyro.clear_param_store()

        self._mean_field_guide = AutoDiagonalNormal(self)
        optimizer = pyro.optim.Adam({"lr": self._config.lr})

        self._svi = SVI(
            self,  # Pass the BNN model.
            self._mean_field_guide,
            optimizer,
            loss=Trace_ELBO()
        )
        pyro.clear_param_store()

        progress_bar = trange(self._config.num_epochs)
        for epoch in progress_bar:
            loss = self._svi.step(x_train, y_train)
            progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

        self.predictive = Predictive(
            self,
            guide=self._mean_field_guide,
            num_samples=num_samples
        )

    def _get_mcmc_predictive_distribution(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            num_samples: int
    ):
        nuts_kernel = NUTS(self, jit_compile=False)
        self._mcmc = MCMC(nuts_kernel, num_samples=num_samples)
        self._mcmc.run(x_train, y_train)

        self.predictive = Predictive(
            model=self,
            posterior_samples=self._mcmc.get_samples()
        )


    def infer(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            num_samples: int,
            inference_method: str
    ) -> tuple[np.ndarray[Any], np.ndarray[Any]]:
        assert inference_method.lower() in self._generate_predictive_distribution.keys(), (
            "Please ensure the specified inference method to one of ('mcmc', 'variational')."
        )
        self._generate_predictive_distribution[inference_method](
            x_train, y_train, num_samples
        )

        predictions = self.predictive(x_test)
        return (
            predictions['obs'].T.detach().numpy().mean(axis=1),
            predictions['obs'].T.detach().numpy().std(axis=1)
        )







