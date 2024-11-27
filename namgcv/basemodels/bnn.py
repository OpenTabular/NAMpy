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
    """
    Bayesian Neural Network (BNN) model class.

    This class implements a Bayesian Neural Network (BNN) model using Pyro.
    The BNN comprises a series of linear layers with optional activation functions,
    and various normalization layers.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            config: DefaultBayesianNNConfig = DefaultBayesianNNConfig(),
            model_name: str = "bayesian_nn",
            independent_network_flag: bool = False
    ):
        """
        Initializes the Bayesian Neural Network model using the specified configuration.

        Parameters
        ----------
        in_dim : int
            Dimension of input features.
        out_dim : int
            Dimension of model output.
        config : DefaultBayesianNNConfig
            Configuration class for the Bayesian Neural Network (optional).
        model_name : str
            Name of the Bayesian Neural Network model (optional).
        """

        self._logger = logging.getLogger(__name__)

        super().__init__()

        self.model_name = model_name
        self._independent_network_flag = independent_network_flag
        self._config = config
        self._activation = self._config.activation

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
                      else [self._config.activation]
                  # Directly add the activation if pre-instantiated
                  ) if not self._config.use_glu else [PyroModule[nn.GLU]()])
                +
                (
                  [PyroModule[nn.Dropout](self._config.dropout)]
                  if self._config.dropout > 0.0 else []
                )
            ]
        )

        # Note: In contrast to pytorch tensors, which have a single .shape attribute,
        # pyro distributions have two additive shape attributes,
        # with .shape = .batch_shape + .event_shape.
        # The dimensions corresponding to batch_shape denote independent RVs,
        # whereas those corresponding to event_shape denote dependent RVs.
        # By calling .to_event(n), the n right-most dimensions of a tensor are declared dependent.
        layer_idx = 0
        for layer in self._layers:
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
        if self._independent_network_flag:
            self._generate_predictive_distribution = {
                "svi": self._get_svi_predictive_distribution,
                "mcmc": self._get_mcmc_predictive_distribution
            }

        self._logger.info(f"Bayesian NN successfully initialized.")

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Bayesian Neural Network model.

        Parameters
        ----------
        x : torch.Tensor
            Input features tensor.
        y : torch.Tensor
            True response tensor (optional). This is only required during training.

        Returns
        -------
        mu : torch.Tensor
            Output of the network.
        """

        if x.dim() == 1:  # This will not be true for interaction networks.
            x = x.reshape(-1, 1)

        for i in range(len(self._layers) - 1):  # Exclude final layer.
            x = self._layers[i](x)
        mu = self._layers[-1](x).squeeze()

        if self._independent_network_flag:
            sigma = pyro.sample(
                name=f"{self.model_name}_sigma",
                fn=dist.Gamma(
                    self._config.gamma_prior_shape,
                    self._config.gamma_prior_scale
                )
            )  # Sample the response variable noise from the specified prior.

            # Sample from Gaussian with mean estimated by NN and variance sampled from variance prior.
            # i.e., the likelihood function is:
            # P(y_i|x_i;\theta) = N(NN_{\theta}(x_i); sigma)) with \sigma ~ Gamma(loc;scale)
            with pyro.plate(name=f"{self.model_name}_data", size=x.shape[0]):
                pyro.sample(
                    name=f"{self.model_name}_obs",
                    fn=dist.Normal(loc=mu, scale=sigma * sigma),
                    obs=y  # True response observation.
                )  # .expand(x.shape[0]) is automatically executed.

        return mu

    def _get_svi_predictive_distribution(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            num_samples: int
    ):
        """
        Optimize the model using Stochastic Variational Inference (SVI) and store the
        predictive distribution.

        Parameters
        ----------
        x_train : torch.Tensor
            Input features tensor.
        y_train : torch.Tensor
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        """

        pyro.clear_param_store()
        self._mean_field_guide = AutoDiagonalNormal(self)

        self._svi = SVI(
            self,  # Pass the BNN model.
            self._mean_field_guide,
            optim=pyro.optim.Adam(
                {
                    "lr": self._config.lr,
                    "weight_decay": self._config.weight_decay
                }
            ),
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
        """
        Optimize the model using Markov Chain Monte Carlo (MCMC) sampling
        via the No U-Turn Sampler (NUTS) and store the predictive distribution.

        Parameters
        ----------
        x_train : torch.Tensor
            Input features tensor.
        y_train : torch.Tensor
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        """

        nuts_kernel = NUTS(
            model=self,
            step_size=self._config.mcmc_step_size,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            jit_compile=False
        )
        self._mcmc = MCMC(
            kernel=nuts_kernel,
            num_samples=num_samples
        )
        self._mcmc.run(x_train, y_train)

        self.predictive = Predictive(
            model=self,
            posterior_samples=self._mcmc.get_samples()
        )

    def train_model(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            num_samples: int,
            inference_method: str
    ):
        """
        Train the Bayesian Neural Network model using the specified inference method.

        Parameters
        ----------
        x_train : torch.Tensor
            Input features tensor.
        y_train: torch.Tensor
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        inference_method : str
            Inference method to use for training the model. Must be one of 'svi' or 'mcmc'.
        """

        assert inference_method.lower() in self._generate_predictive_distribution.keys(), (
            "Please ensure the specified inference method to one of ('mcmc', 'svi'). "
            "Note: The inference method is NOT case-insensitive."
        )
        self._generate_predictive_distribution[inference_method](
            x_train, y_train, num_samples
        )

    def predict(
            self,
            x_test: torch.Tensor
    ) -> tuple[np.ndarray:, np.ndarray:] | np.ndarray:
        predictions = self.predictive(x_test)
        if self._independent_network_flag:
            return (
                predictions[f"{self.model_name}_obs"].detach().numpy().mean(axis=0),
                predictions[f"{self.model_name}_obs"].detach().numpy().std(axis=0)
            )
        else:
            return predictions[f"{self.model_name}_obs"].detach().numpy()  # Shape: [num_samples, batch_size]
