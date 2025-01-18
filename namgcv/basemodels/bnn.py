from __future__ import annotations

from typing import Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import Array

from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam

from tqdm.auto import trange

import logging


logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
nl = "\n"


class BayesianNN:
    """
    Bayesian Neural Network (BNN) model class using NumPyro.

    This class implements a Bayesian Neural Network (BNN) model using NumPyro.
    The BNN comprises a series of linear layers with optional activation functions,
    and various normalization layers.
    """

    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 1,
        config: DefaultBayesianNNConfig = DefaultBayesianNNConfig(),
        model_name: str = "bayesian_nn",
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

        self.model_name = model_name
        self._config = config

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

        activation_functions = {
            'relu': jax.nn.relu,
            'tanh': jnp.tanh,
            'sigmoid': jax.nn.sigmoid,
            'softplus': jax.nn.softplus,
            'leaky_relu': jax.nn.leaky_relu,
            'elu': jax.nn.elu,
            'glu': self._glu_activation,
            'selu': jax.nn.selu,
            'gelu': jax.nn.gelu,
        }
        assert self._config.activation.lower() in activation_functions, (
            f"Please ensure that the activation function specified in the model configuration "
            f"file is one of: {list(activation_functions.keys())}"
        )

        if isinstance(self._config.activation, str):
            self._activation_fn = activation_functions[self._config.activation.lower()]
        else:  # Treat as callable.
            self._activation_fn = self._config.activation

        self.predictive = None
        self._logger.info("Bayesian NN successfully initialized.")

    @staticmethod
    def _glu_activation(x):
        """
        Gated Linear Unit (GLU) activation function.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor.

        Returns
        -------
        z : jnp.ndarray
            Output tensor.
        """

        a, b = jnp.split(x, 2, axis=-1)
        return a * jax.nn.sigmoid(b)

    def batch_norm(
            self,
            name: str,
            x: jnp.ndarray,
            eps: float=1e-5
    ):
        """
        Implements Batch Normalization.

        Parameters
        ----------
        name : str
            Unique name for the batch normalization layer.
        x : jnp.ndarray
            Input tensor.
        eps : float
            Small epsilon value to avoid division by zero.

        Returns
        -------
        normalized_x : jnp.ndarray
            Batch-normalized tensor.
        """

        axis = 0  # Normalize over the batch dimension.
        mean = jnp.mean(x, axis=axis, keepdims=True)
        var = jnp.var(x, axis=axis, keepdims=True)

        gamma = numpyro.param(f"{name}_batch_norm_gamma", jnp.ones(x.shape[1]))
        beta = numpyro.param(f"{name}_batch_norm_beta", jnp.zeros(x.shape[1]))

        normalized_x = gamma * (x - mean) / jnp.sqrt(var + eps) + beta
        return normalized_x

    def layer_norm(
            self,
            name: str,
            x: jnp.ndarray,
            eps: float=1e-5
    ):
        """
        Implements Layer Normalization.

        Parameters
        ----------
        name : str
            Unique name for the layer normalization layer.
        x : jnp.ndarray
            Input tensor.
        eps : float
            Small epsilon value to avoid division by zero.

        Returns
        -------
        normalized_x : jnp.ndarray
            Layer-normalized tensor.
        """

        axis = -1  # Normalize over the layer dimension.
        mean = jnp.mean(x, axis=axis, keepdims=True)
        var = jnp.var(x, axis=axis, keepdims=True)

        gamma = numpyro.param(f"{name}_layer_norm_gamma", jnp.ones(x.shape[-1]))
        beta = numpyro.param(f"{name}_layer_norm_beta", jnp.zeros(x.shape[-1]))

        normalized_x = gamma * (x - mean) / jnp.sqrt(var + eps) + beta
        return normalized_x

    def prior(
            self,
            layer_index: int,
            input_dim: int,
            output_dim: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Define the prior distributions for weights and biases of a given layer,
        allowing the user to choose (via flags) whether weights/biases use
        an isotropic Gaussian prior or a correlated multivariate Gaussian prior.

        Parameters
        ----------
        layer_index : int
            Index of the current layer.
        input_dim : int
            Input dimension of the layer.
        output_dim : int
            Output dimension of the layer.

        Returns
        -------
        w : jnp.ndarray
            Sampled weights from either an isotropic or correlated Gaussian prior.
        b : jnp.ndarray
            Sampled biases from either an isotropic or correlated Gaussian prior.
        """

        # Weights Prior.
        if self._config.use_correlated_weights:
            weight_dim = input_dim * output_dim

            # 1) Sample a Cholesky factor of the correlation matrix using LKJ.
            chol_corr_w = numpyro.sample(
                name=f"{self.model_name}_w{layer_index}_chol_corr",
                fn=dist.LKJCholesky(
                    dimension=weight_dim,
                    concentration=self._config.lkj_concentration
                )
            )

            # 2) Sample per-dimension std dev for the weight vector.
            w_std = numpyro.sample(
                name=f"{self.model_name}_w{layer_index}_std",
                fn=dist.HalfNormal(
                    self._config.w_layer_scale_half_normal_hyperscale
                ).expand([weight_dim])
            )

            # 3) Build the covariance via the Cholesky factor:
            # scale_tril_w = diag(w_std) @ chol_corr_w
            scale_tril_w = jnp.matmul(jnp.diag(w_std), chol_corr_w)

            # 4) Sample from MultivariateNormal and reshape to (input_dim, output_dim).
            w_loc = self._config.gaussian_prior_location * jnp.ones(weight_dim)
            w_vector = numpyro.sample(
                name=f"{self.model_name}_w{layer_index}",
                fn=dist.MultivariateNormal(loc=w_loc, scale_tril=scale_tril_w)
            )
            w = w_vector.reshape((input_dim, output_dim))

        else: # Isotropic Weights.
            # (Original style: Normal(µ, σ^2 I) with dimension [input_dim, output_dim])
            # Optionally: sample a layer-specific scale, or just fix it.
            w_layer_scale = numpyro.sample(
                name=f"{self.model_name}_w{layer_index}_scale",
                fn=dist.HalfNormal(
                    scale=self._config.w_layer_scale_half_normal_hyperscale
                )
            )
            w = numpyro.sample(
                name=f"{self.model_name}_w{layer_index}",
                fn=dist.Normal(
                    loc=self._config.gaussian_prior_location,
                    scale=w_layer_scale * jnp.sqrt(2.0 / input_dim)
                ).expand([input_dim, output_dim]).to_event(2)
            )

        # Biases Prior.
        if self._config.use_correlated_biases:
            b_dim = output_dim

            # 1) Cholesky factor of correlation for the bias vector.
            chol_corr_b = numpyro.sample(
                name=f"{self.model_name}_b{layer_index}_chol_corr",
                fn=dist.LKJCholesky(
                    dimension=b_dim,
                    concentration=self._config.lkj_concentration
                )
            )

            # 2) Per-dimension std dev for the bias vector.
            b_std = numpyro.sample(
                name=f"{self.model_name}_b{layer_index}_std",
                fn=dist.HalfNormal(
                    self._config.b_layer_scale_half_normal_hyperscale
                ).expand([b_dim])
            )

            # 3) Full Cholesky factor of covariance: scale_tril_b.
            scale_tril_b = jnp.matmul(jnp.diag(b_std), chol_corr_b)

            # 4) Sample from MultivariateNormal.
            b_loc = self._config.gaussian_prior_location * jnp.ones(b_dim)
            b = numpyro.sample(
                name=f"{self.model_name}_b{layer_index}",
                fn=dist.MultivariateNormal(loc=b_loc, scale_tril=scale_tril_b)
            )

        else:  # Isotropic Biases.
            b_layer_scale = numpyro.sample(
                name=f"{self.model_name}_b{layer_index}_scale",
                fn=dist.HalfNormal(
                    scale=self._config.b_layer_scale_half_normal_hyperscale
                )
            )
            b = numpyro.sample(
                name=f"{self.model_name}_b{layer_index}",
                fn=dist.Normal(
                    loc=self._config.gaussian_prior_location,
                    scale=b_layer_scale
                ).expand([output_dim]).to_event(1)
            )

        return w, b

    def model(
            self,
            x: jnp.ndarray,
            y: jnp.ndarray=None,
            is_training: bool=True
    ):
        """
        Define the probabilistic model of the Bayesian Neural Network.

        Parameters
        ----------
        x : jnp.ndarray
            Input features tensor.
        y : jnp.ndarray
            True response tensor (optional). This is only required during training.
        is_training : bool
            Flag indicating whether the model is in training mode.

        Returns
        -------
        output : jnp.ndarray
            Output of the network.
        """

        x = x.reshape(-1, 1) if x.ndim == 1 else x
        num_layers = len(self._layer_sizes) - 1

        z = x
        for i in range(num_layers):
            w, b = self.prior(
                layer_index=i,
                input_dim=self._layer_sizes[i],
                output_dim=self._layer_sizes[i + 1]
            )

            z = jnp.dot(z, w) + b

            # Apply BatchNorm or LayerNorm if specified.
            if i < num_layers - 1:
                if self._config.batch_norm:
                    z = self.batch_norm(name=f"batch_norm_{i}", x=z)
                if self._config.layer_norm:
                    z = self.layer_norm(name=f"layer_norm_{i}", x=z)

                z = self._activation_fn(z)

                # Dropout (implemented as a stochastic mask during training).
                if self._config.dropout > 0.0 and is_training:
                    dropout_rate = self._config.dropout
                    rng_key = numpyro.prng_key()
                    dropout_mask = random.bernoulli(rng_key, p=1 - dropout_rate, shape=z.shape)
                    z = z * dropout_mask / (1 - dropout_rate)

            else: # Output layer.
                z = self._activation_fn(z)

        output = z.squeeze()

        # sigma = numpyro.sample(
        #     name=f"{self.model_name}_sigma",
        #     fn=dist.InverseGamma(
        #         self._config.inv_gamma_prior_shape,
        #         self._config.inv_gamma_prior_scale
        #     ),
        # )

        return output

    def _get_svi_predictive_distribution(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        num_samples: int
    ):
        """
        Optimize the model using Stochastic Variational Inference (SVI) and store the
        predictive distribution.

        Parameters
        ----------
        x_train : jnp.ndarray
            Input features tensor.
        y_train : jnp.ndarray
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        """

        rng_key = random.PRNGKey(0)
        optimizer = Adam(
            {
                "step_size": self._config.lr,
                "weight_decay": self._config.weight_decay,
            }
        )

        guide = AutoDiagonalNormal(self.model)

        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(rng_key, x_train, y_train)

        progress_bar = trange(self._config.num_epochs)
        for epoch in progress_bar:
            svi_state, loss = svi.update(svi_state, x_train, y_train)
            progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

        params = svi.get_params(svi_state)
        predictive = Predictive(
            self.model, guide=guide, params=params, num_samples=num_samples
        )
        self.predictive = predictive

    def _get_mcmc_predictive_distribution(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        num_samples: int,
        kernel: str = "NUTS"
    ):
        """
        Optimize the model using Markov Chain Monte Carlo (MCMC) sampling
        via the No U-Turn Sampler (NUTS) and store the predictive distribution.

        Parameters
        ----------
        x_train : jnp.ndarray
            Input features tensor.
        y_train : jnp.ndarray
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        """

        rng_key = random.PRNGKey(0)
        assert kernel.lower() in ["nuts", "sgld"], (
            "Please ensure that the specified kernel is one of ('nuts', 'sgld')."
        )

        if kernel.lower() == "nuts":
            nuts_kernel = NUTS(
                model=self.model,
                step_size=self._config.mcmc_step_size,
                adapt_step_size=True,
                adapt_mass_matrix=True,
                dense_mass=False,
                target_accept_prob=0.8,
            )
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                num_warmup=num_samples//2
            )
            mcmc.run(rng_key, x_train, y_train)
            mcmc.print_summary()
            self.posterior_samples = mcmc.get_samples()
        else:
            pass
            #
            # from jax_sgmc import potential
            # from jax_sgmc.data.numpy_loader import NumpyDataLoader
            # from jax_sgmc import alias
            #
            # data_loader = NumpyDataLoader(x=x_train, y=y_train)
            #
            # potential_fn = potential.minibatch_potential(
            #     prior=self.prior,
            #     likelihood=self.likelihood,
            #     strategy="vmap"
            # )
            # sghmc = alias.sghmc(potential_fn,
            #                     data_loader,
            #                     cache_size=512,
            #                     batch_size=2,
            #                     burn_in=5000,
            #                     accepted_samples=1000,
            #                     integration_steps=5,
            #                     adapt_noise_model=False,
            #                     friction=0.9,
            #                     first_step_size=0.01,
            #                     last_step_size=0.00015,
            #                     diagonal_noise=False)
            # iterations = 3000
            # init_sample = {"w": jnp.zeros((N, 1)), "log_sigma": jnp.array(2.5)}
            # # Run the solver
            # results = sghmc(init_sample, iterations=iterations)
            # # Access the obtained samples from the first Markov chain
            # results = results[0]["samples"]["variables"]

        self.predictive = Predictive(
            self.model,
            posterior_samples=self.posterior_samples
        )

    def train_model(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        num_samples: int,
        inference_method: str,
    ):
        """
        Train the Bayesian Neural Network model using the specified inference method.

        Parameters
        ----------
        x_train : jnp.ndarray
            Input features tensor.
        y_train: jnp.ndarray
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.
        inference_method : str
            Inference method to use for training the model. Must be one of 'svi' or 'mcmc'.
        """

        inference_methods = {
            "svi": self._get_svi_predictive_distribution,
            "mcmc": self._get_mcmc_predictive_distribution,
        }

        if inference_method.lower() not in inference_methods:
            raise ValueError(
                "Please ensure the specified inference method is one of ('mcmc', 'svi'). "
                "Note: The inference method is case-insensitive."
            )

        inference_methods[inference_method.lower()](x_train, y_train, num_samples)

    def predict(
            self,
            x_test: jnp.ndarray
    ) -> tuple[Array, Array] | Array:
        """
        Make predictions using the trained Bayesian Neural Network.

        Parameters
        ----------
        x_test : jnp.ndarray
            Input features tensor for testing.

        Returns
        -------
        mean_pred : np.ndarray
            Mean predictions.
        std_pred : np.ndarray
            Standard deviation of predictions.
        """

        rng_key = random.PRNGKey(1)
        predictions = self.predictive(rng_key, x_test)

        return predictions[f"{self.model_name}_obs"]  # Shape: [num_samples, batch_size]
