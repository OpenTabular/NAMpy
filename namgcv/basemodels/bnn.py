from __future__ import annotations

from typing import Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn

from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import numpyro
import numpyro.distributions as dist

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
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
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
        self.config = config
        self._rng_key = rng_key

        assert len(self.config.hidden_layer_sizes) > 0, (
            "Please ensure that there is at least one hidden layer size defined in the "
            "model configuration file."
        )
        self.layer_sizes = [in_dim] + list(self.config.hidden_layer_sizes) + [out_dim]
        for layer in self.layer_sizes:
            assert layer > 0, (
                "Please ensure that all layers defined in the "
                "model configuration file have size > 0."
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
        assert self.config.activation.lower() in activation_functions, (
            f"Please ensure that the activation function specified in the model configuration "
            f"file is one of: {list(activation_functions.keys())}"
        )

        if isinstance(self.config.activation, str):
            self.activation_fn = activation_functions[self.config.activation.lower()]
        else:  # Treat as callable.
            self.activation_fn = self.config.activation

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

        gamma = numpyro.param(f"{name}_scale", jnp.ones(x.shape[1]))
        beta = numpyro.param(f"{name}_bias", jnp.zeros(x.shape[1]))

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

        gamma = numpyro.param(f"{name}_scale", jnp.ones(x.shape[-1]))
        beta = numpyro.param(f"{name}_bias", jnp.zeros(x.shape[-1]))

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
        if self.config.use_correlated_weights:
            weight_dim = input_dim * output_dim

            # Sample a Cholesky factor of the correlation matrix using LKJ.
            chol_corr_w = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_kernel_chol_corr",
                fn=dist.LKJCholesky(
                    dimension=weight_dim,
                    concentration=self.config.lkj_concentration
                )
            )

            # Sample per-dimension std dev for the weight vector.
            w_std = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_kernel_std",
                fn=dist.HalfNormal(
                    self.config.w_layer_scale_half_normal_hyperscale
                ).expand([weight_dim])
            )

            # Build the covariance via the Cholesky factor:
            # scale_tril_w = diag(w_std) @ chol_corr_w
            scale_tril_w = jnp.matmul(jnp.diag(w_std), chol_corr_w)

            # Sample from MultivariateNormal and reshape to (input_dim, output_dim).
            w_loc = self.config.gaussian_prior_location * jnp.ones(weight_dim)
            w_vector = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_kernel",
                fn=dist.MultivariateNormal(loc=w_loc, scale_tril=scale_tril_w)
            )
            w = w_vector.reshape((input_dim, output_dim))

        else: # Isotropic Weights.
            w_layer_scale = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_kernel_scale",
                fn=dist.HalfNormal(
                    scale=self.config.w_layer_scale_half_normal_hyperscale
                )
            )
            w = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_kernel",
                fn=dist.Normal(
                    loc=self.config.gaussian_prior_location,
                    scale=w_layer_scale * jnp.sqrt(2.0 / input_dim)
                ).expand([input_dim, output_dim]).to_event(2)
            )

        # Biases Prior.
        if self.config.use_correlated_biases:
            b_dim = output_dim

            # Cholesky factor of correlation for the bias vector.
            chol_corr_b = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_bias_chol_corr",
                fn=dist.LKJCholesky(
                    dimension=b_dim,
                    concentration=self.config.lkj_concentration
                )
            )
            # Per-dimension std dev for the bias vector.
            b_std = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_bias_std",
                fn=dist.HalfNormal(
                    self.config.b_layer_scale_half_normal_hyperscale
                ).expand([b_dim])
            )

            # Full Cholesky factor of covariance: scale_tril_b.
            scale_tril_b = jnp.matmul(jnp.diag(b_std), chol_corr_b)

            # Sample from MultivariateNormal.
            b_loc = self.config.gaussian_prior_location * jnp.ones(b_dim)
            b = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_bias",
                fn=dist.MultivariateNormal(loc=b_loc, scale_tril=scale_tril_b)
            )

        else:  # Isotropic Biases.
            b_layer_scale = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_bias_scale",
                fn=dist.HalfNormal(
                    scale=self.config.b_layer_scale_half_normal_hyperscale
                )
            )
            b = numpyro.sample(
                rng_key=self._rng_key,
                name=f"dense_{layer_index}_bias",
                fn=dist.Normal(
                    loc=self.config.gaussian_prior_location,
                    scale=b_layer_scale
                ).expand([output_dim]).to_event(1)
            )

        return w, b

    def model(
            self,
            x: jnp.ndarray,
            y: jnp.ndarray = None,
            is_training: bool = True,
    ):
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        num_layers = len(self.layer_sizes) - 1

        # --------------------
        # 1) Sample all layer weights/biases up front
        # --------------------
        Ws = []
        Bs = []
        for i in range(num_layers):
            w, b = self.prior(
                layer_index=i,
                input_dim=self.layer_sizes[i],
                output_dim=self.layer_sizes[i + 1],
            )
            Ws.append(w)
            Bs.append(b)

        z = x
        for i in range(num_layers):
            w = Ws[i]
            b = Bs[i]

            z = jnp.dot(z, w) + b

            if i < num_layers - 1:
                if self.config.batch_norm:
                    z = self.batch_norm(name=f"batch_norm_{i}", x=z)
                if self.config.layer_norm:
                    z = self.layer_norm(name=f"layer_norm_{i}", x=z)

                z = self.activation_fn(z)

                if self.config.dropout > 0.0 and is_training:
                    dropout_rate = self.config.dropout
                    mask = random.bernoulli(
                        self._rng_key,
                        p=1 - dropout_rate,
                        shape=z.shape
                    )
                    z = z * mask / (1 - dropout_rate)
            else:
                # Output layer activation (if your design uses it)
                z = self.activation_fn(z)

        return z.squeeze()

    # def model(
    #         self,
    #         x: jnp.ndarray,
    #         y: jnp.ndarray=None,
    #         is_training: bool=True,
    #         permute_params: bool=False
    # ):
    #     """
    #     Define the probabilistic model of the Bayesian Neural Network.
    #
    #     Parameters
    #     ----------
    #     x : jnp.ndarray
    #         Input features tensor.
    #     y : jnp.ndarray
    #         True response tensor (optional). This is only required during training.
    #     is_training : bool
    #         Flag indicating whether the model is in training mode.
    #
    #     Returns
    #     -------
    #     output : jnp.ndarray
    #         Output of the network.
    #     """
    #
    #     x = x.reshape(-1, 1) if x.ndim == 1 else x
    #     num_layers = len(self._layer_sizes) - 1
    #
    #     z = x
    #     for i in range(num_layers):
    #         w, b = self.prior(
    #             layer_index=i,
    #             input_dim=self._layer_sizes[i],
    #             output_dim=self._layer_sizes[i + 1]
    #         )
    #
    #         if permute_params:
    #             if is_training:
    #                 raise ValueError("Permutation of parameters is only allowed during inference.")
    #
    #             if i == 1:
    #                 # Randomly choose two hidden units to swap
    #                 key = numpyro.prng_key()
    #                 # 'output_dim' is w.shape[1], i.e. number of hidden units in this layer
    #                 j1, j2 = random.choice(key, a=w.shape[1], shape=(2,), replace=False)
    #
    #                 # --------------------------
    #                 #  1) Swap columns (j1, j2) in w
    #                 # --------------------------
    #                 w_col_j1 = w[:, j1]
    #                 w_col_j2 = w[:, j2]
    #                 w = w.at[:, j1].set(w_col_j2)
    #                 w = w.at[:, j2].set(w_col_j1)
    #
    #                 # --------------------------
    #                 #  2) Swap bias terms (j1, j2)
    #                 # --------------------------
    #                 b_j1 = b[j1]
    #                 b_j2 = b[j2]
    #                 b = b.at[j1].set(b_j2)
    #                 b = b.at[j2].set(b_j1)
    #
    #         z = jnp.dot(z, w) + b
    #
    #         # Apply BatchNorm or LayerNorm if specified.
    #         if i < num_layers - 1:
    #             if self.config.batch_norm:
    #                 z = self.batch_norm(name=f"batch_norm_{i}", x=z)
    #             if self.config.layer_norm:
    #                 z = self.layer_norm(name=f"layer_norm_{i}", x=z)
    #
    #             z = self._activation_fn(z)
    #
    #             # Dropout (implemented as a stochastic mask during training).
    #             if self.config.dropout > 0.0 and is_training:
    #                 dropout_rate = self.config.dropout
    #                 rng_key = numpyro.prng_key()
    #                 dropout_mask = random.bernoulli(rng_key, p=1 - dropout_rate, shape=z.shape)
    #                 z = z * dropout_mask / (1 - dropout_rate)
    #
    #         else: # Output layer.
    #             z = self._activation_fn(z)
    #
    #     output = z.squeeze()
    #
    #     # sigma = numpyro.sample(
    #     #     name=f"{self.model_name}_sigma",
    #     #     fn=dist.InverseGamma(
    #     #         self.config.inv_gamma_prior_shape,
    #     #         self.config.inv_gamma_prior_scale
    #     #     ),
    #     # )
    #
    #     return output


class DeterministicNN(nn.Module):
    layer_sizes: Tuple[int, ...]
    activation: Any  # Can be a string or callable.
    dropout: float
    use_batch_norm: bool
    use_layer_norm: bool
    model_name: str

    def _glu_activation(self, x):
        a, b = jnp.split(x, 2, axis=-1)
        return a * jax.nn.sigmoid(b)

    @nn.compact
    def __call__(
            self,
            x: jnp.ndarray,
            train: bool = True,
            rng: jnp.ndarray = None
    ):
        """
        Forward pass through the deterministic neural network.

        Parameters
        ----------
        x: jnp.ndarray
            Input tensor.
        train: bool
            Flag indicating whether the model is in training mode.
        rng: jnp.ndarray
            Random number generator key.

        Returns
        -------
        x: jnp.ndarray
            Output tensor.
        """

        # Define available activation functions.
        activation_functions = {
            'relu': jax.nn.relu,
            'tanh': jax.nn.tanh,
            'sigmoid': jax.nn.sigmoid,
            'softplus': jax.nn.softplus,
            'leaky_relu': jax.nn.leaky_relu,
            'elu': jax.nn.elu,
            'glu': self._glu_activation,
            'selu': jax.nn.selu,
            'gelu': jax.nn.gelu,
        }

        if isinstance(self.activation, str):
            act_name = self.activation.lower()
            assert act_name in activation_functions, (
                f"Activation function must be one of: "
                f"{list(activation_functions.keys())}"
            )
            act_fn = activation_functions[act_name]
        else:
            act_fn = self.activation

        for i, size in enumerate(self.layer_sizes[1:]):
            x = nn.Dense(features=size)(x)

            # For all but the final layer, apply normalization, activation, and dropout.
            if i < len(self.layer_sizes[1:]) - 1:
                if self.use_batch_norm:
                    x = nn.BatchNorm(use_running_average=not train)(x)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)

                x = act_fn(x)

                if self.dropout > 0.0:
                    x = nn.Dropout(
                        rate=self.dropout
                    )(x, deterministic=not train, rng=rng)

        return x
