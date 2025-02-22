from __future__ import annotations

import logging
from itertools import combinations
from typing import Tuple, Any, Dict

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn
from flax.core.nn import dropout
from flax.linen import Module
import optax

import numpy as np
import numpyro
import numpyro.distributions as dist
from flax.training.train_state import TrainState
from jax import Array
from mile.inference.metrics import RegressionMetrics, MetricsStore
from mile.training.trainer import compute_metrics_regr
from numpy import ndarray, dtype
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value, init_to_uniform

from tqdm import tqdm, trange

from namgcv.basemodels.bnn import BayesianNN
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import seaborn as sns
import matplotlib.pyplot as plt

from namgcv.data_utils.training_utils import (
    single_train_step_wrapper, single_prediction_wrapper, _early_stop_check
)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
nl = "\n"
tab = "\t"

def link_location(x: jnp.ndarray):
    """
    Link function which ensures the location parameter is unconstrained.

    Parameters
    ----------
    x: jnp.ndarray
        output_sum_over_subnetworks[:, 0]

    Returns
    -------
    jnp.ndarray:
        The input as is.
    """
    return x


def link_scale(x: jnp.ndarray):
    """
    Link function which ensures the scale parameter is positive.

    Parameters
    ----------
    x: jnp.ndarray
        output_sum_over_subnetworks[:, 1]

    Returns
    -------
    jnp.ndarray:
        The softplus transformation of the input.
    """
    return jax.nn.softplus(x)


def link_shape(x: jnp.ndarray):
    """
    Link function which ensures the shape parameter is unconstrained.

    Parameters
    ----------
    x: jnp.ndarray
        output_sum_over_subnetworks[:, 2]

    Returns
    -------
    jnp.ndarray:
        The input as is.
    """
    return x


class BayesianNAM:
    """
    Bayesian Neural Additive Model (BNAM) class using NumPyro.

    This class implements a Bayesian Neural Additive Model (BNAM) using NumPyro.
    The model supports both numerical and categorical features as well as interaction terms.
    The BNAM comprises a series of Bayesian Neural Networks (BNNs) for each feature,
    and each interaction term, if specified.
    """

    def __init__(
        self,
        cat_feature_info: Dict[str, dict],
        num_feature_info: Dict[str, dict],
        num_classes: int = 1,
        config: DefaultBayesianNAMConfig = DefaultBayesianNAMConfig(),
        subnetwork_config: DefaultBayesianNNConfig = DefaultBayesianNNConfig(),
        link_location: Any = link_location,
        link_scale: Any = link_scale,
        link_shape: Any = link_shape,
        **kwargs,
    ):
        """
        Initialize the Bayesian Neural Additive Model (BNAM).

        Parameters
        ----------
        cat_feature_info : Dict[str, dict]
            Information about categorical features.
        num_feature_info : Dict[str, dict]
            Information about numerical features.
        num_classes : int
            Number of classes in the target variable.
        config : DefaultBayesianNAMConfig
            Configuration dataclass containing model hyperparameters.
        subnetwork_config : DefaultBayesianNNConfig
            Configuration dataclass containing subnetwork hyperparameters.
        link_location : Any
            Link function for the location parameter.
        link_scale : Any
            Link function for the scale parameter.
        link_shape : Any
            Link function for the shape parameter.
        kwargs : Any
            Additional keyword arguments specifying the hyperparameters of the parent model.
        """

        self._logger = logging.getLogger(__name__)

        self._config = config

        self._cat_feature_info = cat_feature_info
        self._num_feature_info = num_feature_info

        self._subnetwork_config = subnetwork_config

        self._num_feature_networks = {}
        for feature_name, feature_info in num_feature_info.items():
            self._num_feature_networks[feature_name] = BayesianNN(
                in_dim=feature_info["input_dim"],
                out_dim=feature_info["output_dim"],
                config=self._subnetwork_config,
                model_name=f"{feature_name}_num_subnetwork",
            )

        self._cat_feature_networks = {}
        for feature_name, feature_info in cat_feature_info.items():
            self._cat_feature_networks[feature_name] = BayesianNN(
                in_dim=feature_info["input_dim"],
                out_dim=feature_info["output_dim"],
                config=self._subnetwork_config,
                model_name=f"{feature_name}_cat_subnetwork",
            )

        self._interaction_networks = {}
        if (
                self._config.interaction_degree is not None
                and
                self._config.interaction_degree >= 2
        ):
            self._create_interaction_subnetworks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

        self._lnk_fns = [link_location, link_scale, link_shape]

        self.predictive = None
        self.posterior_samples = None
        self._mcmc = None

        self.keys = jax.random.split(
            jax.random.PRNGKey(42),
            num=self._config.num_chains
        ) if self._config.num_chains > 1 else jax.random.PRNGKey(42)

        self._model_initialized = True
        self._logger.info(
            f"{nl}"
            f"+---------------------------------------+{nl}"
            f"| Bayesian NAM successfully initialized.|{nl}"
            f"+---------------------------------------+{nl}"
        )
        self.display_model_architecture()

    def display_model_architecture(self):
        """
        Display the architecture of the Bayesian Neural Additive Model (BNAM), comprising
        the subnetworks for numerical, categorical, and interaction features.
        This method requires the subnetworks to be initialized.

        Raises
        ------
        AssertionError:
            If the model has not yet been initialized.
        """

        assert self._model_initialized, "Model has not yet been initialized."

        for network_type, network_dict in zip(
                ["numerical", "categorical", "interaction"],
                [self._num_feature_networks, self._cat_feature_networks,
                 self._interaction_networks],
        ):
            for sub_network_name, sub_network in network_dict.items():
                num_layers = len(sub_network._layer_sizes) - 1
                architecture_info = ""
                for i in range(num_layers):
                    architecture_info += (
                        f"Layer {i}: "
                        f"Linear({sub_network._layer_sizes[i]} "
                        f"-> "
                        f"{sub_network._layer_sizes[i + 1]}) {nl}"
                    )
                    if i < num_layers - 1:  # Not the last layer
                        architecture_info += \
                            f"{tab}Activation: {sub_network._config.activation} {nl}"
                        if sub_network._config.batch_norm:
                            architecture_info += \
                                f"{tab}BatchNorm {nl}"
                        if sub_network._config.layer_norm:
                            architecture_info += \
                                f"{tab}LayerNorm {nl}"
                        if sub_network._config.dropout > 0.0:
                            architecture_info += \
                                f"{tab}Dropout(p={sub_network._config.dropout}) {nl}"

                self._logger.info(
                    f"{network_type.capitalize()} feature network: {sub_network_name}{nl}"
                    f"Network architecture:{nl}"
                    f"{architecture_info}"
                )

    def _create_interaction_subnetworks(
        self,
        num_feature_info: dict,
        cat_feature_info: dict,
    ):
        """
        Create Bayesian Neural Networks for modeling feature interactions.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        """
        interaction_output_dim = num_feature_info[
            list(num_feature_info.keys())[0]
        ]["output_dim"]  # Same output dimension as the numerical features.
        all_feature_names = list(num_feature_info.keys()) + list(cat_feature_info.keys())

        for degree in range(2, self._config.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                input_dim = 0
                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature]["input_dim"]
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["input_dim"]

                interaction_name = ":".join(interaction)
                self._interaction_networks[interaction_name] = BayesianNN(
                    in_dim=input_dim,
                    out_dim=interaction_output_dim,
                    config=self._subnetwork_config,
                    model_name=f"{interaction_name}_int_subnetwork",
                )

    def likelihood(
            self,
            output_sum_over_subnetworks: jnp.ndarray,
            y: jnp.ndarray,
    ):
        """
        Define the likelihood function.

        Parameters
        ----------
        output_sum_over_subnetworks : jnp.ndarray
            Predictions from the sub models.
        y : jnp.ndarray
            Observed data.

        Returns
        -------
        """

        # TODO: Enable other kinds of distributions in the likelihood.
        with numpyro.plate(name="data", size=output_sum_over_subnetworks.shape[0]):
            numpyro.sample(
                name="obs",
                fn=dist.Normal(
                    loc=output_sum_over_subnetworks[..., 0],
                    scale=output_sum_over_subnetworks[..., 1]
                ),
                obs=y,
            )


    def model(
            self,
            num_features: Dict[str, jnp.ndarray],
            cat_features: Dict[str, jnp.ndarray],
            target: jnp.ndarray = None,
            is_training: bool = True,
            permute_params: bool = False,
    ):
        """
        Method to define the Bayesian Neural Additive Model (BNAM) model in NumPyro.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target :
            True response tensor. (Default value = None)
        is_training : bool
            Flag to indicate whether the model is in training mode. (Default value = True)
        """

        subnet_out_means = {}
        subnet_out_contributions = []
        for subnetwork_type, sub_network_group, feature_group in zip(
                ["numerical", "categorical"],
                [self._num_feature_networks, self._cat_feature_networks],
                [num_features, cat_features]
        ):
            for feature_name, feature_network in sub_network_group.items():
                x = feature_group[feature_name]
                # We use numpyro.handlers.scope to isolate parameter names.
                with handlers.scope(prefix=feature_name):
                    subnet_out_i = feature_network.model(
                        x,
                        y=target,
                        is_training=is_training,
                        permute_params=permute_params
                    )

                subnet_out_means[feature_name] = jnp.mean(subnet_out_i, axis=0)
                subnet_out_contributions.append(subnet_out_i)

        if self._config.interaction_degree is not None and self._config.interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, interaction_network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = jnp.concatenate(
                    [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
                    axis=-1
                )
                with handlers.scope(prefix=interaction_name):
                    subnet_out_i = interaction_network.model(
                        x,
                        y=target,
                        is_training=is_training,
                        permute_params=permute_params
                    )

                subnet_out_means[interaction_name] = jnp.mean(subnet_out_i, axis=0)
                subnet_out_contributions.append(subnet_out_i)

        # Feature dropout (implemented as a stochastic mask during training).
        # After stacking shape: [batch_size, sub_network_out_dim, num_sub_networks].
        subnet_out_contributions = jnp.stack(subnet_out_contributions, axis=-1)
        if self._config.feature_dropout > 0.0 and is_training:
            rng_key = numpyro.prng_key()
            dropout_mask = random.bernoulli(
                rng_key,
                p=1 - self._config.feature_dropout,
                shape=subnet_out_contributions.shape
            )
            # Dropout scaling ensures the scale remains the same during training and inference.
            subnet_out_contributions = (
                    subnet_out_contributions * dropout_mask / (1 - self._config.feature_dropout)
            )

        # Address global unidentifiability by subtracting the global offset from each subnet.
        global_offset = jnp.mean(
            jnp.stack(list(subnet_out_means.values()), axis=-1),
            axis=-1
        )
        for idx, feature_name in enumerate(subnet_out_means.keys()):
            subnet_out_contributions.at[..., idx].set(
                subnet_out_contributions[..., idx] - global_offset
            )
            numpyro.deterministic(
                name=f"contrib_{feature_name}",
                value=subnet_out_contributions[..., idx]
            )

        # After sum shape: [batch_size, sub_network_out_dim].
        output_sum_over_subnetworks = jnp.sum(subnet_out_contributions, axis=-1)

        # Since we subtracted a global offset from each subnetwork,
        # we must add it back via an intercept. Because there are K outputs,
        # we now sample K intercepts (e.g. one for location and one for scale).
        if self._config.intercept:  # Global intercept term.
            # Sample a global intercept vector of shape (K,), independent of batch size.
            intercept = numpyro.sample(
                name="intercept",
                fn=dist.Normal(
                    loc=self._config.intercept_prior_shape,
                    scale=self._config.intercept_prior_scale
                ).expand((output_sum_over_subnetworks.shape[1],)).to_event(1)
            )
            # Broadcast the intercept to match the current batch (test or training)
            intercept = jnp.broadcast_to(
                intercept,
                shape=(
                    output_sum_over_subnetworks.shape[0],
                    output_sum_over_subnetworks.shape[1]
                )
            )
            # Add back num_networks * global_offset to the intercept.
            intercept = intercept + global_offset * subnet_out_contributions.shape[-1]
            output_sum_over_subnetworks += intercept

        # Apply the k desired link function(s) dimension-wise.
        final_params = self._apply_links(
            params=output_sum_over_subnetworks
        )  # shape [batch_size, K]
        numpyro.deterministic(name=f"final_params", value=final_params)

        self.likelihood(
            output_sum_over_subnetworks=final_params,
            y=target
        )

    def _apply_links(
            self,
            params: jnp.ndarray
    ):
        """
        Apply the link functions to the parameters.

        Parameters
        ----------
        params: jnp.ndarray
            The parameters to which the link functions will be applied, of shape [batch_size, K].

        Returns
        -------
        jnp.ndarray:
            The transformed parameters.
        """

        results = []
        for k in range(params.shape[1]):
            results.append(self._lnk_fns[k](params[:, k]))
        return jnp.stack(results, axis=-1)

    def train_model(
        self,
        num_features: Dict[str, jnp.ndarray],
        cat_features: Dict[str, jnp.ndarray],
        target: jnp.ndarray,
    ):
        """
        Optimize the model using Markov Chain Monte Carlo (MCMC) sampling
        via the No U-Turn Sampler (NUTS) and store the predictive distribution.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target : jnp.ndarray
            True response tensor.
        """
        if getattr(self._config, "use_deep_ensemble", False):
            self._logger.info(
                "Deep ensemble initialization enabled. Training deep ensemble..."
            )

            ensemble_params = self.train_deep_ensemble(
                num_features={
                    "train": {
                        k: v[:int(self._config.train_val_split_ratio * v.shape[0])]
                        for k, v in num_features.items()
                    },
                    "val": {
                        k: v[int(self._config.train_val_split_ratio * v.shape[0]):]
                        for k, v in num_features.items()
                    }
                },
                cat_features={
                    "train": {
                        k: v[:int(self._config.train_val_split_ratio * v.shape[0])]
                        for k, v in cat_features.items()
                    },
                    "val": {
                        k: v[int(self._config.train_val_split_ratio * v.shape[0]):]
                        for k, v in cat_features.items()
                    }
                },
                target={
                    "train": target[:int(self._config.train_val_split_ratio * target.shape[0])],
                    "val": target[int(self._config.train_val_split_ratio * target.shape[0]):]
                }
            )
            # Create an initialization strategy that supplies the point estimates for each chain.
            init_strategy = init_to_value(ensemble_params)
        else:
            init_strategy = None

        nuts_kernel = NUTS(
            model=self.model,
            init_strategy=init_strategy if init_strategy is not None else init_to_uniform,
            step_size=self._config.mcmc_step_size,
            target_accept_prob=self._config.target_accept_prob
        )

        self._mcmc = MCMC(
            nuts_kernel,
            num_samples=self._config.num_samples,
            num_warmup=self._config.num_samples,
            num_chains=self._config.num_chains
        )
        self._mcmc.run(self.keys, num_features, cat_features, target)
        self.posterior_samples = self._mcmc.get_samples()

        self.predictive = Predictive(
            self.model,
            posterior_samples=self.posterior_samples
        )

    def get_single_input(
            self,
            num_features: Dict[str, jnp.ndarray] = None,
            cat_features: Dict[str, jnp.ndarray] = None,
            batch_size: int = 1
    ):
        """
        Helper function which returns a random input data of shape (batch_size, ...).

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features:
            Dictionary of categorical features with feature names as keys.
        batch_size : int
            The desired batch size for the input data.

        Returns
        -------
        jnp.ndarray
            A random input data of shape (batch_size, ...).
        """

        def index_generator(
                feature_dict: Dict[str, jnp.ndarray],
                batch_size: int = 1
        ):
            """
            Generator function that yields batches of indices 0, 1, ..., n-1,
            where n is the length of the first array in the feature dictionary.

            Parameters
            ----------
            feature_dict:
                The feature dictionary.
            batch_size:
                The batch size.

            Returns
            -------
            Generator
                A generator yielding the indices.
            """

            first_array = next(iter(feature_dict.values()))
            n = len(first_array)
            for i in range(0, n, batch_size):
                yield list(range(i, min(i + batch_size, n)))

        return {
            "num_features": {
                k: jnp.concatenate(
                    arrays=[
                        v[gen_idx].reshape(-1, 1)
                        for gen_idx in next(index_generator(num_features, batch_size))
                    ], axis=0
                ) for k, v in num_features.items()
            } if num_features is not None else {},
            "cat_features": {
                k: jnp.concatenate(
                    arrays=[
                        v[gen_idx].reshape(-1, 1)
                        for gen_idx in next(index_generator(cat_features, batch_size))
                    ], axis=0
                ) for k, v in cat_features.items()
            } if cat_features is not None else {}
        }

    def train_deep_ensemble(
        self,
        num_features: Dict[str, Dict[str, jnp.ndarray]],
        cat_features: Dict[str, Dict[str, jnp.ndarray]],
        target: Dict[str, jnp.ndarray],
        key: jnp.ndarray = jax.random.PRNGKey(42),
        train_batch_size: int = None,
    ):
        if train_batch_size is None:
            train_batch_size = target["train"].shape[0]  # Default to full batch training.

        self._flax_module = self._get_flax_module()
        self._optimizer = optax.adamw(
            learning_rate=getattr(self._config, "de_lr", 1e-3)
        )

        for idx, step in enumerate(
                jnp.array_split(
                    jnp.arange(self._config.num_chains),
                    self._config.num_chains/jax.device_count()
                )
        ):
            num_devices = len(step)
            if num_devices > 1:  # Replicate the input for multiple devices.
                training_state = jax.pmap(
                    get_initial_state,
                    static_broadcasted_argnums=(2, 3)
                )(
                    jax.random.split(self.keys[idx], num_devices),
                    self.get_single_input(
                        num_features=num_features["train"],
                        cat_features=cat_features["train"],
                        batch_size=train_batch_size
                    )[None, ...].repeat(num_devices, axis=0),
                    self._flax_module,
                    self._optimizer,
                )
            else:
                training_state = get_initial_state(
                    rng=self.keys[idx],
                    x=self.get_single_input(
                        num_features=num_features["train"],
                        cat_features=cat_features["train"],
                        batch_size=train_batch_size
                    ),
                    module=self._flax_module,
                    optimizer=self._optimizer,
                )

            self._logger.info(
                f"Starting warm-start training for chain {idx + 1} of {self._config.num_chains}."
            )
            state, metrics = self._train_ensemble_member(
                training_state=training_state,
                num_devices=num_devices,
                num_features=num_features,
                cat_features=cat_features,
                target=target,
                batch_size=train_batch_size,
                chain_idx=idx
            )
            self._logger.info(
                f"Finished warm-start training for chain {idx + 1} of {self._config.num_chains}."
            )

        return None

    def _train_ensemble_member(
            self,
            training_state: TrainState,
            num_devices: int,
            num_features: Dict[str, Dict[str, jnp.ndarray]],
            cat_features: Dict[str, Dict[str, jnp.ndarray]],
            target: Dict[str, jnp.ndarray],
            batch_size: int,
            chain_idx: int
    ) -> tuple[TrainState | Any, MetricsStore]:
        """
        Train a single deep ensemble member (i.e. a single deterministic NAM instance)
        using a warm-start procedure.

        Parameters
        ----------
        training_state : TrainState
            The initial training state.
        num_devices : int
            The number of devices to use for parallel training.
        num_features: dict
            Nested dictionary of numerical features, containing train, validation, and test sets.
            e.g. {"train": {"feature1": jnp.ndarray, "feature2": jnp.ndarray, ...}, ...}
        cat_features: dict
            Nested dictionary of categorical features, containing train, validation, and test sets.
            e.g. {"train": {"feature1": jnp.ndarray, "feature2": jnp.ndarray, ...}, ...}
        target: dict
            Nested dictionary of target variables, containing train, validation, and test sets.
            e.g. {"train": jnp.ndarray, "val": jnp.ndarray, "test": jnp.ndarray}
        batch_size: int
            The batch size for training. Note: currently, only full-batch training is supported.
            This argument is a placeholder for future implementation of mini-batch training.
        chain_idx: int
            The index of the chain being initialized with by the current ensemble member.
        """

        _model_train_step_func = jax.pmap(single_train_step_wrapper) \
            if num_devices > 1 \
            else jax.jit(single_train_step_wrapper)
        _model_pred_func = jax.pmap(
            single_prediction_wrapper,
            static_broadcasted_argnums=("train", )
        ) \
            if num_devices > 1 \
            else jax.jit(
                single_prediction_wrapper,
                static_argnames="train"
        )

        val_losses = jnp.array([]).reshape(num_devices, 0)
        _stop_n = jnp.repeat(False, num_devices)
        metrics_train, metrics_val, metrics_test = [], [], []
        t = trange(getattr(self._config, "de_num_epochs", 1000))
        for epoch in tqdm(t, position=0, leave=True):
            # TODO:
            #  Currently, only full-batch training is supported.
            #  Implement mini-batch training.
            # --- Train ---
            if jnp.all(_stop_n):
                break  # Early stopping condition.

            batch = self.get_single_input(
                num_features=num_features["train"],
                cat_features=cat_features["train"],
                batch_size=batch_size,
            )
            # Add the target to the batch.
            batch["target"] = target["train"]

            # Perform a single training update step.
            training_state, metrics = _model_train_step_func(
                state=training_state,
                batch=batch,
                rng=self.keys[chain_idx],
                early_stop=_stop_n if num_devices > 1 else _stop_n[0]
            )
            metrics_train.append(metrics)

            # --- Validation ---
            batch = self.get_single_input(
                num_features=num_features["val"],
                cat_features=cat_features["val"],
                batch_size=batch_size
            )
            batch["target"] = target["val"]

            metrics = _model_pred_func(
                state=training_state,
                x={
                    "num_features": batch["num_features"],
                    "cat_features": batch["cat_features"]
                },
                y=batch["target"],
                early_stop=jnp.repeat(
                    False, num_devices
                ) if num_devices > 1 else jnp.bool(False),
                train=False # Validation mode.
            )
            metrics_val.append(metrics)
            if isinstance(metrics, RegressionMetrics):
                t.set_description(
                    f"Epoch {epoch}"
                    f" | "
                    f"NLL=(Train:{metrics.nlll:.3f}, Val: NLL={metrics.nlll:.3f}"
                    f" | "
                    f"RMSE=(Train:{metrics.rmse:.3f}, Val: {metrics.rmse:.3f}"
                )
                val_losses = jnp.append(
                    val_losses,
                    metrics.nlll[..., None]
                    if num_devices > 1
                    else metrics.nlll[..., None, None],
                    axis=-1,
                )
            else: # TODO: Implement logic for ClassificationMetrics
                raise NotImplemented("Classification metrics are not yet implemented.")
            _stop_n = _stop_n + _early_stop_check(
                losses=val_losses,
                patience=self._config.warm_start_early_stop_patience
            )
            # self._logger.info(f"Epoch {epoch}: Early stopping status: {_stop_n}")

        # --- Testing ---
        if "test" in target.keys():
            batch = self.get_single_input(
                num_features=num_features["test"],
                cat_features=cat_features["test"],
                batch_size=batch_size
            )
            batch["target"] = target["test"]
            metrics = _model_pred_func(
                state=training_state,
                batch=batch,
                early_stop=jnp.repeat(
                    False, num_devices
                ) if num_devices > 1 else jnp.bool(False)
            )
            metrics_test.append(metrics)
            if isinstance(metrics, RegressionMetrics):
                self._logger.info(
                    f"Epoch {epoch}: "
                    f"(Test NLL={metrics.nlll:.3f} | Test RMSE={metrics.rmse:.3f})"
                )
            else:  # TODO: Implement logic for ClassificationMetrics
                raise NotImplemented("Classification metrics are not yet implemented.")

        if isinstance(metrics, RegressionMetrics):
            complete_metrics = MetricsStore(
                train=RegressionMetrics.cstack(metrics_train),
                valid=RegressionMetrics.cstack(metrics_val),
                test=RegressionMetrics.cstack(metrics_test) if metrics_test else None
            )
        else:  # TODO: Implement logic for ClassificationMetrics
            raise NotImplemented("Classification metrics are not yet implemented.")

        return training_state, complete_metrics
        

    def _get_flax_module(self):
        """
        Method to create a Flax module for a deterministic equivalent of the
        Bayesian Neural Additive Model, replicating the model's architecture.

        Returns
        -------
        nn.Module:
            The Flax module for the BNAM.
        """

        # Build deterministic subnetworks from the instantiated BayesianNN objects.
        # We assume that each BayesianNN has attributes _layer_sizes and _config.
        num_networks = {}
        for feature_name, bayes_nn in self._num_feature_networks.items():
            num_networks[feature_name] = DeterministicMLP(
                layer_sizes=bayes_nn._layer_sizes,
                activation=bayes_nn._config.activation,
                dropout=bayes_nn._config.dropout,
                use_batch_norm=bayes_nn._config.batch_norm,
                use_layer_norm=bayes_nn._config.layer_norm,
            )

        cat_networks = {}
        for feature_name, bayes_nn in self._cat_feature_networks.items():
            cat_networks[feature_name] = DeterministicMLP(
                layer_sizes=bayes_nn._layer_sizes,
                activation=bayes_nn._config.activation,
                dropout=bayes_nn._config.dropout,
                use_batch_norm=bayes_nn._config.batch_norm,
                use_layer_norm=bayes_nn._config.layer_norm,
            )

        interaction_networks = {}
        for feature_name, bayes_nn in self._interaction_networks.items():
            interaction_networks[feature_name] = DeterministicMLP(
                layer_sizes=bayes_nn._layer_sizes,
                activation=bayes_nn._config.activation,
                dropout=bayes_nn._config.dropout,
                use_batch_norm=bayes_nn._config.batch_norm,
                use_layer_norm=bayes_nn._config.layer_norm,
            )

        # Instantiate the top-level NAM module using the above subnetworks.
        module = NAMFlaxModule(
            num_networks=num_networks,
            cat_networks=cat_networks,
            interaction_networks=interaction_networks,
            config=self._config,
            lnk_fns=self._lnk_fns,
        )

        self._logger.info("Created non-Bayesian NAM Flax module successfully.")
        return module

    def _get_posterior_param_samples(self) -> dict[str, dict[str, np.ndarray] | None]:
        """
        Method to extract the posterior samples for the scale, weight, bias, intercept and
        noise parameters.

        Returns
        -------
        dict[str, dict[str, np.ndarray] | None]: A dictionary containing the posterior samples.
        """

        if not hasattr(self, '_mcmc'):
            raise ValueError("MCMC samples not found. Please train the model using MCMC first.")
        posterior_samples = self._mcmc.get_samples()

        self._logger.info(
            f"All parameter names in the posterior: {nl}"
            f"{list(posterior_samples.keys())}"
        )

        scale_params = {
            param_name: param_vals for param_name, param_vals in posterior_samples.items()
            if "scale" in param_name
        }
        weight_params = {
            param_name: param_vals for param_name, param_vals in posterior_samples.items()
            if "_w" in param_name and "scale" not in param_name
        }
        bias_params = {
            param_name: param_vals for param_name, param_vals in posterior_samples.items()
            if "_b" in param_name and "scale" not in param_name
        }
        noise_params = {
            param_name: param_vals for param_name, param_vals in posterior_samples.items()
            if "sigma" in param_name
        }

        intercept_params = {
            param_name: param_vals for param_name, param_vals in posterior_samples.items()
            if "intercept" in param_name
        } if self._config.intercept else None

        return {
            "noise": noise_params,
            "scale": scale_params,
            "weights": weight_params,
            "biases": bias_params,
            "intercept": intercept_params
        }

    def plot_posterior_samples(self):
        """
        Diagnostic method to inspect the posterior samples for the scale parameters
        If the mean of the _scale parameter is large (away from zero), this is an indication that
        the data is pushing that layer's weight scale outward, leading to less weight shrinkage.
        Thus, if the histogram is spread wide (instead of peaked around zero), the model requires
        larger weights.
        """

        posterior_param_samples_dict = self._get_posterior_param_samples()

        sns.set_style("whitegrid", {"axes.facecolor": ".9"})
        for posterior_params_name, posterior_param_samples in posterior_param_samples_dict.items():
            if not posterior_param_samples:
                continue  # Intercept may be None.

            fig, ax = plt.subplots(
                nrows=len(posterior_param_samples),
                ncols=1,
                figsize=(12, 6*len(posterior_param_samples))
            )
            for i, (param_name, samples_array) in enumerate(posterior_param_samples.items()):
                # samples_array will be shape [num_mcmc_samples, ...]
                mean_val = jnp.mean(samples_array)
                std_val = jnp.std(samples_array)
                self._logger.info(f"{param_name}: mean={mean_val:.3f} | std={std_val:.3f}")

                ax_to_plot = ax[i] if len(posterior_param_samples) > 1 else ax
                sns.distplot(np.array(samples_array), bins=30, ax=ax_to_plot)
                ax_to_plot.set_title(f"Posterior Samples for {param_name}")
                ax_to_plot.set_xlabel("Parameter Value")
                ax_to_plot.set_ylabel("Density")

            plt.tight_layout()
            plt.show()

    def predict(
            self,
            num_features: Dict[str, jnp.ndarray],
            cat_features: Dict[str, jnp.ndarray],
            permute_params: bool = False,
            is_training: bool = False
    ) -> tuple[Any, Any, dict[str, ndarray[Any, dtype[Any]]]]:
        """
        Obtain predictions from the Bayesian Neural Additive Model (BNAM).

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            A tuple containing the predictions and submodel contributions.
        """

        rng_key = random.PRNGKey(1)
        predictions = self.predictive(
            rng_key,
            num_features,
            cat_features,
            target=None,
            permute_params=permute_params,
            is_training=is_training
        )

        pred_samples = predictions["obs"]  # Shape: [num_mcmc_samples, batch_size]
        final_params = predictions["final_params"]  # Shape: [num_mcmc_samples, batch_size, K]

        submodel_output_contributions = {}
        for feature_type, submodel_dict in zip(
                ["numerical", "categorical", "interaction"],
                [self._num_feature_networks, self._cat_feature_networks, self._interaction_networks]
        ):
            if not submodel_dict:
                continue # Interaction networks may be empty if interaction_degree < 2.

            for feature_name, submodel in submodel_dict.items():
                # Shape: [num_mcmc_samples, batch_size, sub_network_out_dim].
                contrib_samples = predictions[f"contrib_{feature_name}"]
                submodel_output_contributions[feature_name] = np.array(contrib_samples)

        return pred_samples, final_params, submodel_output_contributions


class NAMFlaxModule(nn.Module):
    num_networks: dict
    cat_networks: dict
    interaction_networks: dict
    config: Any
    lnk_fns: list

    @nn.compact
    def __call__(
            self,
            num_features: dict[str, jnp.ndarray],
            cat_features: dict[str, jnp.ndarray],
            train: bool = True,
            rng = None
    ):
        contributions = []  # To hold outputs from each subnetwork.
        subnet_means = {}  # To compute the (global) offset.

        # Process numerical feature subnetworks.
        for feature_name, network in self.num_networks.items():
            x = num_features[feature_name]
            out = network(x, train=train, rng=rng)  # shape: [batch, out_dim]
            # Compute the subnetwork "mean" over the batch.
            subnet_means[feature_name] = jnp.mean(out, axis=0)  # shape: [out_dim]
            contributions.append(out)

        # Process categorical feature subnetworks.
        for feature_name, network in self.cat_networks.items():
            x = cat_features[feature_name]
            out = network(x, train=train, rng=rng)
            subnet_means[feature_name] = jnp.mean(out, axis=0)
            contributions.append(out)

        # Process interaction subnetworks (if any).
        if self.interaction_networks:
            # Merge all features into one dictionary.
            all_features = {**num_features, **cat_features}
            for interaction_name, network in self.interaction_networks.items():
                # The interaction name is a colon-separated list of feature names.
                feature_names = interaction_name.split(":")
                # Concatenate each feature’s input along the last axis.
                x_list = [
                    jnp.expand_dims(all_features[name], axis=-1)
                    for name in feature_names
                ]
                x = jnp.concatenate(x_list, axis=-1)
                out = network(x, train=train, rng=rng)
                out = jnp.squeeze(out, axis=1) # Remove the redundant axis for batch size 1.
                subnet_means[interaction_name] = jnp.mean(out, axis=0)
                contributions.append(out)

        # Stack all subnetwork contributions.
        # Shape: [batch, out_dim, num_subnetworks]
        contributions_stack = jnp.stack(contributions, axis=-1)

        # If feature dropout is enabled during training, apply dropout.
        if self.config.feature_dropout > 0.0 and train:
            contributions_stack = nn.Dropout(
                rate=self.config.feature_dropout
            )(
                contributions_stack,
                deterministic=not train
            )

        # Compute the global offset from the subnetwork means.
        # Here we stack the per-network means (each of shape [out_dim]) and average over networks.
        global_offset = jnp.mean(
            jnp.stack(list(subnet_means.values()), axis=-1), axis=-1
        )  # shape: [out_dim]

        # Subtract the global offset from every subnetwork contribution.
        # We reshape global_offset to broadcast over batch and subnetwork dimensions.
        contributions_centered = contributions_stack - global_offset[None, :, None]

        # Sum over the subnetworks to obtain the overall output.
        output_sum = jnp.sum(contributions_centered, axis=-1)  # shape: [batch, out_dim]

        # If the model is configured to include an intercept, add it here.
        if self.config.intercept:
            out_dim = output_sum.shape[-1]
            # Define a global intercept parameter (of shape [out_dim]).
            intercept = self.param("intercept", nn.initializers.normal(), (out_dim,))
            # Adjust the intercept by adding back the (scaled) global offset.
            intercept = intercept + global_offset * contributions_stack.shape[-1]
            output_sum = output_sum + intercept

        # Finally, apply each of the link functions to the corresponding output dimension.
        outputs = []
        for i, fn in enumerate(self.lnk_fns[:output_sum.shape[-1]]):
            outputs.append(fn(output_sum[..., i]))
        final_params = jnp.stack(outputs, axis=-1)
        return final_params

class DeterministicMLP(nn.Module):
    layer_sizes: list[int]
    activation: str
    dropout: float = 0.0
    use_batch_norm: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(
            self,
            x,
            train: bool = True,
            rng: jnp.ndarray = None
    ):
        # Loop over the hidden layers and output layer.
        for i, size in enumerate(self.layer_sizes[1:]):
            x = nn.Dense(features=size)(x)
            # For all but the last layer, add activation (and optional normalization/dropout)
            if i < len(self.layer_sizes[1:]) - 1:
                if self.activation.lower() == "relu":
                    x = nn.relu(x)
                elif self.activation.lower() == "tanh":
                    x = nn.tanh(x)
                else:
                    # Default to relu if unknown
                    x = nn.relu(x)
                if self.use_batch_norm:
                    x = nn.BatchNorm(
                        use_running_average=not train
                    )(x)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                if self.dropout > 0.0:
                    x = nn.Dropout(
                        rate=self.dropout,
                    )(
                        x,
                        deterministic=not train,
                        rng=rng
                    )
        return x

def get_initial_state(
        rng: jnp.ndarray,
        x: jnp.ndarray,
        module: nn.Module,
        optimizer: optax.GradientTransformation,
) -> TrainState:
    """
    Get the initial flax training state.

    Parameters
    ----------
    rng : jnp.ndarray
        Random number generator key.
    x : jnp.ndarray
        Input data.
    module : nn.Module
        Flax module.
    optimizer : optax.GradientTransformation
        Optimizer.

    Returns
    -------
    TrainState:
        The initial training state.
    """

    rng, dropout_rng = jax.random.split(rng)
    rng, batch_norm_rng = jax.random.split(rng)
    rng, layer_norm_rng = jax.random.split(rng)
    rng, params_rng = jax.random.split(rng)
    params = module.init(
        rngs={
            "params": params_rng,
            "dropout": dropout_rng,
            "batch_norm": batch_norm_rng,
            "layer_norm": layer_norm_rng
        },
        num_features=x["num_features"],
        cat_features=x["cat_features"],
        train=True
    )["params"]

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer
    )