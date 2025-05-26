from __future__ import annotations

import logging

from itertools import combinations
from pathlib import Path
import os

from typing import (
    Tuple, Any, Dict, Callable, Optional
)

from frozendict import frozendict
from jax import Array
from mile.config.data import DataConfig, Source, DatasetType, Task
from mile.training.utils import save_params, load_params_batch

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn
from flax.training.train_state import TrainState
import optax

import numpyro
import numpyro.distributions as dist
from numpy import ndarray, dtype
from numpyro import handlers
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive
)
from sklearn.model_selection import KFold

from tqdm import trange
from tqdm.auto import tqdm

from namgcv.basemodels.bnn import BayesianNN, DeterministicNN
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig
from namgcv.data_utils.training_utils import (
    single_train_step_wrapper,
    single_prediction_wrapper,
    early_stop_check,
    get_initial_state,
    get_single_input,
    map_flax_to_numpyro,
    merge_data_dicts,
)
from namgcv.data_utils.jax_dataset import TabularAdditiveModelDataLoader
from namgcv.data_utils.custom_distributions import NaturalNormal

from mile.inference.metrics import (
    RegressionMetrics,
    MetricsStore
)

import seaborn as sns
import matplotlib.pyplot as plt


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO
)
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
        The transformation of the input.
    """
    return jax.nn.softplus(
        jnp.clip(
            x,
            a_min=-8,
            a_max=8
        )
    )

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
        config: DefaultBayesianNAMConfig = DefaultBayesianNAMConfig(),
        subnetwork_config: DefaultBayesianNNConfig = DefaultBayesianNNConfig(),
        link_1: Callable = link_location,
        link_2: Callable = link_scale,
        link_3: Callable = link_shape,
        rng_key: jnp.ndarray = jax.random.PRNGKey(42),
        bayesian_sampling_flag: bool=True,
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

        self._bayesian_sampling = bayesian_sampling_flag
        self._logger = logging.getLogger(__name__)
        self.config = config

        self._cat_feature_info = cat_feature_info
        self._num_feature_info = num_feature_info

        self._subnetwork_config = subnetwork_config

        self._num_feature_networks = {}
        self._cat_feature_networks = {}
        for feature_type, feature_info_dict, networks in (
                ("num", num_feature_info, self._num_feature_networks),
                ("cat", cat_feature_info, self._cat_feature_networks),
        ):
            for feature_name, feature_info in feature_info_dict.items():
                out_dim = feature_info["output_dim"]
                networks[feature_name] = BayesianNN(
                    in_dim=feature_info["input_dim"],
                    out_dim=out_dim,
                    config=self._subnetwork_config,
                    model_name=f"{feature_name}_{feature_type}_subnetwork",
                )

        self._interaction_networks = {}
        if (
                self.config.interaction_degree is not None
                and
                self.config.interaction_degree >= 2
        ):
            self._create_interaction_subnetworks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

        self.lnk_fns = (link_1, link_2, link_3)

        self.predictive = None
        self.posterior_samples = None
        self._mcmc = None

        self._single_rng_key = rng_key
        self._chains_rng_keys = jax.random.split(
            self._single_rng_key,
            num=self.config.num_chains
        ) if self.config.num_chains > 1 else jax.random.PRNGKey(42)

        self._optimizer = None
        self._flax_module = None

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
                [
                    self._num_feature_networks,
                    self._cat_feature_networks,
                    self._interaction_networks
                ],
        ):
            for sub_network_name, sub_network in network_dict.items():
                num_layers = len(sub_network.layer_sizes) - 1
                architecture_info = ""
                for i in range(num_layers):
                    architecture_info += (
                        f"Layer {i}: "
                        f"Linear({sub_network.layer_sizes[i]} "
                        f"-> "
                        f"{sub_network.layer_sizes[i + 1]}) {nl}"
                    )
                    if i < num_layers - 1:  # Not the last layer.
                        if sub_network.config.batch_norm:
                            architecture_info += \
                                f"{tab}BatchNorm {nl}"
                        if sub_network.config.layer_norm:
                            architecture_info += \
                                f"{tab}LayerNorm {nl}"
                        architecture_info += \
                            f"{tab}Activation: {sub_network.config.activation} {nl}"
                        if sub_network.config.dropout > 0.0:
                            architecture_info += \
                                f"{tab}Dropout(p={sub_network.config.dropout}) {nl}"

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
        # TODO: Implement GAMI-Net style interaction selection.
        interaction_output_dim = num_feature_info[
            list(num_feature_info.keys())[0]
        ]["output_dim"]  # Same output dimension as the numerical features.
        all_feature_names = list(num_feature_info.keys()) + list(cat_feature_info.keys())

        for degree in range(2, self.config.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                input_dim = 0
                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature]["input_dim"]
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["input_dim"]

                interaction_name = ":".join(interaction)
                out_dim = interaction_output_dim
                self._interaction_networks[interaction_name] = BayesianNN(
                    in_dim=input_dim,
                    out_dim=out_dim,
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
        output_sum_over_subnetworks:jnp.ndarray
            Predictions from the sub models.
        y:jnp.ndarray
            Observed data.

        Returns
        -------
        """

        theta1 = output_sum_over_subnetworks[..., 0]
        if output_sum_over_subnetworks.shape[1] > 1:
            theta2 = output_sum_over_subnetworks[..., 1]
        else:  # If we are only doing mean regression, we need to sample sigma.
            theta2 = numpyro.sample(
                name="sigma",
                fn=dist.HalfNormal(
                    scale=self.config.sigma_prior_scale
                ),
                rng_key=random.split(self._single_rng_key)[1]
            )

        sampling_dist = dist.Normal(
            loc=output_sum_over_subnetworks[..., 0],
            scale=output_sum_over_subnetworks[..., 1],
        )

        with numpyro.plate(
                name="data",
                size=output_sum_over_subnetworks.shape[0]  # N
        ):
            numpyro.sample(
                name="y",
                fn=sampling_dist,
                obs=y,
                rng_key=random.split(self._single_rng_key)[1]
            )

    def model(
            self,
            data_loader: TabularAdditiveModelDataLoader,
            is_training: bool = True,
    ):
        """
        Method to define the Bayesian Neural Additive Model (BNAM) model in NumPyro.
        Note: the data loader is passed as a function argument to keep the method "pure", and
        conforming with the NumPyro functional programming paradigm.

        Parameters
        ----------
        data_loader: TabularAdditiveModelDataLoader
            The data loader object, containing:
            - num_features: Dict[str, jnp.ndarray],
            - cat_features: Dict[str, jnp.ndarray],
            - target: jnp.ndarray,
        is_training : bool
            Flag to indicate whether the model is in training mode. (Default value = True)
        """

        # Note: Currently, only full-batch training is supported (batch_size=None).
        if is_training:
            batch_iter_train = data_loader.iter(
                split="train",
                batch_size=None
            )
            batch_iter_val = data_loader.iter(
                split="valid",
                batch_size=None
            )
            batch_iter = zip(batch_iter_train, batch_iter_val)
            data_dicts = next(batch_iter)
            data_dict = merge_data_dicts(data_dicts)
        else:
            batch_iter = data_loader.iter(
                split="test",
                batch_size=None
            )
            data_dict = next(batch_iter)

        # data_dict format:
        # {
        #   "feature": {
        #       "numerical": { ... feature_name: jnp.ndarray, ... },
        #       "categorical": { ... feature_name: jnp.ndarray, ... }
        #   },
        #   "target": jnp.ndarray
        # }
        num_features = data_dict["feature"]["numerical"]
        cat_features = data_dict["feature"]["categorical"]
        target = data_dict["target"]

        subnet_out_means = {}
        subnet_out_contributions = []

        # Build a list of tasks, each containing (key, network, input data).
        networks_to_run = []
        for feature_group, network_group in zip(
                [num_features, cat_features],
                [self._num_feature_networks, self._cat_feature_networks]
        ):
            for feature_name, network in network_group.items():
                x = feature_group[feature_name]
                networks_to_run.append((feature_name, network, x))

        if self.config.interaction_degree is not None and self.config.interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = jnp.concatenate(
                    [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
                    axis=-1
                )
                networks_to_run.append((interaction_name, network, x))

        for key, network, x in networks_to_run:
            # Use a scoped handler to isolate parameter names.
            with handlers.scope(prefix=key):
                subnet_out_i = network.model(
                    x,
                    y=target,
                    is_training=is_training,
                )
                subnet_out_means[key] = jnp.mean(subnet_out_i, axis=0)
                subnet_out_contributions.append(
                    subnet_out_i - jnp.mean(subnet_out_i, axis=0)
                )  # Ensure the subnet output is centered at zero.

        # After stacking shape: [batch_size, sub_network_out_dim, num_subnetworks].
        subnet_out_contributions = jnp.stack(
            subnet_out_contributions,
            axis=-1
        )

        # Feature dropout (implemented as a stochastic mask during training).
        if self.config.feature_dropout > 0.0 and is_training:
            dropout_mask = random.bernoulli(
                random.split(self._single_rng_key)[1],
                p=1 - self.config.feature_dropout,
                shape=subnet_out_contributions.shape
            )
            # Dropout scaling ensures the scale remains the same during training and inference.
            subnet_out_contributions = (
                subnet_out_contributions * dropout_mask
            )

        for idx, (feature_name, _, _) in enumerate(networks_to_run):
            """
            eta2 = -1/2 * sigma**2 --> sigma = sqrt(-2 * eta2)
            eta1 = mu/ sigma**2 --> mu = eta1 * sigma**2 = -2 * eta1 * eta2  
            """
            # subnet_out_i = subnet_out_contributions[..., idx]
            numpyro.deterministic(
                name=f"contrib_{feature_name}",
                value=subnet_out_contributions[..., idx]
                # value=jnp.concatenate(
                #     arrays=[
                #         (-1/2 * subnet_out_i[..., 0] / subnet_out_i[..., 1]).reshape(-1, 1),
                #         (jnp.sqrt(-1.0 / (2.0 * subnet_out_i[..., 1]))).reshape(-1, 1)
                #     ], axis=-1
                # )
            )

        # After sum shape: [batch_size, sub_network_out_dim].
        output_sum_over_subnetworks = jnp.sum(
            subnet_out_contributions,
            axis=-1
        )
        if output_sum_over_subnetworks.ndim == 1:
            # Convert to DataFrame.
            output_sum_over_subnetworks = output_sum_over_subnetworks.reshape(-1, 1)

        # We now sample K intercepts (e.g. one for location and one for scale).
        if self.config.intercept:  # Global intercept term.
            # Sample a global intercept vector of shape (K,), independent of batch size.
            intercept = numpyro.sample(
                name="intercept",
                fn=dist.Normal(
                    loc=self.config.intercept_prior_shape,
                    scale=self.config.intercept_prior_scale
                ).expand((output_sum_over_subnetworks.shape[1],)).to_event(1),
                rng_key=random.split(self._single_rng_key)[1]
            )
            output_sum_over_subnetworks += intercept

        # Apply the k desired link function(s) dimension-wise.
        final_params = self._apply_links(
            params=output_sum_over_subnetworks,
        )  # shape [batch_size, K].

        numpyro.deterministic(
            name=f"final_params",
            value=final_params
        )

        self.likelihood(
            output_sum_over_subnetworks=final_params,
            y=target
        )

    def _apply_links(
            self,
            params: jnp.ndarray,
    ) -> Array:
        """
        Apply the link functions to the parameters.

        Parameters
        ----------
        params: jnp.ndarray
            The parameters to which the link functions will be applied, of shape [batch_size, K].

        Returns
        -------
        tuple[Array, Array]:
            The transformed parameters.
        """

        results = []
        for k in range(params.shape[-1]):
            results.append(self.lnk_fns[k](params[:, k]))

        return jnp.stack(results, axis=-1)

    def train_model(
        self,
        num_features: Dict[str, jnp.ndarray] = None,
        cat_features: Dict[str, jnp.ndarray] = None,
        target: jnp.ndarray = None,
        data_loader: TabularAdditiveModelDataLoader = None,
        dataset_name: str="default"
    ) -> TabularAdditiveModelDataLoader:
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
        dataset_name: str
            String specifying the dataset name.
        """

        if data_loader is None:
            if num_features is None or cat_features is None or target is None:
                raise ValueError(
                    "Please provide either an instance of TabularAdditiveModelDataLoader, "
                    "or a combination of num_features, cat_features and target as input."
                )
            data_loader = TabularAdditiveModelDataLoader(
                config=DataConfig(
                    path=dataset_name,
                    source=Source.LOCAL,
                    data_type=DatasetType.TABULAR,
                    task=Task.REGRESSION,
                    target_column="Response",
                    target_len=target.shape[0],
                    datapoint_limit=None,
                    normalize=True,
                    train_split=0.7,
                    valid_split=0.2,
                    test_split=0.1,
                ),
                rng=random.split(self._single_rng_key)[1],
                data_dict={
                    "numerical": num_features,
                    "categorical": cat_features,
                    "target": target
                },
                target_key="target"
            )
            self._logger.info(f"Data loader initialized: {nl} {data_loader}")
        else:
            if num_features is not None or cat_features is not None or target is not None:
                raise ValueError(
                    "Please provide either an instance of TabularAdditiveModelDataLoader, "
                    "or a combination of num_features, cat_features and target as input. "
                    "Not both."
                )
            if not isinstance(data_loader, TabularAdditiveModelDataLoader):
                raise ValueError(
                    f"Expected an instance of TabularAdditiveModelDataLoader. "
                    f"Received: {type(data_loader)}"
                )
            self._logger.info(f"Data loader provided: {nl} {data_loader}")

        if getattr(self.config, "use_deep_ensemble", False):
            self._logger.info(
                "Deep ensemble initialization enabled. Training deep ensemble..."
            )
            warmstart_save_dir = os.path.join(
                os.path.dirname(__file__),
                "..",
                "bnam_de_warmstart_checkpoints",
                "warmstart"
            )
            self.train_deep_ensemble(
                warmstart_checkpoint_save_dir=warmstart_save_dir,
                data_loader=data_loader
            )
            chains = [
                os.path.join(
                    warmstart_save_dir, i
                ) for i in os.listdir(warmstart_save_dir) if i.startswith('params')
            ]

            steps_to_run = jnp.array_split(
                jnp.arange(self.config.num_chains),
                self.config.num_chains / jax.device_count()
            )
            params_list = []
            for idx, step in enumerate(steps_to_run):
                self._logger.info(f"Starting sampling process for chain(s) {step}...")
                params = load_params_batch(
                    params_path=[chains[i] for i in step],
                    tree_path=os.path.join(
                        warmstart_save_dir,
                        "..",
                        "tree"
                    )
                )
                params_list.append(params)

            self._flax_module_params = params_list
        else:
            params_list = None
            self._logger.warning(
                "No warm-start path found. "
                "Sampling will be initialized with random parameters."
            )

        if self._bayesian_sampling:
            self.begin_sampling(
                params_list=params_list,
                data_loader=data_loader
            )
            self._mcmc.print_summary()

        return data_loader

    def begin_sampling(
            self,
            data_loader: TabularAdditiveModelDataLoader,
            params_list: list[dict],
            kernel: str ="NUTS"
    ):
        if kernel.lower() == "nuts":
            nuts_kernel = NUTS(
                model=self.model,
                adapt_mass_matrix=True,
                adapt_step_size=True,
                step_size=self.config.mcmc_step_size,
                find_heuristic_step_size=True,
                target_accept_prob=self.config.target_accept_prob,
            )
            self._mcmc = MCMC(
                nuts_kernel,
                num_samples=self.config.num_samples,
                num_warmup=self.config.num_warmup_samples,
                num_chains=self.config.num_chains,
            )
            init_params = map_flax_to_numpyro(
                flax_params_list=params_list,
                expected_chains=self.config.num_chains,
            ) if params_list is not None else None
            self._mcmc.run(
                jax.random.PRNGKey(42),
                data_loader=data_loader,
                is_training=True,
                init_params=init_params
            )
            self.posterior_samples = self._mcmc.get_samples()
        else:
            raise NotImplementedError(
                f"Currently, only NUTS sampling is supported."
            )

    def train_deep_ensemble(
        self,
        data_loader: TabularAdditiveModelDataLoader,
        warmstart_checkpoint_save_dir: str | Path,
        train_batch_size: int = None,
    ):
        """
        Method to train an ensemble of deterministic NAMs for the purpose of warm-starting the
        MCMC sampler for the Bayesian NAM.

        Parameters
        ----------
        data_loader : TabularAdditiveModelDataLoader
            The data loader object, containing:
            - num_features: Dict[str, jnp.ndarray],
            - cat_features: Dict[str, jnp.ndarray],
            - target: jnp.ndarray,
        train_batch_size:
            The batch size for training.
            Note: currently, only full-batch (equivalent to None) training is supported.
        warmstart_checkpoint_save_dir: str | Path
            The directory where the warm-start checkpoints will be saved.

        Returns
        -------
        TrainState:
            The final training state for the ensemble.
        """

        self._initialize_module()
        self._optimizer = optax.adamw(
            learning_rate=optax.exponential_decay(
                init_value=getattr(self.config, "de_lr", 1e-3),
                transition_steps=getattr(self.config, "de_lr_transition_steps", 100),
                decay_rate=getattr(self.config, "de_lr_decay", 0.9),
                staircase=getattr(self.config, "de_lr_staircase", True)
            )
        )

        if not os.path.exists(warmstart_checkpoint_save_dir):
            os.makedirs(warmstart_checkpoint_save_dir)

        warmstart_metrics_list: list[MetricsStore] = []
        if self.config.num_chains % jax.device_count() != 0:
            raise ValueError(
                'num_chains must be divisible by the number of devices.'
                f'{self.config.num_chains} % {jax.device_count()} != 0.'
            )

        for idx, step in enumerate(
            jnp.array_split(
                jnp.arange(self.config.num_chains),
                self.config.num_chains/jax.device_count()
            )
        ):
            num_parallel = len(step)
            if num_parallel > 1:  # Replicate the input for multiple devices.
                def replicate_leaf(x):
                    # Broadcast the single example (shape (1, ...)) to (num_parallel, ...)
                    return jnp.broadcast_to(x, (num_parallel,) + x.shape)

                # Note: In order to treat the flax module as static,
                # pmap requires the module to be hashable.
                training_state = jax.pmap(
                    get_initial_state,
                    static_broadcasted_argnums=(2, 3)
                )(
                    jax.random.split(self._chains_rng_keys[idx], num_parallel),
                    jax.tree_util.tree_map(
                        replicate_leaf,
                        get_single_input(
                            num_features=data_loader.data_train["numerical"],
                            cat_features=data_loader.data_train["categorical"],
                            batch_size=1
                        )
                    ),
                    self._flax_module,
                    self._optimizer,
                )
            else:
                training_state = get_initial_state(
                    rng=self._chains_rng_keys[idx]
                        if self.config.num_chains > 1
                        else random.split(self._single_rng_key)[1],
                    x=get_single_input(
                        num_features=data_loader.data_train["numerical"],
                        cat_features=data_loader.data_train["categorical"],
                        batch_size=1
                    ),
                    module=self._flax_module,
                    optimizer=self._optimizer,
                )

            self._logger.info(
                f"Starting warm-start training for chain(s) {step}..."
            )
            state, metrics = self._train_ensemble_member(
                training_state=training_state,
                num_parallel=num_parallel,
                batch_size=train_batch_size,
                data_loader=data_loader
            )
            self._logger.info(
                f"Finished warm-start training for chain {idx + 1} of {self.config.num_chains}."
            )
            warmstart_metrics_list.append(metrics)

            # Save Checkpoints of Deep Ensemble Members.
            for i, chain_n in enumerate(step):
                if len(step) > 1:
                    save_params(
                        dir=warmstart_checkpoint_save_dir,
                        params=jax.tree.map(lambda x: x[i], state.params),
                        idx=chain_n,
                    )
                else:
                    save_params(
                        dir=warmstart_checkpoint_save_dir,
                        params=state.params,
                        idx=chain_n
                    )
                self._logger.info(
                    f"Deep Ensemble {chain_n} saved to {warmstart_checkpoint_save_dir}"
                )

        self.warmstart_metrics_list = MetricsStore.vstack(warmstart_metrics_list)
        self._logger.info(
            f"Deep Ensemble warm-start metrics saved to {warmstart_checkpoint_save_dir}"
        )
        self.warmstart_metrics_list.save(
            path=os.path.join(
                warmstart_checkpoint_save_dir,
                "warmstart_metrics.pkl"
            ),
            save_plots=True
        )

    def _train_ensemble_member(
            self,
            training_state: TrainState,
            num_parallel: int,
            batch_size: int,
            data_loader: TabularAdditiveModelDataLoader,
    ) -> tuple[TrainState | Any, MetricsStore]:
        """
        Train a single deep ensemble member (i.e. a deterministic NAM instance)
        using a warm-start procedure.

        Parameters
        ----------
        training_state : TrainState
            The initial training state.
        num_parallel : int
            The number of devices to use for parallel training.
        batch_size: int
            The batch size for training. Note: currently, only full-batch training is supported.
            This argument is a placeholder for future implementation of mini-batch training.
        data_loader : TabularAdditiveModelDataLoader
            The data loader object, containing:
            - num_features: Dict[str, jnp.ndarray],
            - cat_features: Dict[str, jnp.ndarray],
            - target: jnp.ndarray,
        """

        _model_train_step_func = jax.pmap(single_train_step_wrapper) \
            if num_parallel > 1 \
            else jax.jit(single_train_step_wrapper)
        _model_pred_func = jax.pmap(
            single_prediction_wrapper,
            static_broadcasted_argnums=(4,)
        ) \
            if num_parallel > 1 \
            else jax.jit(
                single_prediction_wrapper,
                static_argnames="train"
        )

        val_losses = jnp.array([]).reshape(num_parallel, 0)
        _stop_n = jnp.repeat(False, num_parallel)
        metrics_train, metrics_val, metrics_test = [], [], []
        t = trange(getattr(self.config, "de_num_epochs", 1000))
        for epoch in tqdm(t, position=0, leave=True):
            if jnp.all(_stop_n):
                break  # Early stopping condition.

            # --- Train ---
            data_loader.shuffle(split="train")
            for batch in data_loader.iter(
                    split="train",
                    batch_size=batch_size,
                    n_devices=num_parallel,
            ):
                # Note: Each batch is of the form:
                # {
                #   "feature": {
                #       "numerical": {...},
                #       "categorical": {...}
                #   },
                #   "target": ...
                # }
                for feature_type in batch["feature"].keys():
                    for feature_name in batch["feature"][feature_type].keys():
                        if batch["feature"][feature_type][feature_name].ndim == 1:
                            batch["feature"][feature_type][feature_name] = \
                                batch["feature"][feature_type][feature_name].reshape(-1, 1)

                # Perform a single training update step.
                training_state, metrics = _model_train_step_func(
                    state=training_state,
                    batch=batch,
                    rng=self._chains_rng_keys
                        if num_parallel > 1
                        else random.split(self._single_rng_key)[1],
                    early_stop=_stop_n if num_parallel > 1 else _stop_n[0]
                )
                metrics_train.append(metrics)

            # --- Validation ---
            val_metric_over_batches = []
            for batch in data_loader.iter(
                    split="valid",
                    batch_size=None,  # Enforce full-batch validation.
                    n_devices=num_parallel
            ):
                for feature_type in batch["feature"].keys():
                    for feature_name in batch["feature"][feature_type].keys():
                        if batch["feature"][feature_type][feature_name].ndim == 1:
                            batch["feature"][feature_type][feature_name] = \
                                batch["feature"][feature_type][feature_name].reshape(-1, 1)

                metrics = _model_pred_func(
                    training_state,
                    batch["feature"],
                    batch["target"],
                    jnp.repeat(
                        False, num_parallel
                    ) if num_parallel > 1 else jnp.bool_(False),
                    False  # Static 'train' argument.
                )
                val_metric_over_batches.append(metrics)

            if val_metric_over_batches:
                # For now, take the last batch's metrics as the epoch's metrics.
                # Since we force full-batch validation, this is inconsequential.
                metrics_val_latest = val_metric_over_batches[-1]
                metrics_val.append(metrics_val_latest)

                # If it is a RegressionMetrics:
                if isinstance(metrics_val_latest, RegressionMetrics):
                    if num_parallel == 1:
                        t.set_description(
                            f"Epoch {epoch} | "
                            f"NLL=(Train:{metrics.nlll:.3f}, Val:{metrics_val_latest.nlll:.3f}) "
                            f"| RMSE=(Train:{metrics.rmse:.3f}, Val:{metrics_val_latest.rmse:.3f})"
                        )
                    else:
                        nlll_train_str = ", ".join(f"{x:.3f}" for x in metrics.nlll)
                        nlll_val_str = ", ".join(f"{x:.3f}" for x in metrics_val_latest.nlll)
                        rmse_train_str = ", ".join(f"{x:.3f}" for x in metrics.rmse)
                        rmse_val_str = ", ".join(f"{x:.3f}" for x in metrics_val_latest.rmse)

                        t.set_description(
                            f"Epoch {epoch} | "
                            f"NLL=(Train:[{nlll_train_str}], Val:[{nlll_val_str}]) "
                            f"| RMSE=(Train:[{rmse_train_str}], Val:[{rmse_val_str}])"
                        )

                    # Track losses for early stopping, etc.
                    val_losses = jnp.append(
                        val_losses,
                        metrics_val_latest.nlll[..., None]
                        if num_parallel > 1
                        else metrics_val_latest.nlll[..., None, None],
                        axis=-1,
                    )
                    _stop_n = _stop_n + early_stop_check(
                        losses=val_losses,
                        patience=self.config.warm_start_early_stop_patience
                    )
                else:
                    raise NotImplementedError("Classification metrics not yet implemented.")
            else: # No validation data. Skip.
                pass

        # --- Test ---
        test_metrics_over_batches = []
        for batch in data_loader.iter(
                split="test",
                batch_size=None,  # Enforce full-batch testing.
                n_devices=num_parallel
        ):
            for feature_type in batch["feature"].keys():
                for feature_name in batch["feature"][feature_type].keys():
                    if batch["feature"][feature_type][feature_name].ndim == 1:
                        batch["feature"][feature_type][feature_name] = \
                            batch["feature"][feature_type][feature_name].reshape(-1, 1)

            metrics = _model_pred_func(
                training_state,
                batch["feature"],
                batch["target"],
                jnp.repeat(
                    False, num_parallel
                ) if num_parallel > 1 else jnp.bool_(False),
                False  # Static 'train' argument.
            )
            test_metrics_over_batches.append(metrics)

        if test_metrics_over_batches:
            metrics_test.append(test_metrics_over_batches[-1])  # Last index is the only batch.
            if isinstance(test_metrics_over_batches[-1], RegressionMetrics):
                if num_parallel == 1:
                    self._logger.info(
                        f"(Test NLL={test_metrics_over_batches[-1].nlll:.3f} | "
                        f"Test RMSE={test_metrics_over_batches[-1].rmse:.3f})"
                    )
                else:
                    nlll_test_str = (
                        ", ".join(f"{x:.3f}" for x in test_metrics_over_batches[-1].nlll)
                    )
                    rmse_test_str = (
                        ", ".join(f"{x:.3f}" for x in test_metrics_over_batches[-1].rmse)
                    )
                    self._logger.info(
                        f"(Test NLL=[{nlll_test_str}] | Test RMSE=[{rmse_test_str}])"
                    )

        if isinstance(metrics, RegressionMetrics):
            complete_metrics = MetricsStore(
                train=RegressionMetrics.cstack(metrics_train),
                valid=RegressionMetrics.cstack(metrics_val) if metrics_val else None,
                test=RegressionMetrics.cstack(metrics_test) if metrics_test else None
            )
        else:
            raise NotImplementedError("Classification metrics not yet implemented.")

        return training_state, complete_metrics

    def _initialize_module(self):
        """
        Method to create and store a Flax module for a deterministic equivalent of the
        Bayesian Neural Additive Model, replicating the model's architecture.
        """

        # Build deterministic subnetworks from the instantiated BayesianNN objects.
        # We assume that each BayesianNN has attributes _layer_sizes and _config.
        num_subnetworks = {}
        cat_subnetworks = {}
        int_subnetworks = {}
        for idx, (network_type, source_bayesian_nn, target_deterministic_nn) in enumerate(
                zip(
                    ("num", "cat", "int"),
                    (self._num_feature_networks, self._cat_feature_networks, self._interaction_networks),
                    (num_subnetworks, cat_subnetworks, int_subnetworks),
                )
        ):
            for feature_name, bayes_nn in source_bayesian_nn.items():
                target_deterministic_nn[feature_name] = DeterministicNN(
                    model_name=f"{feature_name}_{network_type}_subnetwork",
                    layer_sizes=tuple(bayes_nn.layer_sizes),
                    activation=bayes_nn.config.activation,
                    dropout=bayes_nn.config.dropout,
                    use_batch_norm=bayes_nn.config.batch_norm,
                    use_layer_norm=bayes_nn.config.layer_norm,
                )

        # Instantiate the top-level NAM module using the above subnetworks.
        # Note:
        # frozendict is used to ensure the flax module is hashable when using pmap for
        # parallelized training.
        self._flax_module = DeterministicNAM(
            num_subnetworks=frozendict(num_subnetworks),
            cat_subnetworks=frozendict(cat_subnetworks),
            int_subnetworks=frozendict(int_subnetworks),
            config=self.config,
            lnk_fns=self.lnk_fns,
        )
        self._logger.info(
            f"{nl}"
            f"+--------------------------------------------+{nl}"
            f"| Deterministic NAM successfully initialized.|{nl}"
            f"+--------------------------------------------+{nl}"
        )

    from typing import Dict, Optional
    import numpy as np

    def _get_posterior_param_samples(
            self,
            group_by_chain: bool = False
    ) -> Dict[str, Optional[Dict[str, np.ndarray]]]:
        """
        Extract the posterior samples for the scale, weight, bias, intercept and noise parameters.

        Parameters
        ----------
        group_by_chain : bool, optional
            If True, return each array with shape (n_chains, n_draws, ...).
            If False, flatten across chains to shape (n_draws, ...).
            Default is False.

        Returns
        -------
        Dict[str, Optional[Dict[str, np.ndarray]]]
            A dictionary with keys:
              - 'noise'     : dict of sigma parameters
              - 'scale'     : dict of kernel scale parameters
              - 'weights'   : dict of kernel weight parameters
              - 'biases'    : dict of kernel bias parameters
              - 'intercept' : dict of intercept parameters, or None if intercepts are not used.

            Each inner dict maps the parameter name to its posterior samples
            as a NumPy array with the shape determined by `group_by_chain`.
        """
        if not hasattr(self, '_mcmc'):
            raise ValueError(
                "MCMC samples not found. Please train the model using MCMC first."
            )

        # grab raw samples, optionally grouped by chain
        posterior_samples = self._mcmc.get_samples(group_by_chain=group_by_chain)

        # filter out the different parameter types
        scale_params = {
            name: vals
            for name, vals in posterior_samples.items()
            if "scale" in name
        }
        weight_params = {
            name: vals
            for name, vals in posterior_samples.items()
            if "_kernel" in name and "scale" not in name
        }
        bias_params = {
            name: vals
            for name, vals in posterior_samples.items()
            if "_bias" in name and "scale" not in name
        }
        noise_params = {
            name: vals
            for name, vals in posterior_samples.items()
            if "sigma" in name
        }

        intercept_params: Optional[Dict[str, np.ndarray]]
        if self.config.intercept:
            intercept_params = {
                name: vals
                for name, vals in posterior_samples.items()
                if "intercept" in name
            }
        else:
            intercept_params = None

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

    def cross_validation(
            self,
            num_features: dict[str, jnp.ndarray],
            cat_features: dict[str, jnp.ndarray],
            target: jnp.ndarray,
    ) -> list[
        dict[
            str, TabularAdditiveModelDataLoader |
            dict[Any, ndarray[Any, dtype[Any]]] |
            Any
        ]
    ]:
        """
        Method to perform cross-validation on the model.

        Parameters
        ----------
        num_features: dict
            Dictionary of numerical features with feature names as keys.
        cat_features: dict
            Dictionary of categorical features with feature names as keys.
        target: jnp.ndarray
            True response tensor.

        Returns
        -------
        list[dict]:
            List of dictionaries containing the cross-validation results.

        """

        num_outer_splits = getattr(self.config, "outer_cv_num_splits", 4)
        num_inner_splits = getattr(self.config, "inner_cv_num_splits", 4)

        if num_features is not None:
            outer_features_to_split = num_features
        elif cat_features is not None:
            outer_features_to_split = cat_features
        else:
            raise ValueError(
                "Please provide at least one of "
                "num_features or cat_features for cross-validation."
            )

        cv_results = []
        kf_outer = KFold(n_splits=num_outer_splits, shuffle=True, random_state=42)
        for k_outer, (outer_train_index, outer_test_index) in enumerate(
                kf_outer.split(
                    list(outer_features_to_split.values())[0]
                )
        ):
            kf_inner = KFold(n_splits=num_inner_splits, shuffle=True, random_state=42)
            for k_inner, (inner_train_index, inner_val_index) in enumerate(
                    kf_inner.split(outer_train_index)
            ):
                cv_data_loader = TabularAdditiveModelDataLoader(
                    config=DataConfig(
                        path=f"cv_{k_outer}_outer_{k_inner}_inner_data_loader",
                        source=Source.LOCAL,
                        data_type=DatasetType.TABULAR,
                        task=Task.REGRESSION,
                        target_column="Response",
                        target_len=target.shape[0],
                        datapoint_limit=None,
                        normalize=True,
                        train_split=0.7,  # Not used due to CV.
                        valid_split=0.2,  # Not used due to CV.
                        test_split=0.1,   # Not used due to CV.
                    ),
                    rng=random.split(self._single_rng_key)[1],
                    data_dict={
                        "numerical": num_features,
                        "categorical": cat_features,
                        "target": target
                    },
                    target_key="target",
                    train_idx=inner_train_index,
                    valid_idx=inner_val_index,
                    test_idx=outer_test_index,
                )
                self.train_model(
                    data_loader=cv_data_loader
                )

                params, submodel_contributions = self.predict(
                    data_loader=cv_data_loader
                )
                cv_results.append({
                    "params": params,
                    "submodel_contributions": submodel_contributions,
                    "data_loader": cv_data_loader,
                })
                break

        return cv_results

    def predict(
            self,
            data_loader: TabularAdditiveModelDataLoader,
            training: bool = False,
            bayesian_sampling: bool = None
    ):
        """
        Generate predictions from the trained Bayesian NAM model.

        Parameters
        ----------
        num_features: dict
            Dictionary of numerical features with feature names as keys.
        cat_features: dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict | Any:
            The final distributional parameters predicted by the model.
        dict | ndarray:
            The contributions of each submodel to the final prediction.
        """

        if bayesian_sampling is None:
            bayesian_sampling = self._bayesian_sampling

        if bayesian_sampling:
            # We need to remove the deterministic sites to force the model to recompute these values.
            # Get all site names that have "contrib" in them.
            deterministic_site_keys = [
                key for key in self._mcmc.get_samples().keys()
                if "contrib" in key or "final" in key
            ]

            for deterministic_site_key in deterministic_site_keys:
                try:
                    self.posterior_samples.pop(deterministic_site_key)
                except KeyError:
                    continue  # Site has already been removed or never existed.

            predictive = Predictive(
                self.model,
                posterior_samples=self.posterior_samples,
            )
            preds = predictive(
                random.split(self._single_rng_key)[1],
                data_loader=data_loader,
                is_training=training
            )

            submodel_output_contributions = {}
            for feature_type, submodel_dict in zip(
                    ["numerical", "categorical", "interaction"],
                    [self._num_feature_networks, self._cat_feature_networks, self._interaction_networks]
            ):
                if not submodel_dict:
                    continue # Interaction networks may be empty if interaction_degree < 2.

                for feature_name, submodel in submodel_dict.items():
                    # Shape: [num_mcmc_samples, batch_size, sub_network_out_dim].
                    contrib_samples = preds[f"contrib_{feature_name}"]
                    submodel_output_contributions[feature_name] = np.array(contrib_samples)

            preds_return = preds["final_params"]
            submodel_output_contributions_return = submodel_output_contributions
        else:
            # Make the predictions using the deterministic NAM.
            # For each entry in the dict (recursively), check if it is a jnp.ndarray.
            # If so, take the ith element of its leading dimension.

            def get_single_nam_parameters(idx: int):
                def index_array(val):
                    return val[idx, ...] \
                        if isinstance(val, jnp.ndarray) and val.ndim > 0 \
                        else val
                def recursive_index(param_dict):
                    if isinstance(param_dict, dict):
                        return {k: recursive_index(v) for k, v in param_dict.items()}
                    else:
                        return index_array(param_dict)
                return recursive_index(self._flax_module_params[0])

            det_submodel_output_contributions = [{} for _ in range(self.config.num_chains)]
            det_preds_list = []
            for i in range(self.config.num_chains):
                params_i = get_single_nam_parameters(i)
                preds_i, contributions_stack_i = self._flax_module.apply(
                    {"params": params_i},
                    data_loader.data_test["numerical"],
                    data_loader.data_test["categorical"],
                    train=False,
                    rng=random.split(self._single_rng_key)[1],
                    return_contributions=True
                )
                det_preds_list.append(preds_i)
                for feature_type, submodel_dict in zip(
                        ["numerical", "categorical", "interaction"],
                        [self._flax_module.num_subnetworks,
                         self._flax_module.cat_subnetworks,
                         self._flax_module.int_subnetworks]
                ):
                    if not submodel_dict:
                        continue  # Interaction networks may be empty if interaction_degree < 2.
                    # Shape of contributions_stack: [batch, out_dim, num_subnetworks].
                    for submodel_idx, (feature_name, submodel) in enumerate(submodel_dict.items()):
                        contrib_samples = contributions_stack_i[:, :, submodel_idx]
                        det_submodel_output_contributions[i][feature_name] = np.array(contrib_samples)

            submodel_output_contributions_return = {
                feature_name: jnp.stack(
                    [det_submodel_output_contributions[i][feature_name] for i in range(self.config.num_chains)],
                    axis=0
                ) for feature_name in det_submodel_output_contributions[0].keys()
            }
            preds_return = jnp.stack(det_preds_list, axis=0)

        return (
            preds_return,
            submodel_output_contributions_return
        )


class DeterministicNAM(nn.Module):
    num_subnetworks: frozendict[str, DeterministicNN]
    cat_subnetworks: frozendict[str, DeterministicNN]
    int_subnetworks: frozendict[str, DeterministicNN]
    config: Any
    lnk_fns: Tuple[Callable, ...]

    @nn.compact
    def __call__(
            self,
            num_features: dict[str, jnp.ndarray],
            cat_features: dict[str, jnp.ndarray],
            train: bool = True,
            rng = None,
            return_contributions: bool = False
    ):
        """
        Forward pass of the NAM.

        Parameters
        ----------
        num_features: dict
            Dictionary of numerical features with feature names as keys.
        cat_features: dict
            Dictionary of categorical features with feature names as keys.
        train: bool
            Flag to indicate whether the model is in training mode. (Default value = True)
        rng:
            Random key for initializing the ensemble.

        Returns
        -------
        jnp.ndarray:
            The final distributional parameters predicted by the model.
        """

        contributions = []  # To hold outputs from each subnetwork.
        subnet_means = {}  # To compute the (global) offset.
        for networks, features in zip(
                (self.num_subnetworks, self.cat_subnetworks),
                (num_features, cat_features)
        ):
            for feature_name, network in networks.items():
                x = features[feature_name]
                if x.ndim == 1:  # Ensure input is of shape [batch, input_dim].
                    x = x.reshape(-1, 1)
                out = network(x, train=train, rng=rng)  # shape: [batch, out_dim]
                subnet_means[feature_name] = jnp.mean(out, axis=0)  # shape: [out_dim]
                contributions.append(out - subnet_means[feature_name])

        if self.int_subnetworks:
            all_features = {**num_features, **cat_features}
            for interaction_name, network in self.int_subnetworks.items():
                feature_names = interaction_name.split(":")
                x_list = [
                    jnp.expand_dims(all_features[name], axis=-1)
                    for name in feature_names
                ]
                x = jnp.concatenate(x_list, axis=-1)
                out = network(x, train=train, rng=rng)
                if out.shape[1] == 1:  # Remove the redundant axis only for batch size 1.
                    out = jnp.squeeze(out, axis=1)
                subnet_means[interaction_name] = jnp.mean(out, axis=0)
                contributions.append(out - subnet_means[interaction_name])

        # Stack all subnetwork contributions.
        # Shape: [batch, out_dim, num_subnetworks]
        contributions_stack = jnp.stack(contributions, axis=-1)

        if self.config.feature_dropout > 0.0 and train:
            contributions_stack = nn.Dropout(
                rate=self.config.feature_dropout
            )(
                contributions_stack,
                deterministic=not train
            )

        output_sum = jnp.sum(contributions_stack, axis=-1)
        if self.config.intercept:
            out_dim = output_sum.shape[-1]
            # Define a global intercept parameter of shape [out_dim].
            intercept = self.param("intercept", nn.initializers.normal(), (out_dim,))
            output_sum = output_sum + intercept

        # Apply the k desired link function(s) dimension-wise.
        outputs = []
        for i, fn in enumerate(self.lnk_fns[:output_sum.shape[-1]]):
            outputs.append(fn(output_sum[..., i]))
        final_params = jnp.stack(outputs, axis=-1)

        if return_contributions:
            return final_params, contributions_stack
        else:
            return final_params
