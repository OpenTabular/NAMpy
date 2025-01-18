from __future__ import annotations

import pickle
from typing import Tuple, Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from jax_tqdm import scan_tqdm
from numpy import ndarray, dtype
from numpyro import handlers

from namgcv.basemodels.bnn import BayesianNN
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam
from tqdm.auto import trange

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
nl = "\n"
tab = "\t"


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
        kwargs : Any
            Additional keyword arguments specifying the hyperparameters of the parent model.
        """

        self._logger = logging.getLogger(__name__)

        self._config = config
        self._num_epochs = self._config.num_epochs
        self._lr = self._config.lr
        self._weight_decay = self._config.weight_decay

        self._intercept_prior_shape = self._config.intercept_prior_shape
        self._intercept_prior_scale = self._config.intercept_prior_scale

        self._cat_feature_info = cat_feature_info
        self._num_feature_info = num_feature_info
        self._num_classes = num_classes
        self._interaction_degree = self._config.interaction_degree

        self._intercept = self._config.intercept

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
                independent_network_flag=False,
            )

        self._interaction_networks = {}
        if self._interaction_degree is not None and self._interaction_degree >= 2:
            self._create_interaction_subnetworks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

        self.predictive = None

        self._model_initialized = True
        self._logger.info(
            f"{nl}"
            f"+--------------------------------------+{nl}"
            f"| Bayesian NAM successfully initialized.|{nl}"
            f"+--------------------------------------+{nl}"
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

        all_feature_names = list(num_feature_info.keys()) + list(cat_feature_info.keys())

        for degree in range(2, self._interaction_degree + 1):
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
                    out_dim=1,
                    config=self._subnetwork_config,
                    model_name=f"{interaction_name}_int_subnetwork",
                    independent_network_flag=False,
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
            is_training: bool = True
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

        subnet_out_contributions = []
        for sub_network_group, feature_group in zip(
                [self._num_feature_networks, self._cat_feature_networks],
                [num_features, cat_features]
        ):
            for feature_name, feature_network in sub_network_group.items():
                x = feature_group[feature_name]
                # Use numpyro.handlers.scope to isolate parameter names.
                with handlers.scope(prefix=feature_name):
                    subnet_out_i = feature_network.model(x, y=target, is_training=is_training)

                subnet_out_i_centered = subnet_out_i - jnp.mean(subnet_out_i, axis=0)
                numpyro.deterministic(name=f"contrib_{feature_name}", value=subnet_out_i_centered)
                subnet_out_contributions.append(subnet_out_i_centered)

        if self._interaction_degree is not None and self._interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, interaction_network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = jnp.concatenate(
                    [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
                    axis=-1
                )
                with handlers.scope(prefix=interaction_name):
                    subnet_out_i = interaction_network.model(x, y=target, is_training=is_training)
                subnet_out_i_centered = subnet_out_i - jnp.mean(subnet_out_i, axis=0)
                numpyro.deterministic(name=f"contrib_{interaction_name}", value=subnet_out_i_centered)
                subnet_out_contributions.append(subnet_out_i_centered)

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

        # After sum shape: [batch_size, sub_network_out_dim].
        output_sum_over_subnetworks = jnp.sum(subnet_out_contributions, axis=-1)

        if self._intercept:  # Global intercept term.
            intercept = numpyro.sample(
                name=f"intercept",
                fn=dist.Normal(
                    loc=self._intercept_prior_shape,
                    scale=self._intercept_prior_scale
                ).expand(
                    [output_sum_over_subnetworks.shape[1]]
                ).to_event(1)  # K-dimensional.
            )
            output_sum_over_subnetworks += intercept

        # TODO: Place the link function in the submodel class?
        #  OR, equivalently, apply it before storing the contributions.
        # --------------------------------------
        # Link functions per dimension
        # e.g. dimension 0 => identity, dimension 1 => exp, dimension 2 => identity
        # (Just an example for a SkewNormal.)
        # --------------------------------------
        def link_0(x):  # location
            return x

        def link_1(x):  # scale => must be positive
            return jax.nn.softplus(x)

        def link_2(x):  # shape => unconstrained
            return x

        link_fns = [link_0, link_1]

        # Apply link dimensionwise
        # shape => [batch_size, K]
        # But we do it dimension by dimension:
        def apply_links(params):
            # params is shape [batch_size, K]
            results = []
            for k in range(params.shape[1]):
                results.append(link_fns[k](params[:, k]))
            return jnp.stack(results, axis=-1)

        final_params = apply_links(output_sum_over_subnetworks)  # shape [batch_size, K]
        numpyro.deterministic(name=f"final_params", value=final_params)

        self.likelihood(
            output_sum_over_subnetworks=final_params,
            y=target
        )


    def _get_svi_predictive_distribution(
        self,
        num_features: Dict[str, jnp.ndarray],
        cat_features: Dict[str, jnp.ndarray],
        target: jnp.ndarray,
    ):
        """
        Optimize the model using Stochastic Variational Inference (SVI) and store the
        predictive distribution.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target : jnp.ndarray
            True response tensor.
        """

        rng_key = random.PRNGKey(0)
        optimizer = Adam(step_size=self._lr)

        # Note: The guide is the variational distribution that we use to approximate the posterior.
        guide = AutoDiagonalNormal(self.model)

        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())
        svi_state = svi.init(rng_key, num_features, cat_features, target)

        progress_bar = trange(self._num_epochs)
        for epoch in progress_bar:
            svi_state, loss = svi.update(
                svi_state,
                num_features, cat_features, target
            )
            progress_bar.set_postfix(loss=f"{loss / target.shape[0]:.3f}")

        self.params = svi.get_params(svi_state)
        predictive = Predictive(
            self.model,
            guide=guide,
            params=self.params,
            num_samples=self._config.num_samples
        )
        self.predictive = predictive

    def _get_mcmc_predictive_distribution(
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
        num_samples : int
            Number of samples to draw from the posterior distribution.
        """
        rng_key = random.PRNGKey(0)
        nuts_kernel = NUTS(
            model=self.model,
            # step_size=self._config.mcmc_step_size,
            # target_accept_prob=0.3,
        )
        self._mcmc = MCMC(
            nuts_kernel,
            num_samples=self._config.num_samples,
            num_warmup=self._config.num_samples,
            num_chains=self._config.num_chains
        )
        self._mcmc.run(rng_key, num_features, cat_features, target)
        self.posterior_samples = self._mcmc.get_samples()

        self.predictive = Predictive(
            self.model,
            posterior_samples=self.posterior_samples
        )

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
        } if self._intercept else None

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

    def train_model(
        self,
        num_features: Dict[str, jnp.ndarray],
        cat_features: Dict[str, jnp.ndarray],
        target: jnp.ndarray,
        inference_method: str = "mcmc",
    ):
        """
        Train the Bayesian Neural Additive Model (BNAM) using the specified inference method.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target : jnp.ndarray
            True response tensor.
        inference_method : str
            Inference method to use for training the model. Must be one of 'svi' or 'mcmc'.
            Default is 'svi'.
        """

        inference_method = inference_method.lower()
        if inference_method == "svi":
            self._get_svi_predictive_distribution(
                num_features=num_features,
                cat_features=cat_features,
                target=target
            )
        elif inference_method == "mcmc":
            self._get_mcmc_predictive_distribution(
                num_features=num_features,
                cat_features=cat_features,
                target=target
            )
        else:
            raise ValueError("Inference method must be either 'svi' or 'mcmc'.")

    def predict(
            self,
            num_features: Dict[str, jnp.ndarray],
            cat_features: Dict[str, jnp.ndarray],
    ) -> tuple[ndarray[Any, dtype[Any]], dict[str, ndarray[Any, dtype[Any]]]]:
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
            target=None
        )

        pred_samples = predictions["obs"]  # Shape: [num_mcmc_samples, batch_size]
        final_params = predictions["final_params"]

        submodel_output_contributions = {}
        for feature_type, submodel_dict in zip(
                ["numerical", "categorical", "intercation"],
                [self._num_feature_networks, self._cat_feature_networks, self._interaction_networks]
        ):
            if not submodel_dict:
                continue # Interaction networks may be empty if interaction_degree < 2.

            for feature_name, submodel in submodel_dict.items():
                # Shape: [num_mcmc_samples, batch_size, sub_network_out_dim].
                contrib_samples = predictions[f"contrib_{feature_name}"]
                submodel_output_contributions[feature_name] = np.array(contrib_samples)

        return pred_samples, final_params, submodel_output_contributions

    def save_model(
            self,
            filepath: str
    ):
        """
        Save the trained model parameters or posterior samples to disk.

        Parameters
        ----------
        filepath : str
            The file path where the model will be saved.
        """

        data = {
            'config': self._config,
            'subnetwork_config': self._subnetwork_config,
            'cat_feature_info': self._cat_feature_info,
            'num_feature_info': self._num_feature_info,
            'interaction_degree': self._interaction_degree,
            'intercept': self._intercept,
        }

        if hasattr(self, 'posterior_samples'):
            data['posterior_samples'] = self.posterior_samples
            data["mcmc"] = self._mcmc
            data['inference_method'] = 'mcmc'
        elif hasattr(self, 'params'):
            data['params'] = self.params
            data['inference_method'] = 'svi'
        else:
            raise ValueError("No trained model found. Please train the model before saving.")

        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._logger.info(f"Model saved to {filepath}.")

    def load_model(
            self,
            filepath: str
    ):
        """
        Load the trained model parameters or posterior samples from disk.

        Parameters
        ----------
        filepath : str
            The file path from where the model will be loaded.
        """

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Restore model configurations and parameters
        self._config = data['config']
        self._subnetwork_config = data['subnetwork_config']
        self._cat_feature_info = data['cat_feature_info']
        self._num_feature_info = data['num_feature_info']
        self._interaction_degree = data['interaction_degree']
        self._intercept = data['intercept']

        inference_method = data['inference_method']

        self.__init__(
            cat_feature_info=self._cat_feature_info,
            num_feature_info=self._num_feature_info,
            config=self._config,
            subnetwork_config=self._subnetwork_config,
        )  # Re-initialize subnetworks.

        if inference_method == 'mcmc':
            self.posterior_samples = data['posterior_samples']
            self.predictive = Predictive(
                self.model,
                posterior_samples=self.posterior_samples
            )
            self._mcmc = data["mcmc"]
        elif inference_method == 'svi':
            # TODO: Fix saving and loading of SVI model.
            #  It currently raises an error during prediction time.
            self.params = data['params']
            self.predictive = Predictive(
                self.model,
                guide=AutoDiagonalNormal(self.model),
                params=self.params,
                num_samples=self._config.num_samples
            )
        else:
            raise ValueError("Invalid inference method found in saved data.")

        self._logger.info(f"Model loaded from {filepath}")
