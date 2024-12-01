import pickle
from typing import Tuple, Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
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
        self._gamma_prior_shape = self._config.gamma_prior_shape
        self._gamma_prior_scale = self._config.gamma_prior_scale

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
                independent_network_flag=False,
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

    def model(
            self,
            num_features: Dict[str, jnp.ndarray],
            cat_features: Dict[str, jnp.ndarray],
            target: jnp.ndarray = None,
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
        """

        contributions = []

        for feature_name, feature_network in self._num_feature_networks.items():
            x = num_features[feature_name]
            # Use numpyro.handlers.scope to isolate parameter names
            with handlers.scope(prefix=feature_name):
                mu_i = feature_network.model(x)
            numpyro.deterministic(f"contrib_{feature_name}", mu_i)
            contributions.append(mu_i)

        for feature_name, feature_network in self._cat_feature_networks.items():
            x = cat_features[feature_name]
            with handlers.scope(prefix=feature_name):
                mu_i = feature_network.model(x)
            numpyro.deterministic(f"contrib_{feature_name}", mu_i)
            contributions.append(mu_i)

        if self._interaction_degree is not None and self._interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, interaction_network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = jnp.concatenate(
                    [jnp.expand_dims(all_features[name], axis=-1) for name in feature_names],
                    axis=-1
                )
                with handlers.scope(prefix=interaction_name):
                    mu_i = interaction_network.model(x)
                numpyro.deterministic(f"contrib_{interaction_name}", mu_i)
                contributions.append(mu_i)

        mu = sum(contributions)

        if self._intercept:
            intercept = numpyro.sample(
                "intercept",
                dist.Normal(
                    loc=0.0,
                    scale=1.0
                )
            )
            mu = mu + intercept

        sigma = numpyro.sample(
            "sigma",
            dist.Gamma(
                self._gamma_prior_shape,
                self._gamma_prior_scale
            )
        )

        with numpyro.plate("data", size=mu.shape[0]):
            numpyro.sample(
                "obs",
                dist.Normal(mu, sigma),
                obs=target,
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
        num_samples: int,
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
            step_size=self._config.mcmc_step_size,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            dense_mass=False,
            target_accept_prob=0.8,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=num_samples//2,
        )
        mcmc.run(rng_key, num_features, cat_features, target)
        self.posterior_samples = mcmc.get_samples()

        self.predictive = Predictive(
            self.model,
            posterior_samples=self.posterior_samples
        )
    
    def train_model(
        self,
        num_features: Dict[str, jnp.ndarray],
        cat_features: Dict[str, jnp.ndarray],
        target: jnp.ndarray,
        inference_method: str = "svi",
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
                num_features,
                cat_features,
                target
            )
        elif inference_method == "mcmc":
            self._get_mcmc_predictive_distribution(
                num_features,
                cat_features,
                target
            )
        else:
            raise ValueError("Inference method must be either 'svi' or 'mcmc'.")

    def predict(
            self,
            num_features: Dict[str, jnp.ndarray],
            cat_features: Dict[str, jnp.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
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
        Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]
            A tuple containing the mean predictions, standard deviations,
            and submodel contributions.
        """

        rng_key = random.PRNGKey(1)
        predictions = self.predictive(
            rng_key,
            num_features,
            cat_features,
            target=None
        )

        pred_samples = predictions["obs"]  # Shape: [num_samples, batch_size]
        pred_means = np.mean(pred_samples, axis=0)
        pred_std = np.std(pred_samples, axis=0)

        submodel_contributions = {}

        for feature_name in self._num_feature_networks.keys():
            contrib_samples = predictions[
                f"contrib_{feature_name}"]  # Shape: [num_samples, batch_size]
            submodel_contributions[feature_name] = np.array(contrib_samples)

        for feature_name in self._cat_feature_networks.keys():
            contrib_samples = predictions[f"contrib_{feature_name}"]
            submodel_contributions[feature_name] = np.array(contrib_samples)

        if self._interaction_degree is not None and self._interaction_degree >= 2:
            for interaction_name in self._interaction_networks.keys():
                contrib_samples = predictions[f"contrib_{interaction_name}"]
                submodel_contributions[interaction_name] = np.array(contrib_samples)

        return pred_means, pred_std, submodel_contributions

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