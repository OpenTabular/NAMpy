from typing import Tuple, Any, Dict

import numpy as np
import torch

from namgcv.basemodels.bnn import BayesianNN
from namgcv.configs.bayesian_nam_config import DefaultBayesianNAMConfig
from namgcv.configs.bayesian_nn_config import DefaultBayesianNNConfig

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange

import torch.nn as nn
from itertools import combinations

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
nl = "\n"



class BayesianNAM(PyroModule):
    """
    Bayesian Neural Additive Model (BNAM) class.

    This class implements a Bayesian Neural Additive Model (BNAM) using Pyro.
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
        super().__init__(**kwargs)

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

        self._intercept = PyroSample(
            dist.Normal(0.0, 1.0).expand([1]).to_event(1)
        ) if self._config.intercept else None

        # Initialize subnetworks
        self._num_feature_networks = PyroModule[nn.ModuleDict]()
        for feature_name, feature_info in num_feature_info.items():
            self._num_feature_networks[feature_name] = BayesianNN(
                in_dim=feature_info["input_dim"],
                out_dim=feature_info["output_dim"],
                config=subnetwork_config,
                model_name=f"{feature_name}_num_subnetwork"
            )

        self._cat_feature_networks = PyroModule[nn.ModuleDict]()
        for feature_name, feature_info in cat_feature_info.items():
            self._cat_feature_networks[feature_name] = BayesianNN(
                in_dim=feature_info["input_dim"],
                out_dim=feature_info["output_dim"],
                config=subnetwork_config,
                model_name=f"{feature_name}_cat_subnetwork"
            )

        self._interaction_networks = PyroModule[nn.ModuleDict]()
        if self._interaction_degree is not None and self._interaction_degree >= 2:
            self._create_interaction_subnetworks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=subnetwork_config,
            )

        self._logger.info(
            f"{nl}"
            f"+--------------------------------------+{nl}"
            f"|Bayesian NAM successfully initialized.|{nl}"
            f"+--------------------------------------+{nl}"
            f"Numerical feature networks: {nl}{self._num_feature_networks}{nl}"
            f"Categorical feature networks: {nl}{self._cat_feature_networks}{nl}"
            f"Interaction networks: {nl}{self._interaction_networks}"
        )

    def _create_interaction_subnetworks(
        self,
        num_feature_info: dict,
        cat_feature_info: dict,
        config=None  # Replace with your actual config class
    ):
        """
        Create Bayesian Neural Networks for modeling feature interactions.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        config : DefaultNAMConfig
            Configuration dataclass containing model hyperparameters.
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
                    config=config,
                    model_name=f"{interaction_name}_int_subnetwork"
                )

    def forward(
        self,
        num_features: Dict[str, torch.Tensor],
        cat_features: Dict[str, torch.Tensor],
        target: torch.Tensor = None
    ):
        """
        Forward pass of the Bayesian Neural Additive Model (BNAM).

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target: torch.Tensor
            True response tensor (optional). This is only required during training.

        Returns
        -------
        mu : torch.Tensor
            Aggregated output of the BNAM.
        """

        contributions = []
        for feature_name, feature_network in self._num_feature_networks.items():
            x = num_features[feature_name]
            mu_i = feature_network(x)
            contributions.append(mu_i)

        for feature_name, feature_network in self._cat_feature_networks.items():
            x = cat_features[feature_name].float()
            mu_i = feature_network(x)
            contributions.append(mu_i)

        if self._interaction_degree is not None and self._interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, interaction_network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = torch.cat(
                    [all_features[name].unsqueeze(-1) for name in feature_names], dim=-1
                )
                mu_i = interaction_network(x)
                contributions.append(mu_i)

        mu = sum(contributions)
        if self._intercept is not None:
            mu = mu + self._intercept

        sigma = pyro.sample(
            "sigma",
            dist.Gamma(
                self._gamma_prior_shape,
                self._gamma_prior_scale
            )
        )

        with pyro.plate("data", size=mu.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mu, sigma),
                obs=target
            )

        return mu

    def train_model(
        self,
        num_features: Dict[str, torch.Tensor],
        cat_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        num_samples: int,
        inference_method: str = "svi"
    ):
        """
        Train the Bayesian Neural Additive Model (BNAM) using the specified inference method.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.
        target : torch.Tensor
            True response tensor.
        num_samples : int
            Number of samples to draw from the posterior distribution.

        inference_method : str
            Inference method to use for training the model. Must be one of 'svi' or 'mcmc'.
            Default is 'svi'.
        """

        inference_method = inference_method.lower()
        if inference_method == "svi":
            guide = AutoDiagonalNormal(self)
            optimizer = pyro.optim.Adam({"lr": self._lr, "weight_decay": self._weight_decay})
            svi = SVI(self, guide, optimizer, loss=Trace_ELBO())
            pyro.clear_param_store()

            progress_bar = trange(self._num_epochs)
            for epoch in progress_bar:
                loss = svi.step(num_features, cat_features, target)
                progress_bar.set_postfix(loss=f"{loss / target.shape[0]:.3f}")

            self.predictive = Predictive(self, guide=guide, num_samples=num_samples)

        elif inference_method == "mcmc":
            nuts_kernel = NUTS(
                model=self,
                step_size=self._config.mcmc_step_size,
                adapt_step_size=True,
                adapt_mass_matrix=True,
                jit_compile=False
            )
            mcmc = MCMC(nuts_kernel, num_samples=num_samples)
            mcmc.run(num_features, cat_features, target)

            self.predictive = Predictive(self, posterior_samples=mcmc.get_samples())

        else:
            raise ValueError("Inference method must be either 'svi' or 'mcmc'.")

    @staticmethod
    def _compute_subnetwork_output(
            subnetwork: BayesianNN,
            x: torch.Tensor,
            samples: dict
    ):
        """
        Helper method to compute the output of a subnetwork for a given input and parameter samples.

        Parameters
        ----------
        subnetwork : BayesianNN
            Subnetwork model.
        x : torch.Tensor
            Input tensor.
        samples : dict
            Dictionary of samples from the predictive distribution.

        Returns
        -------
        contributions : np.ndarray
            Subnetwork contributions.
        """

        outputs = []
        for i in range(samples["obs"].shape[0]):
            # Manually set the parameters of the subnetwork to the i-th sample.
            pyro.get_param_store().clear()
            subnetwork_samples = {
                k: v[i] for k, v in samples.items() if k.startswith(subnetwork.model_name)
            }
            with pyro.poutine.trace() as tr:
                with pyro.poutine.condition(data=subnetwork_samples):
                    output = subnetwork(x)
            outputs.append(output.detach().numpy())

        return np.array(outputs)  # Shape: [num_samples, batch_size]

    def predict(
            self,
            num_features: Dict[str, torch.Tensor],
            cat_features: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Obtain predictions from the Bayesian Neural Additive Model (BNAM).
        First, we obtain samples from the predictive distribution, after which we compute the mean
        and standard deviation of the predictions. Finally, we extract the individual contributions
        of each subnetwork.

        Parameters
        ----------
        num_features
        cat_features

        Returns
        -------
        pred_means : np.ndarray
            Mean predictions of the BNAM.
        pred_std : np.ndarray
            Standard deviations of the predictions.
        submodel_contributions : Dict[str, np.ndarray]
            Contributions of each subnetwork.
        """

        # Obtain samples from the predictive distribution.
        samples = self.predictive(
            num_features=num_features,
            cat_features=cat_features,
            target=None
        )
        # Extract the total predictions from the BNAM.
        pred_samples = samples["obs"]  # Shape: [num_samples, batch_size]
        pred_means = pred_samples.mean(axis=0).detach().numpy()
        pred_std = pred_samples.std(axis=0).detach().numpy()

        # Extract individual contributions of each subnetwork.
        submodel_contributions = {}
        num_samples = pred_samples.shape[0]
        batch_size = pred_samples.shape[1]

        for feature_name, feature_network in self._num_feature_networks.items():
            x = num_features[feature_name]
            contributions = self._compute_subnetwork_output(
                subnetwork=feature_network,
                x=x,
                samples=samples
            )
            submodel_contributions[feature_name] = contributions

        for feature_name, feature_network in self._cat_feature_networks.items():
            x = cat_features[feature_name].float()
            contributions = self._compute_subnetwork_output(
                subnetwork=feature_network,
                x=x,
                samples=samples
            )
            submodel_contributions[feature_name] = contributions

        if self._interaction_degree is not None and self._interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for interaction_name, interaction_network in self._interaction_networks.items():
                feature_names = interaction_name.split(":")
                x = torch.cat(
                    [all_features[name].unsqueeze(-1) for name in feature_names], dim=-1
                )
                contributions = self._compute_subnetwork_output(
                    subnetwork=interaction_network,
                    x=x,
                    samples=samples
                )
                submodel_contributions[interaction_name] = contributions

        return pred_means, pred_std, submodel_contributions
