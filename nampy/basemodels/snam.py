import torch
import torch.nn as nn
from ..configs.snam_config import DefaultSNAMConfig
from .basemodel import BaseModel

from ..splines.neural_splines import CubicSplineLayer
from itertools import combinations


class SNAM(BaseModel):
    """
    Neural Additive Model (NAM) class with CubicSplineLayer.

    This class implements a Neural Additive Model (NAM) using cubic spline layers for feature modeling,
    with support for numerical and categorical features, interaction terms, and various normalization layers.

    Attributes
    ----------
    num_feature_networks : nn.ModuleDict
        Spline networks for each numerical feature.
    cat_feature_networks : nn.ModuleDict
        Spline networks for each categorical feature.
    interaction_networks : nn.ModuleDict
        Networks for modeling feature interactions (if applicable).
    interaction_degree : int, optional
        Degree of interactions to be modeled.
    intercept : torch.nn.Parameter
        Learnable intercept term, if enabled.
    feature_dropout : nn.Dropout
        Dropout layer for regularizing feature contributions.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultSNAMConfig = DefaultSNAMConfig(),
        **kwargs,
    ):
        """
        Initializes the Neural Additive Model (NAM) with cubic spline layers.

        Parameters
        ----------
        cat_feature_info : dict
            Dictionary providing information about categorical features (e.g., input dimensions).
        num_feature_info : dict
            Dictionary providing information about numerical features (e.g., input dimensions).
        num_classes : int, optional
            Number of output classes for classification tasks, by default 1.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters for the model, by default DefaultNAMConfig.
        kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

        # Initialize spline layers for each numerical feature
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=config.n_knots,
                min_val=0,
                max_val=1,
                learn_knots=config.learn_knots,
                identify=config.identify,
            )

        # Initialize spline layers for each categorical feature (one-hot or ordinal encoded)
        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=config.n_knots,
                min_val=0,
                max_val=info["dimension"],
                learn_knots=config.learn_knots,
                identify=config.identify,
            )  # Categorical features are typically encoded as single values

        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

    def _create_interaction_networks(self, num_feature_info, cat_feature_info, config):
        """
        Creates networks for modeling feature interactions using spline layers.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        config : DefaultNAMConfig
            Configuration dataclass containing model hyperparameters.
        """
        self.interaction_networks = nn.ModuleDict()
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        # Add pairwise and higher interactions up to the specified degree
        for degree in range(2, self.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                interaction_name = ":".join(interaction)  # e.g., "feature1_feature2"
                input_dim = 0

                # Calculate input dimension for the interaction
                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature]["dimension"]
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["dimension"]

                self.interaction_networks[interaction_name] = CubicSplineLayer(
                    n_bases=config.n_bases,
                    min_val=0,
                    max_val=input_dim,
                    learn_knots=config.learn_knots,
                    identify=config.identify,
                )

    def _interaction_forward(self, num_features: dict, cat_features: dict):
        """
        Forward pass for the interaction networks.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Outputs from the interaction networks, keyed by interaction names.
        """
        interaction_outputs = {}
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {
                **num_features,
                **cat_features,
            }
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                )
                interaction_output = interaction_network(input_features)
                interaction_outputs[interaction_name] = interaction_output

        return interaction_outputs

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model using cubic splines.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Dictionary containing the output tensor and the original feature values.
        """
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name])
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = self._interaction_forward(
            num_features=num_features, cat_features=cat_features
        )

        # Sum all feature outputs (main effects) and interaction outputs
        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        x = torch.cat(all_outputs, dim=1).sum(dim=1).unsqueeze(-1)

        if self.intercept is not None:
            x += self.intercept

        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result
