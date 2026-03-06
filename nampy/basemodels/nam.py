# nam.py
from itertools import combinations

import torch
import torch.nn as nn

from ..arch_utils.mlp_utils import MLP
from ..configs.nam_config import DefaultNAMConfig
from .basemodel import BaseModel


class NAM(BaseModel):
    """
    Neural Additive Model (NAM) class.

    This class implements a Neural Additive Model (NAM) with support for numerical and
    categorical features, interaction terms, and various normalization layers.

    Attributes
    ----------
    num_feature_networks : nn.ModuleDict
        Sub-networks for each numerical feature.
    cat_feature_networks : nn.ModuleDict
        Sub-networks for each categorical feature.
    interaction_networks : nn.ModuleDict
        Networks for modeling feature interactions (if applicable).
    interaction_degree : int, optional
        Degree of interactions to be modeled.
    intercept : torch.nn.Parameter
        Learnable intercept term, if enabled.
    feature_dropout_p : float
        Probability for feature-level dropout (drops whole feature outputs).
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMConfig = DefaultNAMConfig(),
        **kwargs,
    ):
        """
        Initializes the Neural Additive Model (NAM) with the given configuration.

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

        self.feature_dropout_p = self.hparams.get(
            "feature_dropout", config.feature_dropout
        )

        # Resolve architecture hyperparameters once, with kwargs overriding config
        self.layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        self.activation = self.hparams.get("activation", config.activation)
        self.norm = self.hparams.get("norm", config.norm)
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.batch_norm = self.hparams.get("batch_norm", config.batch_norm)
        self.layer_norm = self.hparams.get("layer_norm", config.layer_norm)
        self.dropout = self.hparams.get("dropout", config.dropout)

        # Initialize sub-networks for each feature
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )  # Categorical features are typically encoded as single values

        reserved = {"output", "intercept"}
        all_feature_names = set(num_feature_info) | set(cat_feature_info)
        if reserved & all_feature_names:
            raise ValueError(
                f"Feature names {sorted(reserved.intersection(all_feature_names))} are reserved."
            )

        self.interaction_networks = nn.ModuleDict()
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

    def _create_subnetwork(self, input_dim: int) -> MLP:
        """Create a subnetwork for a single feature using MLP from mlp_utils."""
        return MLP(
            n_input_units=input_dim,
            hidden_units_list=self.layer_sizes,
            n_output_units=self.num_classes,
            dropout_rate=self.dropout,
            use_skip_layers=self.skip_connections,
            activation_fn=self.activation,
            use_batch_norm=self.batch_norm,
            use_layer_norm=self.layer_norm,
            norm=self.norm,
            use_glu=self.use_glu,
        )

    def _create_interaction_networks(self, num_feature_info, cat_feature_info):
        """
        Creates networks for modeling feature interactions.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        """
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
                        input_dim += num_feature_info[feature][
                            "dimension"
                        ]  # Numerical features
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["dimension"]

                self.interaction_networks[interaction_name] = self._create_subnetwork(
                    input_dim
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
        # Handle interaction networks
        interaction_outputs = {}
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {
                **num_features,
                **cat_features,
            }  # Combine numerical and categorical features
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                )
                interaction_output = interaction_network(input_features.float())
                interaction_outputs[interaction_name] = interaction_output

        return interaction_outputs

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model.

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
            feature_output = feature_network(num_features[feature_name].float())
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = self._interaction_forward(
            num_features=num_features, cat_features=cat_features
        )

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        concatenated = torch.cat(all_outputs, dim=1)
        num_features_total = len(all_outputs)

        if self.feature_dropout_p > 0.0 and self.training:
            shaped = concatenated.view(-1, num_features_total, self.num_classes)
            mask = torch.ones(
                shaped.shape[0], num_features_total, 1,
                device=shaped.device, dtype=shaped.dtype,
            )
            mask = nn.functional.dropout(
                mask, p=self.feature_dropout_p, training=True
            )
            shaped = shaped * mask
        else:
            shaped = concatenated.view(-1, num_features_total, self.num_classes)

        x = shaped.sum(dim=1)  # [batch, num_classes]

        # intercept
        if self.intercept is not None:
            x += self.intercept

        # Combine the output tensor with the original feature values
        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result
