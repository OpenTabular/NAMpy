import torch
import torch.nn as nn
from ..configs.nam_config import DefaultNAMConfig
from .basemodel import BaseModel
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
import matplotlib.pyplot as plt
from itertools import combinations


class NAM(BaseModel):
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
        cat_feature_info : Any
            Information about categorical features.
        num_feature_info : Any
            Information about numerical features.
        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNAMConfig().
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

        # Initialize sub-networks for each feature
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, input_shape in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                input_shape, config
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, input_shape in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork(
                1, config
            )  # Categorical features are typically encoded as single values

        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self.interaction_networks = nn.ModuleDict()
            all_feature_names = list(num_feature_info.keys()) + list(
                cat_feature_info.keys()
            )

            # Add pairwise and higher interactions up to the specified degree
            for degree in range(2, self.interaction_degree + 1):
                for interaction in combinations(all_feature_names, degree):
                    interaction_name = ":".join(
                        interaction
                    )  # e.g., "feature1_feature2"
                    input_dim = 0

                    # Calculate input dimension for the interaction
                    for feature in interaction:
                        if feature in num_feature_info:
                            input_dim += num_feature_info[feature]  # Numerical features
                        elif feature in cat_feature_info:
                            input_dim += cat_feature_info[
                                feature
                            ]  # Categorical features (assumed 1 if encoded)

                    self.interaction_networks[interaction_name] = (
                        self._create_subnetwork(input_dim, config)
                    )

    def _create_subnetwork(self, input_dim, config):
        """
        Creates a subnetwork for a single feature.

        Parameters
        ----------
        input_dim : int
            Dimension of the input feature.
        config : DefaultNAMConfig
            Configuration dataclass containing hyperparameters.

        Returns
        -------
        nn.Sequential
            Subnetwork for the feature.
        """
        layers = nn.Sequential()
        layers.add_module("input", nn.Linear(input_dim, config.layer_sizes[0]))

        if config.batch_norm:
            layers.add_module("batch_norm", nn.BatchNorm1d(config.layer_sizes[0]))

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            layers.add_module("norm", RMSNorm(config.layer_sizes[0]))
        elif norm_layer == "LayerNorm":
            layers.add_module("norm", LayerNorm(config.layer_sizes[0]))
        elif norm_layer == "BatchNorm":
            layers.add_module("norm", BatchNorm(config.layer_sizes[0]))
        elif norm_layer == "InstanceNorm":
            layers.add_module("norm", InstanceNorm(config.layer_sizes[0]))
        elif norm_layer == "GroupNorm":
            layers.add_module("norm", GroupNorm(1, config.layer_sizes[0]))
        elif norm_layer == "LearnableLayerScaling":
            layers.add_module("norm", LearnableLayerScaling(config.layer_sizes[0]))

        if config.use_glu:
            layers.add_module("glu", nn.GLU())
        else:
            layers.add_module(
                "activation", self.hparams.get("activation", config.activation)
            )

        if config.dropout > 0.0:
            layers.add_module("dropout", nn.Dropout(config.dropout))

        for i in range(1, len(config.layer_sizes)):
            layers.add_module(
                f"linear_{i}",
                nn.Linear(config.layer_sizes[i - 1], config.layer_sizes[i]),
            )
            if config.batch_norm:
                layers.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(config.layer_sizes[i])
                )
            if config.layer_norm:
                layers.add_module(
                    f"layer_norm_{i}", nn.LayerNorm(config.layer_sizes[i])
                )
            if config.use_glu:
                layers.add_module(f"glu_{i}", nn.GLU())
            else:
                layers.add_module(
                    f"activation_{i}", self.hparams.get("activation", config.activation)
                )
            if config.dropout > 0.0:
                layers.add_module(f"dropout_{i}", nn.Dropout(config.dropout))

        layers.add_module(
            f"linear_{i+1}",
            nn.Linear(config.layer_sizes[i], self.num_classes),
        )
        return layers

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
            feature_output = feature_network(num_features[feature_name])
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name])
            cat_outputs[feature_name] = feature_output

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
                interaction_output = interaction_network(input_features)
                interaction_outputs[interaction_name] = interaction_output

        # Sum all feature outputs (main effects) and interaction outputs
        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        # feature dropout and sum
        x = self.feature_dropout(torch.cat(all_outputs, dim=1)).sum(dim=1)

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
