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

        # Sum all feature outputs
        all_outputs = list(num_outputs.values()) + list(cat_outputs.values())
        x = torch.stack(all_outputs, dim=1).sum(dim=1)

        # Combine the output tensor with the original feature values
        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)

        return result
