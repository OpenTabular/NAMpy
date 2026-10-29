"""Neural Additive Tabular Transformer forward architecture.

References
----------
- Thielmann, A. F., Reuter, A., Kneib, T., Rügamer, D., and Säfken, B. (2024).
  Interpretable Additive Tabular Transformer Networks.
  https://openreview.net/forum?id=TdJ7lpzAkD
"""

import torch
import torch.nn as nn

from ..configs.natt_config import DefaultNATTConfig
from .components.base_model import BaseModel
from .components.embeddings import EmbeddingLayer
from .components.interactions import resolve_interactions, sum_term_outputs
from .components.mlp import MLP, make_activation
from .components.module_dict import RawKeyModuleDict
from .components.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)
from .components.transformer import CustomTransformerEncoderLayer


class NATT(BaseModel):
    """Attention-based architecture for additive tabular modeling."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNATTConfig | None = None,
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
        config : DefaultNATTConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNATTConfig().
        """
        if config is None:
            config = DefaultNATTConfig()
        super().__init__()

        self.cat_feature_info = cat_feature_info
        self.contextual_feature_names = tuple(cat_feature_info)
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.interaction_degree = config.interaction_degree
        self.interactions = config.interactions
        self.intercept: nn.Parameter | None
        if config.intercept:
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        self.feature_dropout_p = config.feature_dropout

        # Initialize sub-networks for each feature
        self.num_feature_networks = RawKeyModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"], config
            )

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            d_model=config.d_model,
            embedding_activation=config.embedding_activation,
            layer_norm_after_embedding=config.layer_norm_after_embedding,
            use_cls=True,
            cls_position=0,
        )

        self.tabular_head = MLP(
            config.d_model,
            hidden_units_list=config.head_layer_sizes,
            dropout=config.head_dropout,
            use_skip_layers=config.head_skip_layers,
            activation=config.head_activation,
            use_batch_norm=config.head_use_batch_norm,
            n_output_units=num_classes,
        )

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            batch_first=True,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.attn_dropout,
            activation=config.transformer_activation,
            layer_norm_eps=config.layer_norm_eps,
            norm_first=config.norm_first,
            bias=config.bias,
        )

        self.norm_embedding = LayerNorm(config.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers, norm=self.norm_embedding
        )

        if self.interactions is not None or (
            self.interaction_degree is not None and self.interaction_degree >= 2
        ):
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

    def _create_interaction_networks(self, num_feature_info, cat_feature_info, config):
        """
        Creates networks for modeling feature interactions.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        config : DefaultNAMConfig
            Configuration dataclass containing model hyperparameters.
        """

        self.interaction_networks = RawKeyModuleDict()
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        for interaction in resolve_interactions(
            all_feature_names, self.interaction_degree, self.interactions
        ):
            interaction_name = ":".join(interaction)
            input_dim = sum(
                (num_feature_info.get(feature) or cat_feature_info[feature])[
                    "dimension"
                ]
                for feature in interaction
            )
            self.interaction_networks[interaction_name] = self._create_subnetwork(
                input_dim, config
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

        Notes
        -----
        Deliberately hand-rolled rather than ``components.mlp.MLP``: the layer
        sequence is not expressible via MLP parameters. The named ``norm``
        is applied after the first linear layer only (MLP applies it to every
        hidden block), ``batch_norm`` can coexist with a named norm (MLP
        forbids that combination), hidden layers use native ``nn.LayerNorm``
        while the named norm uses the custom components.normalization classes,
        and the GLU path here does not halve the running width as MLP does.
        """
        layers = nn.Sequential()
        layers.add_module("input", nn.Linear(input_dim, config.layer_sizes[0]))

        if config.batch_norm:
            layers.add_module("batch_norm", nn.BatchNorm1d(config.layer_sizes[0]))

        norm_layer = config.norm
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
            layers.add_module("activation", make_activation(config.activation))

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
                layers.add_module(f"activation_{i}", make_activation(config.activation))
            if config.dropout > 0.0:
                layers.add_module(f"dropout_{i}", nn.Dropout(config.dropout))

        layers.add_module(
            "output",
            nn.Linear(config.layer_sizes[-1], self.num_classes),
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
        if cat_features:
            cat_embeddings = self.embedding_layer(None, list(cat_features.values()))
            cat_vals = self.encoder(cat_embeddings)
            cat_vals = self.tabular_head(cat_vals[:, 1:])
            cat_outputs = {
                feature_name: cat_vals[:, index]
                for index, feature_name in enumerate(cat_features)
            }

        interaction_outputs = self._interaction_forward(
            num_features=num_features, cat_features=cat_features
        )

        return sum_term_outputs(
            {**num_outputs, **cat_outputs, **interaction_outputs},
            self.intercept,
            p=self.feature_dropout_p,
            training=self.training,
        )
