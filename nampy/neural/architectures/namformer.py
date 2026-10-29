"""Transformer predictors with identifiable marginal-effect heads.

References
----------
- Thielmann, A., Reuter, A., and Säfken, B. (2025). Beyond Black-Box Predictions:
  Identifying Marginal Feature Effects in Tabular Transformer Networks.
  https://arxiv.org/abs/2504.08712
"""

import torch
import torch.nn as nn

from ..configs.namformer_config import DefaultNAMformerConfig
from .components.base_model import BaseModel
from .components.embeddings import EmbeddingLayer
from .components.interactions import resolve_interactions, sum_term_outputs
from .components.mlp import MLP
from .components.module_dict import RawKeyModuleDict
from .components.normalization import LayerNorm
from .components.transformer import CustomTransformerEncoderLayer


class NAMformer(BaseModel):
    """Transformer architecture with marginal heads and a global contribution."""

    global_term_names = ("global_transformer",)

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMformerConfig | None = None,
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
        config : DefaultNAMformerConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNAMformerConfig().
        """
        if config is None:
            config = DefaultNAMformerConfig()
        super().__init__()

        self.cat_feature_info = cat_feature_info
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
        self.feature_networks = nn.ModuleList(
            [
                nn.Linear(config.d_model, num_classes)
                for _ in range(len(num_feature_info) + len(cat_feature_info))
            ]
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

        self.interaction_networks = RawKeyModuleDict()
        if self.interactions is not None or (
            self.interaction_degree is not None and self.interaction_degree >= 2
        ):
            all_feature_names = list(num_feature_info.keys()) + list(
                cat_feature_info.keys()
            )

            for interaction in resolve_interactions(
                all_feature_names, self.interaction_degree, self.interactions
            ):
                interaction_name = ":".join(interaction)
                input_dim = len(interaction) * config.d_model
                self.interaction_networks[interaction_name] = (
                    self._create_interaction_subnetwork(input_dim, config)
                )

    def _create_interaction_subnetwork(self, input_dim, config):
        return MLP(
            n_input_units=input_dim,
            hidden_units_list=config.head_layer_sizes,
            n_output_units=self.num_classes,
            dropout=config.dropout,
            use_skip_layers=config.skip_connections,
            activation=config.activation,
            use_batch_norm=config.batch_norm,
            use_layer_norm=config.layer_norm,
            norm=config.norm,
            use_glu=config.use_glu,
        )

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

        # Extract embeddings for numerical and categorical features
        embeddings = self.embedding_layer(
            [vals for key, vals in num_features.items()],
            [vals for key, vals in cat_features.items()],
        )

        x = self.encoder(embeddings)
        x = self.tabular_head(x)
        x = x[:, 0]

        # Create a dictionary for feature values, using keys from num_features and cat_features
        nam_outputs = {}

        # EmbeddingLayer places CLS first, then categorical, then numerical tokens.
        cls_offset = 1
        numerical_offset = cls_offset + len(cat_features)

        # Handle numerical features
        for i, feature_name in enumerate(num_features.keys()):
            network_idx = i
            embedding_idx = numerical_offset + i
            nam_outputs[feature_name] = self.feature_networks[network_idx](
                embeddings[:, embedding_idx]
            )

        # Handle categorical features
        for j, feature_name in enumerate(cat_features.keys()):
            network_idx = len(num_features) + j
            embedding_idx = cls_offset + j
            nam_outputs[feature_name] = self.feature_networks[network_idx](
                embeddings[:, embedding_idx]
            )

        # Handle interaction networks
        # Create a dictionary for the embeddings of each feature (numerical + categorical)
        all_embeddings = {
            **{
                key: embeddings[:, numerical_offset + i]
                for i, key in enumerate(num_features.keys())
            },
            **{
                key: embeddings[:, cls_offset + i]
                for i, key in enumerate(cat_features.keys())
            },
        }

        interaction_outputs = {}
        if self.interaction_networks:
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                # Split the interaction name to get feature names
                feature_names = interaction_name.split(":")

                # Use the corresponding embeddings for the input to the interaction network
                input_features = torch.cat(
                    [all_embeddings[fn] for fn in feature_names], dim=-1
                )

                # Pass the concatenated embeddings through the interaction network
                interaction_output = interaction_network(input_features)

                # Store the interaction output
                interaction_outputs[interaction_name] = interaction_output

        # The global transformer head depends on every input token. Return it
        # explicitly so explanations account for the complete predictor.
        return sum_term_outputs(
            {"global_transformer": x, **nam_outputs, **interaction_outputs},
            self.intercept,
            p=self.feature_dropout_p,
            training=self.training,
        )
