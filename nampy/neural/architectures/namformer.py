import torch
import torch.nn as nn

from ..configs.namformer_config import DefaultNAMformerConfig
from .components.base_model import BaseModel
from .components.embeddings import EmbeddingLayer
from .components.interactions import resolve_interactions
from .components.mlp import MLP
from .components.normalization import LayerNorm
from .components.transformer import CustomTransformerEncoderLayer


class NAMformer(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMformerConfig | None = None,
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
        config : DefaultNAMformerConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNAMformerConfig().
        """
        if config is None:
            config = DefaultNAMformerConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        self.interactions = self.hparams.get("interactions", config.interactions)
        self.intercept: nn.Parameter | None
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        # TODO: this is plain elementwise nn.Dropout over concatenated term
        # outputs, unlike the term-mask feature dropout in
        # architectures/components/interactions.py used by NAM/QNAM/TreeNAM/SplineNAM.
        # The semantics differ; do not unify without an explicit decision.
        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

        # Initialize sub-networks for each feature
        self.feature_networks = nn.ModuleList(
            [
                nn.Linear(self.hparams.get("d_model", config.d_model), num_classes)
                for _ in range(len(num_feature_info) + len(cat_feature_info))
            ]
        )

        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            d_model=self.hparams.get("d_model", config.d_model),
            embedding_activation=self.hparams.get(
                "embedding_activation", config.embedding_activation
            ),
            layer_norm_after_embedding=self.hparams.get(
                "layer_norm_after_embedding", config.layer_norm_after_embedding
            ),
            use_cls=True,
            cls_position=0,
        )

        self.tabular_head = MLP(
            self.hparams.get("d_model", config.d_model),
            hidden_units_list=self.hparams.get(
                "head_layer_sizes", config.head_layer_sizes
            ),
            dropout=self.hparams.get("head_dropout", config.head_dropout),
            use_skip_layers=self.hparams.get(
                "head_skip_layers", config.head_skip_layers
            ),
            activation=self.hparams.get("head_activation", config.head_activation),
            use_batch_norm=self.hparams.get(
                "head_use_batch_norm", config.head_use_batch_norm
            ),
            n_output_units=num_classes,
        )

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.hparams.get("d_model", config.d_model),
            nhead=self.hparams.get("n_heads", config.n_heads),
            batch_first=True,
            dim_feedforward=self.hparams.get(
                "transformer_dim_feedforward", config.transformer_dim_feedforward
            ),
            dropout=self.hparams.get("attn_dropout", config.attn_dropout),
            activation=self.hparams.get(
                "transformer_activation", config.transformer_activation
            ),
            layer_norm_eps=self.hparams.get("layer_norm_eps", config.layer_norm_eps),
            norm_first=self.hparams.get("norm_first", config.norm_first),
            bias=self.hparams.get("bias", config.bias),
        )

        self.norm_embedding = LayerNorm(self.hparams.get("d_model", config.d_model))
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.get("n_layers", config.n_layers),
            norm=self.norm_embedding,
        )

        self.interaction_networks = nn.ModuleDict()
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
                input_dim = len(interaction) * self.hparams.get(
                    "d_model", config.d_model
                )
                self.interaction_networks[interaction_name] = (
                    self._create_interaction_subnetwork(input_dim, config)
                )

    def _create_interaction_subnetwork(self, input_dim, config):
        return MLP(
            n_input_units=input_dim,
            hidden_units_list=self.hparams.get(
                "head_layer_sizes", config.head_layer_sizes
            ),
            n_output_units=self.num_classes,
            dropout=self.hparams.get("dropout", config.dropout),
            use_skip_layers=self.hparams.get(
                "skip_connections", config.skip_connections
            ),
            activation=self.hparams.get("activation", config.activation),
            use_batch_norm=self.hparams.get("batch_norm", config.batch_norm),
            use_layer_norm=self.hparams.get("layer_norm", config.layer_norm),
            norm=self.hparams.get("norm", config.norm),
            use_glu=self.hparams.get("use_glu", config.use_glu),
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
        if self.interaction_degree is not None and self.interaction_degree >= 2:
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

        # Sum all feature outputs (main effects) and interaction outputs

        all_outputs = (
            [x] + list(nam_outputs.values()) + list(interaction_outputs.values())
        )

        # Make sure all tensors have the same number of dimensions
        all_outputs = [
            output.unsqueeze(-1) if output.dim() == 2 else output
            for output in all_outputs
        ]

        # Concatenate all feature outputs
        concatenated = torch.cat(all_outputs, dim=-1)
        # Apply feature dropout
        concatenated = self.feature_dropout(concatenated)

        # Sum across features, keeping the num_classes dimension
        num_features_total = len(all_outputs)
        if self.num_classes > 1:
            # Reshape to [batch_size, num_features, num_classes] and sum
            x = concatenated.view(-1, num_features_total, self.num_classes).sum(dim=1)
        else:
            # For single output, sum and keep dimension
            x = concatenated.sum(dim=-1)

        # intercept
        if self.intercept is not None:
            x += self.intercept

        # Combine the output tensor with the original feature values
        result = {"output": x}
        result.update(nam_outputs)
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result
