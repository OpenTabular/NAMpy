import torch
import torch.nn as nn
from ..configs.namformer_config import DefaultNAMformerConfig
from .basemodel import BaseModel
from ..arch_utils.normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..arch_utils.embedding_layer import EmbeddingLayer
from ..arch_utils.mlp_utils import MLP
from ..arch_utils.transformer_utils import CustomTransformerEncoderLayer
from itertools import combinations


class NAMformer(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMformerConfig = DefaultNAMformerConfig(),
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
        self.feature_networks = nn.ModuleList(
            [
                nn.Linear(self.hparams.get("d_model", config.d_model), 1)
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
            cat_encoding=self.hparams.get("cat_encoding", config.cat_encoding),
        )

        self.tabular_head = MLP(
            self.hparams.get("d_model", config.d_model),
            hidden_units_list=self.hparams.get(
                "head_layer_sizes", config.head_layer_sizes
            ),
            dropout_rate=self.hparams.get("head_dropout", config.head_dropout),
            use_skip_layers=self.hparams.get(
                "head_skip_layers", config.head_skip_layers
            ),
            activation_fn=self.hparams.get("head_activation", config.head_activation),
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
                    input_dim = self.interaction_degree * self.hparams.get(
                        "d_model", config.d_model
                    )

                    self.interaction_networks[interaction_name] = (
                        self._create_subnetwork(input_dim, config)
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

        # Handle numerical features
        for i, feature_name in enumerate(num_features.keys()):
            nam_outputs[feature_name] = self.feature_networks[i](embeddings[:, i])

        # Handle categorical features
        for j, feature_name in enumerate(cat_features.keys(), start=len(num_features)):
            nam_outputs[feature_name] = self.feature_networks[j](embeddings[:, j])

        # Handle interaction networks
        # Create a dictionary for the embeddings of each feature (numerical + categorical)
        all_embeddings = {
            **{key: embeddings[:, i] for i, key in enumerate(num_features.keys())},
            **{
                key: embeddings[:, i + len(num_features)]
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
                    [all_embeddings[fn].unsqueeze(-1) for fn in feature_names], dim=-1
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

        # Now concatenate the tensors along the second dimension
        x = self.feature_dropout(torch.cat(all_outputs, dim=-1)).sum(dim=-1)

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
