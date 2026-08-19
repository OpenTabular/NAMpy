# basemodels/nam.py
import torch
import torch.nn as nn

from ..configs.nam_config import DefaultNAMConfig
from .basemodel import BaseModel
from .interactions import (
    apply_feature_dropout,
    create_interaction_networks,
    interaction_forward,
    sum_feature_dims,
)
from .mlp_utils import MLP


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
        config: DefaultNAMConfig | None = None,
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
            Configuration dataclass containing hyperparameters for the model, by default DefaultNAMConfig().
        kwargs : dict
            Additional keyword arguments.
        """
        if config is None:
            config = DefaultNAMConfig()
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
        self.intercept: nn.Parameter | None
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

        self.interaction_networks = create_interaction_networks(
            list(num_feature_info.keys()) + list(cat_feature_info.keys()),
            self.interaction_degree,
            lambda interaction: self._create_subnetwork(
                sum_feature_dims(interaction, num_feature_info, cat_feature_info)
            ),
        )

    def _create_subnetwork(self, input_dim: int) -> MLP:
        """Create a subnetwork for a single feature using MLP from mlp_utils."""
        return MLP(
            n_input_units=input_dim,
            hidden_units_list=self.layer_sizes,
            n_output_units=self.num_classes,
            dropout=self.dropout,
            use_skip_layers=self.skip_connections,
            activation=self.activation,
            use_batch_norm=self.batch_norm,
            use_layer_norm=self.layer_norm,
            norm=self.norm,
            use_glu=self.use_glu,
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
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name].float())
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = interaction_forward(
            self.interaction_networks,
            self.interaction_degree,
            num_features,
            cat_features,
            lambda network, input_features: network(input_features),
        )

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        concatenated = torch.cat(all_outputs, dim=1)
        num_features_total = len(all_outputs)

        shaped = concatenated.view(-1, num_features_total, self.num_classes)
        shaped = apply_feature_dropout(shaped, self.feature_dropout_p, self.training)

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
