import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.qnam_config import DefaultQNAMConfig
from .basemodel import BaseModel
from .interactions import (
    apply_feature_dropout,
    create_interaction_networks,
    interaction_forward,
    sum_feature_dims,
)
from .mlp_utils import MLP


class QNAM(BaseModel):
    """
    Quantile Neural Additive Model (QNAM).

    This is a neural additive model whose outputs are constrained to be
    nondecreasing across the output dimension, which is interpreted as ordered
    quantiles.

    The constraint is enforced at the level of:
    - each main-effect feature contribution,
    - each interaction contribution,
    - the intercept.

    Therefore, the summed final output is also nondecreasing across quantiles.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultQNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultQNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        # Optimization hyperparameters
        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        # Data / task metadata
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )

        # Architecture hyperparameters (same pattern as corrected NAM)
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

        self.feature_dropout_p = self.hparams.get(
            "feature_dropout", config.feature_dropout
        )

        # Raw intercept parameter; transformed to monotone quantile outputs in forward
        self.raw_intercept: nn.Parameter | None
        if self.hparams.get("intercept", config.intercept):
            self.raw_intercept = nn.Parameter(torch.zeros(num_classes))
        else:
            self.raw_intercept = None

        # Main-effect subnetworks
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )

        # Interaction subnetworks
        self.interaction_networks = create_interaction_networks(
            list(num_feature_info.keys()) + list(cat_feature_info.keys()),
            self.interaction_degree,
            lambda interaction: self._create_subnetwork(
                sum_feature_dims(interaction, num_feature_info, cat_feature_info)
            ),
        )

        # Monotone transform function
        self.monotone_transform = self.hparams.get(
            "monotone_transform", config.monotone_transform
        )
        self.min_increment = self.hparams.get("min_increment", config.min_increment)

    def _create_subnetwork(self, input_dim: int) -> MLP:
        """Create one feature/interaction subnetwork."""
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

    def _monotone_transform(
        self, x: torch.Tensor, min_increment: float = 0.00
    ) -> torch.Tensor:
        """
        Transform raw outputs into a nondecreasing sequence across the last dimension.

        For quantiles q1 < q2 < ... < qK:
        - first value is unconstrained,
        - subsequent increments are positive via softplus,
        - cumulative sum enforces ordering.

        Works for shapes [..., num_classes].
        """
        if x.shape[-1] <= 1:
            return x

        first = x[..., :1]
        if self.monotone_transform == "softplus":
            increments = F.softplus(x[..., 1:] + min_increment)
        elif self.monotone_transform == "exponential":
            increments = torch.exp(x[..., 1:] + min_increment)
        else:
            raise ValueError(f"Invalid monotone transform: {self.monotone_transform}")
        return torch.cat(
            [first, first + torch.cumsum(increments, dim=-1)],
            dim=-1,
        )

    def _get_intercept(self):
        """Return the transformed monotone intercept actually used by the model."""
        if self.raw_intercept is None:
            return None
        return self._monotone_transform(self.raw_intercept.unsqueeze(0)).squeeze(0)

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of QNAM.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical feature tensors.
        cat_features : dict
            Dictionary of categorical feature tensors.

        Returns
        -------
        dict
            Dictionary containing:
            - "output": final quantile predictions [batch, num_classes]
            - one entry per feature contribution [batch, num_classes]
            - one entry per interaction contribution [batch, num_classes], if enabled
            - "intercept": transformed monotone intercept [num_classes], if enabled
        """
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            raw_output = feature_network(num_features[feature_name].float())
            num_outputs[feature_name] = self._monotone_transform(raw_output)

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            raw_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = self._monotone_transform(raw_output)

        interaction_outputs = interaction_forward(
            self.interaction_networks,
            self.interaction_degree,
            num_features,
            cat_features,
            lambda network, input_features: self._monotone_transform(
                network(input_features)
            ),
        )

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )

        if not all_outputs:
            raise ValueError("QNAM received no feature contributions to sum.")

        # [B, T, Q]; feature-level dropout drops whole term contributions
        # consistently across all quantiles, preserving monotonicity within
        # each kept term.
        stacked = torch.stack(all_outputs, dim=1)
        stacked = apply_feature_dropout(stacked, self.feature_dropout_p, self.training)

        x = stacked.sum(dim=1)  # [B, Q]

        intercept = self._get_intercept()
        if intercept is not None:
            x = x + intercept

        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)

        if intercept is not None:
            result["intercept"] = intercept

        return result
