from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.qnam_config import DefaultQNAMConfig
from ..layers.mlp_utils import MLP
from .basemodel import BaseModel


class QNAMBase(BaseModel):
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

        reserved = {"output", "intercept", "output_penalty"}
        all_feature_names = set(num_feature_info) | set(cat_feature_info)
        if reserved & all_feature_names:
            raise ValueError(
                f"Feature names {sorted(reserved.intersection(all_feature_names))} are reserved."
            )
        if any(":" in name for name in all_feature_names):
            bad = sorted(name for name in all_feature_names if ":" in name)
            raise ValueError(
                f"Feature names {bad} contain ':', which is reserved for interaction names."
            )

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
        self.interaction_networks = nn.ModuleDict()
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
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

    def _create_interaction_networks(self, num_feature_info, cat_feature_info):
        """Create interaction subnetworks up to the specified interaction degree."""
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        for degree in range(2, self.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                interaction_name = ":".join(interaction)
                input_dim = 0

                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature]["dimension"]
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["dimension"]

                self.interaction_networks[interaction_name] = self._create_subnetwork(
                    input_dim
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

    def _interaction_forward(self, num_features: dict, cat_features: dict) -> dict:
        """Forward pass through interaction subnetworks."""
        interaction_outputs = {}
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {**num_features, **cat_features}
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                ).float()
                raw_output = interaction_network(input_features)
                interaction_outputs[interaction_name] = self._monotone_transform(
                    raw_output
                )

        return interaction_outputs

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

        interaction_outputs = self._interaction_forward(
            num_features=num_features,
            cat_features=cat_features,
        )

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )

        if not all_outputs:
            raise ValueError("QNAM received no feature contributions to sum.")

        # [B, T, Q]
        stacked = torch.stack(all_outputs, dim=1)
        num_terms = stacked.shape[1]

        # Proper feature-level dropout: drop whole term contributions consistently
        # across all quantiles, preserving monotonicity within each kept term.
        if self.feature_dropout_p > 0.0 and self.training:
            mask = torch.ones(
                stacked.shape[0],
                num_terms,
                1,
                device=stacked.device,
                dtype=stacked.dtype,
            )
            mask = F.dropout(mask, p=self.feature_dropout_p, training=True)
            stacked = stacked * mask

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
