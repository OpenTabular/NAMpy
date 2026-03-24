from collections import OrderedDict
from itertools import combinations

import torch
import torch.nn as nn

from ..arch_utils.nbm_utils import ConceptNNBasesNary
from ..configs.nbm_config import DefaultNBMConfig
from .basemodel import BaseModel


class NBM(BaseModel):
    """
    Neural Basis Model (NBM) with optional higher-order interactions.

    Notes
    -----
    - This implementation treats each post-preprocessing scalar column as an atomic feature.
      If an original feature expands to multiple columns (e.g. one-hot / binning / splines),
      each column becomes its own unary term, which is the cleanest way to stay faithful
      to the paper's NBM decomposition.
    - The returned dictionary contains:
        * "output": final prediction tensor of shape [batch, num_classes]
        * one entry per main effect / interaction term, each of shape [batch, num_classes]
        * optionally "intercept"
        * optionally "output_penalty"
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNBMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultNBMConfig()
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

        # NBM hyperparameters
        self.num_bases = self.hparams.get("num_bases", config.num_bases)
        self.num_subnets = self.hparams.get("num_subnets", config.num_subnets)
        self.output_penalty = self.hparams.get(
            "output_penalty", getattr(config, "output_penalty", 0.0)
        )

        self.bases_dropout = nn.Dropout(
            p=self.hparams.get("bases_dropout", config.bases_dropout)
        )

        # Optional intercept (keep this; classifier bias is disabled below)
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(torch.zeros(num_classes))
        else:
            self.intercept = None

        # Architecture hyperparameters
        self.layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        self.activation = self.hparams.get("activation", config.activation)
        self.dropout = self.hparams.get("dropout", config.dropout)
        self.norm = self.hparams.get("norm", config.norm)
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.batch_norm = self.hparams.get("batch_norm", config.batch_norm)
        self.layer_norm = self.hparams.get("layer_norm", config.layer_norm)
        self.feature_dropout_p = self.hparams.get(
            "feature_dropout", config.feature_dropout
        )

        # Preserve deterministic feature order between init and forward
        self.num_feature_keys = list(num_feature_info.keys())
        self.cat_feature_keys = list(cat_feature_info.keys())

        reserved = {"output", "intercept", "output_penalty"}
        all_feature_names = set(self.num_feature_keys) | set(self.cat_feature_keys)
        if reserved & all_feature_names:
            raise ValueError(
                f"Feature names {sorted(reserved & all_feature_names)} are reserved."
            )
        if any(":" in name for name in all_feature_names):
            bad = sorted(name for name in all_feature_names if ":" in name)
            raise ValueError(
                f"Feature names {bad} contain ':', which is reserved for interaction names."
            )

        # Atomic features: each scalar post-preprocessing column becomes one feature
        self.atomic_feature_names = self._build_atomic_feature_names(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
        )
        self.num_atomic_features = len(self.atomic_feature_names)

        # Normalize nary specification to explicit tuples over atomic features
        raw_nary = self.hparams.get("nary", config.nary)
        interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        order = self.hparams.get("order", config.order)

        self.nary = self._normalize_nary(
            nary=raw_nary,
            interaction_degree=interaction_degree,
            order=order,
        )
        self._validate_nary(self.nary)

        # Register tuple index tensors as buffers so they move with the model
        self._nary_buffer_names = {}
        for order_key in self._sorted_order_keys():
            tuples_ = self.nary[order_key]
            idx = torch.tensor(tuples_, dtype=torch.long)
            buffer_name = f"_nary_idx_ord{order_key}"
            self.register_buffer(buffer_name, idx, persistent=False)
            self._nary_buffer_names[order_key] = buffer_name

        # Build metadata for every channel (term x subnet)
        # Each channel gets its own coefficient vector over shared bases.
        self.channel_specs = []
        self.term_to_channel_indices = OrderedDict()

        for order_key in self._sorted_order_keys():
            order = int(order_key)
            tuples_ = self.nary[order_key]

            for subnet in range(self.num_subnets):
                for tup in tuples_:
                    if order == 1:
                        name = self.atomic_feature_names[tup[0]]
                        kind = "main"
                    else:
                        name = ":".join(self.atomic_feature_names[i] for i in tup)
                        kind = "interaction"

                    channel_idx = len(self.channel_specs)
                    spec = {
                        "order": order,
                        "subnet": subnet,
                        "tuple": tuple(tup),
                        "name": name,
                        "kind": kind,
                    }
                    self.channel_specs.append(spec)
                    self.term_to_channel_indices.setdefault(name, []).append(channel_idx)

        self.num_channels = len(self.channel_specs)
        if self.num_channels == 0:
            raise ValueError("NBM requires at least one unary or interaction term.")

        # Shared basis networks: one per order and subnet
        self.bases_nary_models = nn.ModuleDict()
        for order_key in self._sorted_order_keys():
            order_size = int(order_key)
            for subnet in range(self.num_subnets):
                self.bases_nary_models[self.get_key(order_key, subnet)] = (
                    ConceptNNBasesNary(
                        order=order_size,
                        num_bases=self.num_bases,
                        layer_sizes=self.layer_sizes,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.batch_norm,
                        use_layer_norm=self.layer_norm,
                        norm=self.norm,
                        use_glu=self.use_glu,
                        skip_connections=self.skip_connections,
                    )
                )

        # Explicit term-specific coefficients over the shared bases:
        # unary terms use a_{ik}, interactions use b_{ijk}-style coefficients.
        self.term_basis_weights = nn.Parameter(
            torch.empty(self.num_channels, self.num_bases)
        )
        nn.init.xavier_uniform_(self.term_basis_weights)

        # Final class / parameter head.
        # Bias is disabled to avoid double-counting with self.intercept.
        self.classifier = nn.Linear(
            in_features=self.num_channels,
            out_features=self.num_classes,
            bias=False,
        )

    def _sorted_order_keys(self):
        """Return nary order keys sorted numerically as strings."""
        return sorted(self.nary.keys(), key=lambda x: int(x))

    def _build_atomic_feature_names(self, num_feature_info, cat_feature_info):
        """
        Build atomic feature names from post-preprocessing feature blocks.

        If a logical feature has dimension d > 1, it is split into names like
        'feature[0]', 'feature[1]', ..., 'feature[d-1]'.
        """
        atomic_names = []

        for name in self.num_feature_keys:
            dim = int(num_feature_info[name]["dimension"])
            if dim <= 0:
                raise ValueError(f"Numerical feature '{name}' has invalid dimension {dim}.")
            if dim == 1:
                atomic_names.append(name)
            else:
                for j in range(dim):
                    atomic_names.append(f"{name}[{j}]")

        for name in self.cat_feature_keys:
            dim = int(cat_feature_info[name]["dimension"])
            if dim <= 0:
                raise ValueError(
                    f"Categorical feature '{name}' has invalid dimension {dim}."
                )
            if dim == 1:
                atomic_names.append(name)
            else:
                for j in range(dim):
                    atomic_names.append(f"{name}[{j}]")

        return atomic_names

    def _normalize_nary(self, nary=None, interaction_degree=None, order=1):
        """
        Normalize nary specification.

        Precedence
        ----------
        1. explicit nary
        2. interaction_degree -> orders [1, ..., interaction_degree]
        3. order -> [order]

        Accepted explicit nary forms
        ----------------------------
        None
            Uses interaction_degree/order fallback.
        list/tuple
            Example: [1, 2] -> all unary and pairwise terms.
        dict
            Explicit tuple specification over atomic feature indices, e.g.
            {"1": [(0,), (1,)], "2": [(0, 3), (1, 2)]}
        """
        if nary is not None:
            if isinstance(nary, (list, tuple)):
                orders = sorted(set(int(o) for o in nary))
                if not orders:
                    raise ValueError("nary list/tuple must not be empty.")
                return {
                    str(o): list(combinations(range(self.num_atomic_features), o))
                    for o in orders
                }

            if isinstance(nary, dict):
                normalized = {}
                for key, tuples_ in nary.items():
                    o = int(key)
                    normalized[str(o)] = [tuple(map(int, tup)) for tup in tuples_]
                return normalized

            raise TypeError("nary must be None, a list/tuple of orders, or a dict.")

        if interaction_degree is not None:
            if int(interaction_degree) < 1:
                raise ValueError("interaction_degree must be >= 1.")
            orders = list(range(1, int(interaction_degree) + 1))
        else:
            if int(order) < 1:
                raise ValueError("order must be >= 1.")
            orders = [int(order)]

        return {
            str(o): list(combinations(range(self.num_atomic_features), o))
            for o in orders
        }

    def _validate_nary(self, nary):
        """Validate explicit nary tuples."""
        if not nary:
            raise ValueError("nary must contain at least one order.")

        for order_key, tuples_ in nary.items():
            order = int(order_key)
            if order < 1:
                raise ValueError(f"Invalid interaction order '{order}'. Must be >= 1.")

            if len(tuples_) == 0:
                continue

            for tup in tuples_:
                if len(tup) != order:
                    raise ValueError(
                        f"Tuple {tup} does not match declared order {order}."
                    )
                if len(set(tup)) != len(tup):
                    raise ValueError(
                        f"Tuple {tup} repeats indices, which is not allowed."
                    )
                for idx in tup:
                    if idx < 0 or idx >= self.num_atomic_features:
                        raise ValueError(
                            f"Tuple {tup} contains index {idx}, but valid atomic "
                            f"feature indices are [0, {self.num_atomic_features - 1}]."
                        )

    def get_key(self, order, subnet):
        """Generate a unique key for a basis model of a given order and subnet."""
        return f"ord{order}_net{subnet}"

    def _concat_all_features(self, num_features, cat_features):
        """Concatenate numerical and categorical feature tensors in a fixed order."""
        tensors = []

        for feature_name in self.num_feature_keys:
            x = num_features[feature_name]
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            tensors.append(x)

        for feature_name in self.cat_feature_keys:
            x = cat_features[feature_name]
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            tensors.append(x)

        if not tensors:
            raise ValueError("NBM received no input features.")

        return torch.cat(tensors, dim=1).float()

    def _apply_term_dropout(self, term_scores):
        """
        Apply dropout at the logical-term level.

        If the same logical term appears across multiple subnets, all of its channels
        are dropped together.
        """
        if self.feature_dropout_p <= 0.0 or not self.training:
            return term_scores

        batch_size = term_scores.shape[0]
        n_terms = len(self.term_to_channel_indices)

        logical_mask = torch.ones(
            batch_size,
            n_terms,
            device=term_scores.device,
            dtype=term_scores.dtype,
        )
        logical_mask = nn.functional.dropout(
            logical_mask, p=self.feature_dropout_p, training=True
        )

        channel_mask = torch.ones_like(term_scores)
        for logical_idx, channel_indices in enumerate(self.term_to_channel_indices.values()):
            channel_mask[:, channel_indices] = logical_mask[:, logical_idx].unsqueeze(-1)

        return term_scores * channel_mask

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the Neural Basis Model.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical feature tensors.
        cat_features : dict
            Dictionary of categorical feature tensors.

        Returns
        -------
        dict
            A dictionary containing:
            - "output": final prediction tensor [batch, num_classes]
            - one entry per main-effect / interaction term [batch, num_classes]
            - optionally "intercept"
            - optionally "output_penalty"
        """
        all_features = self._concat_all_features(num_features, cat_features)
        batch_size = all_features.shape[0]

        # Compute shared basis outputs for every term channel.
        # After concatenation, bases has shape [batch, num_channels, num_bases].
        basis_chunks = []

        for order_key in self._sorted_order_keys():
            idx = getattr(self, self._nary_buffer_names[order_key])
            if idx.numel() == 0:
                continue

            order = idx.shape[1]
            n_terms_for_order = idx.shape[0]

            # Gather inputs for all tuples of this order:
            # all_features: [B, D]
            # idx: [n_terms, order]
            # x_order: [B, n_terms, order]
            x_order = all_features[:, idx]
            x_order_flat = x_order.reshape(-1, order)

            for subnet in range(self.num_subnets):
                h = self.bases_nary_models[self.get_key(order_key, subnet)](x_order_flat)
                h = self.bases_dropout(h)
                h = h.reshape(batch_size, n_terms_for_order, self.num_bases)
                basis_chunks.append(h)

        if not basis_chunks:
            raise RuntimeError("No basis outputs were created. Check nary configuration.")

        bases = torch.cat(basis_chunks, dim=1)  # [B, num_channels, num_bases]

        # Term scores are the feature-/interaction-specific linear combinations
        # of the shared basis outputs.
        term_scores = (bases * self.term_basis_weights.unsqueeze(0)).sum(dim=-1)
        term_scores = self._apply_term_dropout(term_scores)  # [B, num_channels]

        # Class-/parameter-wise additive contributions:
        # classifier.weight: [num_classes, num_channels]
        # -> transpose to [num_channels, num_classes]
        term_contribs = (
            term_scores.unsqueeze(-1) * self.classifier.weight.transpose(0, 1).unsqueeze(0)
        )  # [B, num_channels, num_classes]

        output = term_contribs.sum(dim=1)  # [B, num_classes]
        if self.intercept is not None:
            output = output + self.intercept

        result = {"output": output}

        # Aggregate channels that correspond to the same logical term
        # (e.g. across multiple subnets).
        for term_name, channel_indices in self.term_to_channel_indices.items():
            result[term_name] = term_contribs[:, channel_indices, :].sum(dim=1)

        if self.intercept is not None:
            result["intercept"] = self.intercept

        if self.output_penalty > 0.0:
            result["output_penalty"] = self.output_penalty * term_scores.pow(2).mean()

        return result