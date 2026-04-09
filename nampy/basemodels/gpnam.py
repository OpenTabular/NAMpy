import math
from typing import Optional

import torch
import torch.nn as nn

from ..configs.gpnam_config import DefaultGPNAMConfig
from .basemodel import BaseModel


class GPNAM(BaseModel):
    """
    Gaussian Process Neural Additive Model (GP-NAM).

    This implementation follows the paper's additive RFF construction:
        g(x) = w0 + sum_i phi(x_i)^T w_i

    where each scalar input dimension x_i gets its own one-dimensional RFF map
    with shared (z_s, c_s) across dimensions and a feature-specific kernel width b_i.

    Notes
    -----
    - The model is additive over scalar post-preprocessing columns.
    - If an original feature expands to multiple columns (e.g. one-hot), each
      scalar column is treated as its own GP shape function and returned as
      `feature[j]`.
    - Training uses your package's optimizer loop, not the paper's conjugate-
      gradient regression solver.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        kernel_width: float = 0.2,
        rff_num_feat: int = 100,
        config: DefaultGPNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultGPNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        # Optimization hyperparameters
        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        # Metadata
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes

        self.num_feature_keys = list(num_feature_info.keys())
        self.cat_feature_keys = list(cat_feature_info.keys())

        reserved = {"output", "intercept", "feature_contribution"}
        all_feature_names = set(self.num_feature_keys) | set(self.cat_feature_keys)
        if reserved & all_feature_names:
            raise ValueError(
                f"Feature names {sorted(reserved.intersection(all_feature_names))} are reserved."
            )
        if any(":" in name for name in all_feature_names):
            bad = sorted(name for name in all_feature_names if ":" in name)
            raise ValueError(
                f"Feature names {bad} contain ':', which is reserved for interaction names."
            )

        # Scalar post-preprocessing dimensions
        self.atomic_feature_names = self._build_atomic_feature_names(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
        )
        self.input_dim = len(self.atomic_feature_names)

        # GP-NAM hyperparameters
        self.rff_num_feat = int(
            self.hparams.get(
                "rff_num_feat", getattr(config, "rff_num_feat", rff_num_feat)
            )
        )
        if self.rff_num_feat <= 0:
            raise ValueError("rff_num_feat must be a positive integer.")

        self.use_deterministic_rff_grid = bool(
            self.hparams.get(
                "use_deterministic_rff_grid",
                getattr(config, "use_deterministic_rff_grid", True),
            )
        )

        # Per-dimension kernel widths b_i
        raw_kernel_widths = self.hparams.get(
            "kernel_widths", getattr(config, "kernel_widths", None)
        )
        if raw_kernel_widths is None:
            scalar_width = float(
                self.hparams.get(
                    "kernel_width",
                    getattr(config, "kernel_width", kernel_width),
                )
            )
            if scalar_width <= 0:
                raise ValueError("kernel_width must be positive.")
            kernel_widths = torch.full(
                (self.input_dim,), scalar_width, dtype=torch.float32
            )
        else:
            kernel_widths = torch.as_tensor(
                raw_kernel_widths, dtype=torch.float32
            ).flatten()
            if kernel_widths.numel() != self.input_dim:
                raise ValueError(
                    f"kernel_widths must have length {self.input_dim}, got {kernel_widths.numel()}."
                )
            if torch.any(kernel_widths <= 0):
                raise ValueError("All kernel_widths must be positive.")

        self.register_buffer("kernel_widths", kernel_widths, persistent=True)

        # Shared RFF parameters z_s, c_s across all scalar dimensions
        z, c = self._build_shared_rff_parameters(
            num_rff=self.rff_num_feat,
            deterministic=self.use_deterministic_rff_grid,
        )
        self.register_buffer("z", z, persistent=True)  # [S]
        self.register_buffer("c", c, persistent=True)  # [S]

        # Learnable per-dimension linear weights and bias:
        # weights[d, s, k] corresponds to feature i weight vector w_i for class/param k.
        self.weights = nn.Parameter(
            torch.zeros(self.input_dim, self.rff_num_feat, self.num_classes)
        )

        if self.hparams.get("intercept", getattr(config, "intercept", True)):
            self.intercept = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.intercept = None

    def _build_atomic_feature_names(self, num_feature_info, cat_feature_info):
        """Build names for scalar post-preprocessing dimensions."""
        atomic_names = []

        for name in self.num_feature_keys:
            dim = int(num_feature_info[name]["dimension"])
            if dim <= 0:
                raise ValueError(
                    f"Numerical feature '{name}' has invalid dimension {dim}."
                )
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

    def _build_shared_rff_parameters(self, num_rff: int, deterministic: bool = True):
        """
        Build shared z_s, c_s.

        Paper-compatible deterministic option:
        - z_s from inverse CDF grid of N(0,1)
        - c_s from a uniform grid on [0, 2pi), randomly permuted
        """
        if deterministic:
            normal = torch.distributions.Normal(0.0, 1.0)
            probs = torch.arange(1, num_rff + 1, dtype=torch.float32) / (num_rff + 1)
            z = normal.icdf(probs)  # [S]

            c_grid = (
                2.0 * math.pi * torch.arange(num_rff, dtype=torch.float32) / num_rff
            )
            perm = torch.randperm(num_rff)
            c = c_grid[perm]
        else:
            z = torch.randn(num_rff, dtype=torch.float32)
            c = 2.0 * math.pi * torch.rand(num_rff, dtype=torch.float32)

        return z, c

    def _concat_all_features(self, num_features, cat_features):
        """Concatenate features in a deterministic order."""
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
            raise ValueError("GPNAM received no input features.")

        return torch.cat(tensors, dim=1).float()

    def _compute_rff_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute phi(x_i) for every scalar input dimension.

        Parameters
        ----------
        x : Tensor of shape [B, D]

        Returns
        -------
        Tensor of shape [B, D, S]
        """
        # scaled_x: [B, D, 1]
        scaled_x = (x / self.kernel_widths.unsqueeze(0)).unsqueeze(-1)

        # z, c shared across dimensions: [1, 1, S]
        phi = math.sqrt(2.0 / self.rff_num_feat) * torch.cos(
            scaled_x * self.z.view(1, 1, -1) + self.c.view(1, 1, -1)
        )
        return phi

    def forward(
        self,
        num_features: dict,
        cat_features: dict,
        feature_of_interest: Optional[str] = None,
    ) -> dict:
        """
        Forward pass of GP-NAM.

        Returns
        -------
        dict
            Contains:
            - "output": [B, C]
            - one exact additive contribution per scalar dimension, each [B, C]
            - optionally "intercept": [C]
            - optionally "feature_contribution" if feature_of_interest is requested
        """
        x = self._concat_all_features(num_features, cat_features)  # [B, D]
        if x.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Expected concatenated input dimension {self.input_dim}, got {x.shape[1]}."
            )

        phi = self._compute_rff_map(x)  # [B, D, S]

        # Exact additive per-dimension contributions:
        # contribs[b, d, c] = sum_s phi[b, d, s] * weights[d, s, c]
        contribs = torch.einsum("bds,dsc->bdc", phi, self.weights)  # [B, D, C]

        output = contribs.sum(dim=1)  # [B, C]
        if self.intercept is not None:
            output = output + self.intercept

        result = {"output": output}

        for d, name in enumerate(self.atomic_feature_names):
            result[name] = contribs[:, d, :]

        if self.intercept is not None:
            result["intercept"] = self.intercept

        if feature_of_interest is not None:
            if feature_of_interest not in result:
                raise KeyError(
                    f"Unknown feature_of_interest={feature_of_interest!r}. "
                    f"Available feature keys: {self.atomic_feature_names}"
                )
            result["feature_contribution"] = result[feature_of_interest]

        return result
