"""Gaussian-process neural additive models with fixed RFF bases."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import torch
import torch.nn as nn

from ..configs.gpnam_config import DefaultGPNAMConfig
from .components.base_model import BaseModel


class GPNAM(BaseModel):
    """Paper-aligned GP-NAM and its pairwise GP-NA2M extension.

    Each scalar architecture input receives a one-dimensional random Fourier
    feature (RFF) map. Frequencies use the inverse-normal grid described in
    Zhang, Barr, and Paisley (2024), while each input dimension receives an
    independently permuted phase grid. Only the final additive coefficients
    and intercept are trainable.
    """

    extra_reserved_feature_names = ("feature_contribution",)
    supports_fixed_linear_regression = True
    estimator_fitted_attributes = ("kernel_widths_",)

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultGPNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultGPNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self._validate_input_representation()
        self.num_classes = int(num_classes)
        self.num_feature_keys = list(num_feature_info)
        self.cat_feature_keys = list(cat_feature_info)
        self.atomic_feature_names = self._build_atomic_feature_names()
        self.input_dim = len(self.atomic_feature_names)

        self.rff_num_feat = int(self.hparams.get("rff_num_feat", config.rff_num_feat))
        if self.rff_num_feat < 1:
            raise ValueError("rff_num_feat must be a positive integer.")
        self.rff_scheme = str(
            self.hparams.get("rff_scheme", config.rff_scheme)
        ).lower()
        if self.rff_scheme not in {"quasi_random", "random"}:
            raise ValueError("rff_scheme must be 'quasi_random' or 'random'.")
        self.rff_random_state = self.hparams.get(
            "rff_random_state", config.rff_random_state
        )

        self.solver = str(self.hparams.get("solver", config.solver)).lower()
        if self.solver not in {"cg", "gradient"}:
            raise ValueError("solver must be 'cg' or 'gradient'.")
        self.ridge = float(self.hparams.get("ridge", config.ridge))
        self.cg_rtol = float(self.hparams.get("cg_rtol", config.cg_rtol))
        self.cg_max_iter = self.hparams.get("cg_max_iter", config.cg_max_iter)
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if self.cg_rtol <= 0:
            raise ValueError("cg_rtol must be positive.")

        self.interaction_names, interaction_indices = self._resolve_interactions(
            interaction_degree=self.hparams.get(
                "interaction_degree", config.interaction_degree
            ),
            requested=self.hparams.get("interactions", config.interactions),
        )
        self.register_buffer(
            "interaction_indices",
            torch.as_tensor(interaction_indices, dtype=torch.long).reshape(-1, 2),
            persistent=True,
        )

        kernel_widths, auto_widths = self._initial_kernel_widths(config)
        self._auto_kernel_widths = auto_widths
        self.register_buffer("kernel_widths", kernel_widths, persistent=True)

        generator = self._rff_generator()
        z, c = self._build_main_rff_parameters(generator)
        interaction_z, interaction_c = self._build_interaction_rff_parameters(
            len(self.interaction_names), generator
        )
        self.register_buffer("z", z, persistent=True)
        self.register_buffer("c", c, persistent=True)
        self.register_buffer("interaction_z", interaction_z, persistent=True)
        self.register_buffer("interaction_c", interaction_c, persistent=True)

        self.weights = nn.Parameter(
            torch.zeros(self.input_dim, self.rff_num_feat, self.num_classes)
        )
        self.interaction_weights = nn.Parameter(
            torch.zeros(
                len(self.interaction_names), self.rff_num_feat, self.num_classes
            )
        )
        if self.hparams.get("intercept", config.intercept):
            self.intercept: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.num_classes)
            )
        else:
            self.intercept = None

    def _validate_input_representation(self) -> None:
        expanded = [
            name
            for name, info in self.num_feature_info.items()
            if int(info["dimension"]) != 1
        ]
        if expanded:
            raise ValueError(
                "GPNAM requires one scalar architecture input per numerical "
                f"feature; expanded numerical features: {expanded}. Use "
                "numerical_preprocessing='none', 'standardization', or 'minmax'."
            )

    def _build_atomic_feature_names(self) -> list[str]:
        names = list(self.num_feature_keys)
        for name in self.cat_feature_keys:
            dimension = int(self.cat_feature_info[name]["dimension"])
            if dimension < 1:
                raise ValueError(f"Categorical feature {name!r} has no columns.")
            if dimension == 1:
                names.append(name)
            else:
                names.extend(f"{name}[{index}]" for index in range(dimension))
        return names

    def _resolve_interactions(self, interaction_degree, requested):
        if interaction_degree not in {None, 1, 2}:
            raise ValueError("GP-NA2M supports pairwise interactions only.")
        if requested is not None and interaction_degree not in {None, 1}:
            raise ValueError(
                "Specify either interactions or interaction_degree=2, not both."
            )
        name_to_index = {
            name: index for index, name in enumerate(self.atomic_feature_names)
        }
        if requested is None:
            pairs = (
                list(combinations(self.atomic_feature_names, 2))
                if interaction_degree == 2
                else []
            )
        else:
            pairs = [tuple(pair) for pair in requested]

        names = []
        indices = []
        seen = set()
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(
                    "Each GPNAM interaction must contain two distinct features."
                )
            unknown = [name for name in pair if name not in name_to_index]
            if unknown:
                raise ValueError(
                    f"Unknown GPNAM interaction features {unknown}; available: "
                    f"{self.atomic_feature_names}."
                )
            ordered = tuple(sorted(pair, key=name_to_index.__getitem__))
            if ordered in seen:
                raise ValueError(f"Duplicate GPNAM interaction {ordered}.")
            seen.add(ordered)
            names.append(":".join(ordered))
            indices.append(tuple(name_to_index[name] for name in ordered))
        return names, indices

    def _initial_kernel_widths(self, config):
        explicit = self.hparams.get("kernel_widths", config.kernel_widths)
        if explicit is not None:
            widths = torch.as_tensor(explicit, dtype=torch.float32).flatten()
            if widths.numel() != self.input_dim:
                raise ValueError(
                    f"kernel_widths must contain {self.input_dim} values, got "
                    f"{widths.numel()}."
                )
            if not torch.isfinite(widths).all() or torch.any(widths <= 0):
                raise ValueError("All kernel_widths must be finite and positive.")
            return widths, False

        width = self.hparams.get("kernel_width", config.kernel_width)
        if isinstance(width, str):
            if width.lower() != "auto":
                raise ValueError("kernel_width must be positive or 'auto'.")
            return torch.full((self.input_dim,), torch.nan), True
        width = float(width)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("kernel_width must be finite and positive.")
        return torch.full((self.input_dim,), width), False

    def _rff_generator(self):
        if self.rff_random_state is None:
            return None
        return torch.Generator().manual_seed(int(self.rff_random_state))

    def _normal_grid(self):
        normal = torch.distributions.Normal(0.0, 1.0)
        probabilities = torch.arange(1, self.rff_num_feat + 1, dtype=torch.float32)
        probabilities = probabilities / (self.rff_num_feat + 1)
        return normal.icdf(probabilities)

    def _phase_grid(self):
        return (
            2.0
            * math.pi
            * torch.arange(self.rff_num_feat, dtype=torch.float32)
            / self.rff_num_feat
        )

    def _build_main_rff_parameters(self, generator):
        if self.rff_scheme == "quasi_random":
            z = self._normal_grid()
            phase_grid = self._phase_grid()
            c = torch.stack(
                [
                    phase_grid[
                        torch.randperm(self.rff_num_feat, generator=generator)
                    ]
                    for _ in range(self.input_dim)
                ]
            )
        else:
            z = torch.randn(self.rff_num_feat, generator=generator)
            c = 2.0 * math.pi * torch.rand(
                self.input_dim, self.rff_num_feat, generator=generator
            )
        return z, c

    def _build_interaction_rff_parameters(self, count, generator):
        if count == 0:
            return torch.empty(0, self.rff_num_feat, 2), torch.empty(
                0, self.rff_num_feat
            )
        if self.rff_scheme == "quasi_random":
            grid = self._normal_grid()
            phase_grid = self._phase_grid()
            z = torch.empty(count, self.rff_num_feat, 2)
            c = torch.empty(count, self.rff_num_feat)
            for index in range(count):
                for dimension in range(2):
                    z[index, :, dimension] = grid[
                        torch.randperm(self.rff_num_feat, generator=generator)
                    ]
                c[index] = phase_grid[
                    torch.randperm(self.rff_num_feat, generator=generator)
                ]
            return z, c
        return (
            torch.randn(count, self.rff_num_feat, 2, generator=generator),
            2.0
            * math.pi
            * torch.rand(count, self.rff_num_feat, generator=generator),
        )

    def _concat_all_features(self, num_features, cat_features):
        tensors = []
        for name in self.num_feature_keys:
            tensor = num_features[name]
            tensors.append(tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor)
        for name in self.cat_feature_keys:
            tensor = cat_features[name]
            tensors.append(tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor)
        if not tensors:
            raise ValueError("GPNAM received no input features.")
        result = torch.cat(tensors, dim=1).float()
        if result.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Expected {self.input_dim} architecture inputs, got "
                f"{result.shape[1]}."
            )
        return result

    def initialize_from_training_data(self, num_features, cat_features) -> None:
        """Fit data-dependent bandwidths from training architecture inputs."""
        if not self._auto_kernel_widths:
            return
        x = self._concat_all_features(num_features, cat_features)
        if x.shape[0] < 2:
            raise ValueError("Automatic GPNAM kernel widths require at least two rows.")
        widths = torch.std(x, dim=0, correction=1) / 24.0
        invalid = ~torch.isfinite(widths) | (widths <= 0)
        if torch.any(invalid):
            bad = [
                self.atomic_feature_names[index]
                for index in torch.nonzero(invalid, as_tuple=False).flatten().tolist()
            ]
            raise ValueError(
                "Automatic GPNAM kernel widths require non-constant finite "
                f"features; invalid: {bad}."
            )
        self.kernel_widths.copy_(widths)

    @property
    def kernel_widths_(self):
        if torch.isnan(self.kernel_widths).any():
            raise RuntimeError("Automatic kernel widths have not been fitted yet.")
        return self.kernel_widths.detach().cpu().numpy().copy()

    def _main_rff_map(self, x):
        if torch.isnan(self.kernel_widths).any():
            raise RuntimeError(
                "Automatic GPNAM kernel widths must be initialized from training data."
            )
        scaled = (x / self.kernel_widths[None, :])[:, :, None]
        return math.sqrt(2.0 / self.rff_num_feat) * torch.cos(
            scaled * self.z[None, None, :] + self.c[None, :, :]
        )

    def _interaction_rff_map(self, x):
        if not self.interaction_names:
            return x.new_empty(x.shape[0], 0, self.rff_num_feat)
        selected = x[:, self.interaction_indices]
        widths = self.kernel_widths[self.interaction_indices]
        scaled = selected / widths[None, :, :]
        arguments = torch.einsum("bpi,psi->bps", scaled, self.interaction_z)
        return math.sqrt(2.0 / self.rff_num_feat) * torch.cos(
            arguments + self.interaction_c[None, :, :]
        )

    def linear_design(self, num_features, cat_features):
        """Return the fixed, intercept-free design matrix used by CG fitting."""
        x = self._concat_all_features(num_features, cat_features)
        blocks = [self._main_rff_map(x).flatten(1)]
        interactions = self._interaction_rff_map(x)
        if interactions.shape[1]:
            blocks.append(interactions.flatten(1))
        return torch.cat(blocks, dim=1)

    def set_linear_coefficients(self, coefficients, intercept=None) -> None:
        """Install a fixed-design solution without replacing Parameters."""
        expected = (
            self.input_dim + len(self.interaction_names)
        ) * self.rff_num_feat
        if coefficients.shape != (expected, self.num_classes):
            raise ValueError(
                f"Expected coefficient shape {(expected, self.num_classes)}, got "
                f"{tuple(coefficients.shape)}."
            )
        split = self.input_dim * self.rff_num_feat
        with torch.no_grad():
            self.weights.copy_(
                coefficients[:split].reshape(
                    self.input_dim, self.rff_num_feat, self.num_classes
                )
            )
            self.interaction_weights.copy_(
                coefficients[split:].reshape(
                    len(self.interaction_names),
                    self.rff_num_feat,
                    self.num_classes,
                )
            )
            if self.intercept is not None:
                if intercept is None:
                    raise ValueError("This GPNAM requires an intercept solution.")
                self.intercept.copy_(intercept.reshape(self.num_classes))

    def basis_metadata(self) -> dict:
        """Return fitted RFF state for reproducibility and diagnostics."""
        return {
            "feature_names": tuple(self.atomic_feature_names),
            "interaction_names": tuple(self.interaction_names),
            "kernel_widths": self.kernel_widths_,
            "frequencies": self.z.detach().cpu().numpy().copy(),
            "phases": self.c.detach().cpu().numpy().copy(),
            "interaction_frequencies": self.interaction_z.detach().cpu().numpy().copy(),
            "interaction_phases": self.interaction_c.detach().cpu().numpy().copy(),
        }

    def forward(
        self,
        num_features: dict,
        cat_features: dict,
        feature_of_interest: Optional[str] = None,
    ) -> dict:
        x = self._concat_all_features(num_features, cat_features)
        main_basis = self._main_rff_map(x)
        main = torch.einsum("bds,dsc->bdc", main_basis, self.weights)
        interaction_basis = self._interaction_rff_map(x)
        interaction = torch.einsum(
            "bps,psc->bpc", interaction_basis, self.interaction_weights
        )

        output = main.sum(dim=1) + interaction.sum(dim=1)
        if self.intercept is not None:
            output = output + self.intercept

        result = {"output": output}
        result.update(
            {
                name: main[:, index, :]
                for index, name in enumerate(self.atomic_feature_names)
            }
        )
        result.update(
            {
                name: interaction[:, index, :]
                for index, name in enumerate(self.interaction_names)
            }
        )
        if self.intercept is not None:
            result["intercept"] = self.intercept
        if feature_of_interest is not None:
            if feature_of_interest not in result:
                available = self.atomic_feature_names + self.interaction_names
                raise KeyError(
                    f"Unknown feature_of_interest={feature_of_interest!r}; "
                    f"available: {available}."
                )
            result["feature_contribution"] = result[feature_of_interest]
        return result


__all__ = ["GPNAM"]
