"""Interpretable Generalized Additive Neural Networks (IGANN)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Lasso, LogisticRegression

from ..configs.igann_config import DefaultIGANNConfig
from .components.base_model import BaseModel


class IGANN(BaseModel):
    """Linear initialization followed by feature-wise ELM boosting.

    This mirrors ``igann.IGANN``'s architecture-specific optimization rather
    than treating its fixed random hidden layers as an ordinary end-to-end
    network. Numerical features receive ``n_hid`` independent random units;
    categorical features are reference-coded and remain linear in every ELM.
    """

    supports_native_training = True
    native_objective_kinds = frozenset({"regression", "binary"})
    estimator_fitted_attributes = (
        "n_estimators_",
        "selected_features_",
        "training_history_",
    )

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultIGANNConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultIGANNConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_feature_names = list(num_feature_info)
        self.cat_feature_names = list(cat_feature_info)
        self.source_feature_names = self.num_feature_names + self.cat_feature_names
        self.num_classes = int(num_classes)
        self.n_num = len(self.num_feature_names)

        self.n_hid = int(self.hparams.get("n_hid", config.n_hid))
        self.n_estimators = int(
            self.hparams.get("n_estimators", config.n_estimators)
        )
        self.boost_rate = float(self.hparams.get("boost_rate", config.boost_rate))
        self.init_reg = float(self.hparams.get("init_reg", config.init_reg))
        self.elm_scale = float(self.hparams.get("elm_scale", config.elm_scale))
        self.elm_alpha = float(self.hparams.get("elm_alpha", config.elm_alpha))
        self.early_stopping = int(
            self.hparams.get("early_stopping", config.early_stopping)
        )
        self.elm_random_state = int(
            self.hparams.get("elm_random_state", config.elm_random_state)
        )
        self.sparse = int(self.hparams.get("sparse", config.sparse))
        self.native_device = str(self.hparams.get("device", config.device))
        self.clip_predictions = float(
            self.hparams.get("clip_predictions", config.clip_predictions)
        )
        self.activation_name = str(
            self.hparams.get("activation", config.activation)
        ).lower()
        self.solver = str(self.hparams.get("solver", config.solver)).lower()
        self._validate_configuration()
        self.activation = self._make_activation()

        self._validate_input_representation()
        (
            self.source_design_slices,
            self.source_basis_slices,
            self.atomic_design_slices,
            self.atomic_basis_slices,
            self.atomic_feature_names,
            self.atomic_source_indices,
        ) = self._build_feature_layout()
        self.design_dim = (
            self.source_design_slices[-1].stop if self.source_design_slices else 0
        )
        self.basis_dim = (
            self.source_basis_slices[-1].stop if self.source_basis_slices else 0
        )
        if self.design_dim == 0:
            raise ValueError("IGANN requires at least one non-constant input column.")

        hidden_weights = self._build_hidden_weights()
        self.register_buffer("hidden_weights", hidden_weights, persistent=True)
        self.register_buffer(
            "active_atomic_mask",
            torch.ones(len(self.atomic_feature_names), dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "n_estimators_fitted_tensor",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )

        self.linear_weights = nn.Parameter(
            torch.zeros(self.design_dim, self.num_classes), requires_grad=False
        )
        self.intercept = nn.Parameter(
            torch.zeros(self.num_classes), requires_grad=False
        )
        self.stage_coefficients = nn.Parameter(
            torch.zeros(self.n_estimators, self.basis_dim, self.num_classes),
            requires_grad=False,
        )
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def _validate_configuration(self) -> None:
        if self.n_hid < 1:
            raise ValueError("n_hid must be a positive integer.")
        if self.n_estimators < 0:
            raise ValueError("n_estimators must be non-negative.")
        if self.boost_rate <= 0:
            raise ValueError("boost_rate must be positive.")
        if self.init_reg < 0 or self.elm_alpha < 0:
            raise ValueError("init_reg and elm_alpha must be non-negative.")
        if self.early_stopping < 0:
            raise ValueError("early_stopping must be non-negative.")
        if self.sparse < 0:
            raise ValueError("sparse must be non-negative.")
        if not math.isfinite(self.elm_scale) or self.elm_scale <= 0:
            raise ValueError("elm_scale must be finite and positive.")
        if not math.isfinite(self.clip_predictions) or self.clip_predictions <= 0:
            raise ValueError("clip_predictions must be finite and positive.")
        if self.activation_name not in {"elu", "relu"}:
            raise ValueError("activation must be 'elu' or 'relu'.")
        if self.solver not in {"auto", "native", "gradient"}:
            raise ValueError("solver must be 'auto', 'native', or 'gradient'.")

    @classmethod
    def uses_native_training(cls, objective, config) -> bool:
        """Select native fitting only where the released updates are defined."""
        solver = str(getattr(config, "solver", "auto")).lower()
        objective_kind = str(getattr(objective, "kind", ""))
        supported = objective_kind in cls.native_objective_kinds
        if solver == "native" and not supported:
            raise NotImplementedError(
                "IGANN solver='native' supports squared-error regression and "
                "binary classification only. Use solver='gradient' or 'auto' "
                "for multiclass and distributional objectives."
            )
        return solver == "native" or (solver == "auto" and supported)

    def prepare_gradient_training(self) -> None:
        """Enable the shared objective engine over the complete fixed ELM basis."""
        if self.sparse > 0:
            raise NotImplementedError(
                "IGANN-Sparse is defined for native training only; use sparse=0 "
                "with solver='gradient'."
            )
        with torch.no_grad():
            self.n_estimators_fitted_tensor.fill_(self.n_estimators)
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _make_activation(self) -> nn.Module:
        return nn.ELU() if self.activation_name == "elu" else nn.ReLU()

    def _validate_input_representation(self) -> None:
        invalid_num = [
            name
            for name, info in self.num_feature_info.items()
            if int(info["dimension"]) != 1
        ]
        invalid_cat = [
            name
            for name, info in self.cat_feature_info.items()
            if int(info["dimension"]) != 1
        ]
        if invalid_num or invalid_cat:
            raise ValueError(
                "IGANN requires scalar numerical inputs and integer-encoded "
                "categorical inputs; invalid numerical features "
                f"{invalid_num}, invalid categorical features {invalid_cat}."
            )

    def _build_feature_layout(self):
        source_design_slices = []
        source_basis_slices = []
        atomic_design_slices = []
        atomic_basis_slices = []
        atomic_names = []
        atomic_sources = []
        design_start = 0
        basis_start = 0
        for source_index, name in enumerate(self.num_feature_names):
            source_design_slices.append(slice(design_start, design_start + 1))
            source_basis_slices.append(slice(basis_start, basis_start + self.n_hid))
            atomic_design_slices.append(slice(design_start, design_start + 1))
            atomic_basis_slices.append(slice(basis_start, basis_start + self.n_hid))
            design_start += 1
            basis_start += self.n_hid
            atomic_names.append(name)
            atomic_sources.append(source_index)
        for source_index, name in enumerate(
            self.cat_feature_names, start=self.n_num
        ):
            cardinality = int(self.cat_feature_info[name].get("n_unique", 0))
            if cardinality < 1:
                raise ValueError(
                    f"IGANN requires fitted categorical cardinality for {name!r}."
                )
            width = max(cardinality - 1, 0)
            source_design_slices.append(slice(design_start, design_start + width))
            source_basis_slices.append(slice(basis_start, basis_start + width))
            for level in range(1, cardinality):
                offset = level - 1
                atomic_design_slices.append(
                    slice(design_start + offset, design_start + offset + 1)
                )
                atomic_basis_slices.append(
                    slice(basis_start + offset, basis_start + offset + 1)
                )
                atomic_names.append(f"{name}[{level}]")
                atomic_sources.append(source_index)
            design_start += width
            basis_start += width
        return (
            source_design_slices,
            source_basis_slices,
            atomic_design_slices,
            atomic_basis_slices,
            atomic_names,
            atomic_sources,
        )

    def _build_hidden_weights(self) -> torch.Tensor:
        if self.n_num == 0 or self.n_estimators == 0:
            return torch.empty(self.n_estimators, self.n_num, self.n_hid)
        stages = []
        for stage in range(self.n_estimators):
            generator = torch.Generator().manual_seed(self.elm_random_state + stage)
            # Mirror upstream ELM_Regressor: draw the full masked matrix before
            # retaining its feature-specific diagonal blocks.
            full = torch.randn(
                self.n_num,
                self.n_num * self.n_hid,
                generator=generator,
            ) * self.elm_scale
            stages.append(
                torch.stack(
                    [
                        full[index, index * self.n_hid : (index + 1) * self.n_hid]
                        for index in range(self.n_num)
                    ]
                )
            )
        return torch.stack(stages)

    def _design_inputs(self, num_features, cat_features) -> torch.Tensor:
        blocks = []
        for name in self.num_feature_names:
            value = num_features[name].float()
            blocks.append(value.unsqueeze(-1) if value.ndim == 1 else value)
        for name in self.cat_feature_names:
            value = cat_features[name]
            if value.ndim > 1:
                value = value.squeeze(-1)
            cardinality = int(self.cat_feature_info[name]["n_unique"])
            # PreTab's integer encoder is one-based and reserves zero for an
            # unseen level. Unknown levels therefore receive the same all-zero
            # reference coding as upstream OneHotEncoder(handle_unknown="ignore").
            encoded_values = value.long()
            unknown = encoded_values == 0
            indices = encoded_values - 1
            if torch.any(indices < -1) or torch.any(indices >= cardinality):
                raise ValueError(
                    f"Categorical feature {name!r} contains an invalid encoded level."
                )
            safe_indices = indices.clamp_min(0)
            one_hot = F.one_hot(safe_indices, num_classes=cardinality).float()
            one_hot[unknown] = 0.0
            blocks.append(one_hot[:, 1:])
        return torch.cat(blocks, dim=1)

    def _basis_for_stage(self, design: torch.Tensor, stage: int) -> torch.Tensor:
        blocks = []
        if self.n_num:
            numerical = design[:, : self.n_num]
            blocks.append(
                self.activation(
                    numerical[:, :, None] * self.hidden_weights[stage][None, :, :]
                ).flatten(1)
            )
        if self.design_dim > self.n_num:
            blocks.append(design[:, self.n_num :])
        return torch.cat(blocks, dim=1)

    def _atomic_masks(self, *, device) -> tuple[torch.Tensor, torch.Tensor]:
        design_mask = torch.zeros(self.design_dim, dtype=torch.bool, device=device)
        basis_mask = torch.zeros(self.basis_dim, dtype=torch.bool, device=device)
        for active, design_slice, basis_slice in zip(
            self.active_atomic_mask.tolist(),
            self.atomic_design_slices,
            self.atomic_basis_slices,
            strict=True,
        ):
            if active:
                design_mask[design_slice] = True
                basis_mask[basis_slice] = True
        return design_mask, basis_mask

    def _select_sparse_features(
        self, design: torch.Tensor, targets: torch.Tensor, objective_kind: str
    ) -> None:
        if self.sparse == 0 or self.sparse >= len(self.atomic_feature_names):
            self.active_atomic_mask.fill_(True)
            return
        if targets.shape[1] != 1:
            raise NotImplementedError(
                "IGANN-Sparse currently supports one target column only."
            )
        try:
            import abess.linear
        except ImportError as error:
            raise ImportError(
                "IGANN sparse>0 requires the optional 'abess' package. "
                "Install abess>=0.4.5 to enable best-subset selection."
            ) from error

        basis = self._basis_for_stage(design, 0).detach().cpu().numpy()
        groups = []
        for atomic_index, basis_slice in enumerate(self.atomic_basis_slices):
            groups.extend([atomic_index] * (basis_slice.stop - basis_slice.start))
        common = {
            "path_type": "gs",
            "cv": 3,
            "s_min": 1,
            "s_max": self.sparse,
            "thread": 0,
            "group": np.asarray(groups),
        }
        if objective_kind == "binary":
            selector = abess.linear.LogisticRegression(**common)
            response = targets[:, 0].detach().cpu().numpy().astype(int)
        else:
            selector = abess.linear.LinearRegression(**common)
            response = targets[:, 0].detach().cpu().numpy()
        selector.fit(basis, response)
        coefficients = np.asarray(selector.coef_).reshape(-1)
        selected = torch.zeros_like(self.active_atomic_mask)
        for atomic_index, basis_slice in enumerate(self.atomic_basis_slices):
            selected[atomic_index] = bool(
                np.any(np.abs(coefficients[basis_slice]) > 0)
            )
        if not torch.any(selected):
            raise RuntimeError("IGANN-Sparse selected no usable feature blocks.")
        self.active_atomic_mask.copy_(selected)

    @staticmethod
    def _as_targets(targets: torch.Tensor, output_dim: int) -> torch.Tensor:
        values = targets.float()
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.shape[1] != output_dim:
            raise ValueError(
                f"IGANN expected {output_dim} target columns, got {values.shape[1]}."
            )
        return values

    @staticmethod
    def _reject_nonuniform_weights(name: str, weights: torch.Tensor | None) -> None:
        if weights is None:
            return
        flat = weights.reshape(-1)
        if not torch.allclose(flat, torch.ones_like(flat)):
            raise NotImplementedError(
                f"IGANN native boosting does not yet support non-uniform {name}."
            )

    def _fit_linear_initialization(
        self,
        design: torch.Tensor,
        targets: torch.Tensor,
        *,
        objective_kind: str,
        random_state: int,
        offset: torch.Tensor | None,
    ) -> torch.Tensor:
        design_mask, _ = self._atomic_masks(device=design.device)
        selected = design[:, design_mask].detach().cpu().numpy()
        response = targets.detach().cpu().numpy()
        if offset is not None:
            response = response - offset.detach().cpu().numpy()

        coefficients = np.zeros((self.design_dim, self.num_classes), dtype=np.float32)
        if objective_kind == "binary":
            if self.init_reg <= 0:
                raise ValueError("Binary IGANN requires init_reg > 0.")
            estimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0 / self.init_reg,
                random_state=int(random_state),
            )
            estimator.fit(selected, response[:, 0].astype(int))
            coefficients[design_mask.cpu().numpy(), 0] = estimator.coef_[0]
            intercept = np.asarray(estimator.intercept_, dtype=np.float32)
        else:
            intercept = np.zeros(self.num_classes, dtype=np.float32)
            for output_index in range(self.num_classes):
                estimator = Lasso(
                    alpha=self.init_reg,
                    random_state=int(random_state),
                )
                estimator.fit(selected, response[:, output_index])
                coefficients[design_mask.cpu().numpy(), output_index] = estimator.coef_
                intercept[output_index] = estimator.intercept_

        with torch.no_grad():
            self.linear_weights.copy_(
                torch.as_tensor(
                    coefficients,
                    device=self.linear_weights.device,
                    dtype=self.linear_weights.dtype,
                )
            )
            self.intercept.copy_(
                torch.as_tensor(
                    intercept,
                    device=self.intercept.device,
                    dtype=self.intercept.dtype,
                )
            )
        return design @ self.linear_weights + self.intercept

    @staticmethod
    def _mean_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        objective_kind: str,
        offset: torch.Tensor | None,
    ) -> torch.Tensor:
        if offset is not None:
            predictions = predictions + offset
        if objective_kind == "binary":
            return F.binary_cross_entropy_with_logits(predictions, targets)
        return F.mse_loss(predictions, targets)

    def fit_native(
        self,
        *,
        train_num_features: Mapping[str, torch.Tensor],
        train_cat_features: Mapping[str, torch.Tensor],
        train_targets: torch.Tensor,
        val_num_features: Mapping[str, torch.Tensor],
        val_cat_features: Mapping[str, torch.Tensor],
        val_targets: torch.Tensor,
        objective: object,
        train_offset: torch.Tensor | None = None,
        val_offset: torch.Tensor | None = None,
        train_sample_weight: torch.Tensor | None = None,
        val_sample_weight: torch.Tensor | None = None,
        random_state: int = 0,
    ) -> Mapping[str, object]:
        """Fit the upstream linear-plus-ELM boosting sequence."""
        objective_kind = str(getattr(objective, "kind", ""))
        if objective_kind not in {"regression", "binary"}:
            raise NotImplementedError(
                "IGANN supports regression and binary classification only."
            )
        self._reject_nonuniform_weights("sample_weight", train_sample_weight)
        self._reject_nonuniform_weights("sample_weight_val", val_sample_weight)
        if objective_kind == "binary" and (
            train_offset is not None or val_offset is not None
        ):
            raise NotImplementedError(
                "Binary IGANN does not support per-sample link offsets."
            )

        device = torch.device(self.native_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("IGANN device='cuda' requested but CUDA is unavailable.")
        self.to(device)
        train_num = {key: value.to(device) for key, value in train_num_features.items()}
        train_cat = {key: value.to(device) for key, value in train_cat_features.items()}
        val_num = {key: value.to(device) for key, value in val_num_features.items()}
        val_cat = {key: value.to(device) for key, value in val_cat_features.items()}
        y_train = self._as_targets(train_targets.to(device), self.num_classes)
        y_val = self._as_targets(val_targets.to(device), self.num_classes)
        offset_train = None if train_offset is None else train_offset.to(device)
        offset_val = None if val_offset is None else val_offset.to(device)

        train_design = self._design_inputs(train_num, train_cat)
        val_design = self._design_inputs(val_num, val_cat)
        with torch.no_grad():
            self.linear_weights.zero_()
            self.intercept.zero_()
            self.stage_coefficients.zero_()
            self.n_estimators_fitted_tensor.zero_()
        self._train_losses = []
        self._val_losses = []
        self._select_sparse_features(train_design, y_train, objective_kind)

        prediction_train = self._fit_linear_initialization(
            train_design,
            y_train,
            objective_kind=objective_kind,
            random_state=random_state,
            offset=offset_train,
        )
        prediction_val = val_design @ self.linear_weights + self.intercept
        best_loss = self._mean_loss(
            prediction_val,
            y_val,
            objective_kind=objective_kind,
            offset=offset_val,
        )
        best_iteration = 0
        without_progress = 0
        _, basis_mask = self._atomic_masks(device=device)

        signed_targets = 2.0 * y_train - 1.0 if objective_kind == "binary" else None
        for stage in range(self.n_estimators):
            if objective_kind == "binary":
                full_prediction = prediction_train
                hessian_sqrt = 0.5 / torch.cosh(
                    0.5 * signed_targets * full_prediction
                )
                pseudo_target = math.sqrt(0.5) * (
                    signed_targets
                    / torch.exp(0.5 * signed_targets * full_prediction)
                )
                multiplier = math.sqrt(0.5) * self.boost_rate * hessian_sqrt
            else:
                full_prediction = prediction_train
                if offset_train is not None:
                    full_prediction = full_prediction + offset_train
                pseudo_target = y_train - full_prediction
                multiplier = torch.full_like(pseudo_target, self.boost_rate)

            train_basis = self._basis_for_stage(train_design, stage)
            val_basis = self._basis_for_stage(val_design, stage)
            weighted_basis = train_basis * multiplier[:, :1]
            gram = weighted_basis.T @ weighted_basis
            gram = gram + self.elm_alpha * torch.eye(
                self.basis_dim, device=device, dtype=gram.dtype
            )
            right_hand_side = weighted_basis.T @ pseudo_target
            coefficients = torch.linalg.solve(gram, right_hand_side)
            coefficients[~basis_mask] = 0.0
            with torch.no_grad():
                self.stage_coefficients[stage].copy_(coefficients)

            prediction_train = torch.clamp(
                prediction_train + self.boost_rate * (train_basis @ coefficients),
                -self.clip_predictions,
                self.clip_predictions,
            )
            prediction_val = torch.clamp(
                prediction_val + self.boost_rate * (val_basis @ coefficients),
                -self.clip_predictions,
                self.clip_predictions,
            )
            train_loss = self._mean_loss(
                prediction_train,
                y_train,
                objective_kind=objective_kind,
                offset=offset_train,
            )
            val_loss = self._mean_loss(
                prediction_val,
                y_val,
                objective_kind=objective_kind,
                offset=offset_val,
            )
            self._train_losses.append(float(train_loss.detach().cpu()))
            self._val_losses.append(float(val_loss.detach().cpu()))

            without_progress += 1
            if val_loss < best_loss:
                best_loss = val_loss
                best_iteration = stage + 1
                without_progress = 0
            if self.early_stopping > 0 and without_progress > self.early_stopping:
                break

        fitted = best_iteration if self.early_stopping > 0 else len(self._val_losses)
        with torch.no_grad():
            self.n_estimators_fitted_tensor.fill_(fitted)
            if fitted < self.n_estimators:
                self.stage_coefficients[fitted:].zero_()
        return {
            "algorithm": "igann_native_elm_boosting",
            "n_estimators_attempted": len(self._val_losses),
            "n_estimators_fitted": fitted,
            "best_validation_loss": float(best_loss.detach().cpu()),
            "selected_features": self.selected_features_,
        }

    @property
    def n_estimators_(self) -> int:
        return int(self.n_estimators_fitted_tensor.item())

    @property
    def selected_features_(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, active in zip(
                self.atomic_feature_names,
                self.active_atomic_mask.tolist(),
                strict=True,
            )
            if active
        )

    @property
    def training_history_(self) -> dict[str, tuple[float, ...]]:
        return {
            "train_loss": tuple(self._train_losses),
            "val_loss": tuple(self._val_losses),
        }

    def basis_metadata(self) -> dict:
        return {
            "source_feature_names": tuple(self.source_feature_names),
            "atomic_feature_names": tuple(self.atomic_feature_names),
            "selected_features": self.selected_features_,
            "hidden_weights": self.hidden_weights[: self.n_estimators_]
            .detach()
            .cpu()
            .numpy()
            .copy(),
            "activation": self.activation_name,
            "elm_scale": self.elm_scale,
        }

    def complexity_metadata(self) -> dict[str, int]:
        active_design = 0
        active_basis = 0
        for active, design_slice, basis_slice in zip(
            self.active_atomic_mask.tolist(),
            self.atomic_design_slices,
            self.atomic_basis_slices,
            strict=True,
        ):
            if active:
                active_design += design_slice.stop - design_slice.start
                active_basis += basis_slice.stop - basis_slice.start
        return {
            "fitted_estimators": self.n_estimators_,
            "selected_features": int(self.active_atomic_mask.sum().item()),
            "effective_fitted_coefficients": int(
                self.num_classes
                * (1 + active_design + self.n_estimators_ * active_basis)
            ),
        }

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        design = self._design_inputs(num_features, cat_features)
        source_outputs = [
            design[:, design_slice] @ self.linear_weights[design_slice]
            for design_slice in self.source_design_slices
        ]
        for stage in range(self.n_estimators_):
            basis = self._basis_for_stage(design, stage)
            for source_index, basis_slice in enumerate(self.source_basis_slices):
                source_outputs[source_index] = source_outputs[source_index] + (
                    self.boost_rate
                    * (basis[:, basis_slice] @ self.stage_coefficients[stage, basis_slice])
                )
        output = torch.stack(source_outputs, dim=1).sum(dim=1) + self.intercept
        result = {"output": output}
        result.update(dict(zip(self.source_feature_names, source_outputs, strict=True)))
        result["intercept"] = self.intercept
        return result


__all__ = ["IGANN"]
