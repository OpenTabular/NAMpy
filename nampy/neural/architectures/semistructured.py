"""Joint spline/neural predictors with weighted training-space identifiability.

A full-cohort projection residualizes each learned neural representation against
all structured columns, including the intercept. New-data prediction uses the
saved training projection, never a query-cohort QR decomposition. This is a
neural objective route with fixed spline penalties, not joint REML training.
"""

import re
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..configs.semistructured_config import DefaultSemiStructuredConfig
from ..configuration import SemiStructuredTrainingConfig
from .components.base_model import BaseModel


class SemiStructured(BaseModel):
    supports_native_training = True
    native_training_config = SemiStructuredTrainingConfig
    native_objective_kinds = frozenset(
        {"regression", "distributional", "binary", "multiclass"}
    )
    estimator_fitted_attributes = (
        "orthogonality_error_",
        "training_history_",
        "n_optimizer_steps_",
    )

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config=None,
        training=None,
    ):
        super().__init__()
        self._validate_features(num_feature_info, cat_feature_info)
        self.config = DefaultSemiStructuredConfig() if config is None else config
        self.training_controls = (
            SemiStructuredTrainingConfig() if training is None else training
        )
        if any(info.get("dimension", 1) != 1 for info in num_feature_info.values()):
            raise NotImplementedError(
                "Composition requires scalar numerical feature blocks."
            )
        if any(
            "onehot" not in info.get("preprocessing", "").replace("-", "").lower()
            and info.get("dimension", 1) <= 1
            for info in cat_feature_info.values()
        ):
            raise NotImplementedError(
                "Composition categorical inputs require fitted one-hot preprocessing."
            )
        self.num_names = tuple(num_feature_info)
        self.cat_dimensions = {
            name: int(info["dimension"]) for name, info in cat_feature_info.items()
        }
        self.feature_names = (*self.num_names, *self.cat_dimensions)
        self.neural_names = self.config.neural_features or self.feature_names
        if not self.neural_names or not set(self.neural_names) <= set(
            self.feature_names
        ):
            raise ValueError(
                "neural_features must select available source feature blocks."
            )
        if re.search(r"\boffset\s*\(", self.config.structured_formula or ""):
            raise NotImplementedError("Composition formula offsets are unsupported.")
        self.num_classes = int(num_classes)
        dimensions = (
            sum(self.cat_dimensions.get(name, 1) for name in self.neural_names),
            *self.config.hidden_dims,
            self.config.representation_dim,
        )
        layers: list[nn.Module] = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)
        self.structured_coefficients = nn.Parameter(torch.empty(0, self.num_classes))
        self.neural_coefficients = nn.Parameter(
            torch.randn(self.config.representation_dim, self.num_classes) * 0.01
        )
        self.register_buffer(
            "projection", torch.empty(0, self.config.representation_dim)
        )

    def prepare_gradient_training(self):
        raise NotImplementedError(
            "Identifiable composition requires SemiStructuredTrainingConfig full-cohort optimization."
        )

    def _frame(self, num_features, cat_features):
        values = {
            name: num_features[name].detach().cpu().numpy().reshape(-1)
            for name in self.num_names
        }
        for name, width in self.cat_dimensions.items():
            encoded = cat_features[name].detach().cpu().numpy()
            if (
                encoded.ndim != 2
                or encoded.shape[1] != width
                or not np.all((encoded == 0) | (encoded == 1))
                or not np.all(encoded.sum(axis=1) == 1)
            ):
                raise ValueError(
                    "Composition categorical inputs need one active fitted category; unknown categories are unsupported."
                )
            values[name] = pd.Categorical(
                np.argmax(encoded, axis=1), categories=range(width)
            )
        return pd.DataFrame(values)

    def _inputs(self, num_features, cat_features):
        features = num_features | cat_features
        return torch.cat(
            [
                features[name].reshape(len(features[name]), -1)
                for name in self.neural_names
            ],
            dim=1,
        ).to(dtype=torch.float64, device="cpu")

    def fit_native(
        self,
        *,
        train_num_features,
        train_cat_features,
        train_targets,
        val_num_features,
        val_cat_features,
        val_targets,
        objective,
        train_offset,
        val_offset,
        train_sample_weight,
        val_sample_weight,
        random_state,
    ):
        from nampy.contracts import AdditiveTermMetadata
        from nampy.gam.design import compile_design

        from ..offsets import add_predictor_offset

        observation_target = getattr(
            getattr(objective, "family", None), "supports_observation_target", False
        )
        if (
            train_targets.ndim > 1
            and train_targets.shape[1] != 1
            and not observation_target
        ):
            raise NotImplementedError(
                "Composition currently supports scalar response objectives only."
            )
        self.to(device="cpu", dtype=torch.float64)
        objective.to(device="cpu", dtype=torch.float64)
        train_y = train_targets.to(device="cpu")
        val_y = val_targets.to(device="cpu")
        n = len(train_y)
        w = (
            torch.ones(n, dtype=torch.float64)
            if train_sample_weight is None
            else train_sample_weight.to(dtype=torch.float64, device="cpu").reshape(-1)
        )
        if not torch.isfinite(w).all() or torch.any(w < 0) or w.sum() <= 0:
            raise ValueError(
                "Composition training needs nonnegative weights with positive sum."
            )
        val_weights = (
            torch.ones(len(val_y), dtype=torch.float64)
            if val_sample_weight is None
            else val_sample_weight.to(dtype=torch.float64, device="cpu").reshape(-1)
        )
        if (
            not len(val_y)
            or not torch.isfinite(val_weights).all()
            or torch.any(val_weights < 0)
            or val_weights.sum() <= 0
        ):
            raise ValueError(
                "Composition validation needs finite nonnegative weights with positive sum."
            )
        if self.config.structured_formula is None and any(
            not name.isidentifier() for name in self.feature_names
        ):
            raise ValueError(
                "Nonidentifier structured feature names require an explicit formula."
            )
        formula = self.config.structured_formula or "~" + "+".join(self.feature_names)
        self.design_ = compile_design(
            self._frame(train_num_features, train_cat_features),
            train_y[:, 0].detach().numpy()
            if observation_target
            else train_y.detach().numpy().reshape(-1),
            formula=formula,
            k=self.config.k,
            basis=self.config.basis,
            smoothing_params=self.training_controls.smoothing_params,
            sample_weight=w.numpy(),
        )
        B = torch.tensor(self.design_.matrix, dtype=torch.float64)
        Bv = torch.tensor(
            self.design_.evaluate(
                self._frame(val_num_features, val_cat_features)
            ).matrix,
            dtype=torch.float64,
        )
        S = torch.tensor(self.design_.penalty, dtype=torch.float64)
        active = w > 0
        sqrt_w = torch.sqrt(w[active])[:, None]
        weighted = B[active] * sqrt_w
        if (
            weighted.shape[0] < weighted.shape[1]
            or torch.linalg.matrix_rank(weighted) != weighted.shape[1]
        ):
            raise NotImplementedError(
                "Composition needs full-rank structured columns under positive training weights."
            )
        Q, R = torch.linalg.qr(weighted, mode="reduced")
        self.structured_coefficients = nn.Parameter(
            torch.zeros(B.shape[1], self.num_classes, dtype=torch.float64)
        )
        self.projection = torch.zeros(
            B.shape[1], self.config.representation_dim, dtype=torch.float64
        )
        train_inputs, val_inputs = (
            self._inputs(train_num_features, train_cat_features),
            self._inputs(val_num_features, val_cat_features),
        )
        optimizer = torch.optim.Adam(
            [
                {"params": [self.structured_coefficients], "weight_decay": 0.0},
                {
                    "params": [*self.encoder.parameters(), self.neural_coefficients],
                    "weight_decay": self.training_controls.neural_weight_decay,
                },
            ],
            lr=self.training_controls.lr,
        )

        def project(representation):
            return torch.linalg.solve_triangular(
                R, Q.T @ (sqrt_w * representation[active]), upper=True
            )

        def prediction(Bx, H, A):
            return (
                Bx @ self.structured_coefficients
                + (H - Bx @ A) @ self.neural_coefficients
            )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best, best_state, best_epoch = float("inf"), None, -1
        initial = [
            parameter.detach().clone() for parameter in self.encoder.parameters()
        ]
        for epoch in range(self.training_controls.epochs):
            optimizer.zero_grad()
            H = self.encoder(train_inputs)
            A = project(H)
            data_loss = objective.compute_loss(
                add_predictor_offset(prediction(B, H, A), train_offset), train_y, w
            )
            penalty = (
                0.5
                * torch.sum(
                    self.structured_coefficients * (S @ self.structured_coefficients)
                )
                / w.sum()
            )
            loss = data_loss + penalty
            if not torch.isfinite(loss):
                raise RuntimeError("Composition training loss is nonfinite.")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.projection.copy_(project(self.encoder(train_inputs)))
                val_loss = objective.compute_loss(
                    add_predictor_offset(
                        prediction(Bv, self.encoder(val_inputs), self.projection),
                        val_offset,
                    ),
                    val_y,
                    val_weights,
                )
                if not torch.isfinite(val_loss):
                    raise RuntimeError("Composition validation loss is nonfinite.")
                history["train_loss"].append(float(data_loss.detach()))
                history["val_loss"].append(float(val_loss))
                if float(val_loss) < best:
                    best, best_epoch = float(val_loss), epoch
                    best_state = deepcopy(self.state_dict())
            if epoch - best_epoch >= self.training_controls.patience:
                break
        if best_state is None:
            raise RuntimeError("No completed composition fit is available.")
        self.load_state_dict(best_state)
        with torch.no_grad():
            residual = self.encoder(train_inputs) - B @ self.projection
            self.orthogonality_error_ = float(
                torch.max(torch.abs(B.T @ (w[:, None] * residual)))
            )
            self.neural_weight_change_ = float(
                sum(
                    torch.sum((parameter - before) ** 2)
                    for parameter, before in zip(
                        self.encoder.parameters(), initial, strict=True
                    )
                )
            )
        self.n_optimizer_steps_ = len(history["train_loss"])
        self.training_history_ = history | {"best_epoch": best_epoch}
        self.component_metadata = {
            "neural_component": AdditiveTermMetadata(
                features=tuple(self.neural_names), kind="global"
            )
        }
        for term in self.design_.terms:
            self.component_metadata[f"structured:{term.label}"] = AdditiveTermMetadata(
                features=term.features,
                kind="interaction" if len(term.features) > 1 else "main",
            )
        # No optimizer, tensors of training observations, validation data or
        # response vector are retained. Compiler-owned fitted basis state stays.
        self.design_ = self.design_.for_prediction()
        return {
            "finished": True,
            "optimizer_steps": self.n_optimizer_steps_,
            "best_val_loss": best,
            "orthogonality_error": self.orthogonality_error_,
        }

    def forward(self, num_features, cat_features):
        if not hasattr(self, "design_"):
            raise RuntimeError("Fit the composition before prediction.")
        B = torch.tensor(
            self.design_.evaluate(self._frame(num_features, cat_features)).matrix,
            dtype=torch.float64,
        )
        H = self.encoder(self._inputs(num_features, cat_features))
        neural = (H - B @ self.projection) @ self.neural_coefficients
        structured = B @ self.structured_coefficients
        output = {
            "output": structured + neural,
            "neural_component": neural,
            "intercept": self.structured_coefficients[
                self.design_.intercept_indices
            ].sum(dim=0),
        }
        for term in self.design_.terms:
            idx = term.coefficient_indices
            output[f"structured:{term.label}"] = (
                B[:, idx] @ self.structured_coefficients[idx]
            )
        return output
