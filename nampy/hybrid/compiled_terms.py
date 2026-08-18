"""Compiled mgcv-parity GAM terms as fixed-lambda Torch components.

The compiler stage of the GAM pipeline (formula -> specs -> compiled model)
runs standalone in numpy: it needs no fit. :class:`CompiledGAMTerms` wraps
that handle — exact mgcv basis construction, constraint absorption, and
penalty matrices — and :class:`CompiledGAMTermsModule` turns it into a Torch
module: coefficients as ``nn.Parameter``, penalty matrices as buffers, and
``sum(lam_k * beta_k' S_k beta_k)`` emitted through the ``*_penalty`` loss
seam.

EXPERIMENTAL, and NOT an mgcv fit: smoothing parameters are FIXED (supplied
or lifted from a fitted GAM) and optimization happens through Torch. Results
will not and should not match mgcv; never compare them in parity suites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def _guard_raw_basis_terms(compiled_model) -> None:
    for term in compiled_model.compiled_terms:
        if getattr(term, "metadata", {}).get("expose_raw_prediction_basis"):
            raise NotImplementedError(
                f"Term {term.label!r} exposes a raw prediction basis whose "
                "width differs from the training basis; the Torch bridge "
                "requires prediction and training bases of equal width."
            )


@dataclass
class CompiledGAMTerms:
    """Numpy-side handle: compiled model, fixed smoothing, preprocessing."""

    compiled_model: object
    lam: np.ndarray
    feature_names: list[str]
    preprocess_state: dict | None
    response: np.ndarray | None
    response_name: str | None
    fit_intercept: bool

    @classmethod
    def from_formula(
        cls,
        formula,
        data,
        *,
        lam,
        default_k=10,
        default_basis="tp",
        fit_intercept=True,
    ) -> CompiledGAMTerms:
        """Compile mgcv-parity terms for ``formula`` on ``data``; no fit."""
        from ..gam.compiler.compile_model import compile_model
        from ..gam.formula.extract import extract_formula_terms
        from ..gam.formula.parse import parse_gam_formula
        from ..gam.specs.build import build_formula_model

        parsed = parse_gam_formula(formula)
        extracted = extract_formula_terms(parsed)
        build_result = build_formula_model(
            extracted,
            data,
            default_k=default_k,
            default_basis=default_basis,
        )
        compiled = compile_model(
            X=build_result.X,
            feature_names=build_result.feature_names,
            predictor_specs=build_result.predictor_specs,
            fit_intercept=fit_intercept,
        )
        _guard_raw_basis_terms(compiled)

        lam = np.asarray(lam, dtype=np.float64).ravel()
        if lam.size != compiled.n_smoothing_params:
            raise ValueError(
                f"lam must have length {compiled.n_smoothing_params} "
                f"(one per smoothing parameter); got {lam.size}."
            )
        if np.any(lam < 0):
            raise ValueError("All lam entries must be non-negative.")

        response = build_result.response
        return cls(
            compiled_model=compiled,
            lam=lam,
            feature_names=list(build_result.feature_names),
            preprocess_state=build_result.preprocess_state,
            response=None if response is None else np.asarray(response),
            response_name=build_result.response_name,
            fit_intercept=bool(fit_intercept),
        )

    @classmethod
    def from_fitted_gam(cls, gam) -> CompiledGAMTerms:
        """Lift the compiled model and fitted smoothing from a fitted GAM.

        The GAM is read, never modified.
        """
        compiled = getattr(gam, "compiled_model_", None)
        if compiled is None:
            raise ValueError("from_fitted_gam requires a fitted GAM.")
        _guard_raw_basis_terms(compiled)

        lam = np.asarray(
            gam.fit_result().smoothing_params, dtype=np.float64
        ).ravel()
        feature_names = getattr(gam, "formula_feature_columns_", None)
        if feature_names is None:
            n_features = int(np.asarray(gam.X_).shape[1])
            feature_names = [f"x{index}" for index in range(n_features)]
        return cls(
            compiled_model=compiled,
            lam=lam,
            feature_names=[str(name) for name in feature_names],
            preprocess_state=getattr(gam, "formula_preprocess_state_", None),
            response=np.asarray(gam.y_),
            response_name=None,
            fit_intercept=bool(gam.fit_intercept),
        )

    @property
    def n_coef(self) -> int:
        return int(self.compiled_model.n_coef)

    def design(self, X_new=None) -> np.ndarray:
        """Training design (``X_new=None``) or the design for new data."""
        if X_new is None:
            return np.asarray(self.compiled_model.design_matrix, dtype=np.float64)

        if hasattr(X_new, "columns") and self.preprocess_state is not None:
            from ..gam.specs.build import apply_formula_preprocess_to_new_data

            X_work = apply_formula_preprocess_to_new_data(
                X_new, self.preprocess_state
            )
            missing = [
                name for name in self.feature_names if name not in X_work.columns
            ]
            if missing:
                raise KeyError(f"Prediction data is missing columns: {missing}")
            X_np = X_work[self.feature_names].to_numpy()
        else:
            X_np = np.asarray(X_new)
        return np.asarray(
            self.compiled_model.build_new_matrix(X_np), dtype=np.float64
        )

    def penalty_payload(self) -> list[tuple[slice, np.ndarray, float]]:
        """(coef_slice, S_k, lam_k) triples in reduced coefficient space."""
        return [
            (
                penalty.coef_slice,
                np.asarray(penalty.matrix, dtype=np.float64),
                float(self.lam[penalty.smoothing_index]),
            )
            for penalty in self.compiled_model.compiled_penalties
        ]


class CompiledGAMTermsModule(nn.Module):
    """Torch module over a fixed compiled-GAM design.

    ``forward`` consumes the design-matrix batch directly and returns the
    linear predictor plus the quadratic smoothness penalty (through the
    ``*_penalty`` loss seam). The penalty uses the FIXED lambdas captured at
    construction; there is no smoothing selection.
    """

    def __init__(self, gam_terms: CompiledGAMTerms, num_classes: int = 1):
        super().__init__()
        self.n_coef = gam_terms.n_coef
        self.num_classes = int(num_classes)
        self.beta = nn.Parameter(torch.zeros(self.n_coef, self.num_classes))
        self.intercept: nn.Parameter | None
        if gam_terms.fit_intercept:
            self.intercept = nn.Parameter(torch.zeros(self.num_classes))
        else:
            self.intercept = None

        self._penalty_slices: list[slice] = []
        self._penalty_lams: list[float] = []
        for index, (coef_slice, matrix, lam_value) in enumerate(
            gam_terms.penalty_payload()
        ):
            self.register_buffer(
                f"S_{index}", torch.tensor(matrix, dtype=torch.float32)
            )
            self._penalty_slices.append(coef_slice)
            self._penalty_lams.append(lam_value)

    def penalty(self) -> torch.Tensor:
        total = self.beta.new_zeros(())
        for index, coef_slice in enumerate(self._penalty_slices):
            S = getattr(self, f"S_{index}")
            lam_value = self._penalty_lams[index]
            block = self.beta[coef_slice, :]
            total = total + lam_value * torch.einsum(
                "ik,ij,jk->", block, S, block
            )
        return total

    def forward(self, design: torch.Tensor) -> dict[str, torch.Tensor]:
        output = design @ self.beta
        if self.intercept is not None:
            output = output + self.intercept
        return {"output": output, "gam_penalty": self.penalty()}
