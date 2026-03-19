import warnings
import numpy as np

from ..base import (
    BaseSmoothTerm,
    _resolve_numeric_by,
    _is_effectively_constant,
    _normalize_knots,
    _normalize_mc,
)
from ..registry import register_smooth
from ..univariate.cubic_regression import SplineTerm1D
from .ops import rowwise_kronecker, tensor_product_penalties


@register_smooth("ti")
class InteractionTensorProductSplineTerm(BaseSmoothTerm):
    term_type = "tensor_interaction"
    basis_name = "ti"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=10,
        basis="cr",
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        mc=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 1:
            raise ValueError("InteractionTensorProductSplineTerm requires at least one feature.")

        super().__init__(
            feature=features,
            label=label or f"ti({', '.join(map(str, features))})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        if np.isscalar(k):
            self.k = [int(k)] * len(features)
        else:
            self.k = [int(v) for v in k]
        if len(self.k) != len(features):
            raise ValueError(
                f"k must have length {len(features)} for features={features}, got {self.k}."
            )

        if isinstance(basis, str):
            self.basis = [basis] * len(features)
        else:
            self.basis = [str(v) for v in basis]
        if len(self.basis) != len(features):
            raise ValueError(
                f"basis must have length {len(features)} for features={features}, got {self.basis}."
            )

        if any(bs != "cr" for bs in self.basis):
            raise NotImplementedError(
                "InteractionTensorProductSplineTerm currently supports only basis='cr' marginals."
            )

        self.mc = mc
        self.fixed = bool(fixed)
        self.knots = _normalize_knots(knots, features)

        self._mc = None
        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._basis_train = None
        self._penalties = None
        self._basis_dims = None
        self._marginal_is_centered = None

        self._by_index = None
        self._by_name = None
        self._by_values = None
        self._by_is_constant = None

    def fit(self, X, feature_names):
        marginals = []
        feature_indices = []
        feature_names_resolved = []

        if self.by is not None:
            self._by_index, self._by_name, self._by_values = _resolve_numeric_by(
                self.by, X, feature_names
            )
            self._by_is_constant = _is_effectively_constant(self._by_values)
        else:
            self._by_index, self._by_name, self._by_values = None, None, None
            self._by_is_constant = True

        self._mc = _normalize_mc(self.mc, len(self.feature))

        for feat, k_i, bs_i, knots_i in zip(self.feature, self.k, self.basis, self.knots):
            term = SplineTerm1D(
                feature=feat,
                k=k_i,
                basis=bs_i,
                label=str(feat),
                smoothing_id=None,
                by=None,
                select=False,
                fixed=False,
                knots=knots_i,
            )
            term.fit(X, feature_names)
            marginals.append(term)
            feature_indices.append(term._feature_index)
            feature_names_resolved.append(term._feature_name)

        marginal_bases = []
        marginal_penalties = []
        basis_dims = []

        if self.by is not None and not self._by_is_constant:
            if self.mc is not None:
                warnings.warn(
                    f"{self.label}: ignoring mc={self.mc} because numeric by={self._by_name!r} "
                    "is non-constant, so automatic identifiability constraints are not applied."
                )
            use_centered = [False] * len(marginals)
        else:
            use_centered = list(self._mc)

        for m, center_this_margin in zip(marginals, use_centered):
            if center_this_margin:
                B = m.basis_train
                S = m.penalties[0]
            else:
                B = m._spline.raw_basis
                S = m._spline.raw_penalty

            marginal_bases.append(np.asarray(B, dtype=np.float64))
            marginal_penalties.append(np.asarray(S, dtype=np.float64))
            basis_dims.append(B.shape[1])

        B_ti = rowwise_kronecker(marginal_bases)
        if self._by_values is not None:
            B_ti = B_ti * self._by_values[:, None]

        self._marginals = marginals
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._basis_dims = basis_dims
        self._marginal_is_centered = list(use_centered)
        self._basis_train = np.asarray(B_ti, dtype=np.float64)
        self._penalties = [] if self.fixed else tensor_product_penalties(
            marginal_penalties, basis_dims=basis_dims
        )

        self.basis_name = "ti(" + ",".join(self.basis) + ")"
        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._penalties is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def transform_new(self, X_new):
        if self._marginals is None or self._marginal_is_centered is None:
            raise RuntimeError("Term is not fitted.")

        X_new = np.asarray(X_new, dtype=np.float64)
        marginal_new = []
        for m, center_this_margin in zip(self._marginals, self._marginal_is_centered):
            xj = X_new[:, m._feature_index]
            if center_this_margin:
                Bj = m._spline.transform_new_centered(xj)
            else:
                Bj = m._spline.transform_new_raw(xj)
            marginal_new.append(Bj)

        B = rowwise_kronecker(marginal_new)
        if self._by_index is not None:
            z = np.asarray(X_new[:, self._by_index], dtype=np.float64).ravel()
            B = B * z[:, None]
        return B
