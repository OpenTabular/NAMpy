import numpy as np

from ..base import BaseSmoothTerm, _normalize_knots
from ..registry import register_smooth
from ..univariate.cubic_regression import SplineTerm1D
from .ops import (
    marginal_range_null_decomposition,
    build_t2_basis_and_penalties,
    materialize_t2_newdata,
)


@register_smooth("t2")
class TensorANOVASplineTerm(BaseSmoothTerm):
    term_type = "tensor_anova"
    basis_name = "t2"
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
        full=False,
        ord=None,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 2:
            raise ValueError("TensorANOVASplineTerm requires at least two features.")

        super().__init__(
            feature=features,
            label=label or f"t2({', '.join(map(str, features))})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        if by is not None:
            raise NotImplementedError(
                "t2 `by` variables are deferred to a later phase."
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
                "TensorANOVASplineTerm currently supports only basis='cr' marginals."
            )

        self.full = bool(full)
        self.ord = ord
        self.fixed = bool(fixed)
        self.knots = _normalize_knots(knots, features)

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._basis_train = None
        self._penalties = None
        self._t2_train = None
        self._marginal_decompositions = None
        self._penalized_specs = None

    def fit(self, X, feature_names):
        marginals = []
        feature_indices = []
        feature_names_resolved = []
        marginal_decompositions = []

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

            dec = marginal_range_null_decomposition(
                term._spline.raw_basis,
                term._spline.raw_penalty,
            )
            marginal_decompositions.append(dec)

        t2_obj = build_t2_basis_and_penalties(
            marginal_decompositions,
            full=self.full,
            ord=self.ord,
            remove_constant_from_null_block=True,
        )

        self._marginals = marginals
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._marginal_decompositions = marginal_decompositions
        self._basis_train = t2_obj["basis"]
        self._penalties = [] if self.fixed else t2_obj["penalties"]
        self._t2_train = t2_obj
        self._penalized_specs = [
            spec for spec in t2_obj["component_specs"] if spec["penalized"]
        ]

        suffix = "full" if self.full else "pars"
        self.basis_name = f"t2({','.join(self.basis)};{suffix})"
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
        if self._marginals is None or self._t2_train is None:
            raise RuntimeError("Term is not fitted.")

        X_new = np.asarray(X_new, dtype=np.float64)

        marginal_new = []
        for m, dec in zip(self._marginals, self._marginal_decompositions):
            xj = X_new[:, m._feature_index]
            B_raw = m._spline.transform_new_raw(xj)

            B_r = (
                B_raw @ dec["T_range"]
                if dec["T_range"].shape[1] > 0
                else np.empty((B_raw.shape[0], 0), dtype=np.float64)
            )
            B_n = (
                B_raw @ dec["T_null"]
                if dec["T_null"].shape[1] > 0
                else np.empty((B_raw.shape[0], 0), dtype=np.float64)
            )

            marginal_new.append(
                {
                    "B_range": B_r,
                    "B_null": B_n,
                }
            )

        allnull_specs = self._t2_train["allnull_specs"]
        allnull_transform = self._t2_train["allnull_transform"]

        penalized_specs = self._t2_train.get("penalized_specs", None)
        if penalized_specs is None:
            raise RuntimeError(
                "Stored t2 fit object does not contain penalized component specifications."
            )

        return materialize_t2_newdata(
            marginal_new,
            allnull_specs=allnull_specs,
            allnull_transform=allnull_transform,
            penalized_specs=penalized_specs,
        )
