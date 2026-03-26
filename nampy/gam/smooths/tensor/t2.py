"""
Tensor-product smooth with null-space reparameterisation (``bs='t2'``).

Like ``te``, but uses an alternative penalty parameterisation where the
penalty directions align with the null space of each marginal penalty.
This can improve numerical conditioning for tensor products of cubic
splines and is the recommended choice when the smoothing parameters for
different axes are very different in magnitude.
"""

import numpy as np

from ..base import (
    BaseSmoothTerm,
    _normalize_knots,
    build_penalty_definition,
    build_selection_penalty_definition,
    column_as_float,
)
from ..registry import register_smooth
from ..univariate.cubic_regression import SplineTerm1D
from ...basis.tensor import (
    build_t2_basis_and_penalties,
    marginal_range_null_decomposition,
    materialize_t2_newdata,
    rescale_tensor_penalties_for_fit,
)
from ...penalties.algebra import null_space_penalty_from_penalty


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
        select=False,
        full=False,
        ord=None,
        fixed=False,
        null_penalty_tol=1e-10,
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

        self.select = bool(select)
        self.full = bool(full)
        self.ord = ord
        self.fixed = bool(fixed)
        self.null_penalty_tol = float(null_penalty_tol)
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
        B_t2 = np.asarray(t2_obj["basis"], dtype=np.float64)
        if not self.fixed:
            pens_t2 = rescale_tensor_penalties_for_fit(
                B_t2,
                [np.asarray(S, dtype=np.float64) for S in t2_obj["penalties"]],
            )
            t2_obj = {**t2_obj, "basis": B_t2, "penalties": pens_t2}
        else:
            t2_obj = {**t2_obj, "basis": B_t2}

        self._marginals = marginals
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._marginal_decompositions = marginal_decompositions
        self._basis_train = t2_obj["basis"]
        self._penalties = [] if self.fixed else t2_obj["penalties"]
        self._t2_train = t2_obj
        self._penalized_specs = [
            spec for spec in t2_obj["component_specs"] if spec["penalized"]
        ]
        self._record_constraint_result(None, None, absorbed_by=None)

        suffix = "full" if self.full else "pars"
        self.basis_name = f"t2({','.join(self.basis)};{suffix})"
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        raw = list(self.penalties)
        if len(raw) == 0:
            return []

        n_raw = len(raw)
        sp_vals = self._normalized_term_sp(n_raw)
        defs = []
        for j, P in enumerate(raw):
            sid = (
                None
                if self.smoothing_id is None
                else (str(self.smoothing_id) if n_raw <= 1 else f"{self.smoothing_id}::{j}")
            )
            sp_j = sp_vals[j] if j < len(sp_vals) else None
            defs.append(
                build_penalty_definition(
                    self,
                    P,
                    kind="smooth",
                    smoothing_id=sid,
                    sp_value_in=sp_j,
                    metadata_extra={"term_sp": sp_j, "is_selection_penalty": False},
                )
            )

        if self.select:
            combined = sum(np.asarray(P, dtype=np.float64) for P in raw)
            S0, meta = null_space_penalty_from_penalty(combined, tol=self.null_penalty_tol)
            if meta["rank"] > 0:
                select_sid = (
                    None
                    if self.smoothing_id is None
                    else f"{self.smoothing_id}::select"
                )
                defs.append(
                    build_selection_penalty_definition(
                        self,
                        S0,
                        rank=meta["rank"],
                        null_space_dim=meta["null_space_dim"],
                        smoothing_id=select_sid,
                    )
                )

        return defs

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

        marginal_new = []
        for m, dec in zip(self._marginals, self._marginal_decompositions):
            xj = column_as_float(X_new, m._feature_index)
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
