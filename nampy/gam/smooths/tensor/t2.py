"""
Tensor-product smooth with null-space reparameterisation (``bs='t2'``).

Like ``te``, but uses an alternative penalty parameterisation where the
penalty directions align with the null space of each marginal penalty.
This can improve numerical conditioning for tensor products of cubic
splines and is the recommended choice when the smoothing parameters for
different axes are very different in magnitude.
"""

import warnings

import numpy as np

from ...constraints.transforms import null_space_basis_from_constraint_matrix
from ...penalties import penalty_id_for_local_index, rescale_tensor_penalties_for_fit
from ..algebra import t2_marginal_reparameterization
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _normalize_knots,
    by_values_from_new_data,
    build_penalty_definition,
)
from .marginals import (
    build_tensor_marginal_terms,
    resolve_tensor_marginal_features,
    tensor_marginal_fit_matrices,
    tensor_marginal_predict_matrix,
    validate_tensor_marginal_bases,
)
from .t2_basis import (
    build_tensor_anova_basis_and_penalties,
    materialize_tensor_anova_newdata,
)


def _normalize_t2_fx_flags(fx, n_penalties: int) -> list[bool]:
    n_penalties = int(n_penalties)
    if n_penalties <= 0:
        return []
    if fx is None:
        return [False] * n_penalties
    if np.isscalar(fx):
        return [bool(fx)] * n_penalties
    vals = [bool(v) for v in np.asarray(fx, dtype=object).ravel()]
    if len(vals) == 1:
        return vals * n_penalties
    if len(vals) != n_penalties:
        warnings.warn("fx length wrong from t2 term: ignored", stacklevel=3)
        return [False] * n_penalties
    return vals


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
        m=None,
        xt=None,
        label=None,
        term_id=None,
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
            term_id=term_id,
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
        self.basis = validate_tensor_marginal_bases(self.basis)
        self.m = m
        self.xt = xt

        self.select = bool(select)
        self.full = bool(full)
        self.ord = ord
        self.fixed = fixed
        self.fixed_flags = None
        self.null_penalty_tol = float(null_penalty_tol)
        self.knots = _normalize_knots(knots, features)

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._basis_train = None
        self._penalties = None
        self._t2_train = None
        self._t2_raw_fit_transform = None
        self._marginal_decompositions = None
        self._penalized_specs = None
        self._by_state = None
        self._prediction_constraint_transform = None
        self.fit_constraint_matrix = None
        self.predict_coefficient_map = None

    def fit(self, X, feature_names):
        self._set_by_state(X, feature_names)
        marginal_shared_setups = self._linked_id_marginal_setups()
        marginals, _, _ = build_tensor_marginal_terms(
            feature=self.feature,
            k=self.k,
            basis=self.basis,
            m=self.m,
            xt=self.xt,
            knots=self.knots,
            centered=False,
            shared_basis_setups=marginal_shared_setups,
        )
        for term in marginals:
            term.fit(X, feature_names)
        feature_indices, feature_names_resolved = resolve_tensor_marginal_features(
            marginals
        )
        marginal_decompositions = []
        for basis_name, term in zip(self.basis, marginals):
            B_i, S_i, _ = tensor_marginal_fit_matrices(term, centered=False)
            dec = t2_marginal_reparameterization(B_i, S_i, basis_name=basis_name)
            marginal_decompositions.append(dec)

        absorb_null_constraint = bool(self._by_state.is_constant)
        t2_obj = build_tensor_anova_basis_and_penalties(
            marginal_decompositions,
            full=self.full,
            ord=self.ord,
            # Match smoothCon(absorb.cons=TRUE) for ordinary intercept-bearing
            # t2 terms by applying the null-block constraint directly to the
            # assembled t2 basis. The generic wrapper QR path rotates columns
            # differently and breaks exact mgcv parity for tensor ANOVA terms.
            remove_constant_from_null_block=absorb_null_constraint,
        )
        B_t2_setup = np.asarray(t2_obj["basis"], dtype=np.float64)
        pens_pre = [
            np.asarray(S, dtype=np.float64) for S in t2_obj["penalties_pre_constraint"]
        ]
        B_pre = np.asarray(t2_obj["basis_pre_constraint"], dtype=np.float64)
        full_transform = t2_obj.get("full_constraint_transform", None)
        penalty_scales = []
        if len(pens_pre) > 0:
            # mgcv smoothCon(scale.penalty=TRUE, absorb.cons=TRUE) scales the
            # assembled t2 penalties before absorbing the null-block constraint,
            # then applies the constraint transform to the scaled blocks.
            pens_scaled_pre, penalty_scales = rescale_tensor_penalties_for_fit(
                B_pre,
                pens_pre,
                return_scales=True,
            )
            if full_transform is not None:
                C_full = np.asarray(full_transform, dtype=np.float64)
                pens_t2 = [
                    0.5 * (C_full.T @ S @ C_full + (C_full.T @ S @ C_full).T)
                    for S in pens_scaled_pre
                ]
            else:
                pens_t2 = pens_scaled_pre
            t2_obj = {**t2_obj, "basis": B_t2_setup, "penalties": pens_t2}
        else:
            t2_obj = {**t2_obj, "basis": B_t2_setup}

        marginal_fit = []
        for m, dec in zip(marginals, marginal_decompositions):
            B_raw = tensor_marginal_predict_matrix(m, X, centered=False)

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

            marginal_fit.append({"B_range": B_r, "B_null": B_n})

        B_t2_raw = materialize_tensor_anova_newdata(
            marginal_fit,
            allnull_specs=t2_obj["allnull_specs"],
            allnull_transform=None,
            penalized_specs=t2_obj["penalized_specs"],
        )
        B_t2_raw = self._apply_cached_by(np.asarray(B_t2_raw, dtype=np.float64))
        if full_transform is not None:
            B_t2_fit = B_t2_raw @ np.asarray(full_transform, dtype=np.float64)
        else:
            B_t2_fit = B_t2_raw
        t2_obj = {
            **t2_obj,
            "basis_raw": np.asarray(B_t2_raw, dtype=np.float64),
            "basis": np.asarray(B_t2_fit, dtype=np.float64),
        }

        self._marginals = marginals
        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._marginal_decompositions = marginal_decompositions
        self._basis_train = t2_obj["basis"]
        self.fixed_flags = _normalize_t2_fx_flags(
            self.fixed,
            len(t2_obj["penalties"]),
        )
        self._penalties = [
            np.asarray(S, dtype=np.float64)
            for S, is_fixed in zip(t2_obj["penalties"], self.fixed_flags)
            if not is_fixed
        ]
        self._t2_train = t2_obj
        self._t2_raw_fit_transform = (
            None
            if full_transform is None
            else np.asarray(full_transform, dtype=np.float64)
        )
        self._penalized_specs = [
            spec for spec in t2_obj["component_specs"] if spec["penalized"]
        ]
        n_pen = int(sum(spec["n_cols"] for spec in self._penalized_specs))
        n_null = int(B_pre.shape[1] - n_pen)
        fit_transform = self._t2_raw_fit_transform
        prediction_basis_map = None
        if absorb_null_constraint and n_null > 0:
            cp = np.asarray(B_pre.mean(axis=0), dtype=np.float64).reshape(1, -1)
            prediction_basis_map, _ = null_space_basis_from_constraint_matrix(
                cp,
                d=B_pre.shape[1],
            )
            prediction_basis_map = np.asarray(
                prediction_basis_map,
                dtype=np.float64,
            )

        self._prediction_constraint_transform = prediction_basis_map
        self.fit_constraint_matrix = None
        self.predict_coefficient_map = None
        self.metadata["prediction_basis_map"] = prediction_basis_map
        self.metadata["expose_raw_prediction_basis"] = False
        self.metadata["prediction_parameterization_wider"] = False
        self.metadata["prediction_replaces_intercept"] = False
        self._record_constraint_result(
            "sum_to_zero" if fit_transform is not None else None,
            fit_transform,
            absorbed_by=("runtime" if fit_transform is not None else None),
        )

        suffix = "full" if self.full else "pars"
        self.basis_name = f"t2({','.join(self.basis)};{suffix})"
        self._set_mgcv_penalty_rescale_factors(
            [
                float(scale)
                for scale, is_fixed in zip(penalty_scales, self.fixed_flags)
                if not is_fixed
            ]
        )
        return self

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        raw = list(self.penalties)
        if len(raw) == 0:
            return []

        n_raw = len(raw)
        sp_vals = self._normalized_t2_term_sp(n_raw)
        defs = []
        for j, P in enumerate(raw):
            sid = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(self.smoothing_id, j, n_penalties=n_raw)
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
                    local_penalty_index=j,
                )
            )

        defs.extend(
            self._build_selection_penalty_definitions(
                raw,
                null_penalty_tol=self.null_penalty_tol,
            )
        )

        return defs

    def _normalized_t2_term_sp(self, n_penalties):
        if n_penalties <= 0:
            return []
        if self.sp is None:
            return [None] * n_penalties
        if np.isscalar(self.sp):
            vals = np.asarray([float(self.sp)], dtype=np.float64)
        else:
            vals = np.asarray(self.sp, dtype=np.float64).ravel()
        if vals.size != n_penalties:
            warnings.warn("length of sp incorrect in t2: ignored", stacklevel=2)
            return [None] * n_penalties
        return [float(v) for v in vals]

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
        self._require_fitted()

        marginal_new = []
        for m, dec in zip(self._marginals, self._marginal_decompositions):
            B_raw = tensor_marginal_predict_matrix(m, X_new, centered=False)

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

        penalized_specs = self._t2_train.get("penalized_specs", None)
        if penalized_specs is None:
            raise RuntimeError(
                "Stored t2 fit object does not contain penalized component specifications."
            )

        B_new = materialize_tensor_anova_newdata(
            marginal_new,
            allnull_specs=self._t2_train["allnull_specs"],
            allnull_transform=None,
            penalized_specs=penalized_specs,
        )
        B_new = np.asarray(B_new, dtype=np.float64)
        z = by_values_from_new_data(X_new, self._by_state)
        B_new = self._apply_by_scale(B_new, z)
        if self._t2_raw_fit_transform is not None:
            B_new = B_new @ self._t2_raw_fit_transform
        return np.asarray(B_new, dtype=np.float64)
