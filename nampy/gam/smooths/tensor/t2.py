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
    build_penalty_definition,
)
from .marginals import (
    build_tensor_marginal_terms,
    resolve_tensor_marginal_features,
    tensor_marginal_fit_matrices,
    tensor_marginal_predict_matrix,
    validate_tensor_marginal_bases,
)
from .t2_basis import build_t2_basis_and_penalties, materialize_t2_newdata


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
        self._by_state = None
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

        t2_obj = build_t2_basis_and_penalties(
            marginal_decompositions,
            full=self.full,
            ord=self.ord,
            # mgcv::smooth.construct.t2.smooth.spec() keeps the raw t2 basis
            # and exposes an explicit constraint matrix affecting only the
            # final unpenalized null block.
            remove_constant_from_null_block=False,
        )
        B_t2_setup = np.asarray(t2_obj["basis"], dtype=np.float64)
        pens_pre = [
            np.asarray(S, dtype=np.float64) for S in t2_obj["penalties_pre_constraint"]
        ]
        B_pre = np.asarray(t2_obj["basis_pre_constraint"], dtype=np.float64)
        full_transform = t2_obj.get("full_constraint_transform", None)
        if len(pens_pre) > 0:
            # mgcv smoothCon(scale.penalty=TRUE, absorb.cons=TRUE) scales the
            # assembled t2 penalties before absorbing the null-block constraint,
            # then applies the constraint transform to the scaled blocks.
            pens_scaled_pre = rescale_tensor_penalties_for_fit(
                B_pre,
                pens_pre,
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

        B_t2 = materialize_t2_newdata(
            marginal_fit,
            allnull_specs=t2_obj["allnull_specs"],
            allnull_transform=t2_obj["allnull_transform"],
            penalized_specs=t2_obj["penalized_specs"],
        )
        B_t2 = self._apply_cached_by(np.asarray(B_t2, dtype=np.float64))
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
        n_pen = int(sum(spec["n_cols"] for spec in self._penalized_specs))
        n_null = int(B_t2_setup.shape[1] - n_pen)
        if n_null > 0 and self._by_state.is_constant:
            Cp = np.sum(B_t2_setup, axis=0, keepdims=True)
            if np.linalg.norm(Cp) > 0.0:
                self.predict_coefficient_map, _ = (
                    null_space_basis_from_constraint_matrix(
                        Cp,
                        d=B_t2_setup.shape[1],
                        tol=self.null_penalty_tol,
                    )
                )
            else:
                self.predict_coefficient_map = None
            if n_null == 1:
                # mgcv::smooth.construct.t2.smooth.spec():
                # ``if (object$null.space.dim==1) C <- ncol(X)``
                # i.e. set final unpenalized coefficient to zero, rather than
                # centering it by row sums.
                C = np.zeros((1, B_t2_setup.shape[1]), dtype=np.float64)
                C[0, -1] = 1.0
                self.fit_constraint_matrix = C
            else:
                C = np.zeros((1, B_t2_setup.shape[1]), dtype=np.float64)
                C[0, n_pen:] = np.sum(B_t2_setup[:, n_pen:], axis=0)
                self.fit_constraint_matrix = C if np.linalg.norm(C) > 0.0 else None
        else:
            self.fit_constraint_matrix = None
            self.predict_coefficient_map = None
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

        allnull_specs = self._t2_train["allnull_specs"]
        allnull_transform = self._t2_train["allnull_transform"]

        penalized_specs = self._t2_train.get("penalized_specs", None)
        if penalized_specs is None:
            raise RuntimeError(
                "Stored t2 fit object does not contain penalized component specifications."
            )

        B_new = materialize_t2_newdata(
            marginal_new,
            allnull_specs=allnull_specs,
            allnull_transform=allnull_transform,
            penalized_specs=penalized_specs,
        )
        return self._apply_constraint_transform_and_by(B_new, X_new)
