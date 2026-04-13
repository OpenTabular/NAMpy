from __future__ import annotations

import numpy as np

from ....splines.mrf import (
    coerce_nb,
    coerce_penalty_matrix,
    combine_duplicate_polys,
    laplacian_penalty_from_nb,
    nat_param_type0,
    polys_to_nb,
)
from ..._mgcv_constants import EIG_TOL_POWER
from ...constraints.absorption import full_term_sum_to_zero_constraint
from ...design.structures import PenaltySpec
from ...penalties import build_null_space_selection_spec
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    apply_numeric_by,
    column_as_object,
    resolve_by_state,
    sync_by_state_attributes,
)
from .categorical_utils import (
    as_object_1d,
    factor_indicator_matrix,
    stable_unique_levels,
)


class MarkovRandomFieldTerm(BaseSmoothTerm):
    """
    mgcv-like bs="mrf" smooth.

    Supported xt inputs
    -------------------
    - xt={"polys": ...}
    - xt={"nb": ...}
    - xt={"penalty": ...}
    - combinations thereof

    Low-rank MRF
    ------------
    If k < number of areas, use the natural-parameter truncation described in
    mgcv's constructor source and documentation.
    """

    term_type = "smooth"
    basis_name = "mrf"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=-1,
        basis="mrf",
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        xt=None,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) != 1:
            raise NotImplementedError(
                "bs='mrf' currently supports exactly one area-label variable."
            )

        super().__init__(
            feature=features,
            label=label or f"mrf({features[0]})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.k = int(k)
        self.select = bool(select)
        self.basis_name = str(basis).lower()
        self.xt = xt
        self.knots = knots

        if self.basis_name != "mrf":
            raise NotImplementedError(
                f"MarkovRandomFieldTerm currently supports only basis='mrf', got {basis!r}."
            )

        self._feature_index = None
        self._feature_name = None

        self._by_state = None

        self._area_names = None
        self._plot_polys = None
        self._nb = None

        self._basis_train = None
        self._penalties = None

        self._P = None
        self._full_penalty = None
        self._rank = None
        self._null_space_dim = None

    def fit(self, X, feature_names):
        X = np.asarray(X, dtype=object)

        idx, fname = _resolve_feature(self.feature[0], feature_names)
        self._feature_index = idx
        self._feature_name = fname

        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)

        x = as_object_1d(X[:, idx])
        self._set_resolved_features([fname])

        if self.xt is None:
            raise ValueError(
                "For bs='mrf', xt must supply at least one of {'polys','nb','penalty'}."
            )
        if not any(key in self.xt for key in ("polys", "nb", "penalty")):
            raise ValueError(
                "For bs='mrf', xt must supply at least one of {'polys','nb','penalty'}."
            )

        if self.knots is None:
            area_names = stable_unique_levels(x)
        else:
            area_names = list(np.asarray(self.knots, dtype=object).ravel())

        if len(area_names) == 0:
            raise ValueError("No area labels available for bs='mrf'.")

        unseen = [v for v in stable_unique_levels(x) if v not in area_names]
        if len(unseen) > 0:
            raise ValueError(
                "Data contain regions that are not contained in the knot specification."
            )

        self._area_names = list(area_names)

        X_full = factor_indicator_matrix(x, self._area_names)

        polys = None
        if self.xt.get("polys", None) is not None:
            polys = combine_duplicate_polys(self.xt["polys"])
            self._plot_polys = polys

        if self.xt.get("penalty", None) is not None:
            S = coerce_penalty_matrix(self.xt["penalty"], self._area_names)
            if self.xt.get("nb", None) is not None:
                self._nb = coerce_nb(self.xt["nb"], self._area_names)
            elif polys is not None:
                self._nb = polys_to_nb(polys)
        else:
            if self.xt.get("nb", None) is not None:
                nb = coerce_nb(self.xt["nb"], self._area_names)
            else:
                if polys is None:
                    raise ValueError("No spatial information provided for bs='mrf'.")
                nb = polys_to_nb(polys)

            S, nb = laplacian_penalty_from_nb(nb, self._area_names)
            self._nb = nb

        self._full_penalty = np.asarray(S, dtype=np.float64)

        n_areas = len(self._area_names)
        bs_dim = n_areas if self.k < 0 else int(self.k)
        if bs_dim > n_areas:
            raise ValueError("MRF basis dimension set too high.")

        if bs_dim < n_areas:
            miss = np.where(np.sum(X_full, axis=0) == 0.0)[0]
            X_aug = X_full
            if miss.size > 0:
                X_aug = np.vstack(
                    [np.zeros((miss.size, n_areas), dtype=np.float64), X_aug]
                )
                for i, j in enumerate(miss):
                    X_aug[i, j] = 1.0

            rp = nat_param_type0(
                X_aug, self._full_penalty, rank=None, tol=None, unit_fnorm=True
            )

            ind = np.arange(n_areas - bs_dim, n_areas, dtype=int)
            X_red = rp["X"][miss.size :, :][:, ind]
            P_red = rp["P"][:, ind]

            D_red = np.zeros(bs_dim, dtype=np.float64)
            rank_full = int(rp["rank"])
            penalized = ind[ind < rank_full]
            if penalized.size > 0:
                D_red[np.where(ind < rank_full)[0]] = rp["D"][penalized]

            S_red = np.diag(D_red)

            # mgcv's scale.penalty=TRUE normalizes using the pre-absorption
            # reduced basis and penalty.
            maXX = float(np.max(np.sum(np.abs(X_red), axis=1)) ** 2)
            maS = float(np.max(np.sum(np.abs(S_red), axis=0)))
            if maS > 1e-12 and maXX > 1e-12:
                S_red = S_red / (maS / maXX)

            self._P = np.asarray(P_red, dtype=np.float64)
            basis_raw = np.asarray(X_red, dtype=np.float64)
            penalty_raw = np.asarray(S_red, dtype=np.float64)
        else:
            # Apply mgcv smoothCon scale_penalty step: normalise using the
            # full-rank X and S.
            maXX = float(np.max(np.sum(np.abs(X_full), axis=1)) ** 2)
            maS = float(np.max(np.sum(np.abs(self._full_penalty), axis=0)))
            S_full = np.asarray(self._full_penalty, dtype=np.float64)
            if maS > 1e-12 and maXX > 1e-12:
                S_full = S_full / (maS / maXX)

            self._P = None
            basis_raw = np.asarray(X_full, dtype=np.float64)
            penalty_raw = np.asarray(S_full, dtype=np.float64)

        if self._by_state.is_constant:
            # Match mgcv::smoothCon(absorb.cons=TRUE): absorb the single MRF
            # identifiability constraint in the term-local coefficient space.
            basis_fit, penalties_fit, transform = full_term_sum_to_zero_constraint(
                basis_raw, [penalty_raw]
            )
            self._record_constraint_result(
                "sum_to_zero", transform, absorbed_by="runtime"
            )
        else:
            basis_fit = basis_raw
            penalties_fit = [penalty_raw]
            self._record_constraint_result(None, None, absorbed_by=None)

        if self._by_state.is_present:
            basis_fit = apply_numeric_by(basis_fit, self._by_state.values)

        penalty_fit = np.asarray(penalties_fit[0], dtype=np.float64)
        ev_fit = np.linalg.eigvalsh(penalty_fit)
        tol_fit = np.finfo(float).eps ** EIG_TOL_POWER * max(np.max(ev_fit), 1.0)
        self._rank = int(np.sum(ev_fit > tol_fit))
        self._basis_train = np.asarray(basis_fit, dtype=np.float64)
        self._penalties = [penalty_fit]
        self._null_space_dim = int(self._basis_train.shape[1] - self._rank)
        return self

    @property
    def basis_train(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._basis_train

    @property
    def penalties(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return self._penalties

    @property
    def n_coef(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")
        return int(self._basis_train.shape[1])

    def get_penalty_definitions(self):
        if self._basis_train is None:
            raise RuntimeError("Term is not fitted.")

        if len(self.penalties) == 0:
            return []

        sp_vals = self._normalized_term_sp(1)
        sp_main = sp_vals[0] if sp_vals else None

        if sp_main is None:
            sp_mode = None
            sp_value = None
        elif sp_main >= 0:
            sp_mode = "fixed"
            sp_value = float(sp_main)
        else:
            sp_mode = "estimate"
            sp_value = None

        defs = [
            PenaltySpec(
                matrix=np.asarray(self.penalties[0], dtype=np.float64),
                smoothing_id=(
                    None if self.smoothing_id is None else str(self.smoothing_id)
                ),
                kind="smooth",
                rank=int(self._rank) if self._rank is not None else None,
                null_space_dim=(
                    int(self._null_space_dim)
                    if self._null_space_dim is not None
                    else None
                ),
                is_null_space_penalty=False,
                sp_mode=sp_mode,
                sp_value=sp_value,
                metadata={
                    "term_type": self.term_type,
                    "basis_name": self.basis_name,
                    "feature": list(self.feature),
                    "label": self.label,
                    "by": self.by,
                    "by_name": self._by_state.feature_name,
                    "area_names": (
                        list(self._area_names) if self._area_names is not None else None
                    ),
                    "has_polys": self._plot_polys is not None,
                    "has_nb": self._nb is not None,
                    "has_penalty": self._full_penalty is not None,
                    "low_rank": self._P is not None,
                    "k": int(self.k),
                },
            )
        ]

        if self.select:
            select_sid = (
                None if self.smoothing_id is None else f"{self.smoothing_id}::select"
            )
            sel_spec = build_null_space_selection_spec(
                main_penalty=np.asarray(self.penalties[0], dtype=np.float64),
                smoothing_id=select_sid,
                metadata={
                    "term_type": self.term_type,
                    "basis_name": self.basis_name,
                    "feature": list(self.feature),
                    "label": self.label,
                    "by": self.by,
                    "by_name": self._by_state.feature_name,
                    "area_names": (
                        list(self._area_names) if self._area_names is not None else None
                    ),
                    "has_polys": self._plot_polys is not None,
                    "has_nb": self._nb is not None,
                    "has_penalty": self._full_penalty is not None,
                    "low_rank": self._P is not None,
                    "k": int(self.k),
                    "is_selection_penalty": True,
                },
            )
            if sel_spec is not None:
                defs.append(sel_spec)

        return defs

    def transform_new(self, X_new):
        self._require_fitted()

        x = as_object_1d(column_as_object(X_new, self._feature_index))

        Xp = factor_indicator_matrix(x, self._area_names)
        if self._P is not None:
            Xp = Xp @ self._P
        Xp = self._apply_constraint_transform_and_by(Xp, X_new)
        return np.asarray(Xp, dtype=np.float64)
