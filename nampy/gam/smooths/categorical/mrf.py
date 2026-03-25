from __future__ import annotations

import numpy as np

from .categorical_utils import factor_indicator_matrix, stable_unique_levels, as_object_1d
from ...design.structures import PenaltySpec
from ...basis.tensor import rescale_tensor_penalties_for_fit
from ..base import (
    BaseSmoothTerm,
    _resolve_feature,
    by_values_from_new_data,
    column_as_object,
    resolve_by_state,
    sync_by_state_attributes,
)
from ....splines.mrf import (
    combine_duplicate_polys,
    polys_to_nb,
    coerce_nb,
    laplacian_penalty_from_nb,
    coerce_penalty_matrix,
    nat_param_type0,
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
        smoothing_id=None,
        by=None,
        sp=None,
        xt=None,
        knots=None,
        metadata=None,
    ):
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) != 1:
            raise NotImplementedError("bs='mrf' currently supports exactly one area-label variable.")

        super().__init__(
            feature=features,
            label=label or f"mrf({features[0]})",
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )

        self.k = int(k)
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
                X_aug = np.vstack([np.zeros((miss.size, n_areas), dtype=np.float64), X_aug])
                for i, j in enumerate(miss):
                    X_aug[i, j] = 1.0

            rp = nat_param_type0(X_aug, self._full_penalty, rank=None, tol=None, unit_fnorm=True)

            ind = np.arange(n_areas - bs_dim, n_areas, dtype=int)
            X_red = rp["X"][miss.size:, :][:, ind]
            P_red = rp["P"][:, ind]

            D_red = np.zeros(bs_dim, dtype=np.float64)
            rank_full = int(rp["rank"])
            penalized = ind[ind < rank_full]
            if penalized.size > 0:
                D_red[np.where(ind < rank_full)[0]] = rp["D"][penalized]

            S_red = np.diag(D_red)
            rank_red = int(np.sum(ind < rank_full))

            self._basis_train = np.asarray(X_red, dtype=np.float64)
            self._penalties = [S_red]
            self._P = np.asarray(P_red, dtype=np.float64)
            self._rank = rank_red
        else:
            ev = np.linalg.eigvalsh(self._full_penalty)
            tol = np.finfo(float).eps ** 0.8 * max(np.max(ev), 1.0)
            rank = int(np.sum(ev > tol))

            # Absorb the centering constraint, matching mgcv's smoothCon behaviour
            # when absorb.cons=TRUE: C = colMeans(X), remove one null-space column
            # via QR so the basis shrinks from n_areas to n_areas-1 columns.
            C = np.mean(X_full, axis=0, keepdims=True)  # (1, n_areas)
            Q, _ = np.linalg.qr(C.T, mode="complete")   # (n_areas, n_areas)
            Z = Q[:, 1:]                                  # (n_areas, n_areas-1)
            X_absorbed = X_full @ Z
            S_absorbed = Z.T @ self._full_penalty @ Z
            S_absorbed = 0.5 * (S_absorbed + S_absorbed.T)

            # Apply mgcv smoothCon scale_penalty step: normalise using the
            # pre-absorption X and S (R scales before absorb.cons, so the same
            # factor is inherited by the absorbed penalty).
            maXX = float(np.max(np.sum(np.abs(X_full), axis=1)) ** 2)
            maS = float(np.max(np.sum(np.abs(self._full_penalty), axis=0)))
            if maS > 1e-12 and maXX > 1e-12:
                S_absorbed = S_absorbed / (maS / maXX)

            self._basis_train = np.asarray(X_absorbed, dtype=np.float64)
            self._penalties = [np.asarray(S_absorbed, dtype=np.float64)]
            self._P = np.asarray(Z, dtype=np.float64)
            self._rank = rank

        if self._by_state.is_present:
            self._basis_train = self._basis_train * self._by_state.values[:, None]

        self._null_space_dim = int(self._basis_train.shape[1] - self._rank)
        self._record_constraint_result(None, None, absorbed_by=None)
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

        return [
            PenaltySpec(
                matrix=np.asarray(self.penalties[0], dtype=np.float64),
                smoothing_id=(None if self.smoothing_id is None else str(self.smoothing_id)),
                kind="smooth",
                rank=int(self._rank) if self._rank is not None else None,
                null_space_dim=int(self._null_space_dim) if self._null_space_dim is not None else None,
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
                    "area_names": list(self._area_names) if self._area_names is not None else None,
                    "has_polys": self._plot_polys is not None,
                    "has_nb": self._nb is not None,
                    "has_penalty": self._full_penalty is not None,
                    "low_rank": self._P is not None,
                    "k": int(self.k),
                },
            )
        ]

    def transform_new(self, X_new):
        if self._area_names is None:
            raise RuntimeError("Term is not fitted.")

        x = as_object_1d(column_as_object(X_new, self._feature_index))

        Xp = factor_indicator_matrix(x, self._area_names)
        if self._P is not None:
            Xp = Xp @ self._P

        z = by_values_from_new_data(X_new, self._by_state)
        if z is not None:
            out = np.zeros_like(Xp)
            ok = np.isfinite(z)
            out[ok, :] = Xp[ok, :] * z[ok][:, None]
            Xp = out

        return np.asarray(Xp, dtype=np.float64)
