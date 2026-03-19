import abc
import numpy as np

from ..design.objects import PenaltyDefinition


def _resolve_feature(feature, feature_names):
    if isinstance(feature, int):
        idx = int(feature)
        if idx < 0 or idx >= len(feature_names):
            raise IndexError(
                f"Feature index {idx} out of range for {len(feature_names)} features."
            )
        return idx, feature_names[idx]

    feature_name = str(feature)
    if feature_name not in feature_names:
        raise KeyError(
            f"Feature {feature_name!r} not found in feature_names={feature_names}."
        )
    return feature_names.index(feature_name), feature_name


def _resolve_numeric_by(by, X, feature_names):
    if by is None:
        return None, None, None

    idx, name = _resolve_feature(by, feature_names)
    z = np.asarray(X[:, idx], dtype=np.float64).ravel()
    if z.ndim != 1:
        raise ValueError("Numeric `by` variables must resolve to a 1D column.")
    return idx, name, z


def _normalize_knots(knots, features):
    if knots is None:
        return [None] * len(features)

    if isinstance(knots, dict):
        return [knots.get(str(f), None) for f in features]

    if isinstance(knots, (list, tuple)):
        if len(knots) == len(features):
            return list(knots)
        if len(features) == 1:
            return [knots]
        raise ValueError(
            f"knots must have length {len(features)} for features={features}, got {len(knots)}."
        )

    if len(features) == 1:
        return [knots]

    raise TypeError(
        "knots must be None, a dict keyed by feature name, or a list/tuple aligned with features."
    )


def _apply_sum_to_zero_constraint(B, penalties):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]

    if B.ndim != 2:
        raise ValueError("Basis matrix must be 2D.")

    if B.shape[1] == 0:
        C = np.eye(0, dtype=np.float64)
        return B, penalties, C

    constraint = B.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(constraint, mode="complete")
    C = q[:, 1:]

    Bc = B @ C
    Sc = [0.5 * (C.T @ S @ C + (C.T @ S @ C).T) for S in penalties]
    return Bc, Sc, C


def _is_effectively_constant(z, tol=1e-12):
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return True
    return bool(np.max(np.abs(z - z[0])) <= tol)


def _full_term_sum_to_zero_constraint(B, penalties):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]

    if B.ndim != 2:
        raise ValueError("Basis matrix must be 2D.")
    if len(penalties) == 0:
        raise ValueError("At least one penalty matrix is required.")

    constraint = B.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(constraint, mode="complete")
    C = q[:, 1:]

    Bc = B @ C
    Sc = [C.T @ S @ C for S in penalties]
    Sc = [0.5 * (S + S.T) for S in Sc]
    return Bc, Sc, C


def _normalize_mc(mc, n_marginals):
    if mc is None:
        return [True] * n_marginals

    if np.isscalar(mc):
        return [bool(mc)] * n_marginals

    mc_list = [bool(v) for v in mc]
    if len(mc_list) != n_marginals:
        raise ValueError(
            f"mc must have length {n_marginals} for a ti term with "
            f"{n_marginals} marginals, got {mc_list}."
        )
    return mc_list


def _normalize_point_constraint(pc, feature_name):
    if pc is None:
        return None

    if np.isscalar(pc):
        return float(pc)

    if isinstance(pc, dict):
        if feature_name in pc:
            return float(pc[feature_name])
        if len(pc) == 1:
            return float(next(iter(pc.values())))
        raise NotImplementedError(
            f"1D point constraint dict must contain key {feature_name!r} "
            f"or have length 1, got keys={list(pc.keys())!r}."
        )

    if isinstance(pc, (list, tuple, np.ndarray)):
        vals = np.asarray(pc, dtype=np.float64).ravel()
        if vals.size != 1:
            raise NotImplementedError(
                "Current runtime supports only 1D point constraints for s(..., bs='cr')."
            )
        return float(vals[0])

    raise NotImplementedError(
        f"Unsupported pc specification type {type(pc)} for 1D s(..., bs='cr')."
    )


def _apply_linear_constraint(B, penalties, constraint_row, tol=1e-12):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    c = np.asarray(constraint_row, dtype=np.float64).reshape(-1, 1)

    if B.ndim != 2:
        raise ValueError("Basis matrix must be 2D.")
    if c.shape[0] != B.shape[1]:
        raise ValueError(
            f"Constraint row has length {c.shape[0]}, but basis width is {B.shape[1]}."
        )

    cn = float(np.linalg.norm(c))
    if cn <= tol:
        C = np.eye(B.shape[1], dtype=np.float64)
        return B, penalties, C

    q, _ = np.linalg.qr(c, mode="complete")
    C = q[:, 1:]

    Bc = B @ C
    Sc = []
    for S in penalties:
        St = C.T @ S @ C
        Sc.append(0.5 * (St + St.T))
    return Bc, Sc, C


class BaseSmoothTerm(abc.ABC):
    term_type = "smooth"
    basis_name = "unknown"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        label=None,
        smoothing_id=None,
        by=None,
        sp=None,
        metadata=None,
    ):
        self.feature = feature
        self.label = label or str(feature)
        self.smoothing_id = smoothing_id
        self.by = by
        self.sp = sp
        self.metadata = dict(metadata or {})

        self.by_done = True
        self.constraints_absorbed = True
        self.fit_constraint_matrix = None
        self.predict_constraint_matrix = None
        self.prediction_offset = None

        self.basis_train_base = None
        self.knots = None

    @abc.abstractmethod
    def fit(self, X, feature_names):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def basis_train(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def penalties(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_coef(self):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_new(self, X_new):
        raise NotImplementedError

    def transform_new_base(self, X_new):
        return self.transform_new(X_new)

    def _normalized_term_sp(self, n_penalties):
        if n_penalties <= 0:
            return []

        if self.sp is None:
            return [None] * n_penalties

        if np.isscalar(self.sp):
            if n_penalties != 1:
                raise NotImplementedError(
                    "For multi-penalty smooths, term-level sp must provide one value "
                    "per penalty."
                )
            vals = [float(self.sp)]
        else:
            vals = np.asarray(self.sp, dtype=np.float64).ravel()
            if vals.size != n_penalties:
                raise ValueError(
                    f"term-level sp must have length {n_penalties}, got {vals.size}."
                )
            vals = [float(v) for v in vals]

        for v in vals:
            if not np.isfinite(v):
                raise ValueError("term-level sp values must be finite.")

        return vals

    def get_penalty_definitions(self):
        raw = list(self.penalties)
        n_raw = len(raw)
        sp_vals = self._normalized_term_sp(n_raw)

        defs = []
        for j, P in enumerate(raw):
            if self.smoothing_id is None:
                sid = None
            elif n_raw <= 1:
                sid = str(self.smoothing_id)
            else:
                sid = f"{self.smoothing_id}::{j}"

            sp_j = sp_vals[j] if j < len(sp_vals) else None
            if sp_j is None:
                sp_mode = None
                sp_value = None
            elif sp_j >= 0:
                sp_mode = "fixed"
                sp_value = float(sp_j)
            else:
                sp_mode = "estimate"
                sp_value = None

            defs.append(
                PenaltyDefinition(
                    matrix=np.asarray(P, dtype=np.float64),
                    smoothing_id=sid,
                    kind="smooth",
                    sp_mode=sp_mode,
                    sp_value=sp_value,
                    metadata={
                        "term_type": self.term_type,
                        "basis_name": self.basis_name,
                        "feature": self.feature,
                        "label": self.label,
                        "by": self.by,
                        "term_sp": sp_j,
                    },
                )
            )
        return defs
