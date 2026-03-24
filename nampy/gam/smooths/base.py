"""
Runtime smooth term base class and shared helpers (Stage 2 of the pipeline).

:class:`BaseSmoothTerm`
    Abstract base for all runtime smooth terms.  Each concrete subclass
    implements the fit/transform/predict interface and provides penalty
    specifications via :meth:`get_penalty_definitions`.

    Key properties (must be valid after :meth:`fit`):
        - ``basis_train``: design matrix built from training data.
        - ``penalties``: list of penalty matrices.
        - ``n_coef``: number of coefficient columns.

    Optional delegation (by-variable and constraint handling may be
    delegated to the Stage-3 wrapper in constructors.py):
        - ``constraint_mode``: ``"auto"``, ``"always"``, ``"never"``, ``"factor_by"``.
        - ``delegate_constraint``: if True, the term returns the raw unconstrained
          basis and lets the wrapper apply sum-to-zero or factor-by absorption.

Supporting helpers:
    :func:`_resolve_feature`, :class:`ByState`, :class:`FeatureMatrixState`
    provide feature-index resolution and by-variable state.
"""

import abc
from dataclasses import dataclass

import numpy as np

from ..penalties import (
    build_null_space_selection_spec,
    make_penalty_spec,
    normalize_penalty_spec,
)


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


@dataclass
class ByState:
    feature_index: int | None
    feature_name: str | None
    values: np.ndarray | None
    is_present: bool
    is_constant: bool

    @property
    def index(self):
        return self.feature_index

    @property
    def name(self):
        return self.feature_name


ByVariableState = ByState


@dataclass
class FeatureMatrixState:
    indices: list[int]
    names: list[str]
    matrix: np.ndarray


def resolve_by_state(by, X, feature_names):
    if by is None:
        return ByState(None, None, None, False, True)
    idx, name, z = _resolve_numeric_by(by, X, feature_names)
    return ByState(idx, name, z, True, _is_effectively_constant(z))


def resolve_feature_matrix_state(features, X, feature_names):
    feats = list(features) if not isinstance(features, (str, int)) else [features]
    indices, names = [], []
    for feat in feats:
        idx, fname = _resolve_feature(feat, feature_names)
        indices.append(idx)
        names.append(fname)
    X_arr = np.asarray(X)
    Xf = np.column_stack([np.asarray(X_arr[:, idx], dtype=np.float64) for idx in indices])
    return FeatureMatrixState(indices=indices, names=names, matrix=Xf)


def sync_by_state_attributes(term, by_state: ByVariableState):
    term._by_state = by_state
    term._by_index = by_state.feature_index
    term._by_name = by_state.feature_name
    term._by_values = by_state.values
    term._by_is_constant = bool(by_state.is_constant)


def by_values_from_new_data(X_new, by_state: ByVariableState):
    if by_state is None or not by_state.is_present:
        return None
    return column_as_float(X_new, by_state.feature_index)


def _column_from_matrix(X, index):
    idx = int(index)
    if hasattr(X, "iloc"):
        return X.iloc[:, idx].to_numpy()
    return X[:, idx]


def column_as_float(X, index):
    return np.asarray(_column_from_matrix(X, index), dtype=np.float64).ravel()


def column_as_object(X, index):
    return np.asarray(_column_from_matrix(X, index), dtype=object).ravel()


def columns_as_float_matrix(X, indices):
    return np.column_stack([column_as_float(X, idx) for idx in list(indices)])


def apply_numeric_by(B, z, allow_missing=False):
    B = np.asarray(B, dtype=np.float64)
    if z is None:
        return B
    z = np.asarray(z, dtype=np.float64).ravel()
    if B.shape[0] != z.shape[0]:
        raise ValueError(
            f"Numeric by vector length {z.shape[0]} does not match basis rows {B.shape[0]}."
        )
    if not allow_missing:
        return B * z[:, None]
    out = np.zeros_like(B)
    ok = np.isfinite(z)
    out[ok, :] = B[ok, :] * z[ok][:, None]
    return out


def _sp_mode_value(sp_j):
    if sp_j is None:
        return None, None
    if sp_j >= 0:
        return "fixed", float(sp_j)
    return "estimate", None


def term_penalty_metadata(term, extra=None, *, is_selection_penalty=False):
    meta = {
        "term_type": term.term_type,
        "basis_name": term.basis_name,
        "feature": term.feature,
        "label": term.label,
        "by": term.by,
        "by_name": getattr(term, "_by_name", None),
        "by_is_constant": bool(getattr(term, "_by_is_constant", True)),
        "is_selection_penalty": bool(is_selection_penalty),
    }
    if extra:
        meta.update(dict(extra))
    return meta


def build_penalty_definition(term, matrix, *, kind="smooth", smoothing_id=None, sp_value_in=None, rank=None, null_space_dim=None, is_null_space_penalty=False, metadata_extra=None):
    sp_mode, sp_value = _sp_mode_value(sp_value_in)
    return make_penalty_spec(
        matrix=np.asarray(matrix, dtype=np.float64),
        smoothing_id=smoothing_id,
        kind=kind,
        sp_mode=sp_mode,
        sp_value=sp_value,
        is_null_space_penalty=bool(is_null_space_penalty),
        metadata=term_penalty_metadata(
            term, extra=metadata_extra, is_selection_penalty=False
        ),
    )


def build_selection_penalty_definition(term, matrix, *, rank, null_space_dim, smoothing_id, metadata_extra=None):
    return make_penalty_spec(
        matrix=np.asarray(matrix, dtype=np.float64),
        smoothing_id=smoothing_id,
        kind="null_space",
        is_null_space_penalty=True,
        sp_mode=None,
        sp_value=None,
        metadata=term_penalty_metadata(
            term, extra=metadata_extra, is_selection_penalty=True
        ),
    )


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
        raise ValueError(f"knots must have length {len(features)} for features={features}, got {len(knots)}.")
    if len(features) == 1:
        return [knots]
    raise TypeError("knots must be None, dict, or list/tuple aligned with features.")


def _is_effectively_constant(z, tol=1e-12):
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return True
    return bool(np.max(np.abs(z - z[0])) <= tol)


def _normalize_mc(mc, n_marginals):
    if mc is None:
        return [True] * n_marginals
    if np.isscalar(mc):
        return [bool(mc)] * n_marginals
    mc_list = [bool(v) for v in mc]
    if len(mc_list) != n_marginals:
        raise ValueError(f"mc must have length {n_marginals}, got {mc_list}.")
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
        raise NotImplementedError("1D point constraint dict incompatible.")
    if isinstance(pc, (list, tuple, np.ndarray)):
        vals = np.asarray(pc, dtype=np.float64).ravel()
        if vals.size != 1:
            raise NotImplementedError("Only 1D point constraints supported.")
        return float(vals[0])
    raise NotImplementedError(f"Unsupported pc type {type(pc)}.")


RUNTIME_TERM_INTERFACE_CHECKLIST = (
    "basis_train",
    "transform_new",
    "get_penalty_definitions",
    "label",
    "basis_name",
    "term_type",
    "feature",
    "constraints_absorbed",
    "fit_constraint_matrix",
    "predict_constraint_matrix",
    "prediction_offset",
    "metadata",
)


class BaseSmoothTerm(abc.ABC):
    """
    Abstract base class for all GAM runtime smooth terms.

    A runtime term is the canonical owner of basis semantics for its smooth
    family (architecture section 3.1).  Subclasses must implement:

        fit(X, feature_names)  ->  self
            Fit the basis and penalties to training data.  Sets ``_basis_train``
            and ``_penalties`` on ``self``.

        transform_new(X_new)   ->  np.ndarray, shape (n_new, n_coef)
            Evaluate the basis at new observations using the same coefficient
            parameterisation as ``basis_train``.

    After fitting, the following properties must be available:

        basis_train              final training design block
        get_penalty_definitions  list of PenaltySpec via penalty subsystem
        label, basis_name, term_type, feature
        constraints_absorbed     True if this term already absorbed its constraints
        fit_constraint_matrix    explicit constraint matrix, or None
        predict_constraint_matrix
        prediction_offset

    Contract
    --------
    - ``basis_train.shape[1] == S.shape[0] == S.shape[1]`` for every penalty S.
    - If a coefficient transform T was applied during fit, penalties satisfy
      ``S_fitted = T.T @ S_raw @ T``.
    - By-variable handling is done either by the runtime term OR delegated to
      the stage-3 construction wrapper via ``by_done = False`` — never both.
    - Constraint absorption is done either by the runtime term OR delegated to
      the stage-3 wrapper via ``constraints_absorbed = False`` — never both.
    """

    term_type = "smooth"
    basis_name = "unknown"
    supports_tensor_marginal = False

    def __init__(self, feature, label=None, smoothing_id=None, by=None, sp=None, metadata=None):
        from uuid import uuid4
        self.feature = feature
        self.label = label or str(feature)
        self.smoothing_id = smoothing_id
        self.term_id = uuid4().hex[:12]
        self.by = by
        self.sp = sp
        self.metadata = dict(metadata or {})
        self.resolved_feature_names = None
        self.by_done = True
        self.constraints_absorbed = True
        self.n_constraints_absorbed = 0
        self.constraint_kind = None
        self.constraint_transform = None
        self.constraints_absorbed_by = None
        self.fit_constraint_matrix = None
        self.predict_constraint_matrix = None
        self.prediction_offset = None
        self.basis_train_base = None
        self.knots = None

    def _set_resolved_features(self, resolved_feature_names):
        if resolved_feature_names is None:
            self.resolved_feature_names = None
        else:
            self.resolved_feature_names = list(resolved_feature_names)

    def _record_constraint_result(self, kind, transform, *, absorbed_by):
        self.constraint_kind = kind
        self.constraint_transform = transform
        self.fit_constraint_matrix = transform
        self.predict_constraint_matrix = transform
        self.constraints_absorbed_by = absorbed_by
        self.constraints_absorbed = bool(transform is not None)
        if transform is None:
            self.n_constraints_absorbed = 0
        else:
            self.n_constraints_absorbed = int(max(0, transform.shape[0] - transform.shape[1]))

    def _apply_constraint_transform_and_by(self, B, X_new):
        """
        Apply constraint transform and by-variable scaling to a raw prediction basis.

        Factor-by order: scale by z first (to match the factor-by fit ordering where
        the by-variable is applied before the QR constraint), then apply T.
        Standard order: apply T first, then scale by z.
        """
        T = getattr(self, "_constraint_transform", None)
        if getattr(self, "_constraint_kind", None) == "factor_by":
            z = by_values_from_new_data(X_new, self._by_state)
            if z is not None:
                B = B * z[:, None]
            if T is not None:
                B = B @ T
            return B
        if T is not None:
            B = B @ T
        z = by_values_from_new_data(X_new, self._by_state)
        if z is not None:
            B = B * z[:, None]
        return B

    @abc.abstractmethod
    def fit(self, X, feature_names):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_new(self, X_new):
        raise NotImplementedError

    def _require_fitted(self):
        if getattr(self, "_basis_train", None) is None:
            raise RuntimeError("Term is not fitted.")

    @property
    def basis_train(self):
        self._require_fitted()
        return self._basis_train

    @property
    def penalties(self):
        self._require_fitted()
        return self._penalties

    @property
    def n_coef(self):
        self._require_fitted()
        return int(self._basis_train.shape[1])

    def transform_new_base(self, X_new):
        return self.transform_new(X_new)

    def _normalized_term_sp(self, n_penalties):
        if n_penalties <= 0:
            return []
        if self.sp is None:
            return [None] * n_penalties
        if np.isscalar(self.sp):
            if n_penalties != 1:
                raise NotImplementedError("Multi-penalty term-level sp must supply one value per penalty.")
            vals = [float(self.sp)]
        else:
            vals = np.asarray(self.sp, dtype=np.float64).ravel()
            if vals.size != n_penalties:
                raise ValueError(f"term-level sp must have length {n_penalties}, got {vals.size}.")
            vals = [float(v) for v in vals]
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
                sp_mode, sp_value = None, None
            elif sp_j >= 0:
                sp_mode, sp_value = "fixed", float(sp_j)
            else:
                sp_mode, sp_value = "estimate", None
            defs.append(
                make_penalty_spec(
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
