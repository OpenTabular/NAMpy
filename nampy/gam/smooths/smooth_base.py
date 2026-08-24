"""
Runtime smooth term base class and shared helpers.

:class:`BaseSmoothTerm`
    Abstract base for all runtime smooth terms.  Each concrete subclass
    implements the fit/transform/predict interface and provides penalty
    specifications via :meth:`get_penalty_definitions`.

    Key properties (must be valid after :meth:`fit`):
        - ``basis_train``: design matrix built from training data.
        - ``penalties``: list of penalty matrices.
        - ``n_coef``: number of coefficient columns.

    Optional delegation (by-variable and constraint handling may be
    delegated to the construction wrapper in ``smooths/construct.py``):
        - ``constraint_mode``: ``"auto"``, ``"always"``, ``"never"``, ``"factor_by"``.
        - ``delegate_constraint``: if True, the term returns the raw unconstrained
          basis and lets the wrapper apply sum-to-zero or factor-by absorption.

Supporting helpers:
    :func:`_resolve_feature` and :class:`ByState`
    provide feature-index resolution and by-variable state.
"""

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constraints.absorption import apply_linear_constraint
from ..penalties import (
    PenaltySpec,
    build_null_space_selection_spec,
    make_penalty_spec,
    normalize_penalty_spec,
    null_space_penalty_from_penalty,
    penalty_id_for_local_index,
    selection_penalty_id,
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
    z = column_as_numeric_array(X, idx)
    return idx, name, z


@dataclass
class ByState:
    feature_index: int | None
    feature_name: str | None
    values: np.ndarray | None
    is_present: bool
    is_constant: bool
    handling: str = "none"
    consumed_by_term: bool = False

    @property
    def index(self):
        return self.feature_index

    @property
    def name(self):
        return self.feature_name


def resolve_by_state(by, X, feature_names):
    if by is None:
        return ByState(None, None, None, False, True, handling="none")
    idx, name, z = _resolve_numeric_by(by, X, feature_names)
    handling = "matrix" if np.asarray(z).ndim == 2 else "numeric"
    return ByState(
        idx,
        name,
        z,
        True,
        _is_effectively_constant(z),
        handling=handling,
    )


def sync_by_state_attributes(term, by_state: ByState):
    term._by_state = by_state


def by_values_from_new_data(X_new, by_state: ByState):
    if by_state is None or not by_state.is_present:
        return None
    return column_as_numeric_array(X_new, by_state.feature_index)


def linear_functional_basis(values, weights, basis_evaluator):
    """Aggregate a pointwise basis into one linear-functional row per sample.

    ``values`` and ``weights`` must be equal-width matrices.  The evaluator is
    called once on the flattened locations and must return one basis row per
    location.  This is the common ``rowSums(L * F(X))`` construction used by
    mgcv and SCAM matrix-valued ``by`` terms.
    """
    locations = np.asarray(values, dtype=np.float64)
    quadrature = np.asarray(weights, dtype=np.float64)
    if locations.ndim != 2 or quadrature.shape != locations.shape:
        raise ValueError(
            "Linear-functional terms require equal-shaped 2D feature and by matrices."
        )
    point_basis = np.asarray(
        basis_evaluator(locations.reshape(-1)), dtype=np.float64
    )
    if point_basis.ndim != 2 or point_basis.shape[0] != locations.size:
        raise ValueError(
            "Linear-functional basis evaluator returned an incompatible row layout."
        )
    reshaped = point_basis.reshape(
        locations.shape[0], locations.shape[1], point_basis.shape[1]
    )
    return np.einsum("nm,nmk->nk", quadrature, reshaped)


def linear_functional_by_state(by_state: ByState) -> ByState:
    """Mark matrix-valued by weights as consumed by basis aggregation."""
    if by_state is None or not by_state.is_present:
        raise ValueError("Linear-functional terms require a matrix-valued by variable.")
    if np.asarray(by_state.values).ndim != 2:
        raise ValueError("Linear-functional by weights must be two-dimensional.")
    return ByState(
        by_state.feature_index,
        by_state.feature_name,
        None,
        True,
        False,
        handling="linear_functional",
        consumed_by_term=True,
    )


def _column_from_matrix(X, index):
    idx = int(index)
    if hasattr(X, "iloc"):
        return X.iloc[:, idx].to_numpy()
    return X[:, idx]


def column_as_float(X, index):
    return np.asarray(_column_from_matrix(X, index), dtype=np.float64).ravel()


def column_as_numeric_array(X, index):
    """Resolve scalar or consistently shaped array-valued numeric columns.

    Array-valued columns are the dataframe representation used for mgcv/SCAM
    linear-functional terms: one vector of quadrature locations or weights per
    observation.
    """
    column = _column_from_matrix(X, index)
    raw = np.asarray(column, dtype=object).reshape(-1)
    has_array_values = any(np.asarray(value).ndim > 0 for value in raw)
    if not has_array_values:
        return np.asarray(column, dtype=np.float64).reshape(-1)
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in raw]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(
            "Array-valued feature rows must all have the same number of entries."
        )
    return np.stack(rows, axis=0)


def column_as_object(X, index):
    return np.asarray(_column_from_matrix(X, index), dtype=object).ravel()


def columns_as_float_matrix(X, indices):
    return np.column_stack([column_as_float(X, idx) for idx in list(indices)])


def apply_numeric_by(B, z, allow_missing=False):
    B = np.asarray(B, dtype=np.float64)
    if z is None:
        return B
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("Numeric by vector must be one-dimensional.")
    z = z.ravel()
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
    _by_state = getattr(term, "_by_state", None)
    meta = {
        "term_type": term.term_type,
        "basis_name": term.basis_name,
        "feature": term.feature,
        "label": term.label,
        "by": term.by,
        "by_name": _by_state.feature_name if _by_state is not None else None,
        "by_is_constant": (
            bool(_by_state.is_constant) if _by_state is not None else True
        ),
        "is_selection_penalty": bool(is_selection_penalty),
    }
    if extra:
        meta.update(dict(extra))
    return meta


def build_penalty_definition(
    term,
    matrix,
    *,
    kind="smooth",
    smoothing_id=None,
    sp_value_in=None,
    rank=None,
    null_space_dim=None,
    is_null_space_penalty=False,
    metadata_extra=None,
    local_penalty_index=0,
):
    sp_mode, sp_value = _sp_mode_value(sp_value_in)
    return make_penalty_spec(
        matrix=np.asarray(matrix, dtype=np.float64),
        smoothing_id=smoothing_id,
        kind=kind,
        sp_mode=sp_mode,
        sp_value=sp_value,
        is_null_space_penalty=bool(is_null_space_penalty),
        metadata=term._penalty_metadata_with_scale(
            term_penalty_metadata(
                term, extra=metadata_extra, is_selection_penalty=False
            ),
            penalty_index=int(local_penalty_index),
        ),
    )


def _normalize_knots(knots, features):
    if knots is None:
        return [None] * len(features)
    if isinstance(knots, dict):
        out = []
        for f in features:
            if isinstance(f, (list, tuple)):
                vals = [knots.get(str(ff), None) for ff in f]
                out.append(None if all(v is None for v in vals) else vals)
            else:
                out.append(knots.get(str(f), None))
        return out
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
        # mgcv accepts pc containers with at least one value for 1D smooths.
        return float(next(iter(pc.values())))
    if isinstance(pc, (list, tuple, np.ndarray)):
        vals = np.asarray(pc, dtype=np.float64).ravel()
        if vals.size == 0:
            raise ValueError("point-constraint sequence cannot be empty.")
        return float(vals[0])
    raise NotImplementedError(f"Unsupported pc type {type(pc)}.")


def _normalize_point_constraint_vector(pc, feature_names):
    if pc is None:
        return None
    names = [str(name) for name in feature_names]
    n = len(names)
    if n <= 0:
        raise ValueError("feature_names must be non-empty.")
    if n == 1:
        return np.asarray([_normalize_point_constraint(pc, names[0])], dtype=np.float64)
    if isinstance(pc, dict):
        missing = [name for name in names if name not in pc]
        if missing:
            raise ValueError(
                "supply a value for each variable for a point constraint"
            )
        return np.asarray([float(pc[name]) for name in names], dtype=np.float64)
    vals = np.asarray(pc, dtype=np.float64).ravel()
    if vals.size < n:
        raise ValueError("supply a value for each variable for a point constraint")
    if vals.size > n:
        vals = vals[:n]
    return vals.astype(np.float64, copy=False)


def _coerce_pc_point_basis(pred_fn, point):
    B_pc = pred_fn(point)
    B_pc = np.asarray(B_pc, dtype=np.float64)
    if B_pc.ndim == 1:
        B_pc = B_pc.reshape(1, -1)
    if B_pc.ndim != 2 or B_pc.shape[0] != 1:
        raise ValueError(
            f"Point-constraint predictor must return a single basis row, got shape {B_pc.shape}."
        )
    return B_pc[0]


RUNTIME_TERM_INTERFACE_CHECKLIST = (
    "basis_train",
    "transform_new",
    "get_penalty_definitions",
    "label",
    "basis_name",
    "term_type",
    "feature",
    "transform_applied",
    "skip_centering",
    "constraint_transform",
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
        transform_applied        True if fit applied an explicit constraint transform
        skip_centering          True if identifiability processing must skip external centering
        constraint_transform     coefficient transform T from raw → constrained space, or None
        prediction_offset

    Contract
    --------
    - ``basis_train.shape[1] == S.shape[0] == S.shape[1]`` for every penalty S.
    - If a coefficient transform T was applied during fit, penalties satisfy
      ``S_fitted = T.T @ S_raw @ T``.
    - Explicit constraint transforms are done either by the runtime term OR
      delegated to the construction wrapper via ``transform_applied = False``.
    - ``skip_centering`` is separate: runtimes may request no predictor-level centering
      even when no explicit coefficient transform was applied.
    """

    term_type = "smooth"
    basis_name = "unknown"
    supports_tensor_marginal = False
    repara = True
    select: bool

    def __init__(
        self,
        feature,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        metadata=None,
    ):
        self.feature = feature
        self.label = label or str(feature)
        self.smoothing_id = smoothing_id
        self.term_id = None if term_id is None else str(term_id)
        self.by = by
        self.sp = sp
        self.metadata = dict(metadata or {})
        self.shared_basis_setup = self.metadata.get("shared_basis_setup", None)
        self.resolved_feature_names = None
        self.transform_applied = False
        self.skip_centering = False
        self.n_constraints_absorbed = 0
        self.constraint_kind = None
        self.constraint_transform = None
        self.constraints_absorbed_by = None
        self.prediction_offset = None
        self.basis_train_base = None
        self.knots = None
        self._penalty_rescale_factors = None

    def _set_resolved_features(self, resolved_feature_names):
        if resolved_feature_names is None:
            self.resolved_feature_names = None
        else:
            self.resolved_feature_names = list(resolved_feature_names)

    def _record_constraint_result(self, kind, transform, *, absorbed_by):
        self.constraint_kind = kind
        self.constraint_transform = transform
        self.constraints_absorbed_by = absorbed_by
        # ``transform_applied`` means fit changed coefficient coordinates.
        # ``skip_centering`` means identifiability handling must not add external centering,
        # even when no explicit transform exists.
        self.transform_applied = bool(transform is not None)
        self.skip_centering = self.transform_applied or (absorbed_by is not None)
        if transform is None:
            self.n_constraints_absorbed = 0
        else:
            self.n_constraints_absorbed = int(
                max(0, transform.shape[0] - transform.shape[1])
            )

    def _set_penalty_rescale_factors(self, factors):
        if factors is None:
            self._penalty_rescale_factors = None
            return
        vals = np.asarray(factors, dtype=np.float64).ravel()
        if vals.size == 0:
            self._penalty_rescale_factors = []
            return
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
            raise ValueError(
                "mgcv penalty rescale factors must be finite and positive."
            )
        self._penalty_rescale_factors = [float(v) for v in vals]

    def _penalty_rescale_factor(self, penalty_index: int) -> float:
        factors = self._penalty_rescale_factors
        if not factors:
            return 1.0
        if penalty_index < 0 or penalty_index >= len(factors):
            raise IndexError(
                f"Penalty index {penalty_index} out of range for {len(factors)} mgcv "
                "rescale factors."
            )
        return float(factors[penalty_index])

    def _penalty_metadata_with_scale(self, metadata, *, penalty_index: int):
        meta = dict(metadata or {})
        meta.setdefault("penalty_rescale_factor", self._penalty_rescale_factor(penalty_index))
        return meta

    def _apply_constraint_transform_and_by(self, B, X_new):
        """
        Apply constraint transform and by-variable scaling to a raw prediction basis.

        For factor-by smooths the constraint (sum-to-zero over all observations)
        is applied to the raw basis first; the level indicator scales the result.
        This mirrors the fit path: center(raw) * indicator.
        All other smooths: apply T first, then scale by z.
        """
        T = self.constraint_transform
        if T is not None:
            B = B @ T
        z = by_values_from_new_data(X_new, self._by_state)
        return self._apply_by_scale(B, z)

    def _set_by_state(self, X, feature_names):
        """
        Resolve and cache the numeric by-state for fit-time use.

        This is the standard location for by-state lifecycle setup.
        """
        self._by_state = resolve_by_state(self.by, X, feature_names)
        sync_by_state_attributes(self, self._by_state)
        return self._by_state

    def _apply_by_scale(self, B, by_values, *, allow_missing=False):
        """
        Apply numeric by scaling with consistent shape/dtype validation.

        Parameters
        ----------
        B : array-like, shape (n, d)
            Design matrix to be scaled.
        by_values : array-like or None
            Numeric by vector (n,) or None.
        allow_missing : bool
            If True, missing entries in by_values zero out rows instead of
            propagating NaNs.
        """
        return apply_numeric_by(B, by_values, allow_missing=allow_missing)

    def _apply_cached_by(self, B, *, allow_missing=False):
        """
        Apply cached fit-time by scaling if a by-state is present.
        """
        if self._by_state is None or not self._by_state.is_present:
            return np.asarray(B, dtype=np.float64)
        return self._apply_by_scale(
            B, self._by_state.values, allow_missing=allow_missing
        )

    def _linked_id_setup(self):
        setup = getattr(self, "shared_basis_setup", None)
        if not isinstance(setup, dict):
            return None
        if str(setup.get("mode", "")).lower() != "linked_id":
            return None
        return setup

    def _linked_id_pooled_columns(self):
        setup = self._linked_id_setup()
        if setup is None:
            return None
        cols = setup.get("pooled_feature_values", None)
        if cols is None:
            return None
        out = [np.asarray(col, dtype=object).ravel() for col in list(cols)]
        if len(out) == 0:
            return None
        n = out[0].shape[0]
        if any(col.shape[0] != n for col in out):
            raise ValueError("linked `id` pooled feature columns must share length.")
        return out

    def _linked_id_setup_matrix(self, feature_names):
        cols = self._linked_id_pooled_columns()
        setup = self._linked_id_setup()
        if cols is None or setup is None:
            return None
        pooled_feature_names = list(setup.get("pooled_feature_names", []))
        if len(pooled_feature_names) != len(cols):
            raise ValueError(
                "linked `id` pooled feature names must align with pooled columns."
            )
        X_shared = np.empty((cols[0].shape[0], len(feature_names)), dtype=object)
        X_shared[:] = 0.0
        for name, col in zip(pooled_feature_names, cols, strict=True):
            if str(name) not in feature_names:
                raise KeyError(
                    f"Feature {name!r} from linked `id` pooled setup not found in "
                    f"feature_names={feature_names}."
                )
            X_shared[:, feature_names.index(str(name))] = col
        return X_shared

    def _linked_id_marginal_setups(self, feature_groups=None):
        cols = self._linked_id_pooled_columns()
        setup = self._linked_id_setup()
        if cols is None or setup is None:
            return None
        pooled_names = [str(name) for name in setup.get("pooled_feature_names", [])]
        if feature_groups is None:
            groups = [(name,) for name in pooled_names]
        else:
            groups = [
                tuple(group) if isinstance(group, (list, tuple)) else (group,)
                for group in feature_groups
            ]
        column_by_name = {
            name: np.asarray(column, dtype=object).copy()
            for name, column in zip(pooled_names, cols, strict=True)
        }
        out = []
        for group in groups:
            names = [str(name) for name in group]
            missing = [name for name in names if name not in column_by_name]
            if missing:
                raise KeyError(
                    f"Linked tensor marginal features {missing!r} are absent from "
                    "the pooled basis setup."
                )
            out.append(
                {
                    "mode": "linked_id",
                    "id": str(setup.get("id")),
                    "pooled_feature_names": names,
                    "pooled_feature_values": [column_by_name[name] for name in names],
                    "n_linked_terms": int(setup.get("n_linked_terms", 0)),
                    "linked_term_labels": list(setup.get("linked_term_labels", [])),
                }
            )
        return out

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

    def resolved_feature_indices(self):
        if hasattr(self, "_feature_index") and self._feature_index is not None:
            return [int(self._feature_index)]
        if hasattr(self, "_feature_indices") and self._feature_indices is not None:
            return [int(v) for v in self._feature_indices]
        raise AttributeError("Runtime term does not expose resolved feature indices.")

    def resolved_feature_names_list(self):
        if hasattr(self, "_feature_name") and self._feature_name is not None:
            return [str(self._feature_name)]
        if hasattr(self, "_feature_names") and self._feature_names is not None:
            return [str(v) for v in self._feature_names]
        if self.resolved_feature_names is not None:
            return [str(v) for v in self.resolved_feature_names]
        raise AttributeError("Runtime term does not expose resolved feature names.")

    def tensor_marginal_fit_matrices(
        self, *, centered=False, apply_np=False, x_train=None
    ):
        del centered, apply_np, x_train
        if len(self.penalties) != 1:
            raise NotImplementedError(
                "Tensor products of smooths with multiple penalties are not supported."
            )
        return (
            np.asarray(self.basis_train, dtype=np.float64),
            np.asarray(self.penalties[0], dtype=np.float64),
            None,
        )

    def tensor_marginal_predict_matrix(
        self, X_new, *, centered=False, np_transform=None
    ):
        del centered
        B = np.asarray(self.transform_new(X_new), dtype=np.float64)
        if np_transform is not None:
            B = B @ np.asarray(np_transform, dtype=np.float64)
        return B

    def _normalized_term_sp(self, n_penalties):
        if n_penalties <= 0:
            return []
        if self.sp is None:
            return [None] * n_penalties
        if np.isscalar(self.sp):
            if n_penalties != 1:
                raise NotImplementedError(
                    "Multi-penalty term-level sp must supply one value per penalty."
                )
            vals = [float(self.sp)]
        else:
            vals = np.asarray(self.sp, dtype=np.float64).ravel()
            if vals.size != n_penalties:
                raise ValueError(
                    f"term-level sp must have length {n_penalties}, got {vals.size}."
                )
            vals = [float(v) for v in vals]
        return vals

    def _build_penalty_block(
        self,
        matrix: np.ndarray,
        *,
        smooth_metadata: dict[str, Any],
        rank: int | None = None,
        null_space_dim: int | None = None,
        selection_metadata: dict[str, Any] | None = None,
        selection_via_subsystem: bool = False,
    ):
        """
        Build penalty definition list for a single main penalty plus optional
        ``select=True`` null-space penalty.

        Subclasses supply ``matrix`` (the main penalty, usually
        ``self.penalties[0]`` after fit) and term-specific ``smooth_metadata``.
        Optional ``rank`` / ``null_space_dim`` are forwarded to the main
        :class:`~nampy.gam.penalties.PenaltySpec` before normalization
        (e.g. TPRS passes a stored rank).
        """
        self._require_fitted()
        main_matrix = np.asarray(matrix, dtype=np.float64)
        selection_defs = []
        if self.select:
            sel_meta = (
                selection_metadata
                if selection_metadata is not None
                else {**smooth_metadata, "is_selection_penalty": True}
            )
            selection_defs = self._build_selection_penalty_definitions(
                [main_matrix],
                selection_metadata=sel_meta,
                null_penalty_tol=float(getattr(self, "null_penalty_tol", 1e-10)),
                fallback_selection_smoothing_id=None,
                selection_via_subsystem=selection_via_subsystem,
            )

        sp_vals = self._normalized_term_sp(1 + len(selection_defs))
        sp_main = sp_vals[0] if sp_vals else None
        if sp_main is None:
            sp_mode, sp_value = None, None
        elif sp_main >= 0:
            sp_mode, sp_value = "fixed", float(sp_main)
        else:
            sp_mode, sp_value = "estimate", None

        sid = None if self.smoothing_id is None else str(self.smoothing_id)
        meta_smooth = {
            **smooth_metadata,
            "term_sp": sp_main,
            "is_selection_penalty": False,
        }
        meta_smooth = self._penalty_metadata_with_scale(meta_smooth, penalty_index=0)
        main_spec = normalize_penalty_spec(
            PenaltySpec(
                matrix=main_matrix,
                smoothing_id=sid,
                kind="smooth",
                rank=rank,
                null_space_dim=null_space_dim,
                is_null_space_penalty=False,
                sp_mode=sp_mode,
                sp_value=sp_value,
                metadata=meta_smooth,
            )
        )
        defs = [main_spec]
        for j, sel in enumerate(selection_defs, start=1):
            sp_sel = sp_vals[j] if j < len(sp_vals) else None
            if sp_sel is None:
                defs.append(sel)
                continue
            if sp_sel >= 0:
                sel_mode, sel_value = "fixed", float(sp_sel)
            else:
                sel_mode, sel_value = "estimate", None
            defs.append(
                normalize_penalty_spec(
                    PenaltySpec(
                        matrix=np.asarray(sel.matrix, dtype=np.float64),
                        smoothing_id=sel.smoothing_id,
                        kind=str(sel.kind),
                        rank=sel.rank,
                        null_space_dim=sel.null_space_dim,
                        is_null_space_penalty=bool(sel.is_null_space_penalty),
                        sp_mode=sel_mode,
                        sp_value=sel_value,
                        metadata=dict(sel.metadata),
                    )
                )
            )
        return defs

    def _build_selection_penalty_definitions(
        self,
        penalty_terms,
        *,
        selection_metadata: dict[str, Any] | None = None,
        null_penalty_tol: float = 1e-10,
        fallback_selection_smoothing_id: str | None = None,
        selection_via_subsystem: bool = True,
    ):
        if not self.select:
            return []

        if not penalty_terms:
            return []

        combined = np.asarray(penalty_terms[0], dtype=np.float64).copy()
        for penalty in penalty_terms[1:]:
            combined = combined + np.asarray(penalty, dtype=np.float64)
        select_sid = selection_penalty_id(
            self.smoothing_id, fallback=fallback_selection_smoothing_id
        )
        if selection_metadata is None:
            meta = {}
        else:
            meta = dict(selection_metadata)
        meta["is_selection_penalty"] = True
        meta.setdefault("penalty_rescale_factor", 1.0)

        if selection_via_subsystem:
            sel = build_null_space_selection_spec(
                main_penalty=combined,
                smoothing_id=select_sid,
                tol=float(null_penalty_tol),
                metadata=meta,
            )
            return [] if sel is None else [sel]

        S0, ns_meta = null_space_penalty_from_penalty(
            combined, tol=float(null_penalty_tol)
        )
        if int(ns_meta["rank"]) <= 0:
            return []
        return [
            make_penalty_spec(
                matrix=S0,
                smoothing_id=select_sid,
                kind="null_space",
                sp_mode=None,
                sp_value=None,
                is_null_space_penalty=True,
                metadata=meta,
            )
        ]

    def _apply_point_constraint(
        self,
        base,
        penalties,
        pc,
        *,
        feature_names,
        point_basis_fn,
        fixed=False,
    ):
        if pc is None:
            return None

        names = [str(n) for n in feature_names]
        if len(names) == 1:
            pc_value = _normalize_point_constraint(pc, names[0])
            pc_point = np.asarray([[pc_value]], dtype=np.float64)
        else:
            pc_point = _normalize_point_constraint_vector(pc, names)[None, :]

        pc_row = _coerce_pc_point_basis(point_basis_fn, pc_point)
        penalties_in = (
            [] if bool(fixed) else [np.asarray(P, dtype=np.float64) for P in penalties]
        )
        Bc, Sc, C = apply_linear_constraint(
            np.asarray(base, dtype=np.float64),
            penalties_in,
            pc_row,
        )
        Bc = self._apply_cached_by(Bc)
        return (
            np.asarray(Bc, dtype=np.float64),
            [np.asarray(S, dtype=np.float64) for S in Sc],
            C,
            pc_row,
        )

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
                sid = penalty_id_for_local_index(
                    self.smoothing_id, j, n_penalties=n_raw
                )
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
                    metadata=self._penalty_metadata_with_scale(
                        {
                            "term_type": self.term_type,
                            "basis_name": self.basis_name,
                            "feature": self.feature,
                            "label": self.label,
                            "by": self.by,
                            "term_sp": sp_j,
                        },
                        penalty_index=j,
                    ),
                )
            )
        return defs
