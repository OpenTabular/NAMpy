"""Runtime terms for SCAM's SCOP-spline basis family."""

from __future__ import annotations

import numpy as np

from ...penalties.algebra import penalty_rescale_factor, scale_penalty
from ...splines.shape import build_scop_univariate_setup, predict_scop_univariate
from ..registry import register_smooth
from ..smooth_base import (
    BaseSmoothTerm,
    _resolve_feature,
    by_values_from_new_data,
    column_as_numeric_array,
    linear_functional_basis,
    linear_functional_by_state,
)


@register_smooth("mpi")
@register_smooth("mpd")
@register_smooth("mdcv")
@register_smooth("mdcx")
@register_smooth("micv")
@register_smooth("micx")
@register_smooth("cv")
@register_smooth("cx")
@register_smooth("po")
@register_smooth("dpo")
@register_smooth("ipo")
@register_smooth("miso")
@register_smooth("mifo")
@register_smooth("mpiby")
@register_smooth("mpdby")
@register_smooth("mdcvby")
@register_smooth("mdcxby")
@register_smooth("micvby")
@register_smooth("micxby")
@register_smooth("cvby")
@register_smooth("cxby")
@register_smooth("cpop")
@register_smooth("lmpi")
@register_smooth("lipl")
class ShapeConstrainedPSplineTerm(BaseSmoothTerm):
    """Monotone SCOP-spline runtime for ``bs='mpi'`` and ``bs='mpd'``."""

    term_type = "shape_constrained_smooth"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        *,
        k=10,
        basis="mpi",
        m=None,
        xt=None,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        knots=None,
        metadata=None,
    ):
        super().__init__(
            feature=feature,
            label=label,
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.metadata["coefficient_covariance_transport"] = "prediction"
        self.basis_name = str(basis).lower()
        self.k = 10 if int(k) < 0 else int(k)
        m_values = np.asarray([2] if m is None else m).reshape(-1)
        self.m = int(m_values[0])
        self.penalty_order = int(
            m_values[0] if m_values.size == 1 else m_values[1]
        )
        self.xt = dict(xt or {})
        self.change_point = self.xt.get("xc")
        self.select = bool(select)
        self.fixed = bool(fixed)
        self.knots = knots
        supported = {
            "mpi",
            "mpd",
            "mdcv",
            "mdcx",
            "micv",
            "micx",
            "cv",
            "cx",
            "po",
            "dpo",
            "ipo",
            "miso",
            "mifo",
            "mpiby",
            "mpdby",
            "mdcvby",
            "mdcxby",
            "micvby",
            "micxby",
            "cvby",
            "cxby",
            "cpop",
            "lmpi",
            "lipl",
        }
        if self.basis_name not in supported:
            raise ValueError(
                f"Unsupported shape basis {self.basis_name!r}; expected {sorted(supported)}."
            )
        self._is_by_basis = self.basis_name.endswith("by")
        if by is not None and not self._is_by_basis:
            raise NotImplementedError(
                f"bs={self.basis_name!r} with by= is not equivalent to SCAM's "
                f"{self.basis_name}By basis and is not enabled yet."
            )
        if self._is_by_basis and by is None:
            raise ValueError(
                f"SCAM's bs={self.basis_name!r} requires a numeric by= variable."
            )
        if self.basis_name in {"lmpi", "lipl"} and self.change_point is None:
            raise ValueError(
                f"SCAM's bs={self.basis_name!r} requires xt={{'xc': change_point}}."
            )
        if self.select:
            raise NotImplementedError(
                "SCAM does not expose mgcv's select= null-space penalty surface."
            )
        self._feature_index = None
        self._feature_name = None
        self._by_state = None
        self._basis_train = None
        self._penalties = None
        self._setup = None
        self.positive_coefficient_mask = None
        self._linear_functional = False
        self._X_train = None

    def fit(self, X, feature_names):
        self._X_train = np.asarray(X, dtype=object).copy()
        index, name = _resolve_feature(self.feature, feature_names)
        self._feature_index = index
        self._feature_name = name
        self._set_resolved_features([name])
        self._set_by_state(X, feature_names)
        values = column_as_numeric_array(X, index)
        self._linear_functional = values.ndim == 2
        if self._linear_functional and not self._is_by_basis:
            raise NotImplementedError(
                "Shape-constrained linear-functional terms require one of SCAM's "
                "*By bases and a matrix-valued by variable."
            )
        by_values = (
            None
            if not self._is_by_basis
            else np.asarray(self._by_state.values, dtype=np.float64)
        )
        if self._linear_functional and by_values.shape != values.shape:
            raise ValueError(
                "Matrix-valued feature and by columns must have identical shape."
            )
        if self._linear_functional:
            self._by_state = linear_functional_by_state(self._by_state)
        self._setup = build_scop_univariate_setup(
            values.reshape(-1),
            basis_code=self.basis_name,
            bs_dim=self.k,
            spline_order=self.m,
            penalty_order=self.penalty_order,
            change_point=self.change_point,
            knots=self.knots,
        )
        setup_basis = np.asarray(self._setup.basis_train, dtype=np.float64)
        if self._linear_functional:
            self._basis_train = linear_functional_basis(
                values,
                by_values,
                lambda points: predict_scop_univariate(points, self._setup),
            )
        else:
            self._basis_train = (
                self._apply_cached_by(setup_basis)
                if self._is_by_basis
                else setup_basis
            )
        raw_penalty = np.asarray(self._setup.penalty, dtype=np.float64)
        self._set_penalty_rescale_factors(
            [penalty_rescale_factor(setup_basis, raw_penalty)]
        )
        self._penalties = (
            [] if self.fixed else [scale_penalty(setup_basis, raw_penalty)]
        )
        self.positive_coefficient_mask = np.asarray(
            self._setup.positive_mask, dtype=bool
        )
        # SCAM constructors center these bases directly and return C with zero
        # rows. A second generic centering transform would destroy the
        # coordinatewise positivity parameterization.
        self._record_constraint_result(
            "scam_constructor_centering", None, absorbed_by="runtime"
        )
        return self

    def derivative_matrix(self, X_new=None, order=1):
        """Return SCAM ``derivative.scam``'s training-data derivative map."""
        self._require_fitted()
        order = int(order)
        if order not in {1, 2}:
            raise ValueError("deriv can be either 1 or 2")
        if self._linear_functional:
            raise NotImplementedError(
                "SCAM derivative extraction currently handles scalar 1D smooths only."
            )
        if X_new is not None:
            raise NotImplementedError(
                "This SCAM derivative provider currently exposes its exact "
                "training-data construction only."
            )
        analytic = {
            "mpi",
            "mpd",
            "cv",
            "cx",
            "mdcv",
            "mdcx",
            "micv",
            "micx",
            "mpiby",
            "mpdby",
            "mdcvby",
            "mdcxby",
            "micvby",
            "micxby",
            "cvby",
            "cxby",
            "po",
        }
        if self.basis_name in analytic:
            derivative_basis = (
                self._setup.derivative_basis_1
                if order == 1
                else self._setup.derivative_basis_2
            )
            difference = np.diff(
                np.eye(self.k - 1, dtype=np.float64), n=order, axis=0
            )
            return np.asarray(
                derivative_basis @ difference @ self._setup.sigma,
                dtype=np.float64,
            )

        # scam/R/derivative.scam.r uses forward differences with eps=1e-7
        # for all other univariate smooth classes and sets numeric by= to one.
        eps = 1e-7
        base = self._X_train.copy()
        if self._by_state is not None:
            base[:, self._by_state.feature_index] = 1.0
        X0 = np.asarray(self.transform_new(base), dtype=np.float64)
        first = base.copy()
        first[:, self._feature_index] = np.asarray(
            first[:, self._feature_index], dtype=np.float64
        ) + eps
        X1 = np.asarray(self.transform_new(first), dtype=np.float64)
        if order == 1:
            return (X1 - X0) / eps
        second = first.copy()
        second[:, self._feature_index] = np.asarray(
            second[:, self._feature_index], dtype=np.float64
        ) + eps
        X2 = np.asarray(self.transform_new(second), dtype=np.float64)
        return (X2 - 2.0 * X1 + X0) / eps**2

    def get_penalty_definitions(self):
        self._require_fitted()
        if self.fixed:
            return []
        metadata = {
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "shape_constraint": self.basis_name,
            "p_ident": self.positive_coefficient_mask.tolist(),
            "m": self.m,
            "penalty_order": self.penalty_order,
            "change_point": self.change_point,
            "knots": np.asarray(self._setup.knots, dtype=np.float64).tolist(),
            "fixed": self.fixed,
            "linear_functional": self._linear_functional,
        }
        return self._build_penalty_block(
            self._penalties[0],
            smooth_metadata=metadata,
            rank=self._setup.rank,
            null_space_dim=self._setup.null_space_dim,
        )

    def transform_new(self, X_new):
        self._require_fitted()
        values = column_as_numeric_array(X_new, self._feature_index)
        if self._linear_functional:
            return linear_functional_basis(
                values,
                by_values_from_new_data(X_new, self._by_state),
                lambda points: predict_scop_univariate(points, self._setup),
            )
        if values.ndim != 1:
            raise ValueError(
                "Prediction data for this scalar smooth must contain scalar feature rows."
            )
        basis = predict_scop_univariate(values, self._setup)
        if self._is_by_basis:
            basis = self._apply_by_scale(
                basis, by_values_from_new_data(X_new, self._by_state)
            )
        return basis


__all__ = ["ShapeConstrainedPSplineTerm"]
