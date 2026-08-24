"""Canonical univariate spline setup surface."""

from .bs import (
    DerivativeBSplineSetup,
    build_derivative_bspline_setup,
    derivative_bspline_design,
    derivative_bspline_knots,
    derivative_penalty_root,
    normalize_bspline_orders,
    predict_derivative_bspline,
)
from .cr import (
    CubicSplines,
    add_full_rank_shrinkage,
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
    place_knots_through_values,
)
from .ps import (
    PSplineBasisSetup,
    bspline_design_matrix,
    build_pspline_term_setup,
    cyclic_pspline_design,
    cyclic_pspline_difference_penalty,
    cyclic_pspline_knots,
    cyclic_wrap,
    predict_pspline_term,
    predict_pspline_term_derivative,
    pspline_difference_penalty,
    pspline_knots,
    pspline_predict_matrix,
)
from .tp import build_tprs_term_setup, predict_tprs_term

__all__ = [
    "DerivativeBSplineSetup",
    "build_derivative_bspline_setup",
    "derivative_bspline_design",
    "derivative_bspline_knots",
    "derivative_penalty_root",
    "normalize_bspline_orders",
    "predict_derivative_bspline",
    "add_full_rank_shrinkage",
    "bspline_design_matrix",
    "cyclic_cubic_bd",
    "cyclic_cubic_predict_matrix",
    "cyclic_pspline_design",
    "cyclic_pspline_difference_penalty",
    "cyclic_pspline_knots",
    "cyclic_wrap",
    "place_knots_through_values",
    "pspline_difference_penalty",
    "pspline_knots",
    "pspline_predict_matrix",
    "CubicSplines",
    "PSplineBasisSetup",
    "build_pspline_term_setup",
    "predict_pspline_term",
    "predict_pspline_term_derivative",
    "build_tprs_term_setup",
    "predict_tprs_term",
]
