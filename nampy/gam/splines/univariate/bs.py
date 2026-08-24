"""Derivative-penalized B-spline primitives for ``mgcv``'s ``bs='bs'``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky_banded, solve

from ...linalg import symmetrize_matrix
from .ps import bspline_design_matrix, pspline_predict_matrix


def _is_missing_order(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() == "NA":
        return True
    try:
        return bool(np.asarray(value).ndim == 0 and np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _integer_order(value, *, position: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "For bs='bs', m entries must be non-negative integers or NA."
        ) from exc
    if not np.isfinite(numeric) or numeric != np.rint(numeric) or numeric < 0:
        raise ValueError("For bs='bs', m entries must be non-negative integers or NA.")
    del position
    return int(numeric)


def normalize_bspline_orders(m) -> tuple[int, ...]:
    """Normalize ``m`` exactly as ``smooth.construct.bs.smooth.spec``."""
    if m is None:
        return (3, 2)
    if np.isscalar(m):
        if _is_missing_order(m):
            return (3, 2)
        degree = _integer_order(m, position=0)
        return (degree, max(0, degree - 1))

    values = list(np.asarray(m, dtype=object).ravel())
    if len(values) == 0:
        raise ValueError("For bs='bs', m must contain a spline degree.")
    if len(values) == 1:
        return normalize_bspline_orders(values[0])

    if _is_missing_order(values[0]):
        if _is_missing_order(values[1]):
            return (3, 2)
        values[0] = _integer_order(values[1], position=1) + 1
    if _is_missing_order(values[1]):
        values[1] = max(0, _integer_order(values[0], position=0) - 1)

    orders = tuple(_integer_order(value, position=i) for i, value in enumerate(values))
    derivative_orders = orders[1:]
    if len(set(derivative_orders)) < len(derivative_orders):
        raise ValueError("multiple penalties of the same order is silly")
    if any(order > orders[0] for order in derivative_orders):
        raise ValueError("requested non-existent derivative in B-spline penalty")
    return orders


def derivative_bspline_knots(x, bs_dim, degree, supplied_knots=None):
    """Port the automatic, endpoint, full, and four-knot constructor rules."""
    values = np.asarray(x, dtype=np.float64).ravel()
    degree = int(degree)
    bs_dim = int(bs_dim)
    nk = bs_dim - degree + 1
    if nk <= 0:
        raise ValueError("basis dimension too small for b-spline order")
    expected = nk + 2 * degree

    knots = None
    if supplied_knots is not None:
        knots = np.asarray(supplied_knots, dtype=np.float64).ravel()

    if knots is not None and knots.size == 4 and knots.size < expected:
        limits = np.sort(knots)
        if nk <= 1:
            raise ValueError(
                "basis dimension too small for automatic knot construction"
            )
        dx = (limits[3] - limits[0]) / float(nk - 1)
        lower_outer = limits[0] - dx * degree
        upper_outer = limits[3] + dx * degree
        lower = np.linspace(lower_outer, limits[0], degree + 1)
        middle = (
            np.linspace(limits[1], limits[2], max(0, nk - 2))
            if nk > 2
            else np.empty(0, dtype=np.float64)
        )
        upper = np.linspace(limits[3], upper_outer, degree + 1)
        return np.concatenate([lower, middle, upper])

    if knots is None or knots.size == 2:
        if knots is None:
            lower = float(np.min(values))
            upper = float(np.max(values))
        else:
            lower = float(np.min(knots))
            upper = float(np.max(knots))
            if lower > np.min(values) or upper < np.max(values):
                raise ValueError("knot range does not include data")
        if nk <= 1:
            raise ValueError(
                "basis dimension too small for automatic knot construction"
            )
        width = upper - lower
        lower -= width * 0.001
        upper += width * 0.001
        dx = (upper - lower) / float(nk - 1)
        return np.linspace(
            lower - dx * degree,
            upper + dx * degree,
            expected,
        )

    if knots.size != expected:
        raise ValueError(f"there should be {expected} supplied knots")
    if np.any(np.diff(knots) < 0):
        raise ValueError("supplied bs knots must be nondecreasing")
    return knots


def derivative_bspline_design(x, knots, degree, deriv=0):
    """Evaluate the constructor basis inside its effective knot interval."""
    values = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    degree = int(degree)
    deriv = int(deriv)
    if deriv < 0 or deriv > degree:
        raise ValueError("requested non-existent derivative in B-spline penalty")
    lower = float(knots[degree])
    upper = float(knots[knots.size - degree - 1])
    if np.min(values) < lower or np.max(values) > upper:
        raise ValueError("x out of range")
    return bspline_design_matrix(
        values,
        knots,
        degree=degree,
        deriv=deriv,
        extrapolate=True,
    )


def derivative_penalty_root(knots, degree, derivative_order):
    """Port mgcv's exact band-Cholesky integrated-derivative penalty root."""
    knots = np.asarray(knots, dtype=np.float64).ravel()
    degree = int(degree)
    derivative_order = int(derivative_order)
    polynomial_degree = degree - derivative_order
    if polynomial_degree < 0:
        raise ValueError("requested non-existent derivative in B-spline penalty")

    n_basis = int(knots.size - degree - 1)
    interior = knots[degree : n_basis + 1]
    widths = np.diff(interior)
    if np.any(widths < 0):
        raise ValueError("supplied bs knots must be nondecreasing")

    if polynomial_degree == 0:
        points = 0.5 * (interior[:-1] + interior[1:])
        design = derivative_bspline_design(
            points,
            knots,
            degree=degree,
            deriv=derivative_order,
        )
        return np.sqrt(widths)[:, None] * design

    steps = np.repeat(widths / polynomial_degree, polynomial_degree)
    points = np.cumsum(np.concatenate(([interior[0]], steps)), dtype=np.float64)
    points = np.clip(points, interior[0], interior[-1])
    design = derivative_bspline_design(
        points,
        knots,
        degree=degree,
        deriv=derivative_order,
    )

    local_nodes = np.linspace(-1.0, 1.0, polynomial_degree + 1)
    vandermonde = local_nodes[:, None] ** np.arange(polynomial_degree + 1)[None, :]
    inverse_vandermonde = solve(
        vandermonde,
        np.eye(polynomial_degree + 1),
        assume_a="gen",
        check_finite=True,
    )
    powers = np.add.outer(
        np.arange(polynomial_degree + 1),
        np.arange(polynomial_degree + 1),
    )
    gram = np.where(powers % 2 == 0, 2.0 / (powers + 1.0), 0.0)
    local_weight = inverse_vandermonde.T @ gram @ inverse_vandermonde

    n_nodes = widths.size * polynomial_degree + 1
    band = np.zeros((polynomial_degree + 1, n_nodes), dtype=np.float64)
    for interval, width in enumerate(widths):
        base = interval * polynomial_degree
        scale = float(width) / 2.0
        for column in range(polynomial_degree + 1):
            for row in range(column, polynomial_degree + 1):
                band[row - column, base + column] += scale * local_weight[column, row]

    factor = cholesky_banded(
        band,
        lower=True,
        overwrite_ab=False,
        check_finite=True,
    )
    root = factor[0, :, None] * design
    for offset in range(1, polynomial_degree + 1):
        root[:-offset, :] += factor[offset, :-offset, None] * design[offset:, :]
    return np.asarray(root, dtype=np.float64)


@dataclass
class DerivativeBSplineSetup:
    feature_index: int
    feature_name: str
    degree: int
    derivative_orders: tuple[int, ...]
    knots: np.ndarray
    basis_train: np.ndarray
    penalty_roots: tuple[np.ndarray, ...]
    penalties: tuple[np.ndarray, ...]
    ranks: tuple[int, ...]
    null_space_dim: int
    bs_dim: int
    orders: tuple[int, ...]


def build_derivative_bspline_setup(
    x,
    *,
    feature_index,
    feature_name,
    bs_dim,
    m=None,
    knots=None,
):
    """Build raw ``Bspline.smooth`` basis and integrated derivative penalties."""
    values = np.asarray(x, dtype=np.float64).ravel()
    orders = normalize_bspline_orders(m)
    degree = int(orders[0])
    derivative_orders = tuple(int(value) for value in orders[1:])
    resolved_bs_dim = max(10, degree) if int(bs_dim) < 0 else int(bs_dim)
    full_knots = derivative_bspline_knots(
        values,
        bs_dim=resolved_bs_dim,
        degree=degree,
        supplied_knots=knots,
    )
    basis = derivative_bspline_design(values, full_knots, degree, deriv=0)
    if np.any(np.sum(basis, axis=0) == 0.0):
        warnings.warn(
            "there is *no* information about some basis coefficients",
            stacklevel=2,
        )
    if np.unique(values).size < resolved_bs_dim:
        warnings.warn(
            "basis dimension is larger than number of unique covariates",
            stacklevel=2,
        )

    roots = tuple(
        derivative_penalty_root(full_knots, degree, order)
        for order in derivative_orders
    )
    penalties = tuple(symmetrize_matrix(root.T @ root) for root in roots)
    ranks = tuple(int(resolved_bs_dim - order) for order in derivative_orders)
    null_space_dim = int(min(derivative_orders))
    return DerivativeBSplineSetup(
        feature_index=int(feature_index),
        feature_name=str(feature_name),
        degree=degree,
        derivative_orders=derivative_orders,
        knots=np.asarray(full_knots, dtype=np.float64),
        basis_train=np.asarray(basis, dtype=np.float64),
        penalty_roots=roots,
        penalties=penalties,
        ranks=ranks,
        null_space_dim=null_space_dim,
        bs_dim=int(resolved_bs_dim),
        orders=orders,
    )


def predict_derivative_bspline(x_new, setup: DerivativeBSplineSetup, deriv=0):
    """Match ``Predict.matrix.Bspline.smooth`` including linear extrapolation."""
    return np.asarray(
        pspline_predict_matrix(
            x_new,
            setup.knots,
            basis_order=int(setup.degree) - 1,
            deriv=int(deriv),
        ),
        dtype=np.float64,
    )
