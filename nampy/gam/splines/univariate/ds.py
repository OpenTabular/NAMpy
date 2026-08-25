"""Duchon regression-spline primitives for ``mgcv``'s ``bs='ds'``."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance_matrix

from ...linalg.qr import r_linpack_qr_no_pivot, r_linpack_qty
from .._low_rank import (
    low_rank_setup_locations,
    normalize_coordinate_knots,
    ordered_unique_numeric_rows,
    parse_low_rank_xt,
    top_eigensystem,
)
from ..basis.tp import tp_T


def normalize_duchon_orders(m, dimension: int) -> tuple[int, float]:
    """Mirror ``smooth.construct.ds.smooth.spec`` normalization of ``m``."""
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("Duchon smooths require at least one covariate.")

    if m is None:
        raw = [np.nan, np.nan]
    elif np.isscalar(m):
        raw = [m, np.nan]
    else:
        raw = list(np.asarray(m, dtype=object).ravel())
        raw = (raw + [np.nan, np.nan])[:2]

    def _missing(value):
        if value is None:
            return True
        if isinstance(value, str) and value.strip().upper() == "NA":
            return True
        try:
            return bool(np.isnan(float(value)))
        except (TypeError, ValueError):
            return False

    try:
        penalty_order = 2 if _missing(raw[0]) else int(np.rint(float(raw[0])))
        shift_order = 0.0 if _missing(raw[1]) else float(np.rint(2 * float(raw[1])) / 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("For bs='ds', m must contain numeric values or NA.") from exc

    penalty_order = max(1, penalty_order)
    if shift_order >= dimension / 2:
        shift_order = (dimension - 1) / 2
        warnings.warn("s value reduced", stacklevel=2)
    if shift_order <= -dimension / 2:
        shift_order = -(dimension - 1) / 2
        warnings.warn("s value increased", stacklevel=2)
    if penalty_order + shift_order <= dimension / 2:
        shift_order = 0.5 + dimension / 2 - penalty_order
        if shift_order >= dimension / 2:
            raise ValueError("No suitable s (i.e. m[2]) try increasing m[1]")
        warnings.warn(
            "s value modified to give continuous function",
            stacklevel=2,
        )
    return int(penalty_order), float(shift_order)


def duchon_null_space_dimension(dimension: int, penalty_order: int) -> int:
    """Return the polynomial null-space size ``choose(m + d - 1, d)``."""
    return math.comb(int(penalty_order) + int(dimension) - 1, int(dimension))


def default_duchon_k(dimension: int, null_space_dim: int) -> int:
    defaults = (10, 30, 100)
    return int(null_space_dim) + defaults[min(int(dimension), len(defaults)) - 1]


def duchon_polynomial_basis(x, penalty_order: int):
    """Port ``DuchonT`` using mgcv's thin-plate polynomial ordering."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    dimension = int(values.shape[1])
    null_space_dim = duchon_null_space_dimension(dimension, penalty_order)
    return np.asarray(
        tp_T(values, null_space_dim, int(penalty_order), dimension),
        dtype=np.float64,
    )


def duchon_kernel(x, knots, penalty_order: int, shift_order: float):
    """Port ``DuchonE`` including its exponent-dependent sign convention."""
    values = np.asarray(x, dtype=np.float64)
    knot_values = np.asarray(knots, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if knot_values.ndim == 1:
        knot_values = knot_values.reshape(-1, 1)
    if values.shape[1] != knot_values.shape[1]:
        raise ValueError("Duchon data and knots must have the same dimension.")

    distances = distance_matrix(values, knot_values)
    exponent_float = 2 * int(penalty_order) + 2 * float(shift_order) - values.shape[1]
    exponent = int(np.rint(exponent_float))
    if not np.isclose(exponent_float, exponent, atol=0.0, rtol=0.0):
        raise ValueError(
            "Duchon kernel exponent must be integral after m normalization."
        )

    if exponent % 2 == 0:
        kernel = np.zeros_like(distances, dtype=np.float64)
        nonzero = distances != 0.0
        kernel[nonzero] = distances[nonzero] ** exponent * np.log(distances[nonzero])
    else:
        kernel = distances**exponent
    sign = 1 - 2 * ((math.floor(exponent / 2) + 1) % 2)
    return np.asarray(kernel * sign, dtype=np.float64)


@dataclass
class DuchonSplineSetup:
    shift: np.ndarray
    knots: np.ndarray
    UZ: np.ndarray
    penalty_order: int
    shift_order: float
    null_space_dim: int
    rank: int
    bs_dim: int
    basis_train: np.ndarray
    penalty: np.ndarray
    used_supplied_knots: bool
    used_subsampling: bool


def build_duchon_spline_setup(X, *, k=-1, m=None, knots=None, xt=None):
    """Port ``smooth.construct.ds.smooth.spec`` and retain prediction state."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Duchon smooth data must be a non-empty numeric matrix.")

    n_obs, dimension = values.shape
    penalty_order, shift_order = normalize_duchon_orders(m, dimension)
    null_space_dim = duchon_null_space_dimension(dimension, penalty_order)
    bs_dim = int(k)
    if bs_dim < 0:
        bs_dim = default_duchon_k(dimension, null_space_dim)
    if bs_dim < null_space_dim + 1:
        bs_dim = null_space_dim + 1
        warnings.warn("basis dimension reset to minimum possible", stacklevel=2)

    unique = ordered_unique_numeric_rows(values)
    if unique.shape[0] < bs_dim:
        raise ValueError(
            "A term has fewer unique covariate combinations than specified "
            "maximum degrees of freedom"
        )

    shift = np.mean(values, axis=0)
    supplied = normalize_coordinate_knots(knots, dimension)
    if supplied is not None and supplied.shape[0] == 0:
        supplied = None
    if supplied is not None and supplied.shape[0] > n_obs:
        warnings.warn(
            "more knots than data in a ds term: knots ignored.",
            stacklevel=2,
        )
        supplied = None
    if supplied is not None:
        supplied = supplied - shift[None, :]

    max_knots, seed = parse_low_rank_xt(xt, basis_name="ds")
    setup_knots, used_subsampling = low_rank_setup_locations(
        values,
        shift,
        supplied,
        max_knots=max_knots,
        seed=seed,
    )
    n_knots = int(setup_knots.shape[0])
    if n_knots < bs_dim:
        raise ValueError(
            "Duchon spline requires at least as many knot locations as basis "
            "coefficients."
        )

    E = duchon_kernel(setup_knots, setup_knots, penalty_order, shift_order)
    T = duchon_polynomial_basis(setup_knots, penalty_order)
    if bs_dim < n_knots:
        eigenvalues, eigenvectors = top_eigensystem(
            E,
            bs_dim,
            tolerance_exponent=0.5,
        )
        diagonal_penalty = np.diag(eigenvalues)
        constraint = (T.T @ eigenvectors).T
    else:
        eigenvectors = np.eye(bs_dim, dtype=np.float64)
        diagonal_penalty = np.asarray(E, dtype=np.float64)
        constraint = np.asarray(T, dtype=np.float64)

    packed_qr, qraux = r_linpack_qr_no_pivot(constraint)
    first = r_linpack_qty(packed_qr, qraux, diagonal_penalty)
    reduced = r_linpack_qty(
        packed_qr,
        qraux,
        first[null_space_dim:, :].T,
    )[null_space_dim:, :]
    penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
    penalty[: bs_dim - null_space_dim, : bs_dim - null_space_dim] = reduced

    UZ = r_linpack_qty(
        packed_qr,
        qraux,
        eigenvectors.T,
    )[null_space_dim:, :].T
    setup = DuchonSplineSetup(
        shift=np.asarray(shift, dtype=np.float64),
        knots=np.asarray(setup_knots, dtype=np.float64),
        UZ=np.asarray(UZ, dtype=np.float64),
        penalty_order=int(penalty_order),
        shift_order=float(shift_order),
        null_space_dim=int(null_space_dim),
        rank=int(bs_dim - null_space_dim),
        bs_dim=int(bs_dim),
        basis_train=np.zeros((n_obs, bs_dim), dtype=np.float64),
        penalty=np.asarray(penalty, dtype=np.float64),
        used_supplied_knots=bool(supplied is not None),
        used_subsampling=bool(used_subsampling),
    )
    setup.basis_train = predict_duchon_spline(values, setup)
    return setup


def predict_duchon_spline(X_new, setup: DuchonSplineSetup):
    """Port ``Predict.matrix.duchon.spline``."""
    values = np.asarray(X_new, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    shifted = values - setup.shift[None, :]
    radial = duchon_kernel(
        shifted,
        setup.knots,
        setup.penalty_order,
        setup.shift_order,
    )
    polynomial = duchon_polynomial_basis(shifted, setup.penalty_order)
    return np.asarray(
        np.column_stack([radial @ setup.UZ, polynomial]),
        dtype=np.float64,
    )


__all__ = [
    "DuchonSplineSetup",
    "build_duchon_spline_setup",
    "default_duchon_k",
    "duchon_kernel",
    "duchon_null_space_dimension",
    "duchon_polynomial_basis",
    "normalize_duchon_orders",
    "predict_duchon_spline",
]
