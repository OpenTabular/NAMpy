"""Low-rank spherical splines matching mgcv ``bs='sos'``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import spence

from ...linalg.qr import r_linpack_qr_no_pivot, r_linpack_qty
from .._low_rank import (
    low_rank_setup_locations,
    normalize_coordinate_knots,
    parse_low_rank_xt,
    top_eigensystem,
)


def normalize_spherical_order(m) -> int:
    """Mirror the order normalization in mgcv's SOS constructor."""
    if m is None:
        return 0
    values = np.asarray(m, dtype=object).ravel()
    if values.size != 1:
        raise ValueError("For bs='sos', m must be a single numeric value.")
    try:
        value = float(values[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("For bs='sos', m must be a single numeric value.") from exc
    if np.isnan(value):
        return 0
    if not np.isfinite(value):
        raise ValueError("For bs='sos', m must be finite.")
    order = int(np.rint(value))
    if order < -2:
        order = -1
    if order > 4:
        order = 4
    return order


def spherical_null_space_dimension(order: int) -> int:
    """Return the literal null-tail size used by mgcv 1.9-4."""
    return 4 if int(order) == -1 else 1


def _spherical_geometry(latitude, longitude, knot_latitude, knot_longitude):
    latitude = np.deg2rad(np.asarray(latitude, dtype=np.float64).ravel())
    longitude = np.deg2rad(np.asarray(longitude, dtype=np.float64).ravel())
    knot_latitude = np.deg2rad(np.asarray(knot_latitude, dtype=np.float64).ravel())
    knot_longitude = np.deg2rad(np.asarray(knot_longitude, dtype=np.float64).ravel())
    cosine = np.sin(latitude)[:, None] * np.sin(knot_latitude)[None, :] + np.cos(
        latitude
    )[:, None] * np.cos(knot_latitude)[None, :] * np.cos(
        longitude[:, None] - knot_longitude[None, :]
    )
    gamma = np.arccos(np.clip(cosine, -1.0, 1.0))
    return latitude, longitude, knot_latitude, knot_longitude, gamma


def spherical_spline_kernel(X, knots, order=0):
    """Port mgcv's ``makeR`` and return its kernel and null-space tail."""
    values = np.asarray(X, dtype=np.float64)
    setup_knots = np.asarray(knots, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            "Spherical spline data must have exactly two columns: latitude, longitude."
        )
    if setup_knots.ndim != 2 or setup_knots.shape[1] != 2:
        raise ValueError(
            "Spherical spline knots must have exactly two columns: latitude, longitude."
        )
    if not np.isfinite(values).all() or not np.isfinite(setup_knots).all():
        raise ValueError("Spherical spline data and knots must be finite.")

    order = int(order)
    latitude, longitude, knot_latitude, knot_longitude, gamma = _spherical_geometry(
        values[:, 0],
        values[:, 1],
        setup_knots[:, 0],
        setup_knots[:, 1],
    )

    if order == -2:
        distance = 2.0 * np.sin(gamma / 2.0)
        distance = np.maximum(distance, np.finfo(np.float64).tiny * 10.0)
        kernel = -distance
        tail = np.ones((values.shape[0], 1), dtype=np.float64)
        constraint_tail = np.ones((setup_knots.shape[0], 1), dtype=np.float64)
    elif order == -1:
        distance = 2.0 * np.sin(gamma / 2.0)
        distance = np.maximum(distance, np.finfo(np.float64).tiny * 10.0)
        with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
            kernel = distance * distance * np.log(distance) / (8.0 * np.pi)
        kernel = np.nan_to_num(kernel, nan=0.0)

        def _tail(lat, lon):
            z = np.sin(lat)
            x = np.cos(lat) * np.sin(lon)
            y = np.cos(lat) * np.cos(lon)
            return np.column_stack([np.ones(lat.size), x, y, z])

        tail = _tail(latitude, longitude)
        constraint_tail = _tail(knot_latitude, knot_longitude)
    elif order == 0:
        cosine = np.cos(gamma)
        argument = np.clip((1.0 + cosine) / 2.0, 0.0, 1.0)
        kernel = (1.0 - np.pi**2 / 6.0 + spence(1.0 - argument)) / (4.0 * np.pi)
        tail = np.ones((values.shape[0], 1), dtype=np.float64)
        constraint_tail = np.ones((setup_knots.shape[0], 1), dtype=np.float64)
    elif order in {1, 2, 3, 4}:
        z = 1.0 - np.cos(gamma)
        z = np.maximum(z, np.finfo(np.float64).eps * 0.0001)
        W = z / 2.0
        C = np.sqrt(W)
        A = np.log(1.0 + 1.0 / C)
        C = 2.0 * C
        if order == 1:
            q = 2.0 * A * W - C + 1.0
            kernel = (q - 0.5) / (2.0 * np.pi)
        elif order == 2:
            W2 = W * W
            q = A * (6.0 * W2 - 2.0 * W) - 3.0 * C * W + 3.0 * W + 0.5
            kernel = (q / 2.0 - 1.0 / 6.0) / (2.0 * np.pi)
        elif order == 3:
            W2 = W * W
            W3 = W2 * W
            q = (
                A * (60.0 * W3 - 36.0 * W2)
                + 30.0 * W2
                + C * (8.0 * W - 30.0 * W2)
                - 3.0 * W
                + 1.0
            ) / 3.0
            kernel = (q / 6.0 - 1.0 / 24.0) / (2.0 * np.pi)
        else:
            W2 = W * W
            W3 = W2 * W
            W4 = W3 * W
            q = (
                A * (70.0 * W4 - 60.0 * W3 + 6.0 * W2)
                + 35.0 * W3 * (1.0 - C)
                + C * 55.0 * W2 / 3.0
                - 12.5 * W2
                - W / 3.0
                + 0.25
            )
            kernel = (q / 24.0 - 1.0 / 120.0) / (2.0 * np.pi)
        tail = np.ones((values.shape[0], 1), dtype=np.float64)
        constraint_tail = np.ones((setup_knots.shape[0], 1), dtype=np.float64)
    else:
        raise ValueError("Spherical spline order must be in {-2,-1,0,1,2,3,4}.")

    return (
        np.asarray(kernel, dtype=np.float64),
        np.asarray(tail, dtype=np.float64),
        np.asarray(constraint_tail, dtype=np.float64),
    )


@dataclass
class SphericalSplineSetup:
    knots: np.ndarray
    UZ: np.ndarray
    order: int
    null_space_dim: int
    rank: int
    bs_dim: int
    basis_train: np.ndarray
    penalty: np.ndarray
    column_scale: np.ndarray
    used_supplied_knots: bool
    used_subsampling: bool


def build_spherical_spline_setup(X, *, k=-1, m=None, knots=None, xt=None):
    """Port ``smooth.construct.sos.smooth.spec`` and retain prediction state."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(
            "Can only deal with a sphere: bs='sos' requires exactly latitude and longitude."
        )
    if not np.isfinite(values).all():
        raise ValueError("Spherical spline data must be finite.")

    n_obs = int(values.shape[0])
    order = normalize_spherical_order(m)
    null_space_dim = spherical_null_space_dimension(order)
    bs_dim = 50 if int(k) < 0 else int(k)
    if bs_dim < null_space_dim + 2:
        raise ValueError(
            f"For bs='sos' with m={order}, k must be at least {null_space_dim + 2}."
        )

    supplied = normalize_coordinate_knots(knots, 2)
    if supplied is not None and supplied.shape[0] == 0:
        supplied = None
    if supplied is not None and supplied.shape[0] > n_obs:
        warnings.warn(
            "more knots than data in an sos term: knots ignored.",
            stacklevel=2,
        )
        supplied = None
    max_knots, seed = parse_low_rank_xt(xt, basis_name="sos")
    setup_knots, used_subsampling = low_rank_setup_locations(
        values,
        np.zeros(2, dtype=np.float64),
        supplied,
        max_knots=max_knots,
        seed=seed,
    )
    n_knots = int(setup_knots.shape[0])
    if bs_dim > n_knots:
        raise ValueError(
            "Spherical spline requires at least as many unique knot locations "
            "as basis coefficients."
        )

    radial, _, constraint_tail = spherical_spline_kernel(
        setup_knots, setup_knots, order
    )
    if bs_dim < n_knots:
        eigenvalues, eigenvectors = top_eigensystem(
            radial,
            bs_dim,
            tolerance_exponent=0.5,
        )
        diagonal_penalty = np.diag(eigenvalues)
        constraint = (constraint_tail.T @ eigenvectors).T
    else:
        eigenvectors = np.eye(bs_dim, dtype=np.float64)
        diagonal_penalty = radial
        constraint = constraint_tail

    packed_qr, qraux = r_linpack_qr_no_pivot(constraint)
    first = r_linpack_qty(packed_qr, qraux, diagonal_penalty)
    reduced = r_linpack_qty(
        packed_qr,
        qraux,
        first[null_space_dim:, :].T,
    )[null_space_dim:, :]
    penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
    rank = int(bs_dim - null_space_dim)
    penalty[:rank, :rank] = reduced
    UZ = r_linpack_qty(
        packed_qr,
        qraux,
        eigenvectors.T,
    )[null_space_dim:, :].T

    setup = SphericalSplineSetup(
        knots=np.asarray(setup_knots, dtype=np.float64),
        UZ=np.asarray(UZ, dtype=np.float64),
        order=int(order),
        null_space_dim=int(null_space_dim),
        rank=rank,
        bs_dim=int(bs_dim),
        basis_train=np.zeros((n_obs, bs_dim), dtype=np.float64),
        penalty=np.asarray(penalty, dtype=np.float64),
        column_scale=np.ones(bs_dim, dtype=np.float64),
        used_supplied_knots=bool(supplied is not None),
        used_subsampling=bool(used_subsampling),
    )
    basis = predict_spherical_spline(values, setup, apply_scale=False)
    standard_deviation = np.std(basis, axis=0, ddof=1)
    standard_deviation[standard_deviation == np.min(standard_deviation)] = 1.0
    column_scale = 1.0 / standard_deviation
    setup.column_scale = np.asarray(column_scale, dtype=np.float64)
    setup.basis_train = np.asarray(basis * column_scale[None, :], dtype=np.float64)
    setup.penalty = np.asarray(
        column_scale[:, None] * penalty * column_scale[None, :],
        dtype=np.float64,
    )
    return setup


def predict_spherical_spline(X_new, setup: SphericalSplineSetup, *, apply_scale=True):
    """Port ``Predict.matrix.sos.smooth`` including knot-sized chunks."""
    values = np.asarray(X_new, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            "Spherical spline prediction requires latitude and longitude columns."
        )
    if not np.isfinite(values).all():
        raise ValueError("Spherical spline prediction data must be finite.")
    n_obs = int(values.shape[0])
    n_knots = int(setup.knots.shape[0])
    out = np.empty((n_obs, setup.bs_dim), dtype=np.float64)
    for start in range(0, n_obs, n_knots):
        stop = min(start + n_knots, n_obs)
        radial, tail, _ = spherical_spline_kernel(
            values[start:stop, :], setup.knots, setup.order
        )
        out[start:stop, :] = np.column_stack([radial @ setup.UZ, tail])
    if apply_scale:
        out *= setup.column_scale[None, :]
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "SphericalSplineSetup",
    "build_spherical_spline_setup",
    "normalize_spherical_order",
    "predict_spherical_spline",
    "spherical_null_space_dimension",
    "spherical_spline_kernel",
]
