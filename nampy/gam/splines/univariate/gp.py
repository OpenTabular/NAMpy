"""Low-rank Gaussian-process smooth construction matching mgcv ``bs='gp'``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .ds import (
    _duchon_setup_locations,
    _duchon_unique_rows,
    _normalize_duchon_knots,
)
from .tp import _top_eigensystem


def normalize_gp_definition(m) -> tuple[np.ndarray, bool]:
    """Normalize mgcv's ``m=(signed type, range, power)`` definition."""
    if m is None:
        values = []
    elif np.isscalar(m):
        values = [m]
    else:
        values = list(np.asarray(m, dtype=object).ravel())

    missing = not values
    if len(values) == 1:
        try:
            missing = bool(np.isnan(float(values[0])))
        except (TypeError, ValueError):
            missing = False

    if missing:
        signed_type = 3.0
        stationary = False
    else:
        try:
            first = float(values[0])
            if not np.isfinite(first):
                raise ValueError
            gp_type = abs(int(np.rint(first)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("incorrect arguments to GP smoother") from exc
        signed_type = float(np.sign(first) * gp_type)
        stationary = bool(first < 0.0)

    try:
        rho = float(values[1]) if len(values) > 1 else -1.0
        power = float(values[2]) if len(values) > 2 else 1.0
    except (TypeError, ValueError) as exc:
        raise ValueError("incorrect arguments to GP smoother") from exc
    return np.asarray([signed_type, rho, power], dtype=np.float64), stationary


def gp_polynomial_basis(x, definition):
    """Port ``gpT``: constant tail for stationary GP, linear tail otherwise."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    definition = np.asarray(definition, dtype=np.float64).ravel()
    constant = np.ones(values.shape[0], dtype=np.float64)
    if definition[0] < 0.0:
        return constant.reshape(-1, 1)
    return np.asarray(np.column_stack([constant, values]), dtype=np.float64)


def gp_kernel(x, knots, definition=None):
    """Port ``gpE`` and return both the covariance and resolved definition."""
    values = np.asarray(x, dtype=np.float64)
    setup_knots = np.asarray(knots, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if setup_knots.ndim == 1:
        setup_knots = setup_knots.reshape(-1, 1)
    if values.shape[1] != setup_knots.shape[1]:
        raise ValueError("GP data and knots must have the same dimension.")

    differences = values[:, None, :] - setup_knots[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    normalized, _ = normalize_gp_definition(definition)
    signed_type, rho, power = map(float, normalized)
    gp_type = abs(int(np.rint(signed_type)))
    if np.isnan(rho) or np.isnan(power):
        raise ValueError("missing value where TRUE/FALSE needed")
    if rho <= 0.0:
        rho = float(np.max(distances))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = distances / rho

    if gp_type not in {1, 2, 3, 4, 5} or power > 2.0 or power <= 0.0:
        raise ValueError("incorrect arguments to GP smoother")

    if gp_type == 1:
        covariance = (1.0 - 1.5 * scaled + 0.5 * scaled**3) * (scaled <= 1.0)
    elif gp_type == 2:
        covariance = np.exp(-(scaled**power))
    else:
        exponential = np.exp(-scaled)
        if gp_type == 3:
            covariance = (1.0 + scaled) * exponential
        elif gp_type == 4:
            covariance = exponential + (scaled * exponential) * (
                1.0 + scaled / 3.0
            )
        else:
            covariance = exponential + (scaled * exponential) * (
                1.0 + 0.4 * scaled + scaled**2 / 15.0
            )

    resolved = np.asarray([signed_type, rho, power], dtype=np.float64)
    return np.asarray(covariance, dtype=np.float64), resolved


def default_gp_k(dimension: int) -> int:
    """Return the literal mgcv default ``d + 1 + c(10,30,100)[d]``."""
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("Gaussian-process smooths require at least one covariate.")
    if dimension > 3:
        raise ValueError(
            "An omitted k for bs='gp' is undefined upstream above three dimensions; "
            "supply k explicitly."
        )
    return dimension + 1 + (10, 30, 100)[dimension - 1]


def _parse_gp_xt(xt):
    max_knots = 2000
    seed = 1
    if xt is None:
        return max_knots, seed
    if not isinstance(xt, dict):
        raise NotImplementedError(
            "For bs='gp', xt must be None or a dict with optional keys "
            "{'max.knots', 'seed'}."
        )
    if xt.get("max.knots") is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed") is not None:
        seed = int(xt["seed"])
    if max_knots < 1:
        raise ValueError("For bs='gp', xt['max.knots'] must be positive.")
    return max_knots, seed


@dataclass
class GaussianProcessSetup:
    shift: np.ndarray
    knots: np.ndarray
    UZ: np.ndarray
    definition: np.ndarray
    null_space_dim: int
    rank: int
    bs_dim: int
    basis_train: np.ndarray
    penalty: np.ndarray
    used_supplied_knots: bool
    used_subsampling: bool


def build_gaussian_process_setup(X, *, k=-1, m=None, knots=None, xt=None):
    """Port ``smooth.construct.gp.smooth.spec`` and retain prediction state."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("GP smooth data must be a non-empty numeric matrix.")

    n_obs, dimension = values.shape
    raw_definition, stationary = normalize_gp_definition(m)
    requested_k = int(k)

    unique = _duchon_unique_rows(values)
    if requested_k >= 0 and unique.shape[0] < requested_k:
        raise ValueError(
            "A term has fewer unique covariate combinations than specified "
            "maximum degrees of freedom"
        )

    supplied = _normalize_duchon_knots(knots, dimension)
    if supplied is not None and supplied.shape[0] == 0:
        supplied = None
    if supplied is not None and supplied.shape[0] > n_obs:
        warnings.warn(
            "more knots than data in an ms term: knots ignored.",
            stacklevel=2,
        )
        supplied = None

    shift = np.mean(values, axis=0)
    if supplied is not None:
        supplied = supplied - shift[None, :]
    max_knots, seed = _parse_gp_xt(xt)
    setup_knots, used_subsampling = _duchon_setup_locations(
        values,
        shift,
        supplied,
        max_knots=max_knots,
        seed=seed,
    )

    covariance, definition = gp_kernel(setup_knots, setup_knots, raw_definition)
    bs_dim = requested_k if requested_k >= 0 else default_gp_k(dimension)
    if bs_dim < dimension + 2:
        bs_dim = dimension + 2
        warnings.warn("basis dimension reset to minimum possible", stacklevel=2)

    null_space_dim = 1 if stationary else dimension + 1
    rank = int(bs_dim - null_space_dim)
    n_knots = int(setup_knots.shape[0])
    if n_knots < rank:
        raise ValueError(
            "Gaussian-process smooth requires at least as many knot locations as "
            "penalized basis coefficients."
        )

    penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
    if rank < n_knots:
        eigenvalues, eigenvectors = _top_eigensystem(
            covariance,
            rank,
            tolerance_exponent=0.5,
        )
        penalty[:rank, :rank] = np.diag(eigenvalues)
    else:
        eigenvectors = np.eye(rank, dtype=np.float64)
        penalty[:rank, :rank] = covariance

    setup = GaussianProcessSetup(
        shift=np.asarray(shift, dtype=np.float64),
        knots=np.asarray(setup_knots, dtype=np.float64),
        UZ=np.asarray(eigenvectors, dtype=np.float64),
        definition=np.asarray(definition, dtype=np.float64),
        null_space_dim=int(null_space_dim),
        rank=int(rank),
        bs_dim=int(bs_dim),
        basis_train=np.zeros((n_obs, bs_dim), dtype=np.float64),
        penalty=np.asarray(penalty, dtype=np.float64),
        used_supplied_knots=bool(supplied is not None),
        used_subsampling=bool(used_subsampling),
    )
    setup.basis_train = predict_gaussian_process(values, setup)
    return setup


def predict_gaussian_process(X_new, setup: GaussianProcessSetup):
    """Port ``Predict.matrix.gp.smooth`` including its knot-sized chunks."""
    values = np.asarray(X_new, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    shifted = values - setup.shift[None, :]
    n_obs = int(shifted.shape[0])
    n_knots = int(setup.knots.shape[0])

    def _block(block):
        covariance, _ = gp_kernel(block, setup.knots, setup.definition)
        tail = gp_polynomial_basis(block, setup.definition)
        return np.asarray(np.column_stack([covariance @ setup.UZ, tail]))

    if n_obs <= n_knots:
        return np.asarray(_block(shifted), dtype=np.float64)
    out = np.empty((n_obs, setup.bs_dim), dtype=np.float64)
    for start in range(0, n_obs, n_knots):
        stop = min(start + n_knots, n_obs)
        out[start:stop, :] = _block(shifted[start:stop, :])
    return out


__all__ = [
    "GaussianProcessSetup",
    "build_gaussian_process_setup",
    "default_gp_k",
    "gp_kernel",
    "gp_polynomial_basis",
    "normalize_gp_definition",
    "predict_gaussian_process",
]
