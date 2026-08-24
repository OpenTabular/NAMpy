"""Duchon regression-spline primitives for ``mgcv``'s ``bs='ds'``."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance_matrix

from ...linalg.qr import r_linpack_qr_no_pivot
from ..basis.tp import tp_T
from .tp import _top_eigensystem


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


def _r_linpack_qty(packed_qr, qraux, values):
    """Apply ``t(Q)`` from base R's LINPACK QR representation."""
    packed = np.asarray(packed_qr, dtype=np.float64)
    aux = np.asarray(qraux, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    for j in range(min(aux.size, packed.shape[1])):
        if aux[j] == 0.0:
            continue
        reflector = packed[j:, j].copy()
        reflector[0] = aux[j]
        denominator = float(reflector[0])
        for column in range(out.shape[1]):
            step = -float(np.dot(reflector, out[j:, column])) / denominator
            out[j:, column] += step * reflector
    return np.asarray(out, dtype=np.float64)


def _parse_duchon_xt(xt):
    max_knots = 2000
    seed = 1
    if xt is None:
        return max_knots, seed
    if not isinstance(xt, dict):
        raise NotImplementedError(
            "For bs='ds', xt must be None or a dict with optional keys "
            "{'max.knots', 'seed'}."
        )
    if xt.get("max.knots") is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed") is not None:
        seed = int(xt["seed"])
    if max_knots < 1:
        raise ValueError("For bs='ds', xt['max.knots'] must be positive.")
    return max_knots, seed


def _normalize_duchon_knots(knots, dimension: int):
    """Collect per-coordinate knot vectors like the upstream constructor."""
    if knots is None:
        return None

    dimension = int(dimension)
    if isinstance(knots, (list, tuple)):
        if dimension == 1 and (not knots or np.isscalar(knots[0])):
            return np.asarray(knots, dtype=np.float64).reshape(-1, 1)
        if len(knots) != dimension or any(value is None for value in knots):
            return None
        columns = [np.asarray(value, dtype=np.float64).ravel() for value in knots]
        if any(column.size != columns[0].size for column in columns[1:]):
            raise ValueError(
                "components of knots relating to a single smooth must be of same length"
            )
        return np.asarray(np.column_stack(columns), dtype=np.float64)

    values = np.asarray(knots, dtype=np.float64)
    if values.ndim == 1:
        if dimension != 1:
            return None
        return values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != dimension:
        return None
    return np.asarray(values, dtype=np.float64)


def _duchon_unique_rows(values):
    """Mirror ``uniquecombs(x, ordered=TRUE)`` for numeric matrices."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 1:
        return np.unique(matrix[:, 0]).reshape(-1, 1)

    first_by_key = {}
    for row in matrix:
        key = "*".join(format(float(value), ".15g") for value in row)
        first_by_key.setdefault(key, np.asarray(row, dtype=np.float64).copy())
    return np.vstack([first_by_key[key] for key in sorted(first_by_key)])


class _RMersenneTwister:
    """Minimal R-compatible MT19937 stream used by ``temp.seed``/``sample``."""

    def __init__(self, seed):
        state = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            state = (69069 * state + 1) & 0xFFFFFFFF
        state = (69069 * state + 1) & 0xFFFFFFFF  # R stores 624 in this slot.
        self._state = []
        for _ in range(624):
            state = (69069 * state + 1) & 0xFFFFFFFF
            self._state.append(state)
        self._index = 624

    def _twist(self):
        for index in range(624):
            word = (self._state[index] & 0x80000000) | (
                self._state[(index + 1) % 624] & 0x7FFFFFFF
            )
            self._state[index] = (
                self._state[(index + 397) % 624]
                ^ (word >> 1)
                ^ (0x9908B0DF if word & 1 else 0)
            )
        self._index = 0

    def uniform(self):
        if self._index >= 624:
            self._twist()
        word = self._state[self._index]
        self._index += 1
        word ^= word >> 11
        word ^= (word << 7) & 0x9D2C5680
        word ^= (word << 15) & 0xEFC60000
        word ^= word >> 18
        return float(word & 0xFFFFFFFF) / float(2**32)

    def uniform_index(self, size):
        size = int(size)
        bits = int(math.ceil(math.log2(size)))
        mask = (1 << bits) - 1
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + math.floor(self.uniform() * 65536)
            value &= mask
            if value < size:
                return int(value)


def _r_sample_without_replacement(size, sample_size, seed):
    """Mirror modern R ``sample.int(..., replace=FALSE)`` rejection sampling."""
    available = list(range(int(size)))
    remaining = int(size)
    rng = _RMersenneTwister(seed)
    selected = []
    for _ in range(int(sample_size)):
        index = rng.uniform_index(remaining)
        selected.append(available[index])
        remaining -= 1
        available[index] = available[remaining]
    return np.asarray(selected, dtype=np.intp)


def _duchon_setup_locations(values, shift, knots, *, max_knots, seed):
    values = np.asarray(values, dtype=np.float64)
    if knots is not None:
        return np.asarray(knots, dtype=np.float64), False
    unique = _duchon_unique_rows(values) - np.asarray(shift, dtype=np.float64)[None, :]
    if values.shape[0] > int(max_knots) and unique.shape[0] > int(max_knots):
        indices = _r_sample_without_replacement(
            unique.shape[0],
            int(max_knots),
            int(seed),
        )
        return np.asarray(unique[indices, :], dtype=np.float64), True
    return np.asarray(unique, dtype=np.float64), False


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

    unique = _duchon_unique_rows(values)
    if unique.shape[0] < bs_dim:
        raise ValueError(
            "A term has fewer unique covariate combinations than specified "
            "maximum degrees of freedom"
        )

    shift = np.mean(values, axis=0)
    supplied = _normalize_duchon_knots(knots, dimension)
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

    max_knots, seed = _parse_duchon_xt(xt)
    setup_knots, used_subsampling = _duchon_setup_locations(
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
        eigenvalues, eigenvectors = _top_eigensystem(
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
    first = _r_linpack_qty(packed_qr, qraux, diagonal_penalty)
    reduced = _r_linpack_qty(
        packed_qr,
        qraux,
        first[null_space_dim:, :].T,
    )[null_space_dim:, :]
    penalty = np.zeros((bs_dim, bs_dim), dtype=np.float64)
    penalty[: bs_dim - null_space_dim, : bs_dim - null_space_dim] = reduced

    UZ = _r_linpack_qty(
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
