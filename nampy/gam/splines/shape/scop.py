"""SCOP-spline primitives ported from ``scam/R/uni.smooth.const.r``.

This module owns raw constructor state only.  It does not perform SCAM's
nonlinear coefficient optimization; that belongs to the fitting backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..univariate.cr import cyclic_wrap
from ..univariate.ps import bspline_design_matrix, pspline_predict_matrix

_CENTERED_SCOP_CODES = frozenset(
    {"mpi", "mpd", "mdcv", "mdcx", "micv", "micx", "cv", "cx"}
)
_POSITIVE_SCOP_CODES = frozenset({"po", "dpo", "ipo"})
_ENDPOINT_ZERO_CODES = frozenset({"miso", "mifo"})
_BY_SCOP_BASE_CODES = {
    "mpiby": "mpi",
    "mpdby": "mpd",
    "mdcvby": "mdcv",
    "mdcxby": "mdcx",
    "micvby": "micv",
    "micxby": "micx",
    "cvby": "cv",
    "cxby": "cx",
}
_SUPPORTED_CODES = (
    _CENTERED_SCOP_CODES
    | _POSITIVE_SCOP_CODES
    | _ENDPOINT_ZERO_CODES
    | frozenset(_BY_SCOP_BASE_CODES)
    | {"cpop"}
    | {"lmpi", "lipl"}
)


@dataclass(frozen=True)
class ShapeConstrainedPSplineSetup:
    """Raw, representation-identified state of a univariate SCOP spline."""

    basis_code: str
    spline_order: int
    penalty_order: int
    bs_dim_requested: int
    knots: np.ndarray
    basis_train: np.ndarray
    center: np.ndarray
    sigma: np.ndarray
    accumulation_matrix: np.ndarray
    difference_matrix: np.ndarray
    penalty: np.ndarray
    positive_mask: np.ndarray
    derivative_basis_1: np.ndarray
    derivative_basis_2: np.ndarray
    rank: int
    null_space_dim: int
    prediction_keep_indices: np.ndarray
    change_point: float | None
    constrained_dimension: int | None

    @property
    def n_coef(self) -> int:
        return int(self.basis_train.shape[1])


def scop_knots(x, *, bs_dim: int, spline_order: int = 2, supplied_knots=None):
    """Construct SCAM's equally spaced, externally extended knot sequence."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("SCOP-spline construction requires at least one value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("SCOP-spline values must be finite.")
    q = int(bs_dim)
    m = int(spline_order)
    if m < 0:
        raise ValueError("spline_order must be non-negative.")
    expected = q + m + 2
    if expected <= 0:
        raise ValueError("bs_dim is too small for spline_order.")

    if supplied_knots is not None:
        knots = np.asarray(supplied_knots, dtype=np.float64).reshape(-1)
        if knots.size != expected:
            raise ValueError(f"there should be {expected} supplied knots")
        return knots.copy()

    n_inner = q - m
    if n_inner < 2:
        # The upstream arithmetic subsequently requires the distance between
        # two adjacent inner knots, so make this unsupported surface explicit.
        raise ValueError("SCOP-splines require bs_dim >= spline_order + 2.")
    inner = np.linspace(float(np.min(values)), float(np.max(values)), n_inner)
    spacing = float(inner[1] - inner[0])
    knots = np.empty(expected, dtype=np.float64)
    knots[m + 1 : q + 1] = inner
    for index in range(m + 1):
        knots[index] = inner[0] - (m + 1 - index) * spacing
    for index in range(q + 1, expected):
        knots[index] = inner[-1] + (index - q) * spacing
    return knots


def _mixed_reduced_accumulation(q: int, basis_code: str) -> np.ndarray:
    n = q - 1
    if basis_code in {"micx", "mdcv"}:
        sigma = np.zeros((n, n), dtype=np.float64)
        for column in range(n):
            sigma[column:, column] = np.arange(1, n - column + 1)
        if basis_code == "mdcv":
            sigma *= -1.0
        return sigma
    if basis_code in {"micv", "mdcx"}:
        sigma = np.zeros((n, n), dtype=np.float64)
        sigma[0, :] = 1.0
        for row in range(1, n):
            split = n - row
            sigma[row, :split] = row + 1
            sigma[row, split:] = np.arange(row, 0, -1)
        if basis_code == "mdcx":
            sigma *= -1.0
        return sigma
    if basis_code in {"cv", "cx"}:
        sigma = np.zeros((n, n), dtype=np.float64)
        sigma[:, 0] = np.arange(1, n + 1)
        for column in range(1, n):
            sigma[column:, column] = -np.arange(1, n - column + 1)
        if basis_code == "cx":
            sigma *= -1.0
        return sigma
    raise ValueError(f"Unsupported mixed SCOP basis {basis_code!r}.")


def _accumulation_matrix(q: int, basis_code: str) -> np.ndarray:
    if basis_code == "mpi":
        sigma = np.tril(np.ones((q, q), dtype=np.float64))
    elif basis_code == "mpd":
        sigma = np.tril(-np.ones((q, q), dtype=np.float64))
        sigma[:, 0] *= -1.0
    elif basis_code in {"mdcv", "mdcx", "micv", "micx", "cv", "cx"}:
        sigma = np.zeros((q, q), dtype=np.float64)
        sigma[:, 0] = 1.0
        sigma[1:, 1:] = _mixed_reduced_accumulation(q, basis_code)
    elif basis_code == "po":
        sigma = np.eye(q, dtype=np.float64)
    elif basis_code == "dpo":
        sigma = np.triu(np.ones((q, q), dtype=np.float64))
    elif basis_code == "ipo":
        sigma = np.tril(np.ones((q, q), dtype=np.float64))
    else:  # pragma: no cover - caller validates before dispatch
        raise ValueError(f"Unsupported SCOP basis {basis_code!r}.")
    return sigma


def _difference_matrix(size: int, order: int) -> np.ndarray:
    return np.diff(np.eye(int(size), dtype=np.float64), n=int(order), axis=0)


def _cpop_knots(x, *, bs_dim: int, supplied_knots=None) -> np.ndarray:
    """Knot setup from ``smooth.construct.cpop.smooth.spec``."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    expected = int(bs_dim) + 1
    if supplied_knots is None:
        return np.linspace(float(np.min(values)), float(np.max(values)), expected)
    supplied = np.asarray(supplied_knots, dtype=np.float64).reshape(-1)
    if supplied.size == 2:
        lower = float(np.min(supplied))
        upper = float(np.max(supplied))
        if lower > np.min(values) or upper < np.max(values):
            raise ValueError("knot range does not include data")
        return np.linspace(lower, upper, expected)
    if supplied.size != expected:
        raise ValueError(f"there should be {expected} supplied knots")
    return supplied.copy()


def _spline_design_zero_outside(x, knots, degree) -> np.ndarray:
    """Evaluate individual B-splines over their full supports.

    Unlike :class:`scipy.interpolate.BSpline`, ``splines::splineDesign`` with
    ``outer.ok=TRUE`` does not extrapolate a combined spline outside the base
    interval. It evaluates every basis function wherever its own knot support
    exists. This Cox--de Boor recursion mirrors that behavior.
    """
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    knot_vector = np.asarray(knots, dtype=np.float64).reshape(-1)
    p = int(degree)
    basis = np.zeros((values.size, knot_vector.size - 1), dtype=np.float64)
    for index in range(knot_vector.size - 1):
        basis[:, index] = (
            (values >= knot_vector[index])
            & (values < knot_vector[index + 1])
        )
    for current_degree in range(1, p + 1):
        width = knot_vector.size - current_degree - 1
        next_basis = np.zeros((values.size, width), dtype=np.float64)
        for index in range(width):
            left_denominator = (
                knot_vector[index + current_degree] - knot_vector[index]
            )
            if left_denominator != 0.0:
                next_basis[:, index] += (
                    (values - knot_vector[index]) / left_denominator
                ) * basis[:, index]
            right_denominator = (
                knot_vector[index + current_degree + 1]
                - knot_vector[index + 1]
            )
            if right_denominator != 0.0:
                next_basis[:, index] += (
                    (knot_vector[index + current_degree + 1] - values)
                    / right_denominator
                ) * basis[:, index + 1]
        basis = next_basis
    return basis


def _cyclic_pspline_design(x, knots, *, spline_order: int) -> np.ndarray:
    """Operation-for-operation port of SCAM's local ``cSplineDes``."""
    values = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    cyclic_knots = np.sort(np.asarray(knots, dtype=np.float64).reshape(-1))
    order = int(spline_order) + 2
    if order < 2:
        raise ValueError("order too low")
    if cyclic_knots.size < order:
        raise ValueError("too few knots")
    lower = float(cyclic_knots[0])
    upper = float(cyclic_knots[-1])
    if np.min(values) < lower or np.max(values) > upper:
        raise ValueError("x out of range")
    wrap_threshold = float(cyclic_knots[cyclic_knots.size - order])
    prefix = lower - (
        upper - cyclic_knots[cyclic_knots.size - order : -1]
    )
    extended_knots = np.concatenate([prefix, cyclic_knots])
    design = _spline_design_zero_outside(
        values, extended_knots, degree=order - 1
    )
    wrap = values > wrap_threshold
    if np.any(wrap):
        shifted = values[wrap] - upper + lower
        design[wrap, :] += _spline_design_zero_outside(
            shifted, extended_knots, degree=order - 1
        )
    return design


def _cyclic_difference_matrix(size: int, order: int) -> np.ndarray:
    size = int(size)
    order = int(order)
    if order < 0 or order > size:
        raise ValueError("penalty order too high for basis dimension")
    expanded = np.eye(size + order, dtype=np.float64)
    for _ in range(order):
        expanded = np.diff(expanded, axis=0)
    if order == 0:
        return expanded
    difference = expanded[:, order:].copy()
    difference[:, size - order : size] += expanded[:, :order]
    return difference


def _local_scop_knots(
    x, *, bs_dim: int, spline_order: int, change_point: float
) -> tuple[np.ndarray, int]:
    """Knot setup shared by ``smooth.construct.lmpi/lipl``."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    m = int(spline_order)
    requested = int(bs_dim)
    lower = float(np.min(values))
    upper = float(np.max(values))
    xc = float(change_point)
    share = (xc - lower) / (upper - lower)
    q1 = max(int(np.ceil(requested * share)), 5)
    q2 = max(requested - q1, 5)
    knots = np.full(q1 + q2 + 2, np.nan, dtype=np.float64)
    knots[m + 1 : q1 + 1] = np.linspace(lower, xc, q1 - m)
    knots[q1 : q1 + q2 - m + 1] = np.linspace(
        xc, upper, q2 - m + 1
    )
    for index in range(m + 1):
        knots[index] = knots[m + 1] - (m + 1 - index) * (
            knots[m + 2] - knots[m + 1]
        )
    for index in range(q1 + q2 - m + 1, q1 + q2 + 2):
        knots[index] = knots[q1 + q2 - m - 1] + (
            index + 1 - q1 - q2 + m
        ) * (knots[q1 + 1] - knots[q1])
    knots = np.concatenate(
        [knots[knots < xc], np.repeat(xc, m), knots[knots > xc]]
    )
    return knots, q1


def build_scop_univariate_setup(
    x,
    *,
    basis_code: str,
    bs_dim: int = 10,
    spline_order: int = 2,
    penalty_order: int | None = None,
    change_point: float | None = None,
    knots=None,
) -> ShapeConstrainedPSplineSetup:
    """Build an identified global univariate SCOP constructor state."""
    code = str(basis_code).lower()
    if code not in _SUPPORTED_CODES:
        raise NotImplementedError(
            f"Raw SCOP constructor {code!r} is not implemented; "
            f"supported codes are {sorted(_SUPPORTED_CODES)}."
        )
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    q = int(bs_dim)
    m = int(spline_order)
    p_order = m if penalty_order is None else int(penalty_order)
    local_q1 = None
    if code in {"lmpi", "lipl"}:
        if m < 1:
            raise ValueError("spline_order must be at least one for local SCOP bases")
        if knots is not None:
            raise ValueError(
                f"{code!r} smooth currently does not work with user-supplied knots"
            )
        if change_point is None:
            raise ValueError("a change point 'xc' is not supplied")
        knot_vector, local_q1 = _local_scop_knots(
            values,
            bs_dim=q,
            spline_order=m,
            change_point=float(change_point),
        )
    elif code == "cpop":
        if q + 1 <= m:
            raise ValueError("basis dimension too small for b-spline order")
        knot_vector = _cpop_knots(values, bs_dim=q, supplied_knots=knots)
    else:
        knot_vector = scop_knots(
            values,
            bs_dim=q,
            spline_order=m,
            supplied_knots=knots,
        )
    if code == "miso":
        # smooth.construct.miso.smooth.spec collapses the next m inner knots
        # onto the first inner knot, avoiding a flat segment at the origin.
        knot_vector[m + 2 : 2 * m + 2] = knot_vector[m + 1]
    elif code == "mifo":
        # smooth.construct.mifo.smooth.spec applies the mirrored operation at
        # the final inner knot.
        knot_vector[q - m : q] = knot_vector[q]

    if code == "cpop":
        raw_basis = _cyclic_pspline_design(
            values, knot_vector, spline_order=m
        )
    else:
        raw_basis = bspline_design_matrix(
            values,
            knot_vector,
            degree=m + 1,
            deriv=0,
            extrapolate=True,
        )
    if code == "lmpi":
        q_total = raw_basis.shape[1]
        accumulation = np.zeros((q_total, q_total), dtype=np.float64)
        accumulation[:local_q1, :local_q1] = np.tril(
            np.ones((local_q1, local_q1), dtype=np.float64)
        )
        accumulation[local_q1:, local_q1:] = np.eye(q_total - local_q1)
        zero_indices = np.empty(0, dtype=np.int64)
        keep = np.arange(q_total, dtype=np.int64)
    elif code == "lipl":
        q_total = raw_basis.shape[1]
        accumulation = np.tril(np.ones((q_total, q_total), dtype=np.float64))
        zero_indices = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.arange(local_q1 - 1, q_total)]
        )
        keep = np.arange(1, local_q1 - 1, dtype=np.int64)
    elif code == "cpop":
        accumulation = np.eye(q, dtype=np.float64)
        zero_indices = np.empty(0, dtype=np.int64)
        keep = np.arange(1, q, dtype=np.int64)
    elif code in _BY_SCOP_BASE_CODES:
        accumulation = _accumulation_matrix(q, _BY_SCOP_BASE_CODES[code])
        zero_indices = np.empty(0, dtype=np.int64)
        keep = np.arange(q, dtype=np.int64)
    elif code in _ENDPOINT_ZERO_CODES:
        accumulation = np.tril(np.ones((q, q), dtype=np.float64))
        if code == "miso":
            zero_indices = np.arange(m + 1, dtype=np.int64)
        else:
            zero_indices = np.arange(q - m - 1, q, dtype=np.int64)
        accumulation[:, zero_indices] = 0.0
        accumulation[zero_indices, :] = 0.0
        keep = np.setdiff1d(
            np.arange(q, dtype=np.int64), zero_indices, assume_unique=True
        )
    else:
        accumulation = _accumulation_matrix(q, code)
        zero_indices = np.empty(0, dtype=np.int64)
        if code == "dpo":
            keep = np.arange(q - 1, dtype=np.int64)
        else:
            keep = np.arange(1, q, dtype=np.int64)
    transformed = raw_basis @ accumulation
    transformed = transformed[:, keep]
    center = (
        np.mean(transformed, axis=0)
        if code in _CENTERED_SCOP_CODES or code in {"lmpi", "lipl"}
        else np.zeros(transformed.shape[1], dtype=np.float64)
    )
    basis = transformed - center[None, :]

    if code == "lmpi":
        q_total = transformed.shape[1]
        difference = np.zeros((q_total - 3, q_total), dtype=np.float64)
        first_difference = _difference_matrix(local_q1, 1)
        difference[1:local_q1, 1 : local_q1 + 1] = first_difference
        unconstrained_size = q_total - local_q1 - 1
        second_difference = _difference_matrix(unconstrained_size, 2)
        difference[local_q1 : q_total - 3, local_q1 + 1 : q_total] = (
            second_difference
        )
        penalty = difference.T @ difference
    elif code == "lipl":
        difference = _difference_matrix(transformed.shape[1], 1)
        penalty = difference.T @ difference
    elif code == "cpop":
        difference = _cyclic_difference_matrix(q - 1, p_order)
        penalty = difference.T @ difference
    elif code in _BY_SCOP_BASE_CODES:
        base_code = _BY_SCOP_BASE_CODES[code]
        if base_code in {"mpi", "mpd"}:
            core_difference = _difference_matrix(q - 1, 1)
            difference = np.pad(core_difference, ((1, 0), (1, 0)))
        else:
            core_difference = _difference_matrix(q - 2, 1)
            difference = np.pad(core_difference, ((2, 0), (2, 0)))
        penalty = difference.T @ difference
    elif code in {"mdcv", "mdcx", "micv", "micx", "cv", "cx"}:
        difference = _difference_matrix(q - 2, 1)
        penalty = np.zeros((q - 1, q - 1), dtype=np.float64)
        penalty[1:, 1:] = difference.T @ difference
    else:
        difference = _difference_matrix(transformed.shape[1], 1)
        penalty = difference.T @ difference
    if code in {"lmpi", "lipl"}:
        sigma = accumulation
    elif code == "cpop":
        sigma = np.eye(q - 1, dtype=np.float64)
    elif code in _BY_SCOP_BASE_CODES:
        sigma = accumulation
    elif code in _ENDPOINT_ZERO_CODES:
        sigma = accumulation[np.ix_(keep, keep)]
    elif code == "dpo":
        sigma = accumulation[:-1, :-1]
    else:
        sigma = accumulation[1:, 1:]

    if code == "lmpi":
        spacing = float(knot_vector[m + 2] - knot_vector[m + 1])
        derivative_1_full = bspline_design_matrix(
            values, knot_vector, degree=m, deriv=0, extrapolate=True
        )
        derivative_1 = derivative_1_full[:, : raw_basis.shape[1] - 1] / spacing
        derivative_2_full = bspline_design_matrix(
            values, knot_vector, degree=m - 1, deriv=0, extrapolate=True
        )
        derivative_2 = derivative_2_full[:, : raw_basis.shape[1] - 2] / spacing**2
    elif code == "lipl":
        derivative_1 = np.empty((values.size, 0), dtype=np.float64)
        derivative_2 = np.empty((values.size, 0), dtype=np.float64)
    elif code == "cpop":
        derivative_1 = np.empty((values.size, 0), dtype=np.float64)
        derivative_2 = np.empty((values.size, 0), dtype=np.float64)
    else:
        spacing = (float(np.max(values)) - float(np.min(values))) / (q - m - 1)
        if code == "miso":
            spacing = float(knot_vector[q - 1] - knot_vector[q - 2])
        derivative_1_full = bspline_design_matrix(
            values,
            knot_vector,
            degree=m,
            deriv=0,
            extrapolate=True,
        )
        if code in _BY_SCOP_BASE_CODES:
            derivative_1 = derivative_1_full[:, : q - 1] / spacing
        elif code == "miso":
            derivative_1 = derivative_1_full[:, m + 1 : q - 1] / spacing
        elif code == "mifo":
            derivative_1 = derivative_1_full[:, : q - m - 2] / spacing
        else:
            derivative_1 = derivative_1_full[:, 1 : q - 1] / spacing
        if m == 0:
            derivative_2 = np.zeros(
                (values.size, derivative_1.shape[1] - 1), dtype=np.float64
            )
        else:
            derivative_2_full = bspline_design_matrix(
                values,
                knot_vector,
                degree=m - 1,
                deriv=0,
                extrapolate=True,
            )
            derivative_2 = derivative_2_full[:, 1 : q - 2] / spacing**2

    return ShapeConstrainedPSplineSetup(
        basis_code=code,
        spline_order=m,
        penalty_order=p_order,
        bs_dim_requested=q,
        knots=np.asarray(knot_vector, dtype=np.float64),
        basis_train=np.asarray(basis, dtype=np.float64),
        center=np.asarray(center, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        accumulation_matrix=np.asarray(accumulation, dtype=np.float64),
        difference_matrix=np.asarray(difference, dtype=np.float64),
        penalty=np.asarray(penalty, dtype=np.float64),
        positive_mask=(
            np.concatenate(
                [np.zeros(1, dtype=bool), np.ones(q - 1, dtype=bool)]
            )
            if code in _BY_SCOP_BASE_CODES
            else
            np.concatenate(
                [np.zeros(1, dtype=bool), np.ones(keep.size - 1, dtype=bool)]
            )
            if code == "mifo"
            else np.concatenate(
                [
                    np.zeros(1, dtype=bool),
                    np.ones(local_q1 - 1, dtype=bool),
                    np.zeros(raw_basis.shape[1] - local_q1, dtype=bool),
                ]
            )
            if code == "lmpi"
            else np.ones(keep.size, dtype=bool)
        ),
        derivative_basis_1=np.asarray(derivative_1, dtype=np.float64),
        derivative_basis_2=np.asarray(derivative_2, dtype=np.float64),
        rank=(
            transformed.shape[1]
            if code in {"lmpi", "lipl"}
            else q - 2
            if code == "cpop"
            else q
            if code in _BY_SCOP_BASE_CODES
            else (keep.size - 1 if code in _ENDPOINT_ZERO_CODES else q - 2)
        ),
        null_space_dim=(1 if code == "cpop" else 2),
        prediction_keep_indices=keep,
        change_point=(
            float(change_point) if code in {"lmpi", "lipl"} else None
        ),
        constrained_dimension=(
            int(local_q1 if code == "lmpi" else local_q1 - 1)
            if code in {"lmpi", "lipl"}
            else None
        ),
    )


def predict_scop_univariate(x_new, setup: ShapeConstrainedPSplineSetup):
    """SCAM ``Predict.matrix.mpi/mpd.smooth`` value matrix."""
    values = np.asarray(x_new, dtype=np.float64).reshape(-1)
    if setup.basis_code == "cpop":
        lower = float(np.min(setup.knots))
        upper = float(np.max(setup.knots))
        if np.min(values) < lower or np.max(values) > upper:
            values = cyclic_wrap(lower, upper, values)
        raw = _cyclic_pspline_design(
            values, setup.knots, spline_order=setup.spline_order
        )
    else:
        raw = pspline_predict_matrix(
            values,
            setup.knots,
            basis_order=setup.spline_order,
            deriv=0,
        )
    transformed = raw @ setup.accumulation_matrix
    result = np.asarray(
        transformed[:, setup.prediction_keep_indices] - setup.center[None, :],
        dtype=np.float64,
    )
    if setup.basis_code == "mifo":
        # Predict.matrix.mifo.smooth asks spline.des for the derivative at a
        # terminal knot with multiplicity m + 1. R's spline machinery returns
        # NA there, so right-side linear extrapolation is NA as well.
        upper = setup.knots[len(setup.knots) - setup.spline_order - 2]
        result[values > upper, :] = np.nan
    return result


__all__ = [
    "ShapeConstrainedPSplineSetup",
    "build_scop_univariate_setup",
    "predict_scop_univariate",
    "scop_knots",
]
