"""Bivariate SCOP-spline primitives ported from SCAM.

The constructors in ``scam/R/bivar.smooth.const.R`` use row tensor products
of equally-spaced P-spline marginals, followed by basis-specific cumulative
coefficient maps. This module owns those identified constructor coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..univariate.ps import bspline_design_matrix, pspline_predict_matrix
from .scop import scop_knots


@dataclass(frozen=True)
class BivariateShapePSplineSetup:
    basis_code: str
    spline_orders: tuple[int, int]
    basis_dimensions: tuple[int, int]
    knots: tuple[np.ndarray, np.ndarray]
    basis_train: np.ndarray
    center: np.ndarray
    sigma: np.ndarray
    constraint_matrix: np.ndarray
    penalties: tuple[np.ndarray, ...]
    positive_mask: np.ndarray
    ranks: tuple[int, ...]
    null_space_dim: int
    drop_first: bool

    @property
    def n_coef(self) -> int:
        return int(self.basis_train.shape[1])


def _pair(value, *, default: int) -> tuple[int, int]:
    if value is None:
        return int(default), int(default)
    values = np.asarray(value).reshape(-1)
    if values.size == 1:
        parsed = int(default) if int(values[0]) < 0 else int(values[0])
        return parsed, parsed
    if values.size != 2:
        raise ValueError("Bivariate SCAM k/m arguments must be scalar or length two.")
    return tuple(
        int(default) if int(item) < 0 else int(item) for item in values
    )


def _row_tensor(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nj->nij", left, right).reshape(left.shape[0], -1)


def _marginal_basis(values, knots, order, *, prediction: bool) -> np.ndarray:
    if prediction:
        return pspline_predict_matrix(values, knots, basis_order=order, deriv=0)
    return bspline_design_matrix(
        values, knots, degree=int(order) + 1, deriv=0, extrapolate=True
    )


def _first_difference_root(dimension: int) -> np.ndarray:
    difference = np.diff(np.eye(dimension - 1, dtype=np.float64), axis=0)
    root = np.zeros((dimension - 1, dimension), dtype=np.float64)
    root[1:, 1:] = difference
    return root


def _double_monotone_state(code: str, q1: int, q2: int):
    lower1 = np.tril(np.ones((q1, q1), dtype=np.float64))
    lower2 = np.tril(np.ones((q2, q2), dtype=np.float64))
    if code == "tedmd":
        lower2 *= -1.0
    sigma = np.kron(lower1, lower2)
    if code == "tedmd":
        sigma[:, 0] = 1.0

    root1 = np.kron(_first_difference_root(q1), np.eye(q2))
    root2 = np.kron(np.eye(q1), _first_difference_root(q2))
    roots = (root1[1:, 1:], root2[1:, 1:])
    penalties = tuple(root.T @ root for root in roots)
    return sigma, penalties


def _second_difference_root(dimension: int) -> np.ndarray:
    difference = np.diff(np.eye(dimension - 2, dtype=np.float64), axis=0)
    root = np.zeros((dimension - 1, dimension), dtype=np.float64)
    root[2:, 2:] = difference
    return root


def _curvature_accumulation(dimension: int, *, convex: bool) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[:, 0] = 1.0
    sign = -1.0 if convex else 1.0
    matrix[1:, 1] = sign * np.arange(1, dimension, dtype=np.float64)
    for column in range(2, dimension):
        values = np.arange(1, dimension - column + 1, dtype=np.float64)
        matrix[column:, column] = -sign * values
    return matrix


def _mixed_monotone_curvature_state(code: str, q1: int, q2: int):
    increasing = code.startswith("temi")
    convex = code in {"temicx", "tedecv"}
    monotone = np.tril(np.ones((q1, q1), dtype=np.float64))
    if not increasing:
        monotone *= -1.0
    curvature = _curvature_accumulation(q2, convex=convex)
    sigma = np.kron(monotone, curvature)

    root1 = np.kron(_first_difference_root(q1), np.eye(q2))
    root2 = np.kron(np.eye(q1), _second_difference_root(q2))
    roots = (root1[1:, 1:], root2[1:, 1:])
    penalties = tuple(root.T @ root for root in roots)
    return sigma, penalties


def _double_curvature_state(code: str, q1: int, q2: int):
    first_convex = code in {"tecxcx", "tecxcv"}
    second_convex = code == "tecxcx"
    sigma = np.kron(
        _curvature_accumulation(q1, convex=first_convex),
        _curvature_accumulation(q2, convex=second_convex),
    )
    # Preserve scam/R/bivar.smooth.const.R: the first marginal root is the
    # embedded first-difference root even for these curvature constructors.
    root1 = np.kron(_first_difference_root(q1), np.eye(q2))
    root2 = np.kron(np.eye(q1), _second_difference_root(q2))
    roots = (root1[1:, 1:], root2[1:, 1:])
    return sigma, tuple(root.T @ root for root in roots)


def _single_curvature_state(code: str, q1: int, q2: int):
    sigma = np.kron(
        np.eye(q1),
        _curvature_accumulation(q2, convex=code == "tescx"),
    )
    root1 = np.kron(np.diff(np.eye(q1), axis=0), np.eye(q2))
    stencil = np.kron(
        np.array([1.0, -2.0, 1.0]),
        np.concatenate([[1.0], np.zeros(q2 - 1)]),
    )
    for index in range(q1 - 2):
        row = q2 * index
        root1[row, row : row + stencil.size] = stencil
    row = q2 * (q1 - 2)
    root1[row, row:] = 0.0
    root2 = np.kron(np.eye(q1), _second_difference_root(q2))
    raw_penalties = (root1.T @ root1, root2.T @ root2)

    dimension = q1 * q2
    constraint = np.delete(np.eye(dimension), 0, axis=1)
    free_indices = np.arange(q1 - 1) * q2
    for index in free_indices:
        constraint[index, index] = -1.0
    for index in free_indices[1:]:
        constraint[index, index - q2] = 1.0
    constraint[(q1 - 1) * q2, (q1 - 2) * q2] = 1.0
    penalties = tuple(constraint.T @ value @ constraint for value in raw_penalties)
    mask = np.ones(dimension - 1, dtype=bool)
    mask[free_indices] = False
    return sigma, constraint, penalties, mask


def _single_monotone_first_state(code: str, q1: int, q2: int):
    accumulation = np.tril(np.ones((q1, q1), dtype=np.float64))
    if code == "tesmd1":
        accumulation *= -1.0
        accumulation[:, 0] = 1.0
    sigma = np.kron(accumulation, np.eye(q2))

    root1 = np.kron(_first_difference_root(q1), np.eye(q2))
    second_first = np.kron(
        np.eye(q1)[[0], :], np.diff(np.eye(q2), n=2, axis=0)
    )
    second_rest = np.kron(
        np.eye(q1)[1:, :], np.diff(np.eye(q2), axis=0)
    )
    root2 = np.vstack([second_first, second_rest])
    raw_penalties = (root1.T @ root1, root2.T @ root2)

    dimension = q1 * q2
    constraint = np.delete(np.eye(dimension), q2 - 1, axis=1)
    constraint[:q2, : q2 - 1] = np.diff(np.eye(q2), axis=0).T
    penalties = tuple(constraint.T @ value @ constraint for value in raw_penalties)
    mask = np.ones(dimension - 1, dtype=bool)
    mask[: q2 - 1] = False
    return sigma, constraint, penalties, mask


def _single_monotone_second_state(code: str, q1: int, q2: int):
    accumulation = np.tril(np.ones((q2, q2), dtype=np.float64))
    if code == "tesmd2":
        accumulation *= -1.0
        accumulation[:, 0] = 1.0
    sigma = np.kron(np.eye(q1), accumulation)

    root1 = np.kron(np.diff(np.eye(q1), axis=0), np.eye(q2))
    stencil = np.kron(
        np.array([1.0, -2.0, 1.0]),
        np.concatenate([[1.0], np.zeros(q2 - 1)]),
    )
    for index in range(q1 - 2):
        row = q2 * index
        root1[row, row : row + stencil.size] = stencil
    row = q2 * (q1 - 2)
    root1[row, row:] = 0.0
    root2 = np.kron(np.eye(q1), _first_difference_root(q2))
    raw_penalties = (root1.T @ root1, root2.T @ root2)

    dimension = q1 * q2
    deleted = (q1 - 1) * q2
    constraint = np.delete(np.eye(dimension), deleted, axis=1)
    free_indices = np.arange(q1 - 1) * q2
    for index in free_indices:
        constraint[index, index] = -1.0
    for index in free_indices[1:]:
        constraint[index, index - q2] = 1.0
    constraint[deleted, deleted - q2] = 1.0
    penalties = tuple(constraint.T @ value @ constraint for value in raw_penalties)
    mask = np.ones(dimension - 1, dtype=bool)
    mask[free_indices] = False
    return sigma, constraint, penalties, mask


def _monotone_interaction_state(code: str, q1: int, q2: int):
    accumulation = np.tril(np.ones((q1, q1), dtype=np.float64))
    if code == "tismd":
        accumulation *= -1.0
        accumulation[:, 0] = 1.0
    sigma = np.kron(accumulation, np.eye(q2))
    selected = [
        first * q2 + second
        for first in range(1, q1)
        for second in range(1, q2)
    ]
    constraint = np.eye(q1 * q2)[:, selected]

    root1 = np.kron(
        np.diff(np.eye(q1 - 1), axis=0), np.eye(q2 - 1)
    )
    root2 = np.vstack(
        [
            np.kron(
                np.eye(q1 - 1)[[0], :],
                np.diff(np.eye(q2 - 1), n=2, axis=0),
            ),
            np.kron(
                np.eye(q1 - 1)[1:, :],
                np.diff(np.eye(q2 - 1), axis=0),
            ),
        ]
    )
    penalties = (root1.T @ root1, root2.T @ root2)
    mask = np.ones((q1 - 1) * (q2 - 1), dtype=bool)
    if code == "tismi":
        mask[: q2 - 1] = False
    return sigma, constraint, penalties, mask


def build_bivariate_shape_setup(
    x,
    z,
    *,
    basis_code: str,
    bs_dim=7,
    spline_order=2,
    knots=None,
) -> BivariateShapePSplineSetup:
    """Build one of SCAM's identified bivariate shape constructors."""
    code = str(basis_code).lower()
    algebraic_codes = {
        "tedmi",
        "tedmd",
        "temicx",
        "temicv",
        "tedecv",
        "tedecx",
        "tecvcv",
        "tecxcx",
        "tecxcv",
        "tescv",
        "tescx",
        "tesmi1",
        "tesmd1",
        "tesmi2",
        "tesmd2",
        "tismi",
        "tismd",
    }
    if code not in algebraic_codes:
        raise NotImplementedError(
            f"Bivariate SCAM constructor {code!r} is not implemented yet."
        )
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if x.shape != z.shape:
        raise ValueError("Arguments of a bivariate smooth must have equal length.")
    q1, q2 = _pair(bs_dim, default=7)
    m1, m2 = _pair(spline_order, default=2)
    supplied = (None, None) if knots is None else tuple(knots)
    if len(supplied) != 2:
        raise ValueError("Bivariate SCAM knots must contain two marginal vectors.")
    knots1 = scop_knots(x, bs_dim=q1, spline_order=m1, supplied_knots=supplied[0])
    knots2 = scop_knots(z, bs_dim=q2, spline_order=m2, supplied_knots=supplied[1])
    marginal1 = _marginal_basis(x, knots1, m1, prediction=False)
    marginal2 = _marginal_basis(z, knots2, m2, prediction=False)
    raw = _row_tensor(marginal1, marginal2)
    dimension = q1 * q2
    constraint = np.delete(np.eye(dimension), 0, axis=1)
    positive_mask = np.ones(dimension - 1, dtype=bool)
    if code in {"tedmi", "tedmd"}:
        sigma, penalties = _double_monotone_state(code, q1, q2)
    elif code in {"temicx", "temicv", "tedecv", "tedecx"}:
        sigma, penalties = _mixed_monotone_curvature_state(code, q1, q2)
    elif code in {"tecvcv", "tecxcx", "tecxcv"}:
        sigma, penalties = _double_curvature_state(code, q1, q2)
    elif code in {"tescv", "tescx"}:
        sigma, constraint, penalties, positive_mask = _single_curvature_state(
            code, q1, q2
        )
    elif code in {"tesmi1", "tesmd1"}:
        sigma, constraint, penalties, positive_mask = _single_monotone_first_state(
            code, q1, q2
        )
    elif code in {"tesmi2", "tesmd2"}:
        sigma, constraint, penalties, positive_mask = _single_monotone_second_state(
            code, q1, q2
        )
    else:
        sigma, constraint, penalties, positive_mask = _monotone_interaction_state(
            code, q1, q2
        )
    transformed = raw @ sigma @ constraint
    center = (
        np.mean(transformed, axis=0)
        if code in {"tedmi", "tedmd", "tecvcv", "tecxcx"}
        else np.zeros(transformed.shape[1], dtype=np.float64)
    )
    basis = transformed - center[None, :]
    rank = int(basis.shape[1] - 1)
    return BivariateShapePSplineSetup(
        basis_code=code,
        spline_orders=(m1, m2),
        basis_dimensions=(q1, q2),
        knots=(knots1, knots2),
        basis_train=np.asarray(basis, dtype=np.float64),
        center=np.asarray(center, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        constraint_matrix=np.asarray(constraint, dtype=np.float64),
        penalties=tuple(np.asarray(value, dtype=np.float64) for value in penalties),
        positive_mask=positive_mask,
        ranks=(rank, rank),
        null_space_dim=3,
        drop_first=bool(np.array_equal(constraint, np.delete(np.eye(dimension), 0, axis=1))),
    )


def predict_bivariate_shape(x, z, setup: BivariateShapePSplineSetup) -> np.ndarray:
    """Evaluate the released SCAM bivariate prediction matrix."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if x.shape != z.shape:
        raise ValueError("Arguments of a bivariate smooth must have equal length.")
    marginal1 = _marginal_basis(
        x, setup.knots[0], setup.spline_orders[0], prediction=True
    )
    marginal2 = _marginal_basis(
        z, setup.knots[1], setup.spline_orders[1], prediction=True
    )
    raw = _row_tensor(marginal1, marginal2)
    transformed = raw @ setup.sigma
    if setup.drop_first:
        transformed = transformed - np.concatenate([[0.0], setup.center])[None, :]
    return np.asarray(transformed, dtype=np.float64)


__all__ = [
    "BivariateShapePSplineSetup",
    "build_bivariate_shape_setup",
    "predict_bivariate_shape",
]
