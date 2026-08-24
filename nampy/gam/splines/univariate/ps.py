import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from ...linalg import numerical_rank, symmetrize_matrix


def bspline_design_matrix(x, knots, degree, deriv=0, extrapolate=True):
    """
    Dense B-spline design matrix using scipy.interpolate.BSpline.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()

    degree = int(degree)
    deriv = int(deriv)

    n_basis = knots.size - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector / degree combination.")

    if deriv > degree:
        return np.zeros((x.size, n_basis), dtype=np.float64)

    X = np.empty((x.size, n_basis), dtype=np.float64)
    for i in range(n_basis):
        c = np.zeros(n_basis, dtype=np.float64)
        c[i] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=extrapolate)
        if deriv > 0:
            spl = spl.derivative(deriv)
        X[:, i] = spl(x)
    return X


def pspline_knots(x, bs_dim, basis_order, supplied_knots=None):
    """
    mgcv::smooth.construct.ps.smooth.spec-style knot setup.

    basis_order = m[0], penalty order handled separately.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    m1 = int(basis_order)

    nk = int(bs_dim) - m1
    if nk <= 0:
        raise ValueError("basis dimension too small for b-spline order")
    if supplied_knots is None and nk < 2:
        raise ValueError(
            "Automatic P-spline knot construction requires bs.dim > basis_order + 1."
        )

    if supplied_knots is None:
        xl = float(np.min(x))
        xu = float(np.max(x))
        xr = xu - xl
        xl = xl - xr * 0.001
        xu = xu + xr * 0.001
        dx = (xu - xl) / float(nk - 1)
        return np.linspace(
            xl - dx * (m1 + 1),
            xu + dx * (m1 + 1),
            nk + 2 * m1 + 2,
        )

    k = np.asarray(supplied_knots, dtype=np.float64).ravel()
    if k.size == 2:
        xl = float(np.min(k))
        xu = float(np.max(k))
        if xl > np.min(x) or xu < np.max(x):
            raise ValueError("knot range does not include data")
        xr = xu - xl
        xl = xl - xr * 0.001
        xu = xu + xr * 0.001
        if nk < 2:
            raise ValueError(
                "Automatic P-spline knot construction requires bs.dim > basis_order + 1."
            )
        dx = (xu - xl) / float(nk - 1)
        return np.linspace(
            xl - dx * (m1 + 1),
            xu + dx * (m1 + 1),
            nk + 2 * m1 + 2,
        )

    expected = nk + 2 * m1 + 2
    if k.size != expected:
        raise ValueError(f"there should be {expected} supplied knots")
    return k


def pspline_difference_penalty(n_coef, diff_order):
    """
    Difference penalty for a P-spline basis.
    """
    n_coef = int(n_coef)
    diff_order = int(diff_order)

    if diff_order < 0:
        raise ValueError("diff_order must be >= 0")
    if diff_order > n_coef - 1:
        raise ValueError("penalty order too high for basis dimension")

    D = np.eye(n_coef, dtype=np.float64)
    if diff_order > 0:
        D = np.diff(D, n=diff_order, axis=0)
    return D.T @ D


def cyclic_pspline_knots(x, bs_dim, basis_order, supplied_knots=None):
    """Port ``smooth.construct.cp.smooth.spec`` knot handling."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nk = int(bs_dim) + 1
    if nk <= int(basis_order):
        raise ValueError("basis dimension too small for b-spline order")

    if supplied_knots is None:
        lower = float(np.min(x))
        upper = float(np.max(x))
        return np.linspace(lower, upper, nk)

    knots = np.asarray(supplied_knots, dtype=np.float64).ravel()
    if knots.size == 2:
        lower = float(np.min(knots))
        upper = float(np.max(knots))
        if lower > np.min(x) or upper < np.max(x):
            raise ValueError("knot range does not include data")
        return np.linspace(lower, upper, nk)
    if knots.size != nk:
        raise ValueError(f"there should be {nk} supplied knots")
    return knots


def cyclic_wrap(x, lower, upper):
    """Port mgcv's ``cwrap`` mapping onto a cyclic interval."""
    values = np.asarray(x, dtype=np.float64).ravel().copy()
    lower = float(lower)
    upper = float(upper)
    width = upper - lower
    above = values > upper
    if np.any(above):
        values[above] = lower + np.mod(values[above] - upper, width)
    below = values < lower
    if np.any(below):
        values[below] = upper - np.mod(lower - values[below], width)
    return values


def _outer_ok_bspline_design(x, knots, degree, deriv=0):
    """Match ``splines::splineDesign(..., outer.ok=TRUE)`` on all knot spans."""
    values = np.asarray(x, dtype=np.float64).ravel()
    knot_vector = np.asarray(knots, dtype=np.float64).ravel()
    degree = int(degree)
    deriv = int(deriv)
    n_basis = int(knot_vector.size - degree - 1)
    if deriv > degree:
        return np.zeros((values.size, n_basis), dtype=np.float64)
    design = np.zeros((values.size, n_basis), dtype=np.float64)
    for index in range(n_basis):
        basis = BSpline.basis_element(
            knot_vector[index : index + degree + 2], extrapolate=False
        )
        if deriv:
            basis = basis.derivative(deriv)
        design[:, index] = np.nan_to_num(
            basis(values), nan=0.0, posinf=0.0, neginf=0.0
        )
    return design


def cyclic_pspline_design(x, knots, basis_order, deriv=0, *, wrap=False):
    """Operation-for-operation port of mgcv's ``cSplineDes``."""
    values = np.asarray(x, dtype=np.float64).ravel().copy()
    cyclic_knots = np.sort(np.asarray(knots, dtype=np.float64).ravel())
    order = int(basis_order) + 2
    deriv = int(deriv)
    if order < 2:
        raise ValueError("order too low")
    if cyclic_knots.size < order:
        raise ValueError("too few knots")

    lower = float(cyclic_knots[0])
    upper = float(cyclic_knots[-1])
    if wrap and (np.min(values) < lower or np.max(values) > upper):
        values = cyclic_wrap(values, lower, upper)
    if np.min(values) < lower or np.max(values) > upper:
        raise ValueError("x out of range")

    wrap_threshold = float(cyclic_knots[cyclic_knots.size - order])
    prefix = lower - (
        upper - cyclic_knots[cyclic_knots.size - order : -1]
    )
    extended_knots = np.concatenate([prefix, cyclic_knots])
    design = _outer_ok_bspline_design(
        values,
        extended_knots,
        degree=order - 1,
        deriv=deriv,
    )
    wrapped_rows = values > wrap_threshold
    if np.any(wrapped_rows):
        shifted = values[wrapped_rows] - upper + lower
        design[wrapped_rows, :] += _outer_ok_bspline_design(
            shifted,
            extended_knots,
            degree=order - 1,
            deriv=deriv,
        )
    return np.asarray(design, dtype=np.float64)


def cyclic_pspline_difference_penalty(n_coef, diff_order):
    """Port the wrapped difference penalty in the cyclic P-spline constructor."""
    n_coef = int(n_coef)
    diff_order = int(diff_order)
    if diff_order < 0:
        raise ValueError("diff_order must be >= 0")
    if diff_order > n_coef - 1:
        raise ValueError("penalty order too high for basis dimension")

    expanded = np.eye(n_coef + diff_order, dtype=np.float64)
    for _ in range(diff_order):
        expanded = np.diff(expanded, axis=0)
    if diff_order == 0:
        difference = expanded
    else:
        difference = expanded[:, diff_order:].copy()
        difference[:, n_coef - diff_order : n_coef] += expanded[:, :diff_order]
    return difference.T @ difference


def pspline_predict_matrix(x, knots, basis_order, deriv=0):
    """
    mgcv::Predict.matrix.pspline.smooth analogue.

    Uses ordinary B-spline evaluation inside the inner knot range and
    linear extrapolation of the smooth outside that range.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    deriv = int(deriv)

    degree = int(basis_order) + 1
    n_basis = knots.size - degree - 1

    if deriv > degree:
        return np.zeros((x.size, n_basis), dtype=np.float64)

    ll = knots[int(basis_order) + 1]
    ul = knots[len(knots) - int(basis_order) - 2]

    X = np.zeros((x.size, n_basis), dtype=np.float64)
    inside = (x >= ll) & (x <= ul)

    if np.any(inside):
        X[inside, :] = bspline_design_matrix(
            x[inside],
            knots,
            degree=degree,
            deriv=deriv,
            extrapolate=True,
        )

    if np.all(inside):
        return X

    if deriv >= 2:
        return X

    B_ll = bspline_design_matrix(
        np.array([ll], dtype=np.float64),
        knots,
        degree=degree,
        deriv=0,
        extrapolate=True,
    )[0]
    dB_ll = bspline_design_matrix(
        np.array([ll], dtype=np.float64),
        knots,
        degree=degree,
        deriv=1,
        extrapolate=True,
    )[0]
    B_ul = bspline_design_matrix(
        np.array([ul], dtype=np.float64),
        knots,
        degree=degree,
        deriv=0,
        extrapolate=True,
    )[0]
    dB_ul = bspline_design_matrix(
        np.array([ul], dtype=np.float64),
        knots,
        degree=degree,
        deriv=1,
        extrapolate=True,
    )[0]

    left = x < ll
    if np.any(left):
        if deriv == 0:
            X[left, :] = B_ll[None, :] + (x[left] - ll)[:, None] * dB_ll[None, :]
        else:
            X[left, :] = np.broadcast_to(dB_ll, (np.sum(left), n_basis))

    right = x > ul
    if np.any(right):
        if deriv == 0:
            X[right, :] = B_ul[None, :] + (x[right] - ul)[:, None] * dB_ul[None, :]
        else:
            X[right, :] = np.broadcast_to(dB_ul, (np.sum(right), n_basis))

    return X


@dataclass
class PSplineBasisSetup:
    feature_index: int
    feature_name: str
    basis_order: int
    penalty_order: int
    knots: np.ndarray
    basis_train: np.ndarray
    penalty: np.ndarray
    bs_dim: int
    rank: int
    basis_name: str = "ps"
    orders: tuple[int, ...] = ()


def build_pspline_term_setup(
    x,
    *,
    feature_index,
    feature_name,
    bs_dim,
    m,
    knots=None,
    basis="ps",
):
    x = np.asarray(x, dtype=np.float64).ravel()
    basis_order, penalty_order = (int(m[0]), int(m[1]))
    basis_name = str(basis).lower()
    if basis_name not in {"ps", "cp"}:
        raise ValueError(f"Unsupported P-spline basis {basis!r}.")
    if basis_order < 0 or penalty_order < 0:
        raise ValueError(f"For bs={basis_name!r}, m entries must be >= 0.")

    if basis_name == "cp":
        k = cyclic_pspline_knots(
            x,
            bs_dim=int(bs_dim),
            basis_order=basis_order,
            supplied_knots=knots,
        )
        B = cyclic_pspline_design(x, k, basis_order, deriv=0)
        if np.any(np.sum(B, axis=0) == 0.0):
            warnings.warn(
                "knot range is so wide that there is *no* information about some "
                "basis coefficients",
                stacklevel=2,
            )
        S = cyclic_pspline_difference_penalty(B.shape[1], penalty_order)
        rank = int(B.shape[1] - 1)
    else:
        k = pspline_knots(
            x,
            bs_dim=int(bs_dim),
            basis_order=basis_order,
            supplied_knots=knots,
        )
        degree = basis_order + 1
        B = bspline_design_matrix(
            x,
            k,
            degree=degree,
            deriv=0,
            extrapolate=True,
        )
        S = pspline_difference_penalty(B.shape[1], penalty_order)
        rank = numerical_rank(S, hermitian=True)
    S = symmetrize_matrix(S)

    return PSplineBasisSetup(
        feature_index=int(feature_index),
        feature_name=str(feature_name),
        basis_order=int(basis_order),
        penalty_order=int(penalty_order),
        knots=np.asarray(k, dtype=np.float64),
        basis_train=np.asarray(B, dtype=np.float64),
        penalty=np.asarray(S, dtype=np.float64),
        bs_dim=int(B.shape[1]),
        rank=rank,
        basis_name=basis_name,
        orders=tuple(int(value) for value in m),
    )


def predict_pspline_term(x_new, setup: PSplineBasisSetup):
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    if str(setup.basis_name).lower() == "cp":
        return cyclic_pspline_design(
            x_new,
            setup.knots,
            basis_order=setup.basis_order,
            deriv=0,
            wrap=True,
        )
    return np.asarray(
        pspline_predict_matrix(
            x_new,
            setup.knots,
            basis_order=setup.basis_order,
            deriv=0,
        ),
        dtype=np.float64,
    )


def predict_pspline_term_derivative(x_new, setup: PSplineBasisSetup, deriv=1):
    """Evaluate an ordinary or cyclic P-spline derivative matrix."""
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    if str(setup.basis_name).lower() == "cp":
        return cyclic_pspline_design(
            x_new,
            setup.knots,
            basis_order=setup.basis_order,
            deriv=deriv,
            wrap=True,
        )
    return pspline_predict_matrix(
        x_new,
        setup.knots,
        basis_order=setup.basis_order,
        deriv=deriv,
    )
