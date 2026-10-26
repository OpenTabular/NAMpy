"""Inequality rows from mgcv's R/scasm.r smooth constructors.

These describe constraints; they do not solve a constrained regression problem.
The separate scasm fitter is needed to enforce ``Ain @ beta >= bin``.
"""

from dataclasses import dataclass

import numpy as np

from .bs import predict_derivative_bspline


@dataclass(frozen=True)
class ShapeInequalities:
    Ain: np.ndarray
    bin: np.ndarray
    bin0: np.ndarray
    skip_centering: bool


def shape_constraint_tokens(xt):
    """Extract the character vector accepted by the upstream xt branch."""
    if xt is None:
        return ()
    if isinstance(xt, dict):
        return ()
    if isinstance(xt, str):
        return (xt,)
    if isinstance(xt, (list, tuple, np.ndarray)):
        values = list(xt)
        if values and isinstance(values[0], (list, tuple, np.ndarray)):
            return ()
        return tuple(value for value in values if isinstance(value, str))
    return ()


def _inequalities(rows, p, tokens):
    matrix = np.concatenate(rows, axis=0) if rows else np.empty((0, p))
    return ShapeInequalities(
        matrix, np.zeros(len(matrix)), np.full(len(matrix), 0.01), "+" in tokens
    )


def sc_inequalities(setup, xt):
    """Mirror ``smooth.construct.sc.smooth.spec`` after its bs construction."""
    con = shape_constraint_tokens(xt)
    p = setup.basis_train.shape[1]
    identity = np.eye(p)
    start, end = setup.knots[setup.degree], setup.knots[-setup.degree - 1]
    rows = []

    def endpoint(x, derivative=0, sign=1):
        rows.append(sign * predict_derivative_bspline([x], setup, deriv=derivative))

    if "c+" in con:
        rows.append(np.diff(identity, n=2, axis=0))
        if "m+" in con:
            endpoint(start, 1)
            if "+" in con:
                endpoint(start)
        elif "m-" in con:
            endpoint(end, 1, -1)
            if "+" in con:
                endpoint(end)
        elif "+" in con:
            rows.append(identity)
    elif "c-" in con:
        rows.append(-np.diff(identity, n=2, axis=0))
        if "m+" in con:
            endpoint(end, 1)
            if "+" in con:
                endpoint(start)
        elif "m-" in con:
            endpoint(start, 1, -1)
            if "+" in con:
                endpoint(end)
        elif "+" in con:
            endpoint(start)
            endpoint(end)
    elif "m+" in con:
        rows.append(np.diff(identity, axis=0))
        if "+" in con:
            endpoint(start)
    elif "m-" in con:
        rows.append(-np.diff(identity, axis=0))
        if "+" in con:
            endpoint(end)
    elif "+" in con:
        rows.append(identity)
    return _inequalities(rows, p, con)


def scad_inequalities(n_coef, xt):
    """Mirror the ordered positive, monotone, then convex scad rows."""
    con = shape_constraint_tokens(xt)
    identity = np.eye(n_coef)
    rows = [identity] if "+" in con else []
    if "m+" in con:
        rows.append(np.diff(identity, axis=0))
    elif "m-" in con:
        rows.append(-np.diff(identity, axis=0))
    if "c+" in con:
        rows.append(np.diff(identity, n=2, axis=0))
    elif "c-" in con:
        rows.append(-np.diff(identity, n=2, axis=0))
    return _inequalities(rows, n_coef, con)
