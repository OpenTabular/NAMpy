"""
Constraint absorption utilities for GAM smooth terms.

These helpers operate on an already-built basis matrix and its associated
penalty matrices.  They implement the coefficient-transform invariant:

    if B is transformed to B @ T,
    then every penalty S must become T.T @ S @ T.

Functions here are used by:
  - runtime terms (gam/smooths/*) for term-local constraint absorption during fit
  - the stage-3 wrapper (gam/design/constructors.py) for delegated absorption
  - gam/constraints/identifiability.py for the centering transform in stage 5

They must not know about compiled predictors, term specs, or fitted results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..penalties import normalize_penalty_spec
from .transforms import (
    apply_coefficient_transform,
    localized_null_space_basis_from_constraint_matrix,
    null_space_basis_from_constraint_matrix,
)


@dataclass
class ConstraintFitResult:
    """
    Return value of :func:`fit_single_penalty_with_constraint_policy`.

    Attributes
    ----------
    basis_train : np.ndarray
        Possibly constrained and by-scaled training basis.
    penalties : list of np.ndarray
        Penalties transformed to match ``basis_train`` coefficient space.
    constraint_kind : str or None
        ``"sum_to_zero"``, ``"factor_by"``, or ``None`` if no constraint absorbed.
    constraint_transform : np.ndarray or None
        Coefficient transform T such that ``basis_train == raw_base @ T``.
        None when no constraint was absorbed.
    """
    basis_train: np.ndarray
    penalties: list
    constraint_kind: str | None
    constraint_transform: np.ndarray | None


def apply_linear_constraint(B, penalties, constraint_row, tol: float = 1e-12):
    """
    Absorb a single linear constraint ``constraint_row @ coef = 0`` into the basis.

    Computes the QR null space of ``constraint_row`` to obtain a transform T such
    that any coefficient vector in the null space satisfies the constraint, then
    returns ``B @ T``, transformed penalties, and T itself.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
    penalties : list of np.ndarray, each shape (d, d)
    constraint_row : array-like, shape (d,)
    tol : float
        Norm threshold below which the constraint is considered trivial.

    Returns
    -------
    Bc : np.ndarray, shape (n, d-1)
    Sc : list of np.ndarray, each shape (d-1, d-1)
    C  : np.ndarray, shape (d, d-1)  — the coefficient transform
    """
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    c = np.asarray(constraint_row, dtype=np.float64).reshape(-1, 1)
    cn = float(np.linalg.norm(c))
    if cn <= tol:
        C = np.eye(B.shape[1], dtype=np.float64)
        return B, penalties, C
    q, _ = np.linalg.qr(c, mode="complete")
    C = q[:, 1:]
    Bc, Sc = apply_coefficient_transform(B, penalties, C)
    return Bc, Sc, C


def full_term_sum_to_zero_constraint(B, penalties):
    B = np.asarray(B, dtype=np.float64)
    mean_row = B.mean(axis=0)
    return apply_linear_constraint(B, penalties, mean_row)


def absorb_explicit_constraints(B, penalty_specs, C, tol: float = 1e-12):
    B = np.asarray(B, dtype=np.float64)
    d = B.shape[1]
    T, n_cons = localized_null_space_basis_from_constraint_matrix(C, d=d, tol=tol)
    mats = [np.asarray(p.matrix, dtype=np.float64) for p in penalty_specs]
    B_new, mats_new = apply_coefficient_transform(B, mats, T)
    out_specs = []
    for p, S_new in zip(penalty_specs, mats_new):
        p_new = copy.copy(p)
        p_new.matrix = S_new
        out_specs.append(normalize_penalty_spec(p_new))
    return B_new, out_specs, T, int(n_cons)


def should_apply_identifiability_constraint(by_state, constraint_mode, *, default_when_auto=False):
    mode = str(constraint_mode).lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "auto":
        return bool(default_when_auto) and bool(by_state.is_constant)
    if mode == "factor_by":
        return False
    raise ValueError("constraint_mode must be one of {'auto', 'factor_by', 'always', 'never'}.")


def fit_single_penalty_with_constraint_policy(base, penalty, by_state, *, constraint_mode, fixed=False, auto_constrain_when=False, apply_numeric_by_fn=None):
    base = np.asarray(base, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    apply_numeric_by_fn = apply_numeric_by_fn or (lambda B, z: B if z is None else B * np.asarray(z)[:, None])

    if str(constraint_mode).lower() == "factor_by":
        if not by_state.is_present:
            raise ValueError("constraint_mode='factor_by' requires a numeric indicator `by` column.")
        base_fb = apply_numeric_by_fn(base, by_state.values)
        penalties_in = [] if bool(fixed) else [penalty]
        mean_row = base_fb.mean(axis=0)
        Bc, Sc, C = apply_linear_constraint(base_fb, penalties_in, mean_row)
        return ConstraintFitResult(
            basis_train=np.asarray(Bc, dtype=np.float64),
            penalties=Sc,
            constraint_kind="factor_by",
            constraint_transform=C,
        )

    should_constrain = should_apply_identifiability_constraint(by_state, constraint_mode, default_when_auto=bool(auto_constrain_when))
    constraint_kind = None
    constraint_transform = None
    if should_constrain:
        penalties_in = [] if bool(fixed) else [penalty]
        mean_row = base.mean(axis=0)
        Bc, Sc, C = apply_linear_constraint(base, penalties_in, mean_row)
        base_out = Bc
        penalties_out = Sc
        constraint_kind = "sum_to_zero"
        constraint_transform = C
    else:
        base_out = base
        penalties_out = [] if bool(fixed) else [penalty]
    base_out = apply_numeric_by_fn(base_out, by_state.values)
    return ConstraintFitResult(
        basis_train=np.asarray(base_out, dtype=np.float64),
        penalties=penalties_out,
        constraint_kind=constraint_kind,
        constraint_transform=constraint_transform,
    )


__all__ = [
    "ConstraintFitResult",
    "apply_linear_constraint",
    "full_term_sum_to_zero_constraint",
    "absorb_explicit_constraints",
    "should_apply_identifiability_constraint",
    "fit_single_penalty_with_constraint_policy",
]
