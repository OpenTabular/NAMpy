from __future__ import annotations

import numpy as np

from ..univariate.cubic_regression import SplineTerm1D
from ..univariate.gp import GPSmoothTerm
from ..univariate.pspline import PSplineTerm1D
from ..univariate.thin_plate import ThinPlateSplineTerm

TENSOR_MARGINAL_BASES = frozenset({"cr", "cs", "cc", "ps", "tp", "ts", "gp"})


def validate_tensor_marginal_bases(bases):
    bases = [str(b).lower() for b in bases]
    bad = [b for b in bases if b not in TENSOR_MARGINAL_BASES]
    if bad:
        raise NotImplementedError(
            "Tensor marginals currently support only bs in "
            f"{sorted(TENSOR_MARGINAL_BASES)}, got {bases!r}."
        )
    return bases


def make_tensor_marginal_term(*, feature, basis, k, m=None, knots=None, centered=False):
    basis = str(basis).lower()
    validate_tensor_marginal_bases([basis])
    constraint_mode = "always" if centered else "never"

    if basis in {"cr", "cs", "cc"}:
        return SplineTerm1D(
            feature=feature,
            k=k,
            basis=basis,
            label=str(feature),
            smoothing_id=None,
            by=None,
            select=False,
            fixed=False,
            constraint_mode=constraint_mode,
            knots=knots,
        )

    if basis == "ps":
        return PSplineTerm1D(
            feature=feature,
            k=k,
            basis=basis,
            m=m,
            label=str(feature),
            smoothing_id=None,
            by=None,
            select=False,
            fixed=False,
            constraint_mode=constraint_mode,
            knots=knots,
        )

    if basis in {"tp", "ts"}:
        return ThinPlateSplineTerm(
            feature=[feature],
            k=k,
            basis=basis,
            m=m,
            label=str(feature),
            smoothing_id=None,
            by=None,
            select=False,
            fixed=False,
            constraint_mode=constraint_mode,
            knots=knots,
        )

    return GPSmoothTerm(
        feature=[feature],
        k=k,
        basis=basis,
        m=m,
        label=str(feature),
        smoothing_id=None,
        by=None,
        select=False,
        fixed=False,
        constraint_mode=constraint_mode,
        knots=knots,
    )


def tensor_marginal_feature_index(term):
    return int(term.resolved_feature_indices()[0])


def tensor_marginal_feature_name(term):
    return str(term.resolved_feature_names_list()[0])


def _tensor_marginal_eval_from_x(term, x):
    idx = tensor_marginal_feature_index(term)
    X_new = np.zeros((len(x), idx + 1), dtype=np.float64)
    X_new[:, idx] = np.asarray(x, dtype=np.float64)
    return np.asarray(term.transform_new(X_new), dtype=np.float64)


def _tensor_np_reparameterization(term, x_train, basis_dim):
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    if x_train.size == 0:
        return None
    x_eval = np.linspace(
        np.min(x_train), np.max(x_train), int(basis_dim), dtype=np.float64
    )
    B_eval = _tensor_marginal_eval_from_x(term, x_eval)
    U, d, Vt = np.linalg.svd(B_eval, full_matrices=False)
    if d.size == 0 or d[0] <= 0.0:
        return None
    if d[-1] / d[0] < np.finfo(np.float64).eps ** 0.66:
        return None
    return Vt.T @ (U.T / d[:, None])


def tensor_marginal_fit_matrices(term, *, centered=False, apply_np=False, x_train=None):
    B, S, XP = term.tensor_marginal_fit_matrices(
        centered=centered,
        apply_np=False,
        x_train=x_train,
    )
    XP = _tensor_np_reparameterization(term, x_train, B.shape[1]) if apply_np else None
    if XP is not None:
        B = B @ XP
        S = XP.T @ S @ XP
    return B, S, XP


def tensor_marginal_predict_matrix(term, X_new, *, centered=False, np_transform=None):
    return term.tensor_marginal_predict_matrix(
        X_new,
        centered=centered,
        np_transform=np_transform,
    )
