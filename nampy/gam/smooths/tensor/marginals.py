from __future__ import annotations

import numpy as np

from ..base import column_as_float
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
    if hasattr(term, "_feature_index"):
        return int(term._feature_index)
    if hasattr(term, "_feature_indices") and term._feature_indices is not None:
        return int(term._feature_indices[0])
    raise AttributeError("Tensor marginal term does not expose a resolved feature index.")


def tensor_marginal_feature_name(term):
    if hasattr(term, "_feature_name"):
        return str(term._feature_name)
    if hasattr(term, "_feature_names") and term._feature_names is not None:
        return str(term._feature_names[0])
    raise AttributeError("Tensor marginal term does not expose a resolved feature name.")


def _tensor_marginal_eval_from_x(term, x):
    idx = tensor_marginal_feature_index(term)
    X_new = np.zeros((len(x), idx + 1), dtype=np.float64)
    X_new[:, idx] = np.asarray(x, dtype=np.float64)
    return np.asarray(term.transform_new(X_new), dtype=np.float64)


def _tensor_np_reparameterization(term, x_train, basis_dim):
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    if x_train.size == 0:
        return None
    x_eval = np.linspace(np.min(x_train), np.max(x_train), int(basis_dim), dtype=np.float64)
    B_eval = _tensor_marginal_eval_from_x(term, x_eval)
    U, d, Vt = np.linalg.svd(B_eval, full_matrices=False)
    if d.size == 0 or d[0] <= 0.0:
        return None
    if d[-1] / d[0] < np.finfo(np.float64).eps ** 0.66:
        return None
    return Vt.T @ (U.T / d[:, None])


def tensor_marginal_fit_matrices(term, *, centered=False, apply_np=False, x_train=None):
    basis_name = str(getattr(term, "basis_name", "")).lower()
    if basis_name in {"cr", "cs"} and getattr(term, "_spline", None) is not None:
        if centered:
            B = np.asarray(term.basis_train, dtype=np.float64)
            S = np.asarray(term.penalties[0], dtype=np.float64)
            XP = term._spline._np_transform_centered
        else:
            B = np.asarray(term._spline.raw_basis, dtype=np.float64)
            S = np.asarray(term._spline.raw_penalty_unscaled, dtype=np.float64)
            XP = term._spline._np_transform
        if XP is not None:
            B = B @ XP
            S = XP.T @ S @ XP
        return B, S, XP

    if len(term.penalties) != 1:
        raise NotImplementedError(
            "Tensor products of smooths with multiple penalties are not supported."
        )

    B = np.asarray(term.basis_train, dtype=np.float64)
    S = np.asarray(term.penalties[0], dtype=np.float64)
    XP = _tensor_np_reparameterization(term, x_train, B.shape[1]) if apply_np else None
    if XP is not None:
        B = B @ XP
        S = XP.T @ S @ XP
    return B, S, XP


def tensor_marginal_predict_matrix(term, X_new, *, centered=False, np_transform=None):
    basis_name = str(getattr(term, "basis_name", "")).lower()
    if basis_name in {"cr", "cs"} and getattr(term, "_spline", None) is not None:
        xj = column_as_float(X_new, term._feature_index)
        if centered:
            B = term._spline.transform_new_centered(xj)
            XP = term._spline._np_transform_centered
        else:
            B = term._spline.transform_new_raw(xj)
            XP = term._spline._np_transform
        if XP is not None:
            B = B @ XP
        return np.asarray(B, dtype=np.float64)

    B = np.asarray(term.transform_new(X_new), dtype=np.float64)
    if np_transform is not None:
        B = B @ np.asarray(np_transform, dtype=np.float64)
    return B
