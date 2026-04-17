from __future__ import annotations

import warnings

import numpy as np

from ...penalties.tensor import normalize_tensor_marginal_penalty
from ..algebra import rowwise_kronecker
from ..smooth_base import column_as_float
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


def make_tensor_marginal_term(
    *,
    feature,
    basis,
    k,
    m=None,
    knots=None,
    centered=False,
    shared_basis_setup=None,
):
    basis = str(basis).lower()
    validate_tensor_marginal_bases([basis])
    constraint_mode = "always" if centered else "never"
    metadata = (
        None
        if shared_basis_setup is None
        else {"shared_basis_setup": shared_basis_setup}
    )

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
            shared_basis_setup=shared_basis_setup,
            knots=knots,
            metadata=metadata,
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
            metadata=metadata,
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
            metadata=metadata,
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
        metadata=metadata,
    )


def _normalize_bool_list(x, n: int):
    if np.isscalar(x):
        return [bool(x)] * n
    vals = [bool(v) for v in x]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {len(vals)}.")
    return vals


def _normalize_tensor_m(m, n: int):
    if m is None:
        return [None] * n
    if isinstance(m, str) or np.isscalar(m):
        return [m] * n

    vals = m.tolist() if isinstance(m, np.ndarray) else list(m)
    nested = any(isinstance(v, (list, tuple, np.ndarray)) for v in vals)

    # Python formula syntax uses lists for both vectors and per-margin lists.
    # Match mgcv's vector behavior for flat length-1 inputs by repeating them.
    if not nested and len(vals) == 1:
        return [vals[0]] * n

    if len(vals) != n:
        warnings.warn("m wrong length and ignored", stacklevel=3)
        return [0] * n

    if nested:
        out = []
        for v in vals:
            if isinstance(v, np.ndarray):
                out.append(v.tolist())
            elif isinstance(v, tuple):
                out.append(list(v))
            else:
                out.append(v)
        return out

    out = []
    for v in vals:
        if v is None:
            out.append(None)
        elif v < 0:
            out.append(0)
        else:
            out.append(v)
    return out


def build_tensor_marginal_terms(
    *,
    feature,
    k,
    basis,
    m=None,
    knots=None,
    centered=False,
    shared_basis_setups=None,
):
    features = list(feature) if not isinstance(feature, (str, int)) else [feature]
    k_list = [int(k)] * len(features) if np.isscalar(k) else [int(v) for v in k]
    if len(k_list) != len(features):
        raise ValueError(
            f"k must have length {len(features)} for features={features}, got {k_list}."
        )
    if isinstance(basis, str):
        basis_list = [str(basis)] * len(features)
    else:
        basis_list = [str(v) for v in basis]
    if len(basis_list) != len(features):
        raise ValueError(
            f"basis must have length {len(features)} for features={features}, got {basis_list}."
        )
    basis_list = validate_tensor_marginal_bases(basis_list)
    knots_list = [knots for _ in features] if np.isscalar(knots) else knots
    if not isinstance(knots_list, (list, tuple)):
        knots_list = [knots_list]
    if len(knots_list) == 1 and len(features) > 1:
        knots_list = [knots_list[0]] * len(features)
    if len(knots_list) != len(features):
        raise ValueError(
            f"knots must have length {len(features)} for features={features}, got {knots_list!r}."
        )
    m_list = _normalize_tensor_m(m, len(features))
    centered_flags = _normalize_bool_list(centered, len(features))
    if shared_basis_setups is None:
        shared_basis_setups = [None] * len(features)
    else:
        shared_basis_setups = list(shared_basis_setups)
        if len(shared_basis_setups) != len(features):
            raise ValueError(
                "shared_basis_setups length mismatch: "
                f"{len(shared_basis_setups)} for {len(features)} features."
            )

    marginals = []
    feature_ids = []
    feature_names = []
    for feat, k_i, bs_i, m_i, knots_i, center_i, shared_i in zip(
        features,
        k_list,
        basis_list,
        m_list,
        knots_list,
        centered_flags,
        shared_basis_setups,
    ):
        term = make_tensor_marginal_term(
            feature=feat,
            basis=bs_i,
            k=k_i,
            m=m_i,
            knots=knots_i,
            centered=center_i,
            shared_basis_setup=shared_i,
        )
        marginals.append(term)
        feature_ids.append(feat)
        feature_names.append(str(feat))
    return marginals, feature_ids, feature_names


def build_tensor_product_components(
    marginals,
    X,
    *,
    use_centered,
    apply_np=True,
):
    use_centered = list(use_centered)
    if len(use_centered) != len(marginals):
        raise ValueError(
            f"centered mask length mismatch: got {len(use_centered)} for {len(marginals)} marginals."
        )

    marginal_setup_bases = []
    marginal_penalties = []
    marginal_np_transforms = []
    marginal_local_bases = []
    for m, center_i in zip(marginals, use_centered):
        x_train = column_as_float(X, tensor_marginal_feature_index(m))
        shared_setup = getattr(m, "shared_basis_setup", None)
        if (
            isinstance(shared_setup, dict)
            and str(shared_setup.get("mode", "")).lower() == "linked_id"
            and shared_setup.get("pooled_feature_values")
        ):
            x_train = np.asarray(
                shared_setup["pooled_feature_values"][0], dtype=np.float64
            ).ravel()
        X_j, S_j, XP_j = tensor_marginal_fit_matrices(
            m, centered=bool(center_i), apply_np=bool(apply_np), x_train=x_train
        )
        S_j = normalize_tensor_marginal_penalty(S_j)
        marginal_setup_bases.append(X_j)
        marginal_penalties.append(S_j)
        marginal_np_transforms.append(XP_j)
        marginal_local_bases.append(
            tensor_marginal_predict_matrix(
                m, X, centered=bool(center_i), np_transform=XP_j
            )
        )
    basis_dims = [int(B.shape[1]) for B in marginal_setup_bases]
    B_raw = rowwise_kronecker(marginal_local_bases)
    B_setup = rowwise_kronecker(marginal_setup_bases)
    return (
        marginal_local_bases,
        marginal_penalties,
        marginal_np_transforms,
        basis_dims,
        B_raw,
        B_setup,
    )


def tensor_predict_matrix(
    marginals,
    X_new,
    *,
    centered=False,
    np_transforms=None,
):
    if np_transforms is None:
        np_transforms = [None] * len(marginals)
    np_transforms = list(np_transforms)
    if len(np_transforms) != len(marginals):
        raise ValueError(
            "np_transforms length mismatch: "
            f"{len(np_transforms)} for {len(marginals)} marginals."
        )
    centered = _normalize_bool_list(centered, len(marginals))
    blocks = []
    for m, center_i, xp in zip(marginals, centered, np_transforms):
        blocks.append(
            tensor_marginal_predict_matrix(m, X_new, centered=center_i, np_transform=xp)
        )
    return rowwise_kronecker(blocks)


def resolve_tensor_marginal_features(marginals):
    feature_indices = []
    feature_names = []
    for term in marginals:
        feature_indices.append(tensor_marginal_feature_index(term))
        feature_names.append(tensor_marginal_feature_name(term))
    return feature_indices, feature_names


def tensor_marginal_feature_index(term):
    idxs = term.resolved_feature_indices()
    if len(idxs) != 1:
        raise AttributeError(
            "Tensor marginal exposes unexpected number of resolved indices."
        )
    return int(idxs[0])


def tensor_marginal_feature_name(term):
    return str(term.resolved_feature_names_list()[0])


def _tensor_marginal_eval_from_x(term, x):
    idx = tensor_marginal_feature_index(term)
    X_new = np.zeros((len(x), idx + 1), dtype=np.float64)
    X_new[:, idx] = np.asarray(x, dtype=np.float64)
    return np.asarray(term.transform_new(X_new), dtype=np.float64)


def _tensor_np_reparameterization(term, x_train, basis_dim):
    if str(getattr(term, "basis_name", "")).lower() in {"cr", "cs", "cc"}:
        return None
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
