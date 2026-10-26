"""Extended-family gam.fit3.post.proc uncertainty with joint theta blocks."""

from __future__ import annotations

import numpy as np

from ...linalg import symmetrize_matrix
from ...linalg.qr import mgcv_pqr_r
from ...model_state import _coef_column_offset, _n_smoothing_params, _penalty_blocks_seq
from ..parameterization import FIT_PARAMETER_SPACE
from ..selection.criteria.pirls.derivatives import (
    _gdi1_kernel,
    gdi2_theta_joint_kernel,
)
from ..solvers.general_family.newton import _vb_corr_root
from .unconditional_covariance import _restore_pirls_dbeta_to_original_parameterization


def extended_unconditional_postfit(model, sol, fit_result, fit_state):
    """Port the joint-theta branch of gam.fit3.post.proc and Vb.corr.

    The latter is called with w=NULL upstream: only penalty derivatives enter
    its Hessian perturbations, including zero blocks for the family parameters.
    This differs from the full observed-Hessian derivatives used for db.drho.
    """
    unavailable = (None, None, FIT_PARAMETER_SPACE)
    if str(getattr(model.family, "name", "")).lower() not in {"scat", "zip", "cnorm"}:
        return unavailable
    method = str(getattr(model, "_optim_method", "")).upper()
    if method not in {"ML", "REML", "LAML"}:
        return unavailable
    if method == "LAML":
        method = "REML"
    nsp = int(_n_smoothing_params(model) or 0)
    fixed = (
        np.zeros(nsp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free = np.flatnonzero(~fixed)
    ntheta = int(getattr(model.family, "n_theta", 0))
    count = ntheta + free.size
    if count == 0:
        return unavailable
    result = getattr(model, "_optim_result", None)
    info = dict(getattr(result, "outer_info", {}) or {})
    hessian = info.get("hess", None)
    if hessian is None:
        return unavailable
    hessian = np.asarray(hessian, dtype=float)
    if hessian.shape != (count, count) or not np.isfinite(hessian).all():
        return unavailable
    dropped = getattr(fit_state, "dropped_column_indices", None)
    if dropped is not None and np.asarray(dropped).size:
        # Vb.corr applies unpivoted Cholesky to the original full Hessian.
        # With dropped coefficients its success depends on rounding: identical
        # serialized R matrices can succeed in R and fail in SciPy, changing
        # identifiable prediction covariance and EDF2, not just null entries.
        raise NotImplementedError(
            f"{model.family.name} smoothing uncertainty with rank-deficient "
            "coefficients lacks portable mgcv parity. Remove redundant "
            "parametric columns before estimating ancillary or smoothing parameters."
        )
    sp = (
        np.asarray(model.smoothing_params, dtype=float).ravel()
        if nsp
        else np.empty(0, dtype=float)
    )
    family_y = getattr(model, "response_", model.y_)
    if ntheta:
        kernel = gdi2_theta_joint_kernel(
            model, family_y, sol, sp, method=method, need_hessian=False
        )
        current = kernel.gdi1.current
        derivatives = kernel.ift.dbeta
        keep = np.r_[np.arange(ntheta), ntheta + free]
    else:
        kernel = _gdi1_kernel(model, family_y, sol, sp, method=method)
        current = kernel.current
        derivatives = kernel.ift.dbeta
        keep = free
    dbeta = np.column_stack(
        [
            _restore_pirls_dbeta_to_original_parameterization(
                current, derivatives[index]
            )
            for index in keep
        ]
    )
    eigenvalues, vectors = np.linalg.eigh(symmetrize_matrix(hessian))
    positive = eigenvalues > 0
    inverse_root = np.zeros_like(eigenvalues)
    inverse_root[positive] = 1 / np.sqrt(eigenvalues[positive])
    root = inverse_root[:, None] * vectors.T
    correction1 = (root @ dbeta.T).T @ (root @ dbeta.T)
    regularized = 1 / np.sqrt(np.maximum(eigenvalues, 0) + 0.1)
    regularized_root = regularized[:, None] * vectors.T
    smoothing_covariance = regularized_root.T @ regularized_root

    # Original coefficient-space penalties, with linked contributions summed
    # into their common working log-smoothing-parameter derivative.
    conditional = np.asarray(fit_result.cov_bayes, dtype=float)
    p = conditional.shape[0]
    offset = int(_coef_column_offset(model))
    penalty_derivatives = [np.zeros((p, p)) for _ in range(nsp)]
    for block in _penalty_blocks_seq(model):
        index = int(block.smoothing_index)
        section = slice(block.coef_slice.start + offset, block.coef_slice.stop + offset)
        penalty_derivatives[index][section, section] += sp[index] * np.asarray(
            block.matrix
        )
    penalty = sum(penalty_derivatives, start=np.zeros((p, p)))
    dH = [np.zeros((p, p)) for _ in range(ntheta)]
    dH.extend(penalty_derivatives[index] for index in free)
    weights = np.asarray(fit_state.fisher_weights, dtype=float)
    X = np.asarray(fit_state.X, dtype=float)
    weighted_root = mgcv_pqr_r(np.sqrt(weights)[:, None] * X)
    rho = np.r_[
        np.asarray(model.family.getTheta(False)).ravel()[:ntheta], np.log(sp[free])
    ]
    scale = float(fit_result.scale)
    correction2 = scale * _vb_corr_root(
        weighted_root,
        L=None,
        lsp0=None,
        S_blocks=[],
        off=None,
        rho=rho,
        Vr=smoothing_covariance,
        penalty_matrix=penalty,
        dH_blocks=dH,
    )
    covariance = symmetrize_matrix(conditional + correction1 + correction2)
    edf2 = np.sum(covariance * (weighted_root.T @ weighted_root), axis=1) / scale
    F = np.asarray(fit_result.H_coef, dtype=float)
    edf1 = 2 * np.diag(F) - np.sum(F * F.T, axis=1)
    if np.sum(edf2) > np.sum(edf1):
        edf2 = edf1.copy()
    return covariance, edf2, FIT_PARAMETER_SPACE
