"""Unconditional covariance / EDF post-fit assembly."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_triangular

from ..._model_state import _n_smoothing_params
from ...linalg import symmetrize_matrix
from ...linalg.qr import mgcv_pqr_r
from ...linalg.reindexing import (
    permute_rows,
    restore_dropped_rows,
)
from ...results import FitResult
from ..parameterization import (
    FIT_PARAMETER_SPACE,
    PREDICTION_PARAMETER_SPACE,
    prediction_parameterization_map,
)

if TYPE_CHECKING:
    from ..state import FitState


def _differentiate_cholesky_factor(dA: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Mirror mgcv/src/mat.c::dchol() for upper-Cholesky factors."""
    dA = np.asarray(dA, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    p = int(R.shape[0])
    dR = np.zeros_like(R, dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            x = 0.0
            for k in range(i):
                x += R[k, i] * dR[k, j] + R[k, j] * dR[k, i]
            if j > i:
                dR[i, j] = (dA[i, j] - x - R[i, j] * dR[i, i]) / R[i, i]
            else:
                dR[i, i] = 0.5 * (dA[i, i] - x) / R[i, i]
    return dR


def _covariance_from_cholesky_derivatives(
    dR_list: list[np.ndarray], Vr: np.ndarray, *, trans: bool
) -> np.ndarray:
    """Mirror mgcv/R/misc.r::vcorr() for dense NumPy arrays."""
    if len(dR_list) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros_like(np.asarray(dR_list[0], dtype=np.float64), dtype=np.float64)
    Vr = np.asarray(Vr, dtype=np.float64)
    for i, dRi in enumerate(dR_list):
        dRi = np.asarray(dRi, dtype=np.float64)
        for j, dRj in enumerate(dR_list):
            w = float(Vr[i, j])
            if w == 0.0:
                continue
            out += w * (dRi.T @ dRj if trans else dRi @ dRj.T)
    return symmetrize_matrix(out)


def _restore_pirls_dbeta_to_original_parameterization(
    current, dbeta_rank
) -> np.ndarray:
    """Undo rank drop/pivot + `T` reparameterization for `gam.fit3`-style `db.drho`."""
    packed = np.asarray(dbeta_rank, dtype=np.float64).reshape(-1, 1)
    pivot1 = np.asarray(current.pivot1, dtype=np.int64)
    dropped = np.asarray(current.dropped_column_indices, dtype=np.int64)
    canonical_T = np.asarray(current.canonical.T, dtype=np.float64)

    unpermuted = permute_rows(packed, pivot1, reverse=True)
    full_canonical = restore_dropped_rows(
        unpermuted,
        int(canonical_T.shape[1]),
        dropped,
    )
    return np.asarray(canonical_T @ full_canonical, dtype=np.float64).ravel()


def _restore_pirls_rank_root_to_original_parameterization(
    current, rank_root: np.ndarray
) -> np.ndarray:
    """Undo rank drop/pivot + `T` reparameterization for `gam.fit3`-style `rV` roots."""
    rank_root = np.asarray(rank_root, dtype=np.float64)
    pivot1 = np.asarray(current.pivot1, dtype=np.int64)
    dropped = np.asarray(current.dropped_column_indices, dtype=np.int64)
    canonical_T = np.asarray(current.canonical.T, dtype=np.float64)

    unpermuted = permute_rows(rank_root, pivot1, reverse=True)
    full_canonical = restore_dropped_rows(
        unpermuted,
        int(canonical_T.shape[1]),
        dropped,
    )
    return np.asarray(canonical_T @ full_canonical, dtype=np.float64)


def _gaussian_unconditional_covariance_space(model) -> str:
    if prediction_parameterization_map(model) is None:
        return FIT_PARAMETER_SPACE
    return PREDICTION_PARAMETER_SPACE


def _gaussian_exact_unconditional_postfit(
    model,
    fit_result: FitResult,
    fit_state: FitState,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Mirror mgcv::gam.fit3.post.proc() EDF2 / unconditional covariance assembly.

    This applies only to Gaussian ML/REML/LAML fits after the final solve, where
    mgcv recomputes `edf2` from the fitted-model outer Hessian plus `Vb.corr()`.
    """
    if str(getattr(getattr(model, "family", None), "name", "")).lower() != "gaussian":
        return None, None, FIT_PARAMETER_SPACE
    if not bool(getattr(model.family, "supports_closed_form_solve", False)):
        return None, None, FIT_PARAMETER_SPACE

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None, None, FIT_PARAMETER_SPACE

    if (
        fit_state.A is None
        or fit_state.A_inv is None
        or fit_state.XtWX is None
        or fit_result.cov_bayes is None
    ):
        return None, None, FIT_PARAMETER_SPACE

    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return None, None, FIT_PARAMETER_SPACE

    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        return None, None, FIT_PARAMETER_SPACE

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.maximum(sp[free_mask], np.finfo(np.float64).tiny))

    from ...fit.backends import solve_gaussian_given_smoothing
    from ..selection.criteria.dispatch import criterion_hessian
    from ..selection.criteria.gaussian_dyn import (
        criterion_hessian_ml_reml_gaussian_dynamic_joint,
    )
    from ..selection.criteria.pirls.derivatives import _gdi1_kernel
    from ..selection.reparam import build_estimate_gam_setup_state
    from ..solvers.general_family.newton import _vb_corr_root

    optim_result = getattr(model, "_optim_result", None)
    joint_log_sigma2 = None
    if method in {"reml", "laml"}:
        if (
            optim_result is not None
            and getattr(optim_result, "joint_log_sigma2", None) is not None
        ):
            joint_log_sigma2 = float(optim_result.joint_log_sigma2)
        else:
            sigma2_opt = getattr(model, "_gaussian_reml_sigma2_opt_", None)
            if (
                sigma2_opt is not None
                and np.isfinite(float(sigma2_opt))
                and float(sigma2_opt) > 0.0
            ):
                joint_log_sigma2 = float(np.log(float(sigma2_opt)))

    H_outer = None
    if joint_log_sigma2 is not None:
        H_joint = np.asarray(
            criterion_hessian_ml_reml_gaussian_dynamic_joint(
                model,
                model.y_,
                log_sp,
                joint_log_sigma2,
                method=method.upper(),
            ),
            dtype=np.float64,
        )
        if H_joint.shape == (free_idx.size + 1, free_idx.size + 1) and np.all(
            np.isfinite(H_joint)
        ):
            H_outer = symmetrize_matrix(H_joint)

    if H_outer is None:
        Hsp = np.asarray(
            criterion_hessian(model, model.y_, log_sp, method=method), dtype=np.float64
        )
        if Hsp.shape != (free_idx.size, free_idx.size) or not np.all(np.isfinite(Hsp)):
            return None, None, FIT_PARAMETER_SPACE
        H_outer = symmetrize_matrix(Hsp)
        if optim_result is not None:
            optim_result.hess = Hsp.copy()

    evals, evecs = np.linalg.eigh(H_outer)
    pos = evals > 0.0

    beta = np.asarray(fit_result.coef_full, dtype=np.float64).ravel()
    H_coef = np.asarray(fit_result.H_coef, dtype=np.float64)
    scale = float(fit_result.scale)
    if not (np.isfinite(scale) and scale > 0.0):
        return None, None, FIT_PARAMETER_SPACE

    kernel = _gdi1_kernel(
        model,
        model.y_,
        solve_gaussian_given_smoothing(model, model.y_, sp),
        sp,
        method=method.upper(),
    )

    db_cols = [
        _restore_pirls_dbeta_to_original_parameterization(
            kernel.current,
            kernel.ift.dbeta[int(j)],
        )
        for j in free_idx.tolist()
    ]
    if len(db_cols) == 0:
        return None, None, FIT_PARAMETER_SPACE
    db_drho = np.column_stack(db_cols)
    M = int(db_drho.shape[1])
    rank_root = getattr(kernel.current, "rank_root", None)
    if rank_root is None:
        rank = int(kernel.current.R.shape[0])
        rank_root = solve_triangular(
            np.asarray(kernel.current.R, dtype=np.float64),
            np.eye(rank, dtype=np.float64),
            lower=False,
            check_finite=False,
        )
    else:
        rank_root = np.asarray(rank_root, dtype=np.float64)
    Vb_root = _restore_pirls_rank_root_to_original_parameterization(
        kernel.current,
        rank_root,
    )
    Vb = np.asarray(scale * (Vb_root @ Vb_root.T), dtype=np.float64)

    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[pos] = 1.0 / np.sqrt(evals[pos])
    rV = np.asarray((inv_sqrt[:, None] * evecs.T)[:, :M], dtype=np.float64)
    Vc1 = np.asarray((rV @ db_drho.T).T @ (rV @ db_drho.T), dtype=np.float64)

    reg_root = np.asarray(evals, dtype=np.float64).copy()
    reg_root[~pos] = 0.0
    reg_root = 1.0 / np.sqrt(reg_root + 0.1)
    Vr_full = np.asarray(
        (reg_root[:, None] * evecs.T).T @ (reg_root[:, None] * evecs.T),
        dtype=np.float64,
    )

    setup = build_estimate_gam_setup_state(model)
    p_full = int(beta.size)
    S_blocks_full = []
    for S_local, off_i in zip(
        list(setup.S),
        np.asarray(setup.off, dtype=np.int64),
        strict=True,
    ):
        S_local = np.asarray(S_local, dtype=np.float64)
        S_full = np.zeros((p_full, p_full), dtype=np.float64)
        start = int(off_i) - 1
        stop = start + int(S_local.shape[0])
        S_full[start:stop, start:stop] = S_local
        S_blocks_full.append(S_full)

    rho = np.log(np.maximum(sp[free_mask], np.finfo(np.float64).tiny))
    if setup.L is None:
        lam = np.exp(rho + np.asarray(setup.lsp0, dtype=np.float64)[: rho.size])
        P_derivs = [float(lam[i]) * S_blocks_full[i] for i in range(rho.size)]
    else:
        L = np.asarray(setup.L, dtype=np.float64)
        lsp0 = np.asarray(setup.lsp0, dtype=np.float64).ravel()
        lam = np.exp(L @ rho + lsp0[: L.shape[0]])
        P_derivs = []
        for j in range(L.shape[1]):
            Pj = np.zeros((p_full, p_full), dtype=np.float64)
            for i, Si in enumerate(S_blocks_full):
                wij = float(L[i, j])
                if wij == 0.0:
                    continue
                Pj += float(lam[i]) * wij * Si
            P_derivs.append(Pj)

    if len(P_derivs) != free_idx.size:
        return None, None, FIT_PARAMETER_SPACE

    scale_est = bool(H_outer.shape[0] == M + 1)
    rho_full = np.asarray(rho, dtype=np.float64)
    lsp0_full = np.asarray(setup.lsp0, dtype=np.float64)
    L_vcorr = None if setup.L is None else np.asarray(setup.L, dtype=np.float64)
    if scale_est:
        rho_full = np.concatenate(
            [rho_full, [float(np.log(max(scale, float(np.finfo(np.float64).tiny))))]]
        )
        lsp0_full = np.concatenate([lsp0_full, [0.0]])
        if L_vcorr is not None:
            # mgcv::Vb.corr(scale.est=TRUE) expects `L` to carry an extra final
            # row/column for the joint log-scale parameter before dropping it.
            L_aug = np.zeros(
                (L_vcorr.shape[0] + 1, L_vcorr.shape[1] + 1),
                dtype=np.float64,
            )
            L_aug[:-1, :-1] = L_vcorr
            L_aug[-1, -1] = 1.0
            L_vcorr = L_aug

    weights = fit_state.fisher_weights
    if weights is None:
        weights = fit_state.working_weights
    if weights is None:
        return None, None, FIT_PARAMETER_SPACE
    weights = np.asarray(weights, dtype=np.float64).ravel()
    # mgcv::gam.fit3.post.proc() receives `G$X` from the original setup and
    # forms `R` from `sqrt(weights) * X`. For aliased parameterizations the
    # fitted state can carry a transformed solve matrix, so prefer the setup
    # matrix here.
    setup_X = getattr(setup, "X", None)
    X_full = (
        np.asarray(setup_X, dtype=np.float64)
        if setup_X is not None
        else np.asarray(fit_state.X, dtype=np.float64)
    )
    if weights.size == 1 and X_full.shape[0] > 1:
        weights = np.full(X_full.shape[0], float(weights[0]), dtype=np.float64)
    if weights.size != X_full.shape[0]:
        return None, None, FIT_PARAMETER_SPACE
    WX = np.sqrt(weights)[:, None] * X_full
    R = mgcv_pqr_r(WX)

    Vc2 = scale * _vb_corr_root(
        R,
        L=L_vcorr,
        lsp0=lsp0_full,
        S_blocks=S_blocks_full,
        off=None,
        rho=rho_full,
        Vr=Vr_full,
        scale_est=scale_est,
    )

    Vc = symmetrize_matrix(Vb + Vc1 + Vc2)
    # mgcv::gam.fit3.post.proc() uses `rowSums(Vc * crossprod(R)) / scale`,
    # where `R` is the weighted-QR factor of `WX`, not the assembled XtWX
    # stored in the fit state. These differ on aliased parameterizations, and
    # the QR-based crossproduct is the parity-sensitive one.
    RTR = np.asarray(R.T @ R, dtype=np.float64)

    edf1 = 2.0 * np.diag(H_coef) - np.sum(H_coef * H_coef.T, axis=1)
    edf2 = np.sum(Vc * RTR, axis=1) / scale
    if float(np.sum(edf2)) > float(np.sum(edf1)):
        edf2 = np.asarray(edf1, dtype=np.float64).copy()

    return (
        np.asarray(Vc, dtype=np.float64),
        np.asarray(edf2, dtype=np.float64),
        _gaussian_unconditional_covariance_space(model),
    )


def _fixed_sp_edf2_from_qr(
    model,
    fit_result: FitResult,
    fit_state: FitState,
) -> np.ndarray | None:
    """Mirror fixed-sp ``mgcv::gam.fit3.post.proc()`` EDF2 assembly."""
    family_name = str(getattr(getattr(model, "family", None), "name", "")).lower()
    if family_name not in {"gaussian", "binomial", "poisson", "gamma", "negbin"}:
        return None

    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return None
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if getattr(model, "smoothing_fixed_mask_", None) is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    if fixed_mask.shape != (n_sp,) or not bool(np.all(fixed_mask)):
        return None
    optim_method = str(getattr(model, "_optim_method", "")).lower()
    theta_only_fixed_reml = (
        family_name == "negbin"
        and bool(getattr(getattr(model, "family", None), "estimate_theta", False))
        and optim_method in {"reml", "laml"}
    )
    if optim_method != "fixed" and not theta_only_fixed_reml:
        return None
    if fit_result.cov_bayes is None or fit_result.H_coef is None or fit_state.X is None:
        return None

    scale = float(fit_result.scale)
    if not (np.isfinite(scale) and scale > 0.0):
        return None

    X = np.asarray(fit_state.X, dtype=np.float64)
    weights = fit_state.fisher_weights
    if weights is None:
        weights = fit_state.working_weights
    if weights is None:
        weights = np.ones(X.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if weights.size == 1 and X.shape[0] > 1:
        weights = np.full(X.shape[0], float(weights[0]), dtype=np.float64)
    if weights.size != X.shape[0] or np.any(weights < 0.0):
        return None

    WX = np.sqrt(weights)[:, None] * X
    R_wx = mgcv_pqr_r(WX)
    RTR_wx = np.asarray(R_wx.T @ R_wx, dtype=np.float64)
    Vb = np.asarray(fit_result.cov_bayes, dtype=np.float64)
    edf2 = np.asarray(np.sum(Vb * RTR_wx, axis=1) / scale, dtype=np.float64)

    H_coef = np.asarray(fit_result.H_coef, dtype=np.float64)
    if H_coef.shape == Vb.shape:
        edf1 = 2.0 * np.diag(H_coef) - np.sum(H_coef * H_coef.T, axis=1)
        if float(np.sum(edf2)) > float(np.sum(edf1)):
            edf2 = np.asarray(edf1, dtype=np.float64).copy()
    return edf2


def _gaussian_fixed_sp_edf2_from_qr(
    model,
    fit_result: FitResult,
    fit_state: FitState,
) -> np.ndarray | None:
    """Backward-compatible alias for the generalized fixed-sp EDF2 path."""
    return _fixed_sp_edf2_from_qr(model, fit_result, fit_state)


def _pirls_exact_unconditional_postfit(
    model,
    sol,
    fit_result: FitResult,
    fit_state: FitState,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Mirror `mgcv::gam.fit3.post.proc()` unconditional `Vc` / `edf2` for
    ordinary PIRLS fits with exact outer derivatives.

    Current strict support matches the implemented exact ordinary-family PIRLS
    ML/REML/LAML path: noncanonical Gaussian, binomial, poisson, gamma, and
    fixed-theta negbin.
    """

    def _embed_setup_penalties(setup_state, p_full: int) -> list[np.ndarray] | None:
        blocks = []
        for S_local, off_i in zip(
            list(getattr(setup_state, "S", [])),
            np.asarray(getattr(setup_state, "off", []), dtype=np.int64),
            strict=True,
        ):
            S_local = np.asarray(S_local, dtype=np.float64)
            if S_local.ndim != 2 or S_local.shape[0] != S_local.shape[1]:
                return None
            S_full = np.zeros((p_full, p_full), dtype=np.float64)
            start = int(off_i) - 1
            stop = start + int(S_local.shape[0])
            if start < 0 or stop > p_full:
                return None
            S_full[start:stop, start:stop] = S_local
            blocks.append(S_full)
        return blocks

    def _map_db_drho_to_working(db_drho, L, free_mask_arr):
        db_drho = np.asarray(db_drho, dtype=np.float64)
        if db_drho.ndim == 1:
            db_drho = db_drho[:, None]
        if L is None:
            return db_drho
        L_arr = np.asarray(L, dtype=np.float64)
        if L_arr.ndim != 2:
            return None
        n_cols = int(db_drho.shape[1])
        if L_arr.shape[0] == free_mask_arr.size and n_cols == int(
            np.sum(free_mask_arr)
        ):
            L_work = L_arr[np.asarray(free_mask_arr, dtype=bool), :]
        elif L_arr.shape[0] >= n_cols:
            # mgcv::gam.fit3.post.proc() uses the leading `M` rows of `L`,
            # where `M` is the pre-link `db.drho` column count.
            L_work = L_arr[:n_cols, :]
        else:
            return None
        if L_work.shape[0] != n_cols:
            return None
        return np.asarray(db_drho @ L_work, dtype=np.float64)

    def _candidate_vc_and_edf2(
        *,
        Hsp,
        db_drho,
        rho,
        Vp,
        scale,
        R_wx,
        RTR_wx,
        edf1,
        setup_state,
        S_blocks_full,
        free_mask_arr,
        vr_ridge,
    ):
        Hsp = np.asarray(Hsp, dtype=np.float64)
        if (
            Hsp.ndim != 2
            or Hsp.shape[0] != Hsp.shape[1]
            or not np.all(np.isfinite(Hsp))
        ):
            return None
        db_work = _map_db_drho_to_working(
            db_drho, getattr(setup_state, "L", None), free_mask_arr
        )
        if db_work is None:
            return None
        db_work = np.asarray(db_work, dtype=np.float64)
        if db_work.ndim != 2 or db_work.shape[0] != Vp.shape[0]:
            return None
        n_work = int(db_work.shape[1])
        scale_est = bool(Hsp.shape[0] == n_work + 1)
        if Hsp.shape[0] not in {n_work, n_work + 1}:
            return None
        rho = np.asarray(rho, dtype=np.float64).ravel()
        if rho.shape[0] != Hsp.shape[0]:
            return None

        Hsp = symmetrize_matrix(Hsp)
        evals, evecs = np.linalg.eigh(Hsp)
        pos = evals > 0.0
        if not np.any(pos):
            return None

        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[pos] = 1.0 / np.sqrt(evals[pos])
        rV = np.asarray((inv_sqrt[:, None] * evecs.T)[:, :n_work], dtype=np.float64)
        Vc1 = np.asarray((rV @ db_work.T).T @ (rV @ db_work.T), dtype=np.float64)

        reg = np.asarray(evals, dtype=np.float64).copy()
        reg[~pos] = 0.0
        reg = 1.0 / np.sqrt(reg + float(vr_ridge))
        Vr = np.asarray(
            (reg[:, None] * evecs.T).T @ (reg[:, None] * evecs.T),
            dtype=np.float64,
        )

        from ..solvers.general_family.newton import _vb_corr_root

        rho_vcorr = np.asarray(rho, dtype=np.float64)
        lsp0_vcorr = np.asarray(getattr(setup_state, "lsp0", []), dtype=np.float64)
        L_vcorr = (
            None
            if getattr(setup_state, "L", None) is None
            else np.asarray(setup_state.L, dtype=np.float64)
        )
        if scale_est:
            lsp0_vcorr = np.concatenate([lsp0_vcorr, [0.0]])
            if L_vcorr is not None:
                # mgcv::gam.fit3.post.proc() / Vb.corr() carry an extra final
                # row/column in `L` for the joint log-scale parameter and drop
                # it internally when `scale.est=TRUE`.
                L_aug = np.zeros(
                    (L_vcorr.shape[0] + 1, L_vcorr.shape[1] + 1),
                    dtype=np.float64,
                )
                L_aug[:-1, :-1] = L_vcorr
                L_aug[-1, -1] = 1.0
                L_vcorr = L_aug

        Vc2 = scale * _vb_corr_root(
            R_wx,
            L=L_vcorr,
            lsp0=lsp0_vcorr,
            S_blocks=S_blocks_full,
            off=None,
            rho=rho_vcorr,
            Vr=Vr,
            scale_est=scale_est,
        )

        Vc = symmetrize_matrix(np.asarray(Vp + Vc1 + Vc2, dtype=np.float64))
        edf2 = np.asarray(np.sum(Vc * RTR_wx, axis=1) / scale, dtype=np.float64)
        if float(np.sum(edf2)) > float(np.sum(edf1)):
            edf2 = np.asarray(edf1, dtype=np.float64).copy()
        return Vc, edf2

    family = getattr(model, "family", None)
    family_name = str(getattr(family, "name", "")).lower()
    if family_name not in {"gaussian", "binomial", "poisson", "gamma", "negbin"}:
        return None, None, FIT_PARAMETER_SPACE

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None, None, FIT_PARAMETER_SPACE

    if (
        fit_result.cov_bayes is None
        or fit_result.H_coef is None
        or fit_state.X is None
        or fit_state.P is None
    ):
        return None, None, FIT_PARAMETER_SPACE

    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return None, None, FIT_PARAMETER_SPACE

    from ..selection.criteria.dispatch import criterion_hessian
    from ..selection.criteria.pirls.derivatives import _gdi1_kernel
    from ..selection.reparam import (
        build_estimate_gam_setup_state,
        can_use_simple_ml_reml_structure,
    )

    if not can_use_simple_ml_reml_structure(model):
        return None, None, FIT_PARAMETER_SPACE
    if not bool(getattr(family, "supports_exact_pirls_second_derivatives", False)):
        return None, None, FIT_PARAMETER_SPACE

    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        return None, None, FIT_PARAMETER_SPACE

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.maximum(sp[free_mask], np.finfo(np.float64).tiny))

    optim_result = getattr(model, "_optim_result", None)
    outer_info = dict(getattr(optim_result, "outer_info", {}) or {})
    Hsp_fit = None
    rho_fit = np.asarray(log_sp, dtype=np.float64)
    Hsp_outer = outer_info.get("hess", None)
    if Hsp_outer is not None:
        Hsp_outer = np.asarray(Hsp_outer, dtype=np.float64)
        if np.all(np.isfinite(Hsp_outer)):
            if Hsp_outer.shape == (free_idx.size, free_idx.size):
                Hsp_fit = Hsp_outer
            elif (
                Hsp_outer.shape == (free_idx.size + 1, free_idx.size + 1)
            ):
                joint_log_scale = getattr(optim_result, "joint_log_phi", None)
                if joint_log_scale is None:
                    joint_log_scale = getattr(
                        optim_result, "joint_log_sigma2", None
                    )
                if joint_log_scale is not None and np.isfinite(
                    float(joint_log_scale)
                ):
                    Hsp_fit = Hsp_outer
                    rho_fit = np.concatenate([rho_fit, [float(joint_log_scale)]])
    if Hsp_fit is None:
        Hsp_fit = np.asarray(
            criterion_hessian(model, model.y_, log_sp, method=method),
            dtype=np.float64,
        )
    if Hsp_fit.shape not in {
        (free_idx.size, free_idx.size),
        (free_idx.size + 1, free_idx.size + 1),
    } or not np.all(np.isfinite(Hsp_fit)):
        return None, None, FIT_PARAMETER_SPACE

    kernel = _gdi1_kernel(
        model,
        model.y_,
        sol,
        sp,
        method=("REML" if method in {"reml", "laml"} else "ML"),
    )

    db_cols = [
        _restore_pirls_dbeta_to_original_parameterization(
            kernel.current,
            kernel.ift.dbeta[int(j)],
        )
        for j in free_idx.tolist()
    ]
    if len(db_cols) == 0:
        return None, None, FIT_PARAMETER_SPACE
    db_drho_fit = np.column_stack(db_cols)

    # Mirror mgcv/R/gam.fit3.r::gam.fit3.post.proc(), which forms
    # `WX <- sqrt(object$weights) * X` using the reported Fisher weights,
    # not the PIRLS working weights.
    weights = fit_state.fisher_weights
    if weights is None:
        weights = fit_state.working_weights
    if weights is None:
        return None, None, FIT_PARAMETER_SPACE

    X = np.asarray(fit_state.X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != X.shape[0]:
        return None, None, FIT_PARAMETER_SPACE

    scale = float(fit_result.scale)
    if not (np.isfinite(scale) and scale > 0.0):
        return None, None, FIT_PARAMETER_SPACE

    WX = np.sqrt(w)[:, None] * X
    R_wx = mgcv_pqr_r(WX)
    RTR_wx = np.asarray(R_wx.T @ R_wx, dtype=np.float64)

    setup = build_estimate_gam_setup_state(model)
    p_full = int(np.asarray(fit_result.coef_full, dtype=np.float64).size)
    S_blocks_full = _embed_setup_penalties(setup, p_full)
    if S_blocks_full is None:
        return None, None, FIT_PARAMETER_SPACE

    Vp = np.asarray(fit_result.cov_bayes, dtype=np.float64)
    H_coef = np.asarray(fit_result.H_coef, dtype=np.float64)
    edf1 = 2.0 * np.diag(H_coef) - np.sum(H_coef * H_coef.T, axis=1)
    fitted_out = _candidate_vc_and_edf2(
        Hsp=Hsp_fit,
        db_drho=db_drho_fit,
        rho=rho_fit,
        Vp=Vp,
        scale=scale,
        R_wx=R_wx,
        RTR_wx=RTR_wx,
        edf1=edf1,
        setup_state=setup,
        S_blocks_full=S_blocks_full,
        free_mask_arr=free_mask,
        vr_ridge=0.1,
    )
    if fitted_out is None:
        return None, None, FIT_PARAMETER_SPACE

    Vc_fit, edf2_fit = fitted_out
    Vc_final = np.asarray(Vc_fit, dtype=np.float64)
    edf2_final = np.asarray(edf2_fit, dtype=np.float64)

    Hsp_edge = outer_info.get("hess1", None)
    db_drho_edge = outer_info.get("db_drho1", None)
    rho_edge = outer_info.get("lsp1", None)
    if Hsp_edge is not None and db_drho_edge is not None and rho_edge is not None:
        edge_out = _candidate_vc_and_edf2(
            Hsp=np.asarray(Hsp_edge, dtype=np.float64),
            db_drho=np.asarray(db_drho_edge, dtype=np.float64),
            rho=np.asarray(rho_edge, dtype=np.float64),
            Vp=Vp,
            scale=scale,
            R_wx=R_wx,
            RTR_wx=RTR_wx,
            edf1=edf1,
            setup_state=setup,
            S_blocks_full=S_blocks_full,
            free_mask_arr=free_mask,
            vr_ridge=1e-7,
        )
        if edge_out is not None:
            Vc_final = np.asarray(edge_out[0], dtype=np.float64)

    return Vc_final, edf2_final, FIT_PARAMETER_SPACE


def apply_unconditional_postfit(model, sol, fit_result, fit_state):
    if str(getattr(model, "smoothing_optimizer", "")).lower() in {"efs", "optim"}:
        family_class = str(
            getattr(getattr(model, "family", None), "family_class", "")
        ).lower()
        if family_class == "general":
            # mgcv::gam.fit5.post.proc() always returns Vc: with no derivative
            # state (efsud/optim final fits run at deriv=0) the correction is
            # zero and Vc == Vb (mgcv/R/gam.fit4.r:1685-1690), with edf2 from
            # rowSums(Vc * crossprod(R)) capped at edf1 (gam.fit4.r:1714-1715).
            # The general-family solver already produced exactly that state.
            return fit_result
        # mgcv::gam.fit3.post.proc(): without db.drho (efsudr and the
        # optim/nlm gam2objective final fits run at deriv=0),
        # `V.sp <- edf2 <- Vc <- NULL` (mgcv/R/gam.fit3.r:1053).
        return replace(fit_result, cov_unconditional=None, edf2=None)

    cov_unconditional_post = (
        None
        if fit_result.cov_unconditional is None
        else np.asarray(fit_result.cov_unconditional, dtype=np.float64).copy()
    )
    edf2_post = (
        None
        if fit_result.edf2 is None
        else np.asarray(fit_result.edf2, dtype=np.float64).copy()
    )
    cov_unconditional_space = getattr(
        fit_result,
        "cov_unconditional_space",
        FIT_PARAMETER_SPACE,
    )

    Vc_pirls, edf2_pirls, cov_space_pirls = _pirls_exact_unconditional_postfit(
        model, sol, fit_result, fit_state
    )
    if Vc_pirls is not None:
        cov_unconditional_post = np.asarray(Vc_pirls, dtype=np.float64).copy()
        cov_unconditional_space = cov_space_pirls
    if edf2_pirls is not None:
        edf2_post = np.asarray(edf2_pirls, dtype=np.float64).copy()

    Vc_gauss, edf2_gauss, cov_space_gauss = _gaussian_exact_unconditional_postfit(
        model, fit_result, fit_state
    )
    if Vc_gauss is not None:
        cov_unconditional_post = np.asarray(Vc_gauss, dtype=np.float64).copy()
        cov_unconditional_space = cov_space_gauss
    if edf2_gauss is not None:
        edf2_post = np.asarray(edf2_gauss, dtype=np.float64).copy()

    if edf2_post is None:
        edf2_fixed = _fixed_sp_edf2_from_qr(model, fit_result, fit_state)
        if edf2_fixed is not None:
            edf2_post = np.asarray(edf2_fixed, dtype=np.float64).copy()

    return replace(
        fit_result,
        cov_unconditional=cov_unconditional_post,
        edf2=edf2_post,
        cov_unconditional_space=cov_unconditional_space,
    )


__all__ = [
    "_gaussian_exact_unconditional_postfit",
    "_differentiate_cholesky_factor",
    "_covariance_from_cholesky_derivatives",
    "_pirls_exact_unconditional_postfit",
    "_restore_pirls_dbeta_to_original_parameterization",
    "_restore_pirls_rank_root_to_original_parameterization",
    "apply_unconditional_postfit",
]
