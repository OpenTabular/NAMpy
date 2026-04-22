"""NCV / QNCV smoothing criteria mirrored from mgcv `gam.fit3.r` / `gam.fit4.r`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from ..._model_state import _n_smoothing_params
from ...fit.solve_ops import solve_pirls_given_smoothing
from ...fit.solvers.general_family_solver import (
    run_general_family_fixed_smoothing,
    sl_initial_repara,
)
from ...fit.solvers.general_newton_solver import (
    _sl_ldetS,
    _sl_mult,
    _sl_repa,
    _sl_repara,
    postprocess_general_newton_fit,
)
from ...fit.state import _restore_pirls_dbeta_to_original_parameterization
from ...linalg import symmetrize_matrix
from .pirls_deriv import (
    _gdi1_kernel,
    _gdi2_joint_kernel,
    _negbin_ddeta_logtheta,
    _prior_weights,
)


@dataclass
class _NCVKernelState:
    sp: np.ndarray
    sol: object
    kernel: object
    free_mask: np.ndarray
    eta: np.ndarray
    mu: np.ndarray
    deta: np.ndarray
    dw: np.ndarray
    dA_cols: list[np.ndarray]
    dVkk: np.ndarray
    kernel_dbeta: np.ndarray | None = None
    drop_grad_weights: np.ndarray | None = None
    direct_eta_weight_derivs: np.ndarray | None = None
    n_non_sp: int = 0


@dataclass
class _NCVFoldResult:
    eta_cv: np.ndarray
    deta_cv: np.ndarray
    dbeta_cv: np.ndarray | None = None


@dataclass
class _GeneralNCVState:
    sp: np.ndarray
    run: dict[str, Any]
    post: dict[str, Any]
    rp_state: dict[str, Any] | None
    free_mask: np.ndarray
    X: np.ndarray
    jj: list[np.ndarray]
    beta: np.ndarray
    H: np.ndarray
    Hi: np.ndarray
    eta: np.ndarray
    llf: dict[str, Any]
    deta: np.ndarray
    dbeta: np.ndarray
    dH: list[np.ndarray]
    dVkk: np.ndarray
    offset_list: list[np.ndarray | None] | None


def _expand_smoothing_params_from_log(model, log_free_sp):
    n_sp = int(_n_smoothing_params(model) or 0)
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    log_free_sp = np.asarray(log_free_sp, dtype=np.float64).ravel()
    n_free = int(np.sum(~fixed_mask))
    if log_free_sp.shape != (n_free,):
        raise ValueError(
            f"Expected {n_free} free log smoothing parameters, got {log_free_sp.shape}."
        )

    sp = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    if n_free > 0:
        sp[~fixed_mask] = np.exp(log_free_sp)
    if model.min_sp_ is not None:
        sp = np.maximum(sp, np.asarray(model.min_sp_, dtype=np.float64))
    return sp


def _column_stack_or_empty(cols, nrow):
    if len(cols) == 0:
        return np.empty((int(nrow), 0), dtype=np.float64)
    return np.column_stack([np.asarray(col, dtype=np.float64) for col in cols])


def _free_mask(model):
    n_sp = int(_n_smoothing_params(model) or 0)
    if model.smoothing_fixed_mask_ is None:
        return np.ones(n_sp, dtype=bool)
    return ~np.asarray(model.smoothing_fixed_mask_, dtype=bool)


def _infer_index_base(values, n_obs):
    values = np.asarray(values, dtype=np.int64).ravel()
    if values.size == 0:
        return 0
    if np.any(values == 0):
        return 0
    if np.any(values == int(n_obs)):
        return 1
    return 0


def _coerce_index_vector(values, n_obs, name, *, index_base):
    out = np.asarray(values, dtype=np.int64).ravel()
    if index_base not in {0, 1}:
        raise ValueError(f"{name} index_base must be 0 or 1, got {index_base!r}.")
    if index_base == 1:
        out = out - 1
    if np.any(out < 0) or np.any(out >= int(n_obs)):
        raise ValueError(
            f"{name} must index observations in [0, {int(n_obs) - 1}], got {values!r}."
        )
    return out


def _coerce_cumulative(values, total, name):
    out = np.asarray(values, dtype=np.int64).ravel()
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D cumulative index vector.")
    if out.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(np.diff(out) < 0):
        raise ValueError(f"{name} must be non-decreasing.")
    if int(out[-1]) != int(total):
        raise ValueError(
            f"{name} must end at {int(total)}, got final value {int(out[-1])}."
        )
    return out


def _resolve_nei(model, n_obs):
    nei = getattr(model, "nei", None)
    if nei is None:
        idx = np.arange(int(n_obs), dtype=np.int64)
        cum = np.arange(1, int(n_obs) + 1, dtype=np.int64)
        return {"d": idx, "md": cum, "a": idx, "ma": cum, "jackknife": 0}

    if not isinstance(nei, dict):
        raise TypeError("nei must be a dict mirroring mgcv's neighborhood structure.")

    missing = [key for key in ("d", "md", "a", "ma") if key not in nei]
    if missing:
        raise ValueError(f"nei is missing required keys: {missing}.")

    index_base = nei.get("index_base", None)
    if index_base is None:
        index_base = _infer_index_base(np.concatenate([nei["d"], nei["a"]]), n_obs)
    index_base = int(index_base)

    d = _coerce_index_vector(nei["d"], n_obs, "nei['d']", index_base=index_base)
    a = _coerce_index_vector(nei["a"], n_obs, "nei['a']", index_base=index_base)
    md = _coerce_cumulative(nei["md"], len(d), "nei['md']")
    ma = _coerce_cumulative(nei["ma"], len(a), "nei['ma']")

    if md.shape != ma.shape:
        raise ValueError("nei['md'] and nei['ma'] must have the same shape.")

    jackknife = int(nei.get("jackknife", 0))
    return {"d": d, "md": md, "a": a, "ma": ma, "jackknife": jackknife}


def _subset_offset_list(offset_list, idx):
    if offset_list is None:
        return None
    out = []
    for off in offset_list:
        if off is None:
            out.append(None)
        else:
            out.append(np.asarray(off, dtype=np.float64)[idx])
    return out


def _stack_general_eta_derivatives(X, jj, dbeta):
    X = np.asarray(X, dtype=np.float64)
    dbeta = np.asarray(dbeta, dtype=np.float64)
    n = int(X.shape[0])
    nsp = 0 if dbeta.ndim != 2 else int(dbeta.shape[1])
    if nsp == 0:
        return np.empty((n * len(jj), 0), dtype=np.float64)
    deta = np.zeros((n * len(jj), nsp), dtype=np.float64)
    for i, cols in enumerate(jj):
        deta[i * n : (i + 1) * n, :] = X[:, cols] @ dbeta[cols, :]
    return np.asarray(deta, dtype=np.float64)


def _i2f(i, j, K):
    if i > j:
        i, j = j, i
    return int((i * (2 * K - i + 1)) // 2 + j - i)


def _i3f(i, j, k, K):
    while not (k >= j and j >= i):
        if i > j:
            i, j = j, i
        if j > k:
            j, k = k, j
    return int(
        (i * (3 * K * (K + 1) + (i - 1) * (i - 3 * K - 2))) // 6
        + ((j - i) * (2 * K + 1 - i - j)) // 2
        + k
        - j
    )


def _build_ncv_kernel_state(model, y, log_sp):
    sp = _expand_smoothing_params_from_log(model, log_sp)
    sol = solve_pirls_given_smoothing(model, y, sp)
    kernel = _gdi1_kernel(model, y, sol, sp, method="REML")
    current = kernel.current
    ift = kernel.ift

    eta = np.asarray(current.X @ current.beta, dtype=np.float64)
    mu = np.asarray(sol["mu"], dtype=np.float64)
    dbeta = _column_stack_or_empty(ift.dbeta, current.beta.size)
    deta = _column_stack_or_empty(ift.deta, eta.size)
    dw = _column_stack_or_empty(
        [] if ift.dW_obs is None else ift.dW_obs,
        eta.size,
    )
    dA_cols = [np.asarray(dAj, dtype=np.float64) for dAj in ift.dA]
    if dbeta.size == 0:
        dVkk = np.empty((0,), dtype=np.float64)
    else:
        Adb = np.asarray(current.A, dtype=np.float64) @ dbeta
        dVkk = np.sum(dbeta * Adb, axis=0)

    return _NCVKernelState(
        sp=np.asarray(sp, dtype=np.float64),
        sol=sol,
        kernel=kernel,
        free_mask=_free_mask(model),
        eta=eta,
        mu=mu,
        deta=deta,
        dw=dw,
        dA_cols=dA_cols,
        dVkk=np.asarray(dVkk, dtype=np.float64),
    )


def _build_general_ncv_state(model, y, log_sp, *, need_gradient):
    sp = _expand_smoothing_params_from_log(model, log_sp)
    deriv = 2 if need_gradient else 0
    run = run_general_family_fixed_smoothing(
        model,
        y,
        sp,
        weights=getattr(model, "prior_weights_", None),
        deriv=deriv,
        score_type="REML",
    )
    setup = run["setup"]
    fit = run["fit"]
    post = postprocess_general_newton_fit(
        fit,
        Sl=setup.Sl,
        L_map=None,
        lsp0=None,
        S_blocks=setup.S_blocks,
        off=[1] * len(setup.S_blocks),
        outer_hess=None,
        smoothing_params=setup.smoothing_params,
        penalty_matrix=setup.St,
        penalty_derivatives=setup.penalty_derivatives,
    )

    beta = np.asarray(
        fit.get("coef_fit_space", fit["coef"]),
        dtype=np.float64,
    )

    dbeta = np.empty((beta.size, 0), dtype=np.float64)
    dH = []
    if need_gradient:
        dbeta_raw = fit.get("db_drho_fit_space", fit.get("db_drho", None))
        if dbeta_raw is None:
            raise RuntimeError("General-family NCV gradient path requires `db_drho`.")
        dbeta_raw = np.asarray(dbeta_raw, dtype=np.float64)
        if dbeta_raw.ndim == 1:
            dbeta_raw = dbeta_raw[:, None]
        dbeta = np.asarray(dbeta_raw, dtype=np.float64)

    rp_state = None
    X_fit = np.asarray(setup.X_initial, dtype=np.float64)
    beta_fit = np.asarray(beta, dtype=np.float64)
    dbeta_fit = np.asarray(dbeta, dtype=np.float64)
    penalty_derivs = [
        np.zeros((beta_fit.size, beta_fit.size), dtype=np.float64)
        for _ in range(int(dbeta_fit.shape[1]))
    ]
    if len(setup.Sl) > 0:
        rp_state = _sl_ldetS(
            setup.Sl,
            rho=np.asarray(setup.log_sp, dtype=np.float64),
            fixed=np.zeros_like(np.asarray(setup.log_sp, dtype=np.float64), dtype=bool),
            np_=int(setup.X_full.shape[1]),
            root=True,
            Stot=True,
            deriv=0,
        )
        X_fit = np.asarray(_sl_repara(rp_state["rp"], X_fit), dtype=np.float64)
        eye_p = np.eye(beta_fit.size, dtype=np.float64)
        penalty_derivs = [
            np.asarray(
                _sl_mult(rp_state["Sl"], eye_p, i + 1, full=True), dtype=np.float64
            )
            for i in range(int(dbeta_fit.shape[1]))
        ]

    offset_list = run.get("offset_list", None)
    eta = np.asarray(
        model.family._stacked_eta(X_fit, setup.jj, beta_fit, offset=offset_list),
        dtype=np.float64,
    )
    weights = _prior_weights(model, y)
    llf = model.family.ll(
        np.asarray(y, dtype=np.float64),
        np.asarray(X_fit, dtype=np.float64),
        setup.jj,
        beta_fit,
        weights,
        offset=offset_list,
        deriv=3 if need_gradient else 1,
        d1b=dbeta_fit if need_gradient else 0,
        eta=eta,
        ncv=True,
    )
    deta = _stack_general_eta_derivatives(X_fit, setup.jj, dbeta_fit)
    if need_gradient:
        dH = [
            np.asarray(llf["d1H"][i], dtype=np.float64)
            - np.asarray(penalty_derivs[i], dtype=np.float64)
            for i in range(dbeta_fit.shape[1])
        ]

    H = symmetrize_matrix(
        -np.asarray(fit["lbb"], dtype=np.float64)
        + np.asarray(fit["St_full"], dtype=np.float64)
    )
    try:
        Hi = symmetrize_matrix(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        Hi = symmetrize_matrix(np.linalg.pinv(H))
    if dbeta_fit.size == 0:
        dVkk = np.empty((0,), dtype=np.float64)
    else:
        Hdb = H @ dbeta_fit
        dVkk = np.sum(dbeta_fit * Hdb, axis=0)

    return _GeneralNCVState(
        sp=np.asarray(sp, dtype=np.float64),
        run=run,
        post=post,
        rp_state=rp_state,
        free_mask=_free_mask(model),
        X=np.asarray(X_fit, dtype=np.float64),
        jj=list(setup.jj),
        beta=np.asarray(beta_fit, dtype=np.float64),
        H=H,
        Hi=Hi,
        eta=np.asarray(eta, dtype=np.float64),
        llf=llf,
        deta=np.asarray(deta, dtype=np.float64),
        dbeta=np.asarray(dbeta_fit, dtype=np.float64),
        dH=[np.asarray(dHi, dtype=np.float64) for dHi in dH],
        dVkk=np.asarray(dVkk, dtype=np.float64),
        offset_list=offset_list,
    )


def _build_negbin_joint_ncv_state(model, y, log_sp, log_theta):
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        sp = _expand_smoothing_params_from_log(model, log_sp)
        sol = solve_pirls_given_smoothing(model, y, sp)
        gdi2 = _gdi2_joint_kernel(
            model,
            y,
            sol,
            sp,
            method="REML",
            need_hessian=False,
        )
        current = gdi2.gdi1.current
        ift = gdi2.ift
        eta = np.asarray(current.X @ current.beta, dtype=np.float64)
        mu = np.asarray(sol["mu"], dtype=np.float64)
        dbeta = _column_stack_or_empty(ift.dbeta, current.beta.size)
        deta = _column_stack_or_empty(ift.deta, eta.size)
        dd = _negbin_ddeta_logtheta(
            model.family,
            y,
            mu,
            _prior_weights(model, y),
            deriv=1,
        )
        dw = np.zeros((eta.size, dbeta.shape[1]), dtype=np.float64)
        if dbeta.shape[1] > 0:
            dw[:, 0] = 0.5 * (
                np.asarray(dd["Deta3"], dtype=np.float64) * deta[:, 0]
                + np.asarray(dd["Deta2th"], dtype=np.float64)
            )
        if dbeta.shape[1] > 1:
            dw[:, 1:] = (
                0.5
                * np.asarray(dd["Deta3"], dtype=np.float64)[:, None]
                * np.asarray(deta[:, 1:], dtype=np.float64)
            )
        dA_cols = [np.asarray(dAj, dtype=np.float64) for dAj in ift.dA]
        if dbeta.size == 0:
            dVkk = np.empty((0,), dtype=np.float64)
        else:
            Adb = np.asarray(current.A, dtype=np.float64) @ dbeta
            dVkk = np.sum(dbeta * Adb, axis=0)
        return _NCVKernelState(
            sp=np.asarray(sp, dtype=np.float64),
            sol=sol,
            kernel=gdi2.gdi1,
            free_mask=_free_mask(model),
            eta=eta,
            mu=mu,
            deta=np.asarray(deta, dtype=np.float64),
            dw=np.asarray(dw, dtype=np.float64),
            dA_cols=dA_cols,
            dVkk=np.asarray(dVkk, dtype=np.float64),
            kernel_dbeta=np.asarray(dbeta, dtype=np.float64),
            drop_grad_weights=np.asarray(
                -np.asarray(dd["Deta"], dtype=np.float64) / 2.0
            ),
            direct_eta_weight_derivs=np.asarray(
                np.asarray(dd["Detath"], dtype=np.float64).reshape(-1, 1) / 2.0,
                dtype=np.float64,
            ),
            n_non_sp=1,
        )
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)


def _chol_up(R, u, *, up, eps):
    """
    Port of `mgcv/src/mat.c::chol_up()` on an upper-triangular dense matrix.

    Returns ``True`` when the update/downdate remains positive definite and
    ``False`` for the mgcv downdate failure signal.
    """
    R = np.asarray(R, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).ravel()
    n = int(R.shape[0])
    if n == 0:
        return True

    c = np.zeros(max(n - 2, 0), dtype=np.float64)
    s = np.zeros(max(n - 2, 0), dtype=np.float64)
    n1 = n - 1
    c0 = 0.0
    s0 = 0.0

    if up:
        for j in range(n):
            z = float(u[j])
            for i in range(j - 1):
                z0 = z
                z = c[i] * z - s[i] * R[i, j]
                R[i, j] = s[i] * z0 + c[i] * R[i, j]
            if j:
                z0 = z
                z = c0 * z - s0 * R[j - 1, j]
                R[j - 1, j] = s0 * z0 + c0 * R[j - 1, j]
                if j < n1:
                    c[j - 1] = c0
                    s[j - 1] = s0

            z0 = math.hypot(z, float(R[j, j]))
            c0 = float(R[j, j]) / z0
            s0 = z / z0
            R[j, j] = s0 * z + c0 * R[j, j]
    else:
        for j in range(n):
            z = float(u[j])
            for i in range(j - 1):
                z0 = z
                z = c[i] * z - s[i] * R[i, j]
                R[i, j] = -s[i] * z0 + c[i] * R[i, j]
            if j:
                z0 = z
                z = c0 * z - s0 * R[j - 1, j]
                R[j - 1, j] = -s0 * z0 + c0 * R[j - 1, j]
                if j < n1:
                    c[j - 1] = c0
                    s[j - 1] = s0

            z0 = z / float(R[j, j])
            if abs(z0) >= 1.0:
                if n > 1:
                    R[1, 0] = -2.0
                return False
            if z0 > 1.0 - eps:
                z0 = 1.0 - eps
            c0 = 1.0 / math.sqrt(1.0 - z0 * z0)
            s0 = c0 * z0
            R[j, j] = -s0 * z + c0 * R[j, j]

    return True


def _minres(R, u, b):
    """
    Port of `mgcv/src/ncv.c::minres()`.

    Solves ``(R'R - uu') x = b`` with ``R`` as preconditioner.
    """
    R = np.asarray(R, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    p = int(R.shape[0])
    if p == 0:
        return np.empty((0,), dtype=np.float64), 0

    if u is None:
        u = np.empty((p, 0), dtype=np.float64)
    else:
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            u = u.reshape(p, 1)
    m = int(u.shape[1])

    x = b.copy()
    maxb = math.sqrt(float(np.dot(x, x)))
    x = solve_triangular(R, x, lower=False, trans="T", check_finite=False)
    if m == 0:
        x = solve_triangular(R, x, lower=False, check_finite=False)
        return np.asarray(x, dtype=np.float64), 0

    u1 = solve_triangular(R, u, lower=False, trans="T", check_finite=False)
    dum = np.asarray(u1.T @ x, dtype=np.float64)
    v1 = np.asarray(u1 @ dum, dtype=np.float64)
    beta1 = math.sqrt(float(np.dot(v1, v1)))
    epsilon = beta1
    eta = beta1
    gamma0 = 1.0
    gamma1 = 1.0
    sig0 = 0.0
    sig1 = 0.0
    v = np.zeros(p, dtype=np.float64)
    w = np.zeros(p, dtype=np.float64)
    w1 = np.zeros(p, dtype=np.float64)
    w2 = np.zeros(p, dtype=np.float64)
    if beta1 == 0.0:
        x = solve_triangular(R, x, lower=False, check_finite=False)
        return np.asarray(x, dtype=np.float64), 0

    j_out = 0
    for j in range(200):
        v1 = v1 / beta1
        z = v1.copy()
        dum = np.asarray(u1.T @ v1, dtype=np.float64)
        z = z - np.asarray(u1 @ dum, dtype=np.float64)
        alpha = float(np.dot(v1, z))
        v2 = z - alpha * v1 - beta1 * v
        beta2_sq = float(np.dot(v2, v2))
        delta = gamma1 * alpha - gamma0 * sig1 * beta1
        rho1 = math.sqrt(delta * delta + beta2_sq)
        beta2 = math.sqrt(beta2_sq)
        rho2 = sig1 * alpha + gamma0 * gamma1 * beta1
        rho3 = sig0 * beta1
        gamma2 = delta / rho1
        sig2 = beta2 / rho1
        xx = gamma2 * eta
        w2 = (v1 - rho3 * w - rho2 * w1) / rho1
        x = x + xx * w2
        epsilon *= abs(sig2)
        j_out = j
        if epsilon < maxb * 1e-10:
            break
        eta *= -sig2

        v, v1, v2 = v1, v2, v
        w, w1, w2 = w1, w2, w
        sig0 = sig1
        sig1 = sig2
        beta1 = beta2
        gamma0 = gamma1
        gamma1 = gamma2

    x = solve_triangular(R, x, lower=False, check_finite=False)
    return np.asarray(x, dtype=np.float64), int(j_out)


def _cg(A, Mi, b, x, *, tol):
    """Port of `mgcv/src/ncv.c::CG()`."""
    A = np.asarray(A, dtype=np.float64)
    Mi = np.asarray(Mi, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    n = int(b.size)
    if n == 0:
        return np.empty((0,), dtype=np.float64), 0

    p = np.zeros(n, dtype=np.float64)
    r = b.copy()
    r1 = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    z1 = np.zeros(n, dtype=np.float64)
    bmax = float(np.max(np.abs(b))) if b.size else 0.0

    r = r - A @ x
    z = Mi @ r
    p[:] = z
    if bmax == 0.0:
        return np.asarray(x, dtype=np.float64), 0

    k_out = 0
    for k in range(200):
        z1 = A @ p
        rz = float(np.dot(r, z))
        pAp = float(np.dot(p, z1))
        if pAp == 0.0:
            return np.asarray(x, dtype=np.float64), -int(k)
        alpha = rz / pAp
        x = x + alpha * p
        r1 = r - alpha * z1
        rmax = float(np.max(np.abs(r1))) if r1.size else 0.0
        k_out = k
        if rmax < float(tol) * bmax:
            break
        z1 = Mi @ r1
        r1z1 = float(np.dot(r1, z1))
        if rz == 0.0:
            return np.asarray(x, dtype=np.float64), -int(k)
        beta = r1z1 / rz
        p = z1 + beta * p
        z, z1 = z1, z
        r, r1 = r1, r

    return np.asarray(x, dtype=np.float64), int(k_out)


def _solve_from_upper_chol(R, rhs):
    rhs = np.asarray(rhs, dtype=np.float64).ravel()
    out = solve_triangular(R, rhs, lower=False, trans="T", check_finite=False)
    out = solve_triangular(R, out, lower=False, check_finite=False)
    return np.asarray(out, dtype=np.float64)


def _compute_fold_predictions(state, nei):
    current = state.kernel.current
    X = np.asarray(current.X, dtype=np.float64)
    R = np.asarray(current.R, dtype=np.float64)
    beta = np.asarray(current.beta, dtype=np.float64)
    n_sp = int(state.deta.shape[1])
    w1 = np.asarray(state.drop_grad_weights, dtype=np.float64)
    w2 = np.asarray(current.W, dtype=np.float64)
    db = np.asarray(state.kernel_dbeta, dtype=np.float64)
    dw = np.asarray(state.dw, dtype=np.float64)
    dH = [np.asarray(dAj, dtype=np.float64) for dAj in state.dA_cols]
    nth = int(getattr(state, "n_non_sp", 0) or 0)
    dlet = getattr(state, "direct_eta_weight_derivs", None)
    if nth > 0:
        dlet = np.asarray(dlet, dtype=np.float64)
        if dlet.ndim == 1:
            dlet = dlet.reshape(-1, 1)
    eps = float(np.finfo(np.float64).eps)

    eta_cv = np.empty(nei["d"].shape[0], dtype=np.float64)
    deta_cv = np.empty((nei["d"].shape[0], n_sp), dtype=np.float64)
    dbeta_cv = np.empty((nei["ma"].shape[0], X.shape[1]), dtype=np.float64)

    a0 = d0 = 0
    for i in range(nei["ma"].shape[0]):
        a1 = int(nei["ma"][i])
        d1 = int(nei["md"][i])
        omit = np.asarray(nei["a"][a0:a1], dtype=np.int64)
        pred = np.asarray(nei["d"][d0:d1], dtype=np.int64)
        a0 = a1
        d0 = d1

        R0 = np.asarray(R, dtype=np.float64).copy()
        g = np.zeros(X.shape[1], dtype=np.float64)
        ddbuf: list[np.ndarray] = []
        pdef = True

        for ki in omit:
            xi = np.asarray(X[ki, :], dtype=np.float64)
            g = g + w1[ki] * xi
            alpha = math.sqrt(abs(float(w2[ki])))
            d = xi * alpha
            Rb = np.asarray(R0, dtype=np.float64).copy()
            # Mirror `mgcv/src/ncv.c::Rncv()`: ordinary NCV always reaches the
            # downdate branch because `alpha = sqrt(fabs(w2ki)) >= 0`.
            ok = _chol_up(R0, d, up=False, eps=eps)
            if not ok:
                pdef = False
                R0[:, :] = Rb
                ddbuf.append(np.asarray(d, dtype=np.float64).copy())

        if pdef:
            d_beta = _solve_from_upper_chol(R0, g)
        else:
            u = (
                np.column_stack(ddbuf)
                if len(ddbuf) > 0
                else np.empty((X.shape[1], 0), dtype=np.float64)
            )
            d_beta, _ = _minres(R0, u, g)
        dbeta_cv[i, :] = np.asarray(d_beta, dtype=np.float64)
        eta_cv[d0 - pred.shape[0] : d0] = np.asarray(
            X[pred, :] @ (beta - d_beta),
            dtype=np.float64,
        )

        if n_sp > 0:
            for sp_idx in range(n_sp):
                dg = np.zeros(X.shape[1], dtype=np.float64)
                for ki in omit:
                    xi = np.asarray(X[ki, :], dtype=np.float64)
                    xx = float(xi @ db[:, sp_idx])
                    dg = dg - (w2[ki] * xi) * xx
                    if sp_idx < nth:
                        dg = dg - xi * float(dlet[ki, sp_idx])
                for ki in omit:
                    xi = np.asarray(X[ki, :], dtype=np.float64)
                    xx = float(xi @ d_beta)
                    dg = dg + (dw[ki, sp_idx] * xi) * xx
                dg = dg - np.asarray(dH[sp_idx] @ d_beta, dtype=np.float64)

                if pdef:
                    d1_beta = _solve_from_upper_chol(R0, dg)
                else:
                    u = (
                        np.column_stack(ddbuf)
                        if len(ddbuf) > 0
                        else np.empty((X.shape[1], 0), dtype=np.float64)
                    )
                    d1_beta, _ = _minres(R0, u, dg)
                deta_cv[d0 - pred.shape[0] : d0, sp_idx] = np.asarray(
                    X[pred, :] @ (db[:, sp_idx] - d1_beta),
                    dtype=np.float64,
                )

    return _NCVFoldResult(
        eta_cv=np.asarray(eta_cv, dtype=np.float64),
        deta_cv=np.asarray(deta_cv, dtype=np.float64),
        dbeta_cv=np.asarray(dbeta_cv, dtype=np.float64),
    )


def _compute_general_fold_predictions_ncvls(state, nei, *, need_gradient):
    X = np.asarray(state.X, dtype=np.float64)
    H = np.asarray(state.H, dtype=np.float64)
    Hi = np.asarray(state.Hi, dtype=np.float64)
    beta = np.asarray(state.beta, dtype=np.float64)
    l1 = np.asarray(state.llf["l1"], dtype=np.float64)
    l2 = np.asarray(state.llf["l2"], dtype=np.float64)
    l3 = None if not need_gradient else np.asarray(state.llf["l3"], dtype=np.float64)
    deta = np.asarray(state.deta, dtype=np.float64)
    db = np.asarray(state.dbeta, dtype=np.float64)
    dH = [np.asarray(dHi, dtype=np.float64) for dHi in state.dH]
    n = int(X.shape[0])
    p = int(X.shape[1])
    nlp = int(len(state.jj))
    no = int(nei["d"].shape[0])
    nsp = int(db.shape[1])

    eta_cv = np.empty((no, nlp), dtype=np.float64)
    deta_cv = np.empty((no * nlp, nsp), dtype=np.float64)
    dbeta_cv = np.empty((nei["ma"].shape[0], p), dtype=np.float64)

    a0 = d0 = 0
    for i in range(nei["ma"].shape[0]):
        a1 = int(nei["ma"][i])
        d1 = int(nei["md"][i])
        omit = np.asarray(nei["a"][a0:a1], dtype=np.int64)
        pred = np.asarray(nei["d"][d0:d1], dtype=np.int64)
        a0 = a1
        d0 = d1

        g = np.zeros(p, dtype=np.float64)
        Hp = np.asarray(H, dtype=np.float64).copy()
        for ki in omit:
            for lp_idx in range(nlp):
                jjl = np.asarray(state.jj[lp_idx], dtype=np.int64)
                g[jjl] += X[ki, jjl] * l1[ki, lp_idx]
                for q in range(lp_idx, nlp):
                    jjq = np.asarray(state.jj[q], dtype=np.int64)
                    xx = (
                        np.asarray(X[ki, jjl], dtype=np.float64)[:, None]
                        * np.asarray(X[ki, jjq], dtype=np.float64)[None, :]
                        * float(l2[ki, _i2f(lp_idx, q, nlp)])
                    )
                    Hp[np.ix_(jjl, jjq)] += xx
                    if q > lp_idx:
                        Hp[np.ix_(jjq, jjl)] += xx.T

        d_init = np.asarray(Hi @ g, dtype=np.float64)
        d_beta, _ = _cg(Hp, Hi, g, d_init, tol=1e-13)
        dbeta_cv[i, :] = np.asarray(d_beta, dtype=np.float64)
        fold_offsets = _subset_offset_list(state.offset_list, pred)
        beta_drop = beta - d_beta
        for lp_idx in range(nlp):
            jjl = np.asarray(state.jj[lp_idx], dtype=np.int64)
            eta_seg = np.asarray(
                X[pred][:, jjl] @ beta_drop[jjl],
                dtype=np.float64,
            )
            if (
                fold_offsets is not None
                and lp_idx < len(fold_offsets)
                and fold_offsets[lp_idx] is not None
            ):
                eta_seg = eta_seg + np.asarray(fold_offsets[lp_idx], dtype=np.float64)
            eta_cv[d0 - pred.shape[0] : d0, lp_idx] = eta_seg

        if need_gradient and nsp > 0:
            for sp_idx in range(nsp):
                dg = np.asarray(dH[sp_idx] @ d_beta, dtype=np.float64)
                for ki in omit:
                    for j in range(nlp):
                        jjl = np.asarray(state.jj[j], dtype=np.int64)
                        for q in range(nlp):
                            jjq = np.asarray(state.jj[q], dtype=np.int64)
                            v = float(
                                sum(
                                    l3[ki, _i3f(j, q, r, nlp)]
                                    * deta[r * n + ki, sp_idx]
                                    for r in range(nlp)
                                )
                            )
                            xx = float(np.dot(X[ki, jjq], d_beta[jjq])) * v
                            dg[jjl] -= X[ki, jjl] * xx
                            xx = (
                                float(np.dot(X[ki, jjq], db[jjq, sp_idx]))
                                * l2[ki, _i2f(j, q, nlp)]
                            )
                            dg[jjl] += X[ki, jjl] * xx

                d1_init = np.asarray(Hi @ dg, dtype=np.float64)
                d1_beta, _ = _cg(Hp, Hi, dg, d1_init, tol=1e-13)
                for q in range(nlp):
                    jjq = np.asarray(state.jj[q], dtype=np.int64)
                    deta_cv[
                        (q * no) + (d0 - pred.shape[0]) : (q * no) + d0,
                        sp_idx,
                    ] = np.asarray(
                        X[pred][:, jjq] @ (db[jjq, sp_idx] - d1_beta[jjq]),
                        dtype=np.float64,
                    )

    return _NCVFoldResult(
        eta_cv=np.asarray(eta_cv, dtype=np.float64),
        deta_cv=np.asarray(deta_cv, dtype=np.float64),
        dbeta_cv=np.asarray(dbeta_cv, dtype=np.float64),
    )


def _compute_general_fold_predictions_rncvls(state, nei, *, need_gradient):
    X = np.asarray(state.X, dtype=np.float64)
    H = np.asarray(state.H, dtype=np.float64)
    beta = np.asarray(state.beta, dtype=np.float64)
    l1 = np.asarray(state.llf["l1"], dtype=np.float64)
    l2 = np.asarray(state.llf["l2"], dtype=np.float64)
    l3 = None if not need_gradient else np.asarray(state.llf["l3"], dtype=np.float64)
    deta = np.asarray(state.deta, dtype=np.float64)
    db = np.asarray(state.dbeta, dtype=np.float64)
    dH = [np.asarray(dHi, dtype=np.float64) for dHi in state.dH]
    n = int(X.shape[0])
    p = int(X.shape[1])
    nlp = int(len(state.jj))
    no = int(nei["d"].shape[0])
    nsp = int(db.shape[1])
    eps = float(np.finfo(np.float64).eps)

    H_chol = np.linalg.cholesky(symmetrize_matrix(H)).T
    eta_cv = np.empty((no, nlp), dtype=np.float64)
    deta_cv = np.empty((no * nlp, nsp), dtype=np.float64)
    dbeta_cv = np.empty((nei["ma"].shape[0], p), dtype=np.float64)

    a0 = d0 = 0
    for i in range(nei["ma"].shape[0]):
        a1 = int(nei["ma"][i])
        d1 = int(nei["md"][i])
        omit = np.asarray(nei["a"][a0:a1], dtype=np.int64)
        pred = np.asarray(nei["d"][d0:d1], dtype=np.int64)
        a0 = a1
        d0 = d1

        g = np.zeros(p, dtype=np.float64)
        R0 = np.asarray(H_chol, dtype=np.float64).copy()
        ddbuf: list[np.ndarray] = []
        pdef = True
        for ki in omit:
            b = np.zeros(nlp, dtype=np.float64)
            for kk in range(2):
                for lp_idx in range(nlp):
                    jjl = np.asarray(state.jj[lp_idx], dtype=np.int64)
                    if kk == 0:
                        g[jjl] += X[ki, jjl] * l1[ki, lp_idx]
                    alpha0 = float(l2[ki, _i2f(lp_idx, lp_idx, nlp)])
                    for q in range(lp_idx + 1, nlp):
                        jjq = np.asarray(state.jj[q], dtype=np.int64)
                        alpha = float(l2[ki, _i2f(lp_idx, q, nlp)])
                        if kk == 0:
                            b[lp_idx] += alpha
                            b[q] += alpha
                        if (kk == 0 and alpha > 0.0) or (kk == 1 and alpha < 0.0):
                            xx = math.sqrt(abs(alpha))
                            d = np.zeros(p, dtype=np.float64)
                            d[jjl] = X[ki, jjl] * xx
                            d[jjq] += X[ki, jjq] * xx
                            Rb = np.asarray(R0, dtype=np.float64).copy()
                            ok = _chol_up(R0, d, up=alpha > 0.0, eps=eps)
                            if not ok:
                                pdef = False
                                R0[:, :] = Rb
                                ddbuf.append(np.asarray(d, dtype=np.float64).copy())
                    alpha = alpha0 - b[lp_idx]
                    if (kk == 0 and alpha > 0.0) or (kk == 1 and alpha < 0.0):
                        xx = math.sqrt(abs(alpha))
                        d = np.zeros(p, dtype=np.float64)
                        d[jjl] = xx * X[ki, jjl]
                        Rb = np.asarray(R0, dtype=np.float64).copy()
                        ok = _chol_up(R0, d, up=alpha > 0.0, eps=eps)
                        if not ok:
                            pdef = False
                            R0[:, :] = Rb
                            ddbuf.append(np.asarray(d, dtype=np.float64).copy())

        if pdef:
            d_beta = _solve_from_upper_chol(R0, g)
        else:
            u = (
                np.column_stack(ddbuf)
                if len(ddbuf) > 0
                else np.empty((p, 0), dtype=np.float64)
            )
            d_beta, _ = _minres(R0, u, g)
        dbeta_cv[i, :] = np.asarray(d_beta, dtype=np.float64)
        fold_offsets = _subset_offset_list(state.offset_list, pred)
        beta_drop = beta - d_beta
        for lp_idx in range(nlp):
            jjl = np.asarray(state.jj[lp_idx], dtype=np.int64)
            eta_seg = np.asarray(
                X[pred][:, jjl] @ beta_drop[jjl],
                dtype=np.float64,
            )
            if (
                fold_offsets is not None
                and lp_idx < len(fold_offsets)
                and fold_offsets[lp_idx] is not None
            ):
                eta_seg = eta_seg + np.asarray(fold_offsets[lp_idx], dtype=np.float64)
            eta_cv[d0 - pred.shape[0] : d0, lp_idx] = eta_seg

        if need_gradient and nsp > 0:
            for sp_idx in range(nsp):
                dg = np.asarray(dH[sp_idx] @ d_beta, dtype=np.float64)
                for ki in omit:
                    for j in range(nlp):
                        jjl = np.asarray(state.jj[j], dtype=np.int64)
                        for q in range(nlp):
                            jjq = np.asarray(state.jj[q], dtype=np.int64)
                            v = float(
                                sum(
                                    l3[ki, _i3f(j, q, r, nlp)]
                                    * deta[r * n + ki, sp_idx]
                                    for r in range(nlp)
                                )
                            )
                            xx = float(np.dot(X[ki, jjq], d_beta[jjq])) * v
                            dg[jjl] -= X[ki, jjl] * xx
                            xx = (
                                float(np.dot(X[ki, jjq], db[jjq, sp_idx]))
                                * l2[ki, _i2f(j, q, nlp)]
                            )
                            dg[jjl] += X[ki, jjl] * xx

                if pdef:
                    d1_beta = _solve_from_upper_chol(R0, dg)
                else:
                    u = (
                        np.column_stack(ddbuf)
                        if len(ddbuf) > 0
                        else np.empty((p, 0), dtype=np.float64)
                    )
                    d1_beta, _ = _minres(R0, u, dg)
                for q in range(nlp):
                    jjq = np.asarray(state.jj[q], dtype=np.int64)
                    deta_cv[
                        (q * no) + (d0 - pred.shape[0]) : (q * no) + d0,
                        sp_idx,
                    ] = np.asarray(
                        X[pred][:, jjq] @ (db[jjq, sp_idx] - d1_beta[jjq]),
                        dtype=np.float64,
                    )

    return _NCVFoldResult(
        eta_cv=np.asarray(eta_cv, dtype=np.float64),
        deta_cv=np.asarray(deta_cv, dtype=np.float64),
        dbeta_cv=np.asarray(dbeta_cv, dtype=np.float64),
    )


def _ordinary_qapprox_weights(model, state, y):
    family = model.family
    eta = np.asarray(state.eta, dtype=np.float64)
    mu = np.asarray(state.mu, dtype=np.float64)
    weights = _prior_weights(model, y)

    mevg = np.asarray(family.mu_eta(eta), dtype=np.float64)
    var_mu = np.asarray(family.variance(mu), dtype=np.float64)
    ww = weights * (np.asarray(y, dtype=np.float64) - mu) * mevg / var_mu
    ww[~np.isfinite(ww)] = 0.0

    w1 = np.asarray(state.kernel.current.W, dtype=np.float64)
    fisher_w = np.asarray(state.sol["fisher_weights"], dtype=np.float64)
    V1 = np.asarray(family.dvar(mu), dtype=np.float64) / var_mu
    g2 = np.asarray(family.d2link(mu), dtype=np.float64) * mevg
    fisher = bool(
        np.allclose(
            w1,
            fisher_w,
            rtol=1e-12,
            atol=1e-12 * max(1.0, float(np.max(np.abs(fisher_w)))),
        )
    )
    if fisher:
        alpha1 = np.zeros_like(w1, dtype=np.float64)
    else:
        V2 = np.asarray(family.d2var(mu), dtype=np.float64) / var_mu
        g3 = np.asarray(family.d3link(mu), dtype=np.float64) * mevg
        alpha = np.divide(
            w1,
            fisher_w,
            out=np.zeros_like(w1, dtype=np.float64),
            where=np.abs(fisher_w) > 0.0,
        )
        alpha1 = np.divide(
            -(V1 + g2)
            + (np.asarray(y, dtype=np.float64) - mu) * (V2 - V1 * V1 + g3 - g2 * g2),
            alpha,
            out=np.zeros_like(alpha, dtype=np.float64),
            where=np.abs(alpha) > 0.0,
        )
    w3 = w1 * mevg * (alpha1 - V1 - 2.0 * g2)
    w3[~np.isfinite(w3)] = 0.0
    return ww, w1, w3


def _ordinary_ncv_score(model, y, state, fold, *, qapprox):
    family = model.family
    nei = _resolve_nei(model, len(y))
    d_idx = np.asarray(nei["d"], dtype=np.int64)
    y_d = np.asarray(y, dtype=np.float64)[d_idx]
    weights_d = _prior_weights(model, y)[d_idx]
    eta_d = np.asarray(state.eta[d_idx], dtype=np.float64)
    deta_d = np.asarray(state.deta[d_idx, :], dtype=np.float64)
    gamma = float(model.score_gamma)

    if qapprox:
        ww, w1, w3 = _ordinary_qapprox_weights(model, state, y)
        delta_eta = np.asarray(fold.eta_cv - eta_d, dtype=np.float64)
        dev0 = np.asarray(
            family.deviance_obs(y_d, state.mu[d_idx], weights_d), dtype=np.float64
        )
        score = float(
            np.sum(dev0)
            + gamma
            * np.sum(
                -2.0 * np.asarray(ww[d_idx], dtype=np.float64) * delta_eta
                + np.asarray(w1[d_idx], dtype=np.float64) * delta_eta * delta_eta
            )
        )
        if deta_d.shape[1] == 0:
            return score, np.empty((0,), dtype=np.float64), None
        else:
            ncv1 = (
                -2.0
                * np.asarray(ww[d_idx], dtype=np.float64)[:, None]
                * (
                    (1.0 - gamma) * deta_d
                    + gamma * np.asarray(fold.deta_cv, dtype=np.float64)
                )
                + 2.0
                * gamma
                * np.asarray(w1[d_idx], dtype=np.float64)[:, None]
                * (np.asarray(fold.deta_cv, dtype=np.float64) * delta_eta[:, None])
                + gamma
                * np.asarray(w3[d_idx], dtype=np.float64)[:, None]
                * deta_d
                * (delta_eta[:, None] ** 2)
            )
            grad = np.asarray(np.sum(ncv1, axis=0), dtype=np.float64)
        return score, grad, np.asarray(ncv1.T @ ncv1, dtype=np.float64)

    eta_cv = np.asarray(fold.eta_cv, dtype=np.float64)
    deta_cv = np.asarray(fold.deta_cv, dtype=np.float64)
    if True:
        eta_cv = gamma * eta_cv - (gamma - 1.0) * eta_d
        if deta_cv.shape[1] > 0 and gamma != 1.0:
            deta_cv = gamma * deta_cv - (gamma - 1.0) * deta_d
        gamma = 1.0

    mu_cv = np.asarray(family.inverse_link(eta_cv), dtype=np.float64)
    score = float(
        np.sum(np.asarray(family.deviance_obs(y_d, mu_cv, weights_d), dtype=np.float64))
    )
    if deta_cv.shape[1] == 0:
        return score, np.empty((0,), dtype=np.float64), None

    var_cv = np.asarray(family.variance(mu_cv), dtype=np.float64)
    mevg_cv = np.asarray(family.mu_eta(eta_cv), dtype=np.float64)
    ww1 = weights_d * (y_d - mu_cv) * mevg_cv / var_cv
    ww1[~np.isfinite(ww1)] = 0.0
    ncv1 = -2.0 * ww1[:, None] * deta_cv * gamma
    grad = np.asarray(np.sum(ncv1, axis=0), dtype=np.float64)
    return score, grad, np.asarray(ncv1.T @ ncv1, dtype=np.float64)


def _fixed_theta_negbin_ncv_score(model, y, state, fold, *, qapprox):
    family = model.family
    nei = _resolve_nei(model, len(y))
    d_idx = np.asarray(nei["d"], dtype=np.int64)
    y_d = np.asarray(y, dtype=np.float64)[d_idx]
    weights = _prior_weights(model, y)
    weights_d = weights[d_idx]
    eta_d = np.asarray(state.eta[d_idx], dtype=np.float64)
    deta_d = np.asarray(state.deta[d_idx, :], dtype=np.float64)
    gamma = float(model.score_gamma)

    dd = _negbin_ddeta_logtheta(
        family,
        y,
        state.mu,
        weights,
        deriv=2 if qapprox else 1,
    )
    dev0 = np.asarray(
        family.deviance_obs(y_d, state.mu[d_idx], weights_d),
        dtype=np.float64,
    )
    ls0 = family.ls(
        y_d,
        weights_d,
        theta=float(family.getTheta(False)),
        scale=1.0,
    )

    if qapprox:
        delta_eta = np.asarray(fold.eta_cv - eta_d, dtype=np.float64)
        qdev = (
            dev0
            + gamma * np.asarray(dd["Deta"][d_idx], dtype=np.float64) * delta_eta
            + 0.5
            * gamma
            * np.asarray(dd["Deta2"][d_idx], dtype=np.float64)
            * delta_eta
            * delta_eta
        )
        score = float(np.sum(qdev) / 2.0 - float(ls0["ls"]))
        if deta_d.shape[1] == 0:
            return score, np.empty((0,), dtype=np.float64), None
        ncv1 = (
            np.asarray(dd["Deta"][d_idx], dtype=np.float64)[:, None]
            * (
                (1.0 - gamma) * deta_d
                + gamma * np.asarray(fold.deta_cv, dtype=np.float64)
            )
            + gamma
            * np.asarray(dd["Deta2"][d_idx], dtype=np.float64)[:, None]
            * np.asarray(fold.deta_cv, dtype=np.float64)
            * delta_eta[:, None]
            + 0.5
            * gamma
            * np.asarray(dd["Deta3"][d_idx], dtype=np.float64)[:, None]
            * deta_d
            * (delta_eta[:, None] ** 2)
        ) / 2.0
        return (
            score,
            np.asarray(np.sum(ncv1, axis=0), dtype=np.float64),
            np.asarray(ncv1.T @ ncv1, dtype=np.float64),
        )

    mu_cv = np.asarray(family.inverse_link(fold.eta_cv), dtype=np.float64)
    dev_cv = np.asarray(family.deviance_obs(y_d, mu_cv, weights_d), dtype=np.float64)
    score = float(np.sum(dev_cv) / 2.0 - float(ls0["ls"]))
    dev_base = float(np.sum(dev0) / 2.0 - float(ls0["ls"]))
    if gamma != 1.0:
        score = gamma * score - (gamma - 1.0) * dev_base
    if deta_d.shape[1] == 0:
        return score, np.empty((0,), dtype=np.float64), None

    dd_cv = _negbin_ddeta_logtheta(
        family,
        y_d,
        mu_cv,
        weights_d,
        deriv=1,
    )
    grad = (
        np.asarray(dd_cv["Deta"], dtype=np.float64)[:, None]
        * np.asarray(fold.deta_cv, dtype=np.float64)
    ) / 2.0
    if gamma != 1.0:
        dev1 = (np.asarray(dd["Deta"][d_idx], dtype=np.float64)[:, None] * deta_d) / 2.0
        grad = gamma * grad - (gamma - 1.0) * dev1
    return (
        score,
        np.asarray(np.sum(grad, axis=0), dtype=np.float64),
        np.asarray(grad.T @ grad, dtype=np.float64),
    )


def _joint_negbin_ncv_score(model, y, state, fold, *, qapprox):
    family = model.family
    nei = _resolve_nei(model, len(y))
    d_idx = np.asarray(nei["d"], dtype=np.int64)
    y_d = np.asarray(y, dtype=np.float64)[d_idx]
    weights = _prior_weights(model, y)
    weights_d = weights[d_idx]
    eta_d = np.asarray(state.eta[d_idx], dtype=np.float64)
    deta_d = np.asarray(state.deta[d_idx, :], dtype=np.float64)
    gamma = float(model.score_gamma)

    dd = _negbin_ddeta_logtheta(
        family,
        y,
        state.mu,
        weights,
        deriv=2 if qapprox else 1,
    )
    dev0 = np.asarray(
        family.deviance_obs(y_d, state.mu[d_idx], weights_d),
        dtype=np.float64,
    )
    ls0 = family.ls(
        y_d,
        weights_d,
        theta=float(family.getTheta(False)),
        scale=1.0,
    )

    if qapprox:
        delta_eta = np.asarray(fold.eta_cv - eta_d, dtype=np.float64)
        qdev = (
            dev0
            + gamma * np.asarray(dd["Deta"][d_idx], dtype=np.float64) * delta_eta
            + 0.5
            * gamma
            * np.asarray(dd["Deta2"][d_idx], dtype=np.float64)
            * delta_eta
            * delta_eta
        )
        score = float(np.sum(qdev) / 2.0 - float(ls0["ls"]))
        ncv1 = (
            np.asarray(dd["Deta"][d_idx], dtype=np.float64)[:, None]
            * (
                (1.0 - gamma) * deta_d
                + gamma * np.asarray(fold.deta_cv, dtype=np.float64)
            )
            + gamma
            * np.asarray(dd["Deta2"][d_idx], dtype=np.float64)[:, None]
            * np.asarray(fold.deta_cv, dtype=np.float64)
            * delta_eta[:, None]
            + 0.5
            * gamma
            * np.asarray(dd["Deta3"][d_idx], dtype=np.float64)[:, None]
            * deta_d
            * (delta_eta[:, None] ** 2)
        ) / 2.0
        ncv1[:, 0] = (
            ncv1[:, 0]
            - np.asarray(ls0["LSTH1"], dtype=np.float64).ravel()
            + (
                np.asarray(dd["Dth"][d_idx], dtype=np.float64)
                + gamma * np.asarray(dd["Detath"][d_idx], dtype=np.float64) * delta_eta
                + 0.5
                * gamma
                * np.asarray(dd["Deta2th"][d_idx], dtype=np.float64)
                * delta_eta
                * delta_eta
            )
            / 2.0
        )
        return (
            score,
            np.asarray(np.sum(ncv1, axis=0), dtype=np.float64),
            np.asarray(ncv1.T @ ncv1, dtype=np.float64),
        )

    mu_cv = np.asarray(family.inverse_link(fold.eta_cv), dtype=np.float64)
    dev_cv = np.asarray(family.deviance_obs(y_d, mu_cv, weights_d), dtype=np.float64)
    score = float(np.sum(dev_cv) / 2.0 - float(ls0["ls"]))
    dev_base = float(np.sum(dev0) / 2.0 - float(ls0["ls"]))
    dd_cv = _negbin_ddeta_logtheta(
        family,
        y_d,
        mu_cv,
        weights_d,
        deriv=1,
    )
    ncv1 = (
        np.asarray(dd_cv["Deta"], dtype=np.float64)[:, None]
        * np.asarray(fold.deta_cv, dtype=np.float64)
    ) / 2.0
    ncv1[:, 0] = (
        ncv1[:, 0]
        + np.asarray(dd_cv["Dth"], dtype=np.float64) / 2.0
        - np.asarray(ls0["LSTH1"], dtype=np.float64).ravel()
    )
    if gamma != 1.0:
        dev1 = (np.asarray(dd["Deta"][d_idx], dtype=np.float64)[:, None] * deta_d) / 2.0
        dev1[:, 0] = (
            dev1[:, 0]
            + np.asarray(dd["Dth"][d_idx], dtype=np.float64) / 2.0
            - np.asarray(ls0["LSTH1"], dtype=np.float64).ravel()
        )
        ncv1 = gamma * ncv1 - (gamma - 1.0) * dev1
        score = gamma * score - (gamma - 1.0) * dev_base
    return (
        score,
        np.asarray(np.sum(ncv1, axis=0), dtype=np.float64),
        np.asarray(ncv1.T @ ncv1, dtype=np.float64),
    )


def _general_ncv_score(model, y, state, fold, *, qapprox, need_gradient):
    family = model.family
    nei = _resolve_nei(model, len(y))
    d_idx = np.asarray(nei["d"], dtype=np.int64)
    weights = _prior_weights(model, y)
    nm = int(d_idx.size)
    n = int(state.X.shape[0])
    nlp = int(len(state.jj))
    nsp = int(state.dbeta.shape[1])
    gamma = float(model.score_gamma)
    dev = (
        -float(np.sum(np.asarray(state.llf["l0"], dtype=np.float64)[d_idx]))
        if (gamma != 1.0 or qapprox)
        else 0.0
    )

    if qapprox:
        delta = np.asarray(fold.eta_cv, dtype=np.float64) - np.asarray(
            state.eta[d_idx, :], dtype=np.float64
        )
        score = dev - gamma * float(
            np.sum(np.asarray(state.llf["l1"], dtype=np.float64)[d_idx, :] * delta)
        )
        for i in range(nlp):
            for j in range(i, nlp):
                score -= (
                    0.5
                    * gamma
                    * (1 + (i != j))
                    * float(
                        np.sum(
                            np.asarray(state.llf["l2"], dtype=np.float64)[
                                d_idx, _i2f(i, j, nlp)
                            ]
                            * delta[:, i]
                            * delta[:, j]
                        )
                    )
                )
        if not need_gradient or nsp == 0:
            return float(score), np.empty((0,), dtype=np.float64), None

        ncv1 = np.zeros((nm, nsp), dtype=np.float64)
        ll1 = np.asarray(state.llf["l1"], dtype=np.float64)
        ll2 = np.asarray(state.llf["l2"], dtype=np.float64)
        ll3 = np.asarray(state.llf["l3"], dtype=np.float64)
        deta = np.asarray(state.deta, dtype=np.float64)
        deta_cv = np.asarray(fold.deta_cv, dtype=np.float64)

        for lp_idx in range(nlp):
            row_block = slice(lp_idx * nm, (lp_idx + 1) * nm)
            full_rows = np.arange(lp_idx * n, lp_idx * n + nm, dtype=np.int64)
            ncv1 -= np.asarray(ll1[d_idx, lp_idx], dtype=np.float64)[:, None] * (
                deta_cv[row_block, :] * gamma + (1.0 - gamma) * deta[full_rows, :]
            )

        kk = 0
        jj = 0
        for j in range(nlp):
            rowj = np.arange(j * nm, j * nm + nm, dtype=np.int64)
            deta_j_drop = deta[d_idx + j * n, :]
            delta_j = delta[:, j][:, None]
            for k in range(j, nlp):
                rowk = np.arange(k * nm, k * nm + nm, dtype=np.int64)
                deta_k_drop = deta[d_idx + k * n, :]
                delta_k = delta[:, k][:, None]
                l2_col = np.asarray(ll2[d_idx, kk], dtype=np.float64)[:, None]
                ncv1 -= (
                    l2_col
                    * (
                        (deta[rowk, :] + deta_cv[rowk, :]) * delta_j
                        + delta_k * (deta_cv[rowj, :] - deta_j_drop)
                    )
                    * gamma
                    * 0.5
                )
                if j != k:
                    ncv1 -= (
                        l2_col
                        * (
                            (deta[rowj, :] + deta_cv[rowj, :]) * delta_k
                            + delta_j * (deta_cv[rowk, :] - deta_k_drop)
                        )
                        * gamma
                        * 0.5
                    )
                for lp_idx in range(k, nlp):
                    l3_col = np.asarray(ll3[d_idx, jj], dtype=np.float64)[:, None]
                    delta_l = delta[:, lp_idx][:, None]
                    deta_l_drop = deta[d_idx + lp_idx * n, :]
                    ncv1 -= (
                        (1 + (j != k))
                        * gamma
                        * 0.5
                        * l3_col
                        * deta_l_drop
                        * delta_k
                        * delta_j
                    )
                    if lp_idx != k:
                        ncv1 -= (
                            (1 + (lp_idx != j and j != k))
                            * gamma
                            * 0.5
                            * l3_col
                            * deta_k_drop
                            * delta_l
                            * delta_j
                        )
                    if lp_idx != j:
                        ncv1 -= (
                            (1 + (lp_idx != k and j != k))
                            * gamma
                            * 0.5
                            * l3_col
                            * deta_j_drop
                            * delta_k
                            * delta_l
                        )
                    jj += 1
                kk += 1
        return (
            float(score),
            np.asarray(np.sum(ncv1, axis=0), dtype=np.float64),
            np.asarray(ncv1.T @ ncv1, dtype=np.float64),
        )

    offi = _subset_offset_list(state.offset_list, d_idx)
    ll_cv = family.ll(
        np.asarray(y, dtype=np.float64)[d_idx],
        np.asarray(state.X[d_idx, :], dtype=np.float64),
        state.jj,
        np.asarray(state.beta, dtype=np.float64),
        weights[d_idx],
        offset=offi,
        deriv=1,
        eta=np.asarray(fold.eta_cv, dtype=np.float64),
        ncv=True,
    )
    score = -float(ll_cv["l"])
    score = gamma * score - (gamma - 1.0) * dev
    if not need_gradient or nsp == 0:
        return float(score), np.empty((0,), dtype=np.float64), None

    ncv1 = np.zeros((nm, nsp), dtype=np.float64)
    dev1 = np.zeros((nm, nsp), dtype=np.float64)
    for i in range(nlp):
        rows_i = slice(i * nm, (i + 1) * nm)
        full_rows_i = slice(i * n, (i + 1) * n)
        ncv1 -= np.asarray(ll_cv["l1"], dtype=np.float64)[:, i][:, None] * np.asarray(
            fold.deta_cv[rows_i, :], dtype=np.float64
        )
        if gamma != 1.0:
            dev1 -= (
                np.asarray(state.llf["l1"], dtype=np.float64)[:, i][:, None]
                * np.asarray(state.deta[full_rows_i, :], dtype=np.float64)
            )[d_idx, :]
    ncv1 = gamma * ncv1 - (gamma - 1.0) * dev1
    return (
        float(score),
        np.asarray(np.sum(ncv1, axis=0), dtype=np.float64),
        np.asarray(ncv1.T @ ncv1, dtype=np.float64),
    )


def _jackknife_weights(family_class, n_obs, nei):
    nk = np.diff(
        np.concatenate(
            [np.array([0], dtype=np.int64), np.asarray(nei["ma"], dtype=np.int64)]
        )
    ).astype(np.float64)
    if str(family_class).lower() == "glm":
        return np.asarray(((float(n_obs) - nk) / float(n_obs)) ** 0.4 / np.sqrt(nk))
    return np.asarray(np.sqrt((float(n_obs) - nk) / (float(n_obs) * nk)))


def _store_ncv_result(model, score, fold, *, Vg=None, dd=None):
    model._ncv_result_ = {
        "criterion_value": float(score),
        "eta_cv": None if fold is None else np.asarray(fold.eta_cv, dtype=np.float64),
        "deta_cv": None if fold is None else np.asarray(fold.deta_cv, dtype=np.float64),
        "Vg": None if Vg is None else np.asarray(Vg, dtype=np.float64),
        "dd": None if dd is None else np.asarray(dd, dtype=np.float64),
    }


def _restore_general_ncv_jackknife_dd(state, fold_dbeta):
    dd = np.asarray(fold_dbeta, dtype=np.float64).T
    if state.rp_state is not None:
        dd = np.asarray(_sl_repa(state.rp_state["rp"], dd, l=-1), dtype=np.float64)
    if len(state.run["setup"].Sl) > 0:
        cols = [
            np.asarray(
                sl_initial_repara(
                    state.run["setup"].Sl,
                    dd[:, j],
                    inverse=True,
                    both_sides=False,
                    cov=False,
                ),
                dtype=np.float64,
            )
            for j in range(dd.shape[1])
        ]
        dd = (
            np.column_stack(cols)
            if len(cols) > 0
            else np.empty((dd.shape[0], 0), dtype=np.float64)
        )
    return np.asarray(dd.T, dtype=np.float64)


def _criterion_ncv_full(model, y, log_sp, *, qapprox, need_gradient):
    y = np.asarray(y, dtype=np.float64).ravel()
    family = getattr(model, "family", None)
    family_class = str(getattr(family, "family_class", "")).lower()
    family_name = str(getattr(family, "name", "")).lower()
    nei = _resolve_nei(model, len(y))
    if need_gradient and int(nei.get("jackknife", 0)) > 2:
        raise ValueError("jackknife and derivatives requested together")

    if family_name == "negbin" and bool(getattr(family, "estimate_theta", False)):
        raise NotImplementedError(
            "Joint negative-binomial NCV/QNCV criteria require the joint "
            "`(log sp, log theta)` objective."
        )

    if family_class == "general":
        if family_name not in {"gaulss", "gammals", "ziplss", "gevlss", "shashlss"}:
            raise NotImplementedError(
                f"NCV/QNCV is not implemented for family={family_name!r}."
            )
        state_g = _build_general_ncv_state(
            model, y, log_sp, need_gradient=need_gradient
        )
        try:
            fold = _compute_general_fold_predictions_rncvls(
                state_g, nei, need_gradient=need_gradient
            )
        except np.linalg.LinAlgError:
            fold = _compute_general_fold_predictions_ncvls(
                state_g, nei, need_gradient=need_gradient
            )
        score, full_grad, Vg = _general_ncv_score(
            model,
            y,
            state_g,
            fold,
            qapprox=qapprox,
            need_gradient=need_gradient,
        )
        dd = None
        if int(nei.get("jackknife", 0)) > 2:
            dd = _jackknife_weights(family_class, len(y), nei)[
                :, None
            ] * _restore_general_ncv_jackknife_dd(
                state_g,
                fold.dbeta_cv,
            )
        _store_ncv_result(model, score, fold, Vg=Vg, dd=dd)
        if need_gradient:
            free_grad = np.asarray(full_grad[state_g.free_mask], dtype=np.float64)
            model._pirls_reml_derivative_kernel_state_ = {
                "dVkk": np.asarray(state_g.dVkk[state_g.free_mask], dtype=np.float64)
            }
            return float(score), np.asarray(full_grad, dtype=np.float64), free_grad
        return float(score), None, None

    if family_class not in {"glm", "extended"}:
        raise NotImplementedError(
            f"NCV/QNCV is not implemented for family_class={family_class!r}."
        )
    if family_class == "extended" and family_name != "negbin":
        raise NotImplementedError(
            f"NCV/QNCV is implemented only for negbin among extended families, got {family_name!r}."
        )

    state = _build_ncv_kernel_state(model, y, log_sp)
    state.kernel_dbeta = _column_stack_or_empty(
        state.kernel.ift.dbeta,
        state.kernel.current.beta.size,
    )
    if family_class == "glm":
        ww = (
            _prior_weights(model, y)
            * (y - state.mu)
            * model.family.mu_eta(state.eta)
            / model.family.variance(state.mu)
        )
        ww[~np.isfinite(ww)] = 0.0
        state.drop_grad_weights = np.asarray(ww, dtype=np.float64)
    else:
        dd0 = _negbin_ddeta_logtheta(
            family, y, state.mu, _prior_weights(model, y), deriv=0
        )
        state.drop_grad_weights = np.asarray(-dd0["Deta"] / 2.0, dtype=np.float64)

    fold = _compute_fold_predictions(state, nei)
    if family_class == "glm":
        score, full_grad, Vg = _ordinary_ncv_score(
            model, y, state, fold, qapprox=qapprox
        )
    else:
        score, full_grad, Vg = _fixed_theta_negbin_ncv_score(
            model,
            y,
            state,
            fold,
            qapprox=qapprox,
        )

    dd = None
    if int(nei.get("jackknife", 0)) > 2:
        weights = _jackknife_weights(family_class, len(y), nei)
        restored = [
            _restore_pirls_dbeta_to_original_parameterization(
                state.kernel.current,
                np.asarray(row, dtype=np.float64),
            )
            for row in np.asarray(fold.dbeta_cv, dtype=np.float64)
        ]
        dd = weights[:, None] * np.vstack(restored)

    _store_ncv_result(model, score, fold, Vg=Vg, dd=dd)
    if need_gradient:
        free_grad = np.asarray(full_grad[state.free_mask], dtype=np.float64)
        model._pirls_reml_derivative_kernel_state_ = {
            "dVkk": np.asarray(state.dVkk[state.free_mask], dtype=np.float64)
        }
        return float(score), np.asarray(full_grad, dtype=np.float64), free_grad
    return float(score), None, None


def criterion_ncv(model, y, log_sp, *, qapprox=False):
    score, _full_grad, _free_grad = _criterion_ncv_full(
        model,
        y,
        log_sp,
        qapprox=bool(qapprox),
        need_gradient=False,
    )
    return float(score)


def criterion_gradient_ncv(model, y, log_sp, *, qapprox=False):
    _score, _full_grad, free_grad = _criterion_ncv_full(
        model,
        y,
        log_sp,
        qapprox=bool(qapprox),
        need_gradient=True,
    )
    return np.asarray(free_grad, dtype=np.float64)


def criterion_ncv_negbin_joint(model, y, log_sp, log_theta, *, qapprox=False):
    y = np.asarray(y, dtype=np.float64).ravel()
    nei = _resolve_nei(model, len(y))
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        state = _build_negbin_joint_ncv_state(model, y, log_sp, log_theta)
        fold = _compute_fold_predictions(state, nei)
        score, _full_grad, Vg = _joint_negbin_ncv_score(
            model,
            y,
            state,
            fold,
            qapprox=bool(qapprox),
        )
        dd = None
        if int(nei.get("jackknife", 0)) > 2:
            weights = _jackknife_weights("extended", len(y), nei)
            restored = [
                _restore_pirls_dbeta_to_original_parameterization(
                    state.kernel.current,
                    np.asarray(row, dtype=np.float64),
                )
                for row in np.asarray(fold.dbeta_cv, dtype=np.float64)
            ]
            dd = weights[:, None] * np.vstack(restored)
        _store_ncv_result(model, score, fold, Vg=Vg, dd=dd)
        return float(score)
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)


def criterion_gradient_ncv_negbin_joint(model, y, log_sp, log_theta, *, qapprox=False):
    y = np.asarray(y, dtype=np.float64).ravel()
    nei = _resolve_nei(model, len(y))
    if int(nei.get("jackknife", 0)) > 2:
        raise ValueError("jackknife and derivatives requested together")
    prev_log_theta = float(model.family.getTheta(False))
    prev_disable_theta_efs = bool(getattr(model.family, "_disable_theta_efs", False))
    try:
        model.family.putTheta(float(log_theta))
        model.family._disable_theta_efs = True
        state = _build_negbin_joint_ncv_state(model, y, log_sp, log_theta)
        fold = _compute_fold_predictions(state, nei)
        score, full_grad, Vg = _joint_negbin_ncv_score(
            model,
            y,
            state,
            fold,
            qapprox=bool(qapprox),
        )
        _store_ncv_result(model, score, fold, Vg=Vg, dd=None)
        free_mask = np.asarray(state.free_mask, dtype=bool)
        dVkk = np.concatenate(
            [
                np.asarray(state.dVkk[1:][free_mask], dtype=np.float64),
                np.array([float(state.dVkk[0])], dtype=np.float64),
            ]
        )
        model._pirls_reml_derivative_kernel_state_ = {"dVkk": dVkk}
        return np.concatenate(
            [
                np.asarray(full_grad[1:][free_mask], dtype=np.float64),
                np.array([float(full_grad[0])], dtype=np.float64),
            ]
        )
    finally:
        model.family.putTheta(prev_log_theta)
        model.family._disable_theta_efs = bool(prev_disable_theta_efs)
