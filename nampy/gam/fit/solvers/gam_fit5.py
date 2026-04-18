"""
Newton penalized log-likelihood fitter for general families.

Mirrors mgcv ``gam.fit5`` from mgcv/R/gam.fit4.r.

Entry points
------------
``gam_fit5``
    Inner Newton loop: fits coefficients at fixed log smoothing parameters.
    Returns fit object including REML score, gradient, and Hessian w.r.t. lsp.

``gam_fit5_post_proc``
    Post-processing: Bayesian/frequentist covariance matrices, EDF.
    Mirrors mgcv ``gam.fit5.post.proc``.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from ..state import _mgcv_dchol, _mgcv_vcorr

# ---------------------------------------------------------------------------
# Control parameters  (mgcv: gam.control)
# ---------------------------------------------------------------------------


@dataclass
class GamFit5Control:
    maxit: int = 200
    epsilon: float = 1e-7
    trace: bool = False


# ---------------------------------------------------------------------------
# Main Newton fitter  (mgcv: gam.fit5)
# ---------------------------------------------------------------------------


def gam_fit5(
    X: np.ndarray,
    y: np.ndarray,
    jj: list[np.ndarray],
    lsp: np.ndarray,
    St: np.ndarray,
    S_blocks: list[np.ndarray],
    ldetS: float,
    ldetS1: np.ndarray | None,
    ldetS2: np.ndarray | None,
    family: Any,
    *,
    weights: np.ndarray | None = None,
    offset: Any = None,
    deriv: int = 2,
    score_type: str = "REML",
    control: GamFit5Control | None = None,
    Mp: int = -1,
    start: np.ndarray | None = None,
    gamma: float = 1.0,
    Sl: Any | None = None,
) -> dict[str, Any]:
    """
    Fit a penalized log-likelihood model for a general (GAMLSS-style) family
    by Newton iteration.

    Parameters
    ----------
    X : (n, p) stacked model matrix.
    y : (n,) response.
    jj : list of K integer index arrays (``jj[k]`` = columns for predictor k).
        Mirrors mgcv ``attr(x, "lpi")``.
    lsp : (m,) log smoothing parameters.
    St : (p, p) total penalty = sum_k exp(lsp[k]) * S_k (coefficient-space only,
        no intercept row/column).
    S_blocks : list of (p, p) unscaled penalty matrices, one per smoothing parameter.
    ldetS : float  log|S|_+.
    ldetS1 : (m,) or None  first derivatives of log|S|_+ w.r.t. lsp.
    ldetS2 : (m, m) or None  second derivatives.
    family : GamlssFamily
        Must implement ``family.ll(y, X, jj, coef, weights, offset, deriv, ...)``
        and ``family.initialize(y, X, jj, offset, weights, E)``.
    weights : (n,) or None.
    offset : per-predictor offsets (list of arrays) or scalar or None.
    deriv : 0 → fit only; 1 → +gradient of REML; 2 → +Hessian.
    score_type : "REML" or "ML".
    control : GamFit5Control.
    Mp : dimension of penalty null space (−1 to ignore).
    start : (p,) starting coefficients.
    gamma : EDF multiplier (default 1.0).

    Returns
    -------
    dict with keys: coef, lbb, L, D, bdrop, St_full,
                    l, REML, REML1, REML2, db_drho, dH, iter, warn, rank, ldetHp.

    Mirrors mgcv ``gam.fit5``.
    """
    if control is None:
        control = GamFit5Control()

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    lsp = np.asarray(lsp, dtype=np.float64).ravel()
    n, p = X.shape
    q = p
    nSp = len(lsp)
    warn: list[str] = []

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()

    use_exact_sl = Sl is not None and _sl_length(Sl) > 0
    rp_state: dict[str, Any] | None = None

    if use_exact_sl:
        rp_state = _sl_ldetS(
            Sl,
            rho=lsp,
            fixed=np.zeros(nSp, dtype=bool),
            np_=q,
            root=True,
            Stot=True,
            deriv=deriv,
        )
        X = _sl_repara(rp_state["rp"], X)
        St = np.asarray(rp_state["S"], dtype=np.float64)
        E = np.asarray(rp_state["E"], dtype=np.float64)
        Sb = _sl_repa(rp_state["rp"], _sl_total_penalty_matrix(Sl), l=-2, r=-1)
        if start is not None:
            start = _sl_repara(rp_state["rp"], np.asarray(start, dtype=np.float64))
        ldetS = float(rp_state["ldetS"])
        ldetS1 = np.asarray(rp_state["ldet1"], dtype=np.float64)
        ldetS2 = np.asarray(rp_state["ldet2"], dtype=np.float64)
    elif Sl is not None and _sl_length(Sl) == 0:
        deriv = 0
        St = np.zeros((q, q), dtype=np.float64)
        E = np.empty((0, q), dtype=np.float64)
        Sb = np.zeros((q, q), dtype=np.float64)
        rp_state = {
            "ldetS": 0.0,
            "ldet1": np.zeros(nSp, dtype=np.float64),
            "ldet2": np.zeros((nSp, nSp), dtype=np.float64),
            "rp": [],
            "Sl": [],
            "E": E,
            "S": St,
        }
        ldetS = 0.0
        ldetS1 = np.zeros(nSp, dtype=np.float64)
        ldetS2 = np.zeros((nSp, nSp), dtype=np.float64)
    else:
        St = np.asarray(St, dtype=np.float64)
        E = _build_root_penalty(St).T
        Sb = np.asarray(St, dtype=np.float64)

    if start is None:
        start = family.initialize(y, X, jj, offset=offset, weights=weights, E=E)
    coef = np.asarray(start, dtype=np.float64).copy()

    llf = family.ll

    ll = llf(y, X, jj, coef, weights, offset=offset, deriv=1)
    ll0 = float(ll["l"]) - 0.5 * float(coef @ St @ coef)
    grad = np.asarray(ll["lb"], dtype=np.float64).ravel() - St @ coef
    iconv = bool(np.max(np.abs(grad)) < control.epsilon * max(abs(ll0), 1e-300))
    Hp = -np.asarray(ll["lbb"], dtype=np.float64) + St

    rank = q
    converged = False
    drop: list[int] | None = None
    bdrop = np.zeros(q, dtype=bool)
    perturbed = 0
    rank_checked = False
    eigen_fix = False
    L: np.ndarray | None = None
    D: np.ndarray | None = None
    iter_ = 0

    for iter_ in range(1, 2 * control.maxit + 1):
        # --- diagonal preconditioner ---
        diag_Hp = np.diag(Hp[:rank, :rank]).copy()
        if not np.all(np.isfinite(diag_Hp)):
            raise RuntimeError("Non-finite values in penalized Hessian diagonal.")

        D_thresh = max(np.max(diag_Hp), 0.0) * np.sqrt(np.finfo(np.float64).eps)
        indefinite = False
        if np.min(diag_Hp) <= 0.0:
            if -np.min(diag_Hp) < D_thresh:
                diag_Hp = np.where(diag_Hp < D_thresh, D_thresh, diag_Hp)
            else:
                indefinite = True

        Hp_r = Hp[:rank, :rank]
        if indefinite:
            min_d = abs(np.min(np.diag(Hp_r)))
            max_d = abs(np.max(np.diag(Hp_r)))
            Ip = np.eye(rank) * max_d * np.finfo(np.float64).eps ** 0.5
            Ib = np.eye(rank) * min_d
            Hp_work = Hp_r + Ip + Ib
            D_arr = np.ones(rank, dtype=np.float64)
        else:
            D_arr = diag_Hp ** (-0.5)
            Hp_work = D_arr[:, None] * Hp_r * D_arr[None, :]
            Ip = np.eye(rank) * np.finfo(np.float64).eps ** 0.5

        # Cholesky factorization with jitter if needed
        L_arr, ok = _safe_cholesky(Hp_work, Ip, eigen_fix=eigen_fix)
        if not ok:
            indefinite = True

        D = D_arr
        L = L_arr

        if converged:
            break

        # --- Newton step: step = D * L^{-1} L^{-T} * (D * grad[:rank]) ---
        rhs = D * grad[:rank]
        step = D * _chol_solve(L, rhs)

        c_norm = float(np.sqrt(np.sum(coef**2)))
        if c_norm > 0.0:
            s_norm = float(np.sqrt(np.sum(step**2)))
            if s_norm > 0.1 * c_norm:
                step = step * 0.1 * c_norm / s_norm
        s_norm = float(np.sqrt(np.sum(step**2)))

        # --- step halving ---
        coef1 = coef.copy()
        coef1[:rank] += step
        ll_try = llf(y, X, jj, coef1, weights, offset=offset, deriv=0)
        ll1 = float(ll_try["l"]) - 0.5 * float(coef1 @ St @ coef1)
        ll_old = ll

        khalf = 0
        fac = 2.0
        no_change = 0
        while (not np.isfinite(ll1) or ll1 <= ll0) and khalf < 25:
            step /= fac
            coef1 = coef.copy()
            coef1[:rank] += step
            ll_try = llf(y, X, jj, coef1, weights, offset=offset, deriv=0)
            ll1 = float(ll_try["l"]) - 0.5 * float(coef1 @ St @ coef1)
            if np.isfinite(ll1) and ll1 >= ll0:
                ll_try = llf(y, X, jj, coef1, weights, offset=offset, deriv=1)
            if np.isfinite(ll1) and ll1 == ll0:
                no_change += 1
            max_chg = np.max(np.abs(coef)) * np.finfo(np.float64).eps
            if np.max(np.abs(coef - coef1)) < max_chg or no_change > 1:
                khalf = 100
            khalf += 1
            if khalf > 5:
                fac = 5.0

        # --- steepest ascent fallback ---
        if not np.isfinite(ll1) or (ll1 <= ll0 and not iconv):
            gnorm = float(np.sqrt(np.sum(grad[:rank] ** 2))) + 1e-300
            step = grad[:rank] * s_norm / gnorm

        no_change = 0
        khalf_sa = 0
        while (not np.isfinite(ll1) or (ll1 <= ll0 and not iconv)) and khalf_sa < 25:
            step /= 10.0
            coef1 = coef.copy()
            coef1[:rank] += step
            ll_try = llf(y, X, jj, coef1, weights, offset=offset, deriv=0)
            ll1 = float(ll_try["l"]) - 0.5 * float(coef1 @ St @ coef1)
            if np.isfinite(ll1) and ll1 >= ll0:
                ll_try = llf(y, X, jj, coef1, weights, offset=offset, deriv=1)
            if np.isfinite(ll1) and ll1 == ll0:
                no_change += 1
            max_chg = np.max(np.abs(coef)) * np.finfo(np.float64).eps
            if np.max(np.abs(coef - coef1)) < max_chg or no_change > 1:
                khalf_sa = 100
            khalf_sa += 1

        step_ok = (
            np.isfinite(ll1)
            and ll1 >= ll0
            and (khalf < 25 or indefinite or khalf_sa < 25)
        )

        if step_ok:
            coef = coef1.copy()
            ll = (
                ll_try
                if ll_try.get("lb") is not None
                else llf(y, X, jj, coef, weights, offset=offset, deriv=1)
            )
            grad = np.asarray(ll["lb"], dtype=np.float64).ravel() - St @ coef
            Hp = -np.asarray(ll["lbb"], dtype=np.float64) + St

            # convergence test
            ok = iter_ == control.maxit or float(
                np.max(np.abs(grad))
            ) < control.epsilon * max(abs(ll0), 1e-300)
            if ok:
                if indefinite:
                    if perturbed == 5:
                        raise RuntimeError(
                            "Indefinite penalized likelihood in gam_fit5."
                        )
                    if iter_ < 4 or rank_checked:
                        perturbed += 1
                        coef = _perturb_coef(coef, perturbed)
                        ll = llf(y, X, jj, coef, weights, offset=offset, deriv=1)
                        ll0 = float(ll["l"]) - 0.5 * float(coef @ St @ coef)
                        grad = (
                            np.asarray(ll["lb"], dtype=np.float64).ravel() - St @ coef
                        )
                        Hp = -np.asarray(ll["lbb"], dtype=np.float64) + St
                    else:
                        rank_checked = True
                        rank, drop, bdrop = _detect_rank_drop(ll["lbb"], Sb, coef, q, rank)
                        if rank < q:
                            coef, St, X, jj = _apply_rank_drop(
                                coef, St, X, jj, bdrop, q
                            )
                            if Sb.shape[0] >= np.sum(~bdrop):
                                Sb = Sb[np.ix_(~bdrop, ~bdrop)]
                            ll = llf(y, X, jj, coef, weights, offset=offset, deriv=1)
                            ll0 = float(ll["l"]) - 0.5 * float(coef @ St @ coef)
                            grad = (
                                np.asarray(ll["lb"], dtype=np.float64).ravel()
                                - St @ coef
                            )
                            Hp = -np.asarray(ll["lbb"], dtype=np.float64) + St
                else:
                    converged = True
            else:
                ll0 = ll1
        else:
            ll = ll_old
            if drop is None:
                bdrop = np.zeros(q, dtype=bool)
            if iconv and iter_ == 1:
                converged = True
                coef = np.asarray(start, dtype=np.float64).copy()
            else:
                warn.append(
                    f"gam_fit5 step failed at iter {iter_}: "
                    f"max|grad| = {float(np.max(np.abs(grad))):.4g}"
                )
            break

        iconv = bool(np.max(np.abs(grad)) < control.epsilon * max(abs(ll0), 1e-300))

    if iter_ == 2 * control.maxit and not converged:
        warn.append(
            f"gam_fit5 iteration limit reached: max|grad| = {float(np.max(np.abs(grad))):.4g}"
        )

    assert L is not None and D is not None
    ldetHp = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L), 1e-300)))) - 2.0 * float(
        np.sum(np.log(np.maximum(D, 1e-300)))
    )

    # Full coefficient vector (with zeros for dropped parameters)
    rank_eff = int(np.sum(~bdrop)) if np.any(bdrop) else rank
    if np.any(bdrop):
        fcoef = np.zeros(q, dtype=np.float64)
        fcoef[~bdrop] = coef
    else:
        fcoef = coef.copy()

    # Full St (with zeros for dropped)
    if np.any(bdrop):
        St_full = np.zeros((q, q), dtype=np.float64)
        St_full[np.ix_(~bdrop, ~bdrop)] = St
    else:
        St_full = St

    # ----------------------------------------------------------------
    # Implicit differentiation for REML derivatives  (mgcv lines 1244ff)
    # ----------------------------------------------------------------
    d1b = fd1b = None
    d2b = None
    d1ldetH = d2ldetH = None
    d1bSb = d2pen = None
    d2l = None
    dH = None
    REML1 = REML2 = None
    trHid2H = None

    keep = ~bdrop

    if deriv > 0 and nSp > 0 and rank_eff > 0:
        m = nSp
        d1b = np.zeros((rank_eff, m), dtype=np.float64)
        sp = np.exp(lsp)

        if use_exact_sl and rp_state is not None:
            Sib = _sl_term_mult(rp_state["Sl"], fcoef, full=True)
            for i in range(m):
                v = np.asarray(Sib[i], dtype=np.float64)[keep]
                d1b[:, i] = -D * _chol_solve(L, D * v)
        else:
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                )
                v = Si_r @ coef
                d1b[:, i] = -D * _chol_solve(L, D * v)

        fd1b = np.zeros((q, m), dtype=np.float64)
        fd1b[keep[:q], :] = d1b

        Hp_inv = D[:, None] * _chol_solve_matrix(L, np.diag(D))
        ll_trace = llf(
            y,
            X,
            jj,
            coef,
            weights,
            offset=offset,
            deriv=2,
            d1b=d1b,
            fh=Hp_inv,
        )
        d1ldetH_trace = ll_trace.get("d1H")
        dH = None

        if deriv > 1:
            ll_d3 = llf(y, X, jj, coef, weights, offset=offset, deriv=3, d1b=d1b)
            dH = ll_d3.get("d1H")

            d2b = np.zeros((rank_eff, m * (m + 1) // 2), dtype=np.float64)
            kk = 0
            for i in range(m):
                for j in range(i, m):
                    dH_i_v = np.zeros(rank_eff, dtype=np.float64)
                    if isinstance(dH, list) and len(dH) > i:
                        dH_i_v = (
                            np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                            @ d1b[:, j]
                        )
                    if use_exact_sl and rp_state is not None:
                        v = (
                            -dH_i_v
                            + np.asarray(
                                _sl_mult(rp_state["Sl"], fd1b[:, j], i + 1),
                                dtype=np.float64,
                            )[keep]
                            + np.asarray(
                                _sl_mult(rp_state["Sl"], fd1b[:, i], j + 1),
                                dtype=np.float64,
                            )[keep]
                        )
                    else:
                        Si_r = (
                            float(sp[i])
                            * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                        )
                        Sj_r = (
                            float(sp[j])
                            * S_blocks[j][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                        )
                        v = -dH_i_v + Si_r @ d1b[:, j] + Sj_r @ d1b[:, i]
                    d2b[:, kk] = -D * _chol_solve(L, D * v)
                    if i == j:
                        d2b[:, kk] += d1b[:, i]
                    kk += 1

            # trHid2H via ll with deriv=4
            ll_r = llf(
                y,
                X,
                jj,
                coef,
                weights,
                offset=offset,
                deriv=4,
                d1b=d1b,
                d2b=d2b,
                fh=L,
                D=D,
            )
            trHid2H = ll_r.get("trHid2H")
            d2l = np.zeros((m, m), dtype=np.float64)
            llbb_outer = np.asarray(ll_d3["lbb"], dtype=np.float64)
            for i in range(m):
                for j in range(i, m):
                    d2l[i, j] = d2l[j, i] = float(
                        d1b[:, i] @ (llbb_outer[:rank_eff, :rank_eff] @ d1b[:, j])
                    )

        if use_exact_sl and rp_state is not None:
            Skb = _sl_term_mult(rp_state["Sl"], fcoef, full=True)
            d1bSb = np.zeros(m, dtype=np.float64)
            for i in range(m):
                Skb[i] = np.asarray(Skb[i], dtype=np.float64)[keep]
                d1bSb[i] = float(np.sum(coef * Skb[i]))

            if deriv > 1:
                d2pen = np.zeros((m, m), dtype=np.float64)
                for i in range(m):
                    Sd1b = St @ d1b[:, i]
                    for j in range(i, m):
                        val_ij = 2.0 * float(
                            np.sum(
                                d1b[:, i] * Skb[j]
                                + d1b[:, j] * Skb[i]
                                + d1b[:, j] * Sd1b
                            )
                        )
                        if i == j:
                            val_ij += float(np.sum(coef * Skb[i]))
                        d2pen[i, j] = d2pen[j, i] = val_ij
        else:
            d1bSb = np.zeros(m, dtype=np.float64)
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                )
                d1bSb[i] = float(coef @ (Si_r @ coef))

            if deriv > 1:
                d2pen = np.zeros((m, m), dtype=np.float64)
                for i in range(m):
                    Si_r = (
                        float(sp[i])
                        * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                    )
                    for j in range(i, m):
                        val_ij = 2.0 * float(d1b[:, j] @ (Si_r @ coef))
                        if i == j:
                            val_ij += float(coef @ (Si_r @ coef))
                        d2pen[i, j] = val_ij
                        if j > i:
                            Sj_r = (
                                float(sp[j])
                                * S_blocks[j][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                            )
                            d2pen[j, i] = 2.0 * float(d1b[:, i] @ (Sj_r @ coef))
                d2pen = 0.5 * (d2pen + d2pen.T)

        d1ldetH = np.zeros(m, dtype=np.float64)
        d1Hp_list = []
        if use_exact_sl and rp_state is not None:
            eye_q = np.eye(q, dtype=np.float64)
            if (
                deriv == 1
                and d1ldetH_trace is not None
                and not isinstance(d1ldetH_trace, list)
            ):
                d1ldetH = -np.asarray(d1ldetH_trace, dtype=np.float64).ravel()
                for i in range(m):
                    A_full = np.asarray(
                        _sl_mult(rp_state["Sl"], eye_q, i + 1, full=True),
                        dtype=np.float64,
                    )[np.ix_(keep, keep)]
                    bind = np.sum(np.abs(A_full), axis=1) != 0.0
                    A = A_full[:, bind]
                    A = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A)
                    if np.any(bind):
                        d1ldetH[i] += float(np.trace(A[bind, :]))
                    if isinstance(dH, list):
                        dH_i = (
                            np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                            if len(dH) > i
                            else np.zeros((rank_eff, rank_eff))
                        )
                        A_hp = -dH_i + A_full
                        d1Hp_list.append(
                            D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A_hp)
                        )
            else:
                if dH is not None and not isinstance(dH, list):
                    d1ldetH = d1ldetH - np.asarray(dH, dtype=np.float64)
                for i in range(m):
                    A_full = np.asarray(
                        _sl_mult(rp_state["Sl"], eye_q, i + 1, full=True),
                        dtype=np.float64,
                    )[np.ix_(keep, keep)]
                    if isinstance(dH, list):
                        dH_i = (
                            np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                            if len(dH) > i
                            else np.zeros((rank_eff, rank_eff))
                        )
                        A_hp = -dH_i + A_full
                    else:
                        A_hp = A_full
                    Ai = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A_hp)
                    d1ldetH[i] = float(np.trace(Ai))
                    d1Hp_list.append(Ai)
        elif deriv == 1 and d1ldetH_trace is not None:
            d1ldetH = -np.asarray(d1ldetH_trace, dtype=np.float64).ravel()
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                )
                Ai = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ Si_r)
                d1ldetH[i] += float(np.trace(Ai))
                if isinstance(dH, list):
                    dH_i = (
                        np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                        if len(dH) > i
                        else np.zeros((rank_eff, rank_eff))
                    )
                    A = -dH_i + Si_r
                    d1Hp_list.append(D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A))
        elif isinstance(dH, list):
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                )
                dH_i = (
                    np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                    if len(dH) > i
                    else np.zeros((rank_eff, rank_eff))
                )
                A = -dH_i + Si_r
                Ai = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A)
                d1ldetH[i] = float(np.trace(Ai))
                d1Hp_list.append(Ai)
        else:
            if dH is not None:
                d1ldetH = d1ldetH - np.asarray(dH, dtype=np.float64)
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                )
                Ai = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ Si_r)
                d1ldetH[i] += float(np.trace(Ai))
                d1Hp_list.append(Ai)

        if deriv > 1 and trHid2H is not None and d1Hp_list:
            d2ldetH = np.zeros((m, m), dtype=np.float64)
            kk = 0
            for i in range(m):
                for j in range(i, m):
                    thr = float(trHid2H[kk]) if kk < len(trHid2H) else 0.0
                    d2ldetH[i, j] = d2ldetH[j, i] = (
                        -float(np.sum(d1Hp_list[i] * d1Hp_list[j].T)) - thr
                    )
                    if i == j:
                        if use_exact_sl and rp_state is not None:
                            A_full = np.asarray(
                                _sl_mult(rp_state["Sl"], np.eye(q, dtype=np.float64), i + 1),
                                dtype=np.float64,
                            )[np.ix_(keep, keep)]
                            bind = np.sum(np.abs(A_full), axis=1) != 0.0
                            A = A_full[:, bind]
                            A = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ A)
                            if np.any(bind):
                                d2ldetH[i, j] += float(np.trace(A[bind, :]))
                        else:
                            Si_r = (
                                float(sp[i])
                                * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                            )
                            Ai = D[:, None] * _chol_solve_matrix(L, np.diag(D) @ Si_r)
                            d2ldetH[i, j] += float(np.trace(Ai))
                    kk += 1

    # ----------------------------------------------------------------
    # Outer score  (mgcv lines 1409-1414; LAML shares REML branch here)
    # ----------------------------------------------------------------
    ll_val = float(ll["l"])
    bSb = 0.5 * float(fcoef @ St_full @ fcoef)
    score_name = str(score_type).upper()
    score_const = Mp * np.log(2.0 * np.pi) / 2.0 - np.log(max(gamma, 1e-300)) / 2.0
    score = -((ll_val - bSb) / gamma + float(ldetS) / 2.0 - ldetHp / 2.0 + score_const)
    REML = float(score)

    if deriv > 0 and d1bSb is not None and d1ldetH is not None:
        _ldetS1 = (
            np.asarray(ldetS1, dtype=np.float64)
            if ldetS1 is not None
            else np.zeros(nSp)
        )
        score1 = -(-d1bSb / (2.0 * gamma) + _ldetS1 / 2.0 - d1ldetH / 2.0)
        REML1 = np.asarray(score1, dtype=np.float64)

    if deriv > 1 and d2pen is not None and d2ldetH is not None:
        _ldetS2 = (
            np.asarray(ldetS2, dtype=np.float64)
            if ldetS2 is not None
            else np.zeros((nSp, nSp))
        )
        if use_exact_sl:
            score2 = -((d2l - d2pen / 2.0) / gamma + _ldetS2 / 2.0 - d2ldetH / 2.0)
        else:
            score2 = (d2pen - _ldetS2 + d2ldetH) / (2.0 * gamma)
        REML2 = np.asarray(score2, dtype=np.float64)

    if control.trace:
        print(
            f"gam_fit5: iter={iter_}  ll={ll_val:.6g}  {score_name}={score:.6g}  bSb={bSb:.6g}"
        )
        if REML1 is not None:
            print(f"  {score_name}1={REML1}")

    coef_out = fcoef
    db_drho_out = fd1b
    if use_exact_sl and rp_state is not None:
        coef_out = np.asarray(_sl_repara(rp_state["rp"], fcoef, inverse=True), dtype=np.float64)
        if db_drho_out is not None:
            db_drho_out = np.asarray(
                _sl_repa(rp_state["rp"], db_drho_out, l=-1),
                dtype=np.float64,
            )

    return {
        "coef": coef_out,
        "lbb": ll["lbb"],
        "L": L,
        "D": D,
        "bdrop": bdrop,
        "St_full": St_full,
        "l": ll_val,
        "REML": REML,
        "REML1": REML1,
        "REML2": REML2,
        "score_type": score_name,
        "score": float(score),
        "score1": REML1,
        "score2": REML2,
        "db_drho": db_drho_out,
        "dH": dH,
        "iter": iter_,
        "warn": warn,
        "rank": rank_eff,
        "ldetHp": ldetHp,
        "rp": [] if rp_state is None else list(rp_state["rp"]),
    }


# ---------------------------------------------------------------------------
# Post-processing: Vb, Ve, EDF  (mgcv: gam.fit5.post.proc)
# ---------------------------------------------------------------------------


def gam_fit5_post_proc(
    fit: dict,
    *,
    Sl: Any | None = None,
    L_map: np.ndarray | None = None,
    lsp0: np.ndarray | None = None,
    S_blocks: list[np.ndarray] | None = None,
    off: list[int] | None = None,
    outer_hess: np.ndarray | None = None,
    smoothing_params: np.ndarray | None = None,
) -> dict:
    """
    Compute Bayesian and frequentist covariance matrices and EDF.

    Mirrors mgcv ``gam.fit5.post.proc``.
    """
    lbb = -np.asarray(fit["lbb"], dtype=np.float64)
    L = fit["L"]
    D = fit["D"]
    bdrop = np.asarray(fit["bdrop"], dtype=bool)
    St_full = np.asarray(fit["St_full"], dtype=np.float64)
    q = St_full.shape[0]
    keep = ~bdrop
    p = int(np.sum(keep))

    lbb_r = lbb[:p, :p] if lbb.shape[0] >= p else lbb
    D_r = D[:p]
    lbb_c = D_r[:, None] * lbb_r * D_r[None, :]

    try:
        R_chol = cholesky(lbb_c, lower=False)
        R = R_chol / D_r[None, :]
    except np.linalg.LinAlgError:
        ev, U = np.linalg.eigh(lbb_c)
        ev = np.where(ev < 0.0, 0.0, ev)
        R_sq = U * np.sqrt(ev)[None, :]
        lbb_c = R_sq @ R_sq.T
        St_r = St_full[np.ix_(keep, keep)]
        Hp_c = lbb_c + D_r[:, None] * St_r * D_r[None, :]
        L = cholesky(Hp_c, lower=False)
        R = (R_sq / D_r[:, None]).T

    Vb = _compute_Hp_inv(L, D_r, p)

    if np.any(bdrop):
        Vb_full = np.zeros((q, q), dtype=np.float64)
        Vb_full[np.ix_(keep, keep)] = Vb
        Vb = Vb_full
        R_full = np.zeros((q, q), dtype=np.float64)
        R_full[np.ix_(keep, keep)] = R
        R = R_full

    if Sl is not None and _sl_length(Sl) > 0:
        Vb = _sl_repara(fit.get("rp", []), Vb, inverse=True)
        Vb = _sl_initial_repara_local(Sl, Vb, inverse=True)
        R = _sl_repa(fit.get("rp", []), R, r=1)
        R = _sl_initial_repara_local(
            Sl,
            R,
            inverse=True,
            both_sides=False,
            cov=False,
        )

    Vc = np.asarray(Vb, dtype=np.float64)
    Hsp = outer_hess
    if Hsp is None and fit.get("REML2", None) is not None:
        Hsp = np.asarray(fit["REML2"], dtype=np.float64)
    db_drho = None if fit.get("db_drho", None) is None else np.asarray(
        fit["db_drho"], dtype=np.float64
    )
    if Hsp is not None and db_drho is not None and db_drho.size > 0:
        Hsp = 0.5 * (np.asarray(Hsp, dtype=np.float64) + np.asarray(Hsp, dtype=np.float64).T)
        if L_map is not None:
            db_drho = np.asarray(
                db_drho @ np.asarray(L_map, dtype=np.float64), dtype=np.float64
            )
        if Sl is not None and _sl_length(Sl) > 0:
            cols = []
            for i in range(db_drho.shape[1]):
                cols.append(
                    np.asarray(
                        _sl_initial_repara_local(
                            Sl,
                            db_drho[:, i],
                            inverse=True,
                            both_sides=False,
                            cov=False,
                        ),
                        dtype=np.float64,
                    )
                )
            db_drho = np.column_stack(cols) if cols else db_drho[:, :0]
        evals, evecs = np.linalg.eigh(Hsp)
        pos = evals > 0.0
        inv_sqrt = np.zeros_like(evals)
        inv_sqrt[pos] = 1.0 / np.sqrt(evals[pos])
        Vsp = np.asarray(evecs @ ((inv_sqrt * inv_sqrt)[:, None] * evecs.T), dtype=np.float64)
        Vc = np.asarray(Vb + db_drho @ Vsp @ db_drho.T, dtype=np.float64)

        if S_blocks is not None and len(S_blocks) > 0:
            reg = np.zeros_like(evals)
            reg[pos] = 1.0 / np.sqrt(evals[pos] + 1.0 / 50.0)
            Vr = np.asarray(evecs @ ((reg * reg)[:, None] * evecs.T), dtype=np.float64)
            if smoothing_params is None:
                raise KeyError(
                    "gam_fit5_post_proc requires smoothing_params for Vb.corr parity."
                )
            Vc += _vb_corr_root(
                R,
                L=L_map,
                lsp0=lsp0,
                S_blocks=S_blocks,
                off=off,
                rho=np.log(
                    np.maximum(np.asarray(smoothing_params, dtype=np.float64), 1e-300)
                ),
                Vr=Vr,
            )

    Vc = 0.5 * (Vc + Vc.T)
    RTR = np.asarray(R.T @ R, dtype=np.float64)
    F = np.asarray(Vb @ RTR, dtype=np.float64)
    Ve = np.asarray(F @ Vb, dtype=np.float64)
    edf = np.asarray(np.diag(F), dtype=np.float64)
    edf1 = np.asarray(2.0 * edf - np.sum(F * F.T, axis=1), dtype=np.float64)
    edf2 = np.asarray(np.sum(Vc * RTR, axis=1), dtype=np.float64)
    if float(np.sum(edf2)) > float(np.sum(edf1)):
        edf2 = np.asarray(edf1, dtype=np.float64).copy()

    return {
        "Vp": Vb,
        "Ve": Ve,
        "Vc": Vc,
        "edf": edf,
        "edf1": edf1,
        "edf2": edf2,
        "R": R,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sl_blocks(Sl: Any) -> list[Any]:
    if Sl is None:
        return []
    blocks = getattr(Sl, "blocks", None)
    if blocks is not None:
        return list(blocks)
    return list(Sl)


def _sl_length(Sl: Any) -> int:
    return len(_sl_blocks(Sl))


def _sl_total_penalty_matrix(Sl: Any) -> np.ndarray:
    total = getattr(Sl, "S", None)
    if total is None:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(total, dtype=np.float64)


def _sl_initial_repara_local(Sl: Any, X: np.ndarray, **kwargs) -> np.ndarray:
    from .general_fit5 import sl_initial_repara

    return sl_initial_repara(Sl, X, **kwargs)


def _vb_corr_root(
    X_root: np.ndarray,
    *,
    L: np.ndarray | None,
    lsp0: np.ndarray | None,
    S_blocks: list[np.ndarray],
    off: list[int] | None,
    rho: np.ndarray,
    Vr: np.ndarray,
) -> np.ndarray:
    """Dense NumPy port of ``mgcv::Vb.corr`` for the ``w is NULL`` case."""

    rho = np.asarray(rho, dtype=np.float64).ravel()
    if rho.size == 0 or len(S_blocks) == 0:
        return np.zeros((X_root.shape[1], X_root.shape[1]), dtype=np.float64)

    if lsp0 is None:
        lsp0 = (
            np.zeros_like(rho)
            if L is None
            else np.zeros((np.asarray(L, dtype=np.float64).shape[0],), dtype=np.float64)
        )
    lsp0 = np.asarray(lsp0, dtype=np.float64).ravel()
    if L is None:
        lam = np.exp(rho + lsp0[: rho.size])
    else:
        L = np.asarray(L, dtype=np.float64)
        lam = np.exp(L @ rho + lsp0[: L.shape[0]])

    H = np.asarray(X_root.T @ X_root, dtype=np.float64)
    full_blocks = [np.asarray(Si, dtype=np.float64) for Si in S_blocks]
    if off is None:
        off = [1] * len(full_blocks)
    for i, Si in enumerate(full_blocks):
        H = H + float(lam[i]) * Si

    try:
        R = cholesky(H, lower=False)
    except np.linalg.LinAlgError:
        return np.zeros_like(H)

    dH = [float(lam[i]) * Si for i, Si in enumerate(full_blocks)]
    if L is not None:
        dH_linked = []
        for j in range(L.shape[1]):
            acc = None
            for i in range(L.shape[0]):
                if L[i, j] == 0.0:
                    continue
                term = dH[i] * float(L[i, j])
                acc = term if acc is None else acc + term
            if acc is not None:
                dH_linked.append(acc)
        dH = dH_linked
    if len(dH) == 0:
        return np.zeros_like(H)

    R_inv = solve_triangular(
        R,
        np.eye(R.shape[0], dtype=np.float64),
        lower=False,
        check_finite=False,
    )
    dR_inv = []
    for dHi in dH:
        dRi = _mgcv_dchol(np.asarray(dHi, dtype=np.float64), R)
        dR_inv.append(
            -(
                solve_triangular(R, dRi, lower=False, check_finite=False) @ R_inv
            )
        )
    return np.asarray(_mgcv_vcorr(dR_inv, np.asarray(Vr, dtype=np.float64), trans=False), dtype=np.float64)


def _sl_repara(
    rp: list[dict[str, Any]],
    X: np.ndarray,
    *,
    inverse: bool = False,
    both_sides: bool = True,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64).copy()
    if len(rp) == 0:
        return X_arr

    is_matrix = X_arr.ndim == 2
    if inverse:
        if is_matrix:
            for item in rp:
                if not bool(item.get("repara", False)):
                    continue
                ind = np.asarray(item["ind"], dtype=int)
                if both_sides:
                    X_arr[ind, :] = (
                        np.asarray(item["Ti"], dtype=np.float64) @ X_arr[ind, :]
                        if item.get("Qs", None) is None
                        else np.asarray(item["Qs"], dtype=np.float64) @ X_arr[ind, :]
                    )
                X_arr[:, ind] = (
                    X_arr[:, ind] @ np.asarray(item["Ti"], dtype=np.float64).T
                    if item.get("Qs", None) is None
                    else X_arr[:, ind] @ np.asarray(item["Qs"], dtype=np.float64).T
                )
        else:
            for item in rp:
                if not bool(item.get("repara", False)):
                    continue
                ind = np.asarray(item["ind"], dtype=int)
                X_arr[ind] = (
                    np.asarray(item["Ti"], dtype=np.float64) @ X_arr[ind]
                    if item.get("Qs", None) is None
                    else np.asarray(item["Qs"], dtype=np.float64) @ X_arr[ind]
                )
    else:
        if is_matrix:
            for item in rp:
                if not bool(item.get("repara", False)):
                    continue
                ind = np.asarray(item["ind"], dtype=int)
                X_arr[:, ind] = (
                    X_arr[:, ind] @ np.asarray(item["Ti"], dtype=np.float64)
                    if item.get("Qs", None) is None
                    else X_arr[:, ind] @ np.asarray(item["Qs"], dtype=np.float64)
                )
        else:
            for item in rp:
                if not bool(item.get("repara", False)):
                    continue
                ind = np.asarray(item["ind"], dtype=int)
                X_arr[ind] = (
                    np.asarray(item["T"], dtype=np.float64) @ X_arr[ind]
                    if item.get("Qs", None) is None
                    else np.asarray(item["Qs"], dtype=np.float64).T @ X_arr[ind]
                )

    return np.asarray(X_arr, dtype=np.float64)


def _sl_repa(
    rp: list[dict[str, Any]],
    X: np.ndarray,
    *,
    l: int = 0,  # noqa: E741
    r: int = 0,
) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64).copy()
    if len(rp) == 0:
        return X_arr

    is_matrix = X_arr.ndim == 2
    for item in rp:
        if not bool(item.get("repara", False)):
            continue
        ind = np.asarray(item["ind"], dtype=int)
        if l:
            if item.get("Qs", None) is None:
                if l < 0:
                    T = (
                        np.asarray(item["Ti"], dtype=np.float64).T
                        if l == -2
                        else np.asarray(item["Ti"], dtype=np.float64)
                    )
                else:
                    T = (
                        np.asarray(item["T"], dtype=np.float64).T
                        if l == 2
                        else np.asarray(item["T"], dtype=np.float64)
                    )
            else:
                Qs = np.asarray(item["Qs"], dtype=np.float64)
                if l < 0:
                    T = Qs.T if l == -2 else Qs
                else:
                    T = Qs if l == 2 else Qs.T
            if is_matrix:
                X_arr[ind, :] = T @ X_arr[ind, :]
            else:
                X_arr[ind] = T @ X_arr[ind]
        if r:
            if item.get("Qs", None) is None:
                if r < 0:
                    T = (
                        np.asarray(item["Ti"], dtype=np.float64).T
                        if r == -2
                        else np.asarray(item["Ti"], dtype=np.float64)
                    )
                else:
                    T = (
                        np.asarray(item["T"], dtype=np.float64).T
                        if r == 2
                        else np.asarray(item["T"], dtype=np.float64)
                    )
            else:
                Qs = np.asarray(item["Qs"], dtype=np.float64)
                if r < 0:
                    T = Qs.T if r == -2 else Qs
                else:
                    T = Qs if r == 2 else Qs.T
            if is_matrix:
                X_arr[:, ind] = X_arr[:, ind] @ T
            else:
                X_arr[ind] = X_arr[ind] @ T

    return np.asarray(X_arr, dtype=np.float64)


def _sl_term_mult(sl_blocks: list[Any], A: np.ndarray, *, full: bool = False) -> list[np.ndarray]:
    A_arr = np.asarray(A, dtype=np.float64)
    SA: list[np.ndarray] = []

    for block in sl_blocks:
        base_ind = np.arange(int(block.start) - 1, int(block.stop), dtype=int)
        if len(block.S) == 1:
            ind = (
                base_ind[np.asarray(block.ind, dtype=bool)]
                if bool(block.repara)
                else base_ind
            )
            lam = float(np.asarray(block.lambda_, dtype=np.float64).ravel()[0])
            local = lam * A_arr[ind, ...]
            if not bool(block.repara):
                local = lam * (
                    np.asarray(block.S[0], dtype=np.float64) @ A_arr[ind, ...]
                )
            if full:
                out = np.zeros_like(A_arr)
                out[ind, ...] = local
            else:
                out = np.asarray(local, dtype=np.float64)
            SA.append(np.asarray(out, dtype=np.float64))
            continue

        ind = (
            base_ind[np.asarray(block.ind, dtype=bool)]
            if bool(block.repara)
            else base_ind
        )
        srp = getattr(block, "Srp", None)
        for i in range(len(block.S)):
            if srp is None or not bool(block.repara):
                local_mat = (
                    float(np.asarray(block.lambda_, dtype=np.float64)[i])
                    * np.asarray(block.S[i], dtype=np.float64)
                )
            else:
                local_mat = np.asarray(srp[i], dtype=np.float64)
            local = local_mat @ A_arr[ind, ...]
            if full:
                out = np.zeros_like(A_arr)
                out[ind, ...] = local
            else:
                out = np.asarray(local, dtype=np.float64)
            SA.append(np.asarray(out, dtype=np.float64))

    return SA


def _sl_mult(
    sl_blocks: list[Any],
    A: np.ndarray,
    k: int = 0,
    *,
    full: bool = True,
) -> np.ndarray:
    A_arr = np.asarray(A, dtype=np.float64)
    if len(sl_blocks) == 0:
        return np.zeros_like(A_arr)

    if k <= 0:
        out = np.zeros_like(A_arr)
        for block in sl_blocks:
            base_ind = np.arange(int(block.start) - 1, int(block.stop), dtype=int)
            if len(block.S) == 1:
                if bool(block.repara):
                    ind = base_ind[np.asarray(block.ind, dtype=bool)]
                    lam = float(np.asarray(block.lambda_, dtype=np.float64).ravel()[0])
                    out[ind, ...] = lam * A_arr[ind, ...]
                else:
                    lam = float(np.asarray(block.lambda_, dtype=np.float64).ravel()[0])
                    out[base_ind, ...] = lam * (
                        np.asarray(block.S[0], dtype=np.float64) @ A_arr[base_ind, ...]
                    )
            else:
                ind = (
                    base_ind[np.asarray(block.ind, dtype=bool)]
                    if bool(block.repara)
                    else base_ind
                )
                out[ind, ...] = np.asarray(block.St, dtype=np.float64) @ A_arr[ind, ...]
        return np.asarray(out, dtype=np.float64)

    j = 0
    for block in sl_blocks:
        base_ind = np.arange(int(block.start) - 1, int(block.stop), dtype=int)
        for i in range(len(block.S)):
            j += 1
            if j != k:
                continue

            if len(block.S) == 1:
                if bool(block.repara):
                    ind = base_ind[np.asarray(block.ind, dtype=bool)]
                    lam = float(np.asarray(block.lambda_, dtype=np.float64).ravel()[0])
                    local = lam * A_arr[ind, ...]
                    if full:
                        out = np.zeros_like(A_arr)
                        out[ind, ...] = local
                        return np.asarray(out, dtype=np.float64)
                    return np.asarray(local, dtype=np.float64)

                local = float(np.asarray(block.lambda_, dtype=np.float64).ravel()[0]) * (
                    np.asarray(block.S[0], dtype=np.float64) @ A_arr[base_ind, ...]
                )
                if full:
                    out = np.zeros_like(A_arr)
                    out[base_ind, ...] = local
                    return np.asarray(out, dtype=np.float64)
                return np.asarray(local, dtype=np.float64)

            ind = (
                base_ind[np.asarray(block.ind, dtype=bool)]
                if bool(block.repara)
                else base_ind
            )
            srp = getattr(block, "Srp", None)
            if srp is None or not bool(block.repara):
                local_mat = (
                    float(np.asarray(block.lambda_, dtype=np.float64)[i])
                    * np.asarray(block.S[i], dtype=np.float64)
                )
            else:
                local_mat = np.asarray(srp[i], dtype=np.float64)
            local = local_mat @ A_arr[ind, ...]
            if full:
                out = np.zeros_like(A_arr)
                out[ind, ...] = local
                return np.asarray(out, dtype=np.float64)
            return np.asarray(local, dtype=np.float64)

    return np.zeros_like(A_arr)


def _sl_ldetS(
    Sl: Any,
    *,
    rho: np.ndarray,
    fixed: np.ndarray,
    np_: int,
    root: bool = False,
    Stot: bool = False,
    deriv: int = 2,
) -> dict[str, Any]:
    from ...smoothing_selection.reparam import gam_reparam

    rho = np.asarray(rho, dtype=np.float64).ravel()
    fixed = np.asarray(fixed, dtype=bool).ravel()
    blocks = [copy(block) for block in _sl_blocks(Sl)]
    n_deriv = int(np.sum(~fixed))
    ldS = 0.0
    d1 = np.zeros(n_deriv, dtype=np.float64)
    d2 = np.zeros((n_deriv, n_deriv), dtype=np.float64)
    k_deriv = 0
    k_sp = 0
    rp: list[dict[str, Any]] = []

    E = np.zeros((np_, np_), dtype=np.float64) if root else None
    S = np.zeros((np_, np_), dtype=np.float64) if Stot else None

    for b_idx, block in enumerate(blocks):
        if not bool(getattr(block, "linear", True)):
            raise NotImplementedError(
                "Non-linear general-family Sl blocks are not implemented."
            )

        if len(block.S) == 1:
            rank = int(block.rank if block.rank is not None else np.sum(block.ind))
            ldS += float(block.ldet) + float(rho[k_sp]) * rank
            block.lambda_ = np.array([np.exp(rho[k_sp])], dtype=np.float64)
            if not bool(fixed[k_sp]):
                d1[k_deriv] = rank
                k_deriv += 1

            if bool(block.repara):
                active = (
                    np.arange(int(block.start) - 1, int(block.stop), dtype=int)[
                        np.asarray(block.ind, dtype=bool)
                    ]
                )
                if E is not None and active.size > 0:
                    E[active, active] = np.exp(rho[k_sp] * 0.5)
                if S is not None and active.size > 0:
                    S[active, active] = np.exp(rho[k_sp])
            else:
                raise NotImplementedError(
                    "Non-reparameterized single-penalty general-family Sl blocks are unsupported."
                )

            k_sp += 1
            continue

        m = len(block.S)
        sp_ind = np.arange(k_sp, k_sp + m, dtype=int)
        grp = gam_reparam(block.rS, rho[sp_ind], deriv=deriv)
        block.lambda_ = np.exp(rho[sp_ind])
        block.St = np.asarray(grp["S"], dtype=np.float64)
        block.Srp = [
            float(block.lambda_[i]) * (np.asarray(grp["rS"][i], dtype=np.float64) @ np.asarray(grp["rS"][i], dtype=np.float64).T)
            for i in range(m)
        ]
        ldS += float(block.ldet) + float(grp["det"])

        free = ~fixed[sp_ind]
        nd = int(np.sum(free))
        if nd > 0:
            d1[k_deriv : k_deriv + nd] = np.asarray(grp["det1"], dtype=np.float64)[free]
            if deriv > 1:
                d2_block = np.asarray(grp["det2"], dtype=np.float64)[np.ix_(free, free)]
                d2[k_deriv : k_deriv + nd, k_deriv : k_deriv + nd] = d2_block
            k_deriv += nd

        if bool(block.repara):
            active = (
                np.arange(int(block.start) - 1, int(block.stop), dtype=int)[
                    np.asarray(block.ind, dtype=bool)
                ]
            )
            rp.append(
                {
                    "block": b_idx,
                    "ind": np.asarray(active, dtype=int),
                    "Qs": np.asarray(grp["Qs"], dtype=np.float64),
                    "repara": bool(block.repara),
                }
            )
            if E is not None:
                grp_E = np.asarray(grp["E"], dtype=np.float64)
                ir = np.arange(int(block.start) - 1, int(block.start) - 1 + grp_E.shape[0])
                ic = np.arange(int(block.start) - 1, int(block.start) - 1 + grp_E.shape[1])
                E[np.ix_(ir, ic)] = grp_E
            if S is not None:
                grp_S = np.asarray(grp["S"], dtype=np.float64)
                ir = np.arange(int(block.start) - 1, int(block.start) - 1 + grp_S.shape[0])
                ic = np.arange(int(block.start) - 1, int(block.start) - 1 + grp_S.shape[1])
                S[np.ix_(ir, ic)] = grp_S
        else:
            raise NotImplementedError(
                "Non-reparameterized multi-penalty general-family Sl blocks are unsupported."
            )

        k_sp += m

    if E is not None:
        keep_rows = np.sum(np.abs(E), axis=1) != 0.0
        E = E[keep_rows, :]

    return {
        "ldetS": float(ldS),
        "ldet1": np.asarray(d1, dtype=np.float64),
        "ldet2": np.asarray(d2, dtype=np.float64),
        "Sl": blocks,
        "rp": rp,
        "E": E,
        "S": S,
    }


def _build_root_penalty(St: np.ndarray) -> np.ndarray:
    """Square root of total penalty for initialization regularization."""
    ev, U = np.linalg.eigh(0.5 * (St + St.T))
    pos = ev > 0.0
    if not np.any(pos):
        return np.empty((St.shape[0], 0), dtype=np.float64)
    return U[:, pos] * np.sqrt(ev[pos])[None, :]


def _safe_cholesky(
    Hp_work: np.ndarray, Ip: np.ndarray, eigen_fix: bool = False  # noqa: ARG001
) -> tuple[np.ndarray, bool]:
    """Try Cholesky; apply jitter if needed.  Returns (L, ok)."""
    try:
        return cholesky(Hp_work, lower=False), True
    except np.linalg.LinAlgError:
        pass

    for _ in range(10):
        Ip = Ip * 100.0
        try:
            return cholesky(Hp_work + Ip, lower=False), False
        except np.linalg.LinAlgError:
            continue
    # Last resort: eigen fix
    ev, U = np.linalg.eigh(Hp_work)
    ev = np.abs(ev)
    ev = np.where(ev < ev.max() * 1e-10, ev.max() * 1e-10, ev)
    Hpf = U @ (ev[:, None] * U.T)
    try:
        return cholesky(Hpf, lower=False), False
    except np.linalg.LinAlgError:
        return np.eye(Hp_work.shape[0], dtype=np.float64), False


def _chol_solve(L: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve L.T @ L @ x = rhs (upper Cholesky)."""
    from scipy.linalg import solve_triangular

    z = solve_triangular(L, rhs, lower=False, trans="T")
    return solve_triangular(L, z, lower=False)


def _chol_solve_matrix(L: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve L.T @ L @ X = rhs column-by-column."""
    from scipy.linalg import solve_triangular

    z = solve_triangular(L, rhs, lower=False, trans="T")
    return solve_triangular(L, z, lower=False)


def _compute_Hp_inv(L: np.ndarray, D: np.ndarray, p: int) -> np.ndarray:
    """Hp^{-1} = D * L^{-1} L^{-T} * D (upper Chol L, diag precon D)."""
    eye = np.eye(p, dtype=np.float64)
    sol = _chol_solve_matrix(L, eye)
    return D[:, None] * sol * D[None, :]


def _perturb_coef(coef: np.ndarray, perturbed: int) -> np.ndarray:
    coef = coef.copy()
    alt = np.arange(len(coef)) % 2
    coef = coef * (1.0 + (alt * 0.02 - 0.01) * perturbed)
    coef += (alt - 0.5) * np.mean(np.abs(coef)) * 1e-5 * perturbed
    return coef


def _detect_rank_drop(
    lbb: np.ndarray,
    St: np.ndarray,
    coef: np.ndarray,  # noqa: ARG001
    q: int,
    rank: int,
) -> tuple[int, list[int] | None, np.ndarray]:
    Hb = -np.asarray(lbb, dtype=np.float64)[:rank, :rank]
    if St.shape[0] >= rank:
        Hb_n = Hb / (np.linalg.norm(Hb, "fro") + 1e-300)
        St_n = St[:rank, :rank] / (np.linalg.norm(St[:rank, :rank], "fro") + 1e-300)
        Hbal = Hb_n + St_n
    else:
        Hbal = Hb / (np.linalg.norm(Hb, "fro") + 1e-300)
    D_diag = np.abs(np.diag(Hbal))
    D_diag[D_diag < 1e-50] = 1.0
    D_half = D_diag ** (-0.5)
    Hbal_c = D_half[:, None] * Hbal * D_half[None, :]
    sv = np.linalg.svd(Hbal_c, compute_uv=False)
    eps = np.finfo(np.float64).eps
    new_rank = int(np.sum(sv > sv[0] * eps**0.5))
    if new_rank < rank:
        drop_local = list(np.argsort(sv)[: rank - new_rank])
        bdrop = np.zeros(q, dtype=bool)
        for d in drop_local:
            if d < q:
                bdrop[d] = True
        return new_rank, drop_local, bdrop
    return rank, None, np.zeros(q, dtype=bool)


def _apply_rank_drop(
    coef: np.ndarray,
    St: np.ndarray,
    X: np.ndarray,
    jj: list[np.ndarray],
    bdrop: np.ndarray,
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    keep = ~bdrop
    coef_r = coef[keep]
    St_r = St[np.ix_(keep, keep)]
    X_r = X[:, keep]
    remap = -np.ones(q, dtype=int)
    remap[keep] = np.arange(int(np.sum(keep)))
    jj_r = [np.array([remap[c] for c in j_arr if remap[c] >= 0]) for j_arr in jj]
    return coef_r, St_r, X_r, jj_r
