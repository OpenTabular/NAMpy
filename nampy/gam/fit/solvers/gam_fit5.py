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

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky

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
    St = np.asarray(St, dtype=np.float64)
    n, p = X.shape
    q = p
    nSp = len(lsp)
    warn: list[str] = []

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()

    # --- initialize coefficients ---
    if start is None:
        # `initialize()` expects penalty-root rows that can be stacked under the
        # model matrix, matching mgcv's use of `E` as an extra design block.
        E = _build_root_penalty(St).T
        start = family.initialize(y, X, jj, offset=offset, weights=weights, E=E)
    coef = np.asarray(start, dtype=np.float64).copy()

    llf = family.ll

    # --- first evaluation ---
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
                        rank, drop, bdrop = _detect_rank_drop(
                            ll["lbb"], St, coef, q, rank
                        )
                        if rank < q:
                            coef, St, X, jj = _apply_rank_drop(
                                coef, St, X, jj, bdrop, q
                            )
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
    dH = None
    REML1 = REML2 = None
    trHid2H = None

    keep = ~bdrop

    if deriv > 0 and nSp > 0 and rank_eff > 0:
        m = nSp
        d1b = np.zeros((rank_eff, m), dtype=np.float64)
        sp = np.exp(lsp)

        # d1b[:,i] = -Hp^{-1} S_i coef  (implicit differentiation)
        for i in range(m):
            Si_r = float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
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

            # d2b = -Hp^{-1}(dH_i d1b_j - Si d1b_j - Sj d1b_i)
            d2b = np.zeros((rank_eff, m * (m + 1) // 2), dtype=np.float64)
            kk = 0
            for i in range(m):
                for j in range(i, m):
                    Si_r = (
                        float(sp[i])
                        * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                    )
                    Sj_r = (
                        float(sp[j])
                        * S_blocks[j][np.ix_(keep, keep)][:rank_eff, :rank_eff]
                    )
                    dH_i_v = np.zeros(rank_eff, dtype=np.float64)
                    if isinstance(dH, list) and len(dH) > i:
                        dH_i_v = (
                            np.asarray(dH[i], dtype=np.float64)[:rank_eff, :rank_eff]
                            @ d1b[:, j]
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

        # d1bSb[i] = coef' Si coef
        d1bSb = np.zeros(m, dtype=np.float64)
        for i in range(m):
            Si_r = float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
            d1bSb[i] = float(coef @ (Si_r @ coef))

        if deriv > 1:
            # score1 uses the simplified penalty derivative term coef' Si coef.
            # Differentiate that expression directly for the score2 penalty block.
            d2pen = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                Si_r = (
                    float(sp[i]) * S_blocks[i][np.ix_(keep, keep)][:rank_eff, :rank_eff]
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

        # d1ldetH  (mgcv lines 1347-1365)
        d1ldetH = np.zeros(m, dtype=np.float64)
        d1Hp_list = []
        if d1ldetH_trace is not None:
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
        score2 = (d2pen - _ldetS2 + d2ldetH) / (2.0 * gamma)
        REML2 = np.asarray(score2, dtype=np.float64)

    if control.trace:
        print(
            f"gam_fit5: iter={iter_}  ll={ll_val:.6g}  {score_name}={score:.6g}  bSb={bSb:.6g}"
        )
        if REML1 is not None:
            print(f"  {score_name}1={REML1}")

    return {
        "coef": fcoef,
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
        "db_drho": fd1b,
        "dH": dH,
        "iter": iter_,
        "warn": warn,
        "rank": rank_eff,
        "ldetHp": ldetHp,
    }


# ---------------------------------------------------------------------------
# Post-processing: Vb, Ve, EDF  (mgcv: gam.fit5.post.proc)
# ---------------------------------------------------------------------------


def gam_fit5_post_proc(fit: dict) -> dict:
    """
    Compute Bayesian and frequentist covariance matrices and EDF.

    Mirrors mgcv ``gam.fit5.post.proc``.
    """
    lbb = -np.asarray(fit["lbb"], dtype=np.float64)
    L = fit["L"]
    D = fit["D"]
    bdrop = fit["bdrop"]
    St_full = fit["St_full"]
    q = St_full.shape[0]
    keep = ~bdrop
    p = int(np.sum(keep))

    lbb_r = lbb[:p, :p] if lbb.shape[0] >= p else lbb
    D_r = D[:p]

    # Pre-conditioned lbb for rank check
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
        L_new = cholesky(Hp_c, lower=False)
        L = L_new
        R = (R_sq / D_r[:, None]).T

    # Bayesian covariance Vb = (lbb + St)^{-1} = Hp^{-1}
    Vb = _compute_Hp_inv(L, D_r, p)

    if np.any(bdrop):
        Vb_full = np.zeros((q, q), dtype=np.float64)
        Vb_full[np.ix_(keep, keep)] = Vb
        Vb = Vb_full
        R_full = np.zeros((q, q), dtype=np.float64)
        R_full[np.ix_(keep, keep)] = R
        R = R_full

    F = Vb @ (R.T @ R)
    Ve = F @ Vb
    edf = np.diag(F)
    edf1 = 2.0 * edf - np.sum(F * F.T, axis=1)
    edf2 = np.sum(Vb * (R.T @ R), axis=1)
    edf2 = np.where(edf2 > edf1, edf1, edf2)

    return {"Vp": Vb, "Ve": Ve, "edf": edf, "edf1": edf1, "edf2": edf2, "R": R}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
