"""Theta-free general-family PIRLS derivative kernel."""

import numpy as np

from ....solvers.general_family.fixed_smoothing import (
    run_general_family_fixed_smoothing,
)
from ....solvers.general_family.newton import _sl_term_mult
from .common import _prior_weights
from .derivatives import _GDI2Kernel


def gdi2_general_family_kernel(model, y, sol, sp, *, method, need_hessian):
    """
    Port-shaped ``gam.fit5`` decomposition for theta-free general families.

    Mirrors ``mgcv/R/gam.fit4.r::gam.fit5`` where score splits into
    ``Dp / (2 * gamma) + K`` with ``Dp = -2 * ll + b'Sb``.
    """
    del sol
    gamma = float(model.score_gamma)
    run = run_general_family_fixed_smoothing(
        model,
        y,
        sp,
        weights=_prior_weights(model, y),
        deriv=2 if need_hessian else 1,
        score_type=method,
    )
    fit = run["fit"]
    setup = run["setup"]

    coef = np.asarray(fit["coef"], dtype=np.float64)
    St_full = np.asarray(fit["St_full"], dtype=np.float64)
    ll_val = float(fit["l"])
    penalty_full = float(coef @ (St_full @ coef))
    Dp = float(-2.0 * ll_val + penalty_full)

    Skb = _sl_term_mult(setup.Sl, coef, full=True)
    Dp1 = np.asarray(
        [float(np.sum(coef * np.asarray(Skb_i, dtype=np.float64))) for Skb_i in Skb],
        dtype=np.float64,
    )
    score1 = np.asarray(fit.get("score1", np.zeros_like(Dp1)), dtype=np.float64)
    K1 = np.asarray(score1 - Dp1 / (2.0 * gamma), dtype=np.float64)

    Dp2 = None
    K2 = None
    if need_hessian:
        db_drho = np.asarray(fit["db_drho"], dtype=np.float64)
        llbb = np.asarray(fit["lbb"], dtype=np.float64)
        n_sp = int(db_drho.shape[1])
        d2pen = np.zeros((n_sp, n_sp), dtype=np.float64)
        d2l = np.zeros((n_sp, n_sp), dtype=np.float64)
        for i in range(n_sp):
            Sd1b = St_full @ db_drho[:, i]
            for j in range(i, n_sp):
                val = 2.0 * float(
                    np.sum(
                        db_drho[:, i] * np.asarray(Skb[j], dtype=np.float64)
                        + db_drho[:, j] * np.asarray(Skb[i], dtype=np.float64)
                        + db_drho[:, j] * Sd1b
                    )
                )
                if i == j:
                    val += float(np.sum(coef * np.asarray(Skb[i], dtype=np.float64)))
                d2pen[i, j] = d2pen[j, i] = val
                d2l[i, j] = d2l[j, i] = float(
                    db_drho[:, i] @ (llbb @ db_drho[:, j])
                )
        Dp2 = np.asarray(d2pen - 2.0 * d2l, dtype=np.float64)
        score2 = np.asarray(fit["score2"], dtype=np.float64)
        K2 = np.asarray(score2 - Dp2 / (2.0 * gamma), dtype=np.float64)

    return _GDI2Kernel(
        gdi1=None,
        phi=1.0,
        phi_curv=np.inf,
        Dp=Dp,
        Dp1=Dp1,
        Dp2=Dp2,
        K1_full=K1,
        K2_full=K2,
        extra_name=None,
        extra_value=None,
    )
