"""Tweedie extended family (`mgcv::tw`) with estimated power parameter.

Upstream references:
- `mgcv/R/efam.r::tw` — family object (dev.resids, Dd, ls, aic, theta transform)
- `mgcv/R/gam.fit3.r::ldTweedie` — log density with derivatives
- `mgcv/src/misc.c::tweedious` — Dunn & Smyth (2005) series summation

The series summation mirrors the C `tweedious` routine term-for-term,
including the buffer-index bookkeeping that determines the scaling maximum
and the sweep boundaries, so summation order matches upstream.
"""

import math
import warnings

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .._mgcv_constants import FAMILY_EPS
from ._function_maps import LINK_REGISTRY, TweedieVariance
from .family_base import ExtendedFamily, JointOuterStrategy

_C_INT_MAX = 2147483647


def _tw_p_from_theta(th, a, b):
    """`p = (a+b*exp(th))/(1+exp(th))` with the branch split used upstream."""
    th = np.asarray(th, dtype=np.float64)
    p = np.empty_like(th)
    ind = th > 0
    ethi = np.exp(-th[ind])
    ethni = np.exp(th[~ind])
    p[ind] = (b + a * ethi) / (1.0 + ethi)
    p[~ind] = (b * ethni + a) / (ethni + 1.0)
    return p


def _tw_dp_dth(th, a, b):
    """First and second derivatives of p w.r.t. working theta (upstream split)."""
    th = np.asarray(th, dtype=np.float64)
    dpth1 = np.empty_like(th)
    dpth2 = np.empty_like(th)
    ind = th > 0
    ethi = np.exp(-th[ind])
    ethni = np.exp(th[~ind])
    dpth1[ind] = ethi * (b - a) / (1.0 + ethi) ** 2
    dpth1[~ind] = ethni * (b - a) / (ethni + 1.0) ** 2
    dpth2[ind] = ((a - b) * ethi + (b - a) * ethi**2) / (ethi + 1.0) ** 3
    dpth2[~ind] = ((a - b) * ethni**2 + (b - a) * ethni) / (ethni + 1.0) ** 3
    return dpth1, dpth2


def _tweedious(y, eps, th, rho, a, b):
    """Port of `mgcv/src/misc.c::tweedious` (scalar working parameters).

    Returns (w, w1, w2, w1p, w2p, w2pp, eps_out) where the derivative
    columns follow the C routine: w = log W, w1 = d logW/d rho,
    w2 = d2 logW/d rho2, w1p = d logW/d th, w2p = d2 logW/d th2,
    w2pp = d2 logW/(d th d rho).
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    jal_lim = 50000000
    buffer_run_out = False
    failed = False

    phi = math.exp(rho)
    if th > 0:
        exp_th = math.exp(-th)
        x = 1.0 + exp_th
        p = (b + a * exp_th) / x
        x1 = x * x
        dpth1 = exp_th * (b - a) / x1
        dpth2 = ((a - b) * exp_th + (b - a) * exp_th * exp_th) / (x1 * x)
    else:
        exp_th = math.exp(th)
        x = exp_th + 1.0
        p = (b * exp_th + a) / x
        x1 = x * x
        dpth1 = exp_th * (b - a) / x1
        dpth2 = ((a - b) * exp_th * exp_th + (b - a) * exp_th) / (x * x1)

    log_eps = math.log(eps)
    onep = 1.0 - p
    onep2 = onep * onep
    alpha = (2.0 - p) / onep
    w_base = alpha * math.log(p - 1.0) + rho / onep - math.log(2.0 - p)
    wp_base = (math.log(-onep) + rho) / onep2 - alpha / onep + 1.0 / (2.0 - p)
    wp2_base = (
        2.0 * (math.log(-onep) + rho) / (onep2 * onep)
        - (3.0 * alpha - 2.0) / onep2
        + 1.0 / ((2.0 - p) * (2.0 - p))
    )

    logy = np.log(y)
    alogy = alpha * logy
    logy1p2 = logy / onep2
    logy1p3 = logy1p2 / onep
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    x = ymin ** (2.0 - p) / (phi * (2.0 - p))
    j_lo = int(math.floor(x))
    if j_lo < 1:
        j_lo = 1
    x = ymax ** (2.0 - p) / (phi * (2.0 - p))
    j_hi = int(math.ceil(x))
    if j_hi < j_lo:
        j_hi = j_lo

    j0 = j_lo - 1000
    if j0 < 1:
        j0 = 1
    jal = j_hi + 1000
    jal -= j0 - 1
    j_lo -= j0
    j_hi -= j0

    term_cache: dict[int, tuple[float, float, float, float, float]] = {}

    def _terms(j):
        """(wb, wb1, wp1, wp2, wpp) at true index j (y-independent parts)."""
        got = term_cache.get(j)
        if got is not None:
            return got
        wb = j * w_base - math.lgamma(j + 1.0) - math.lgamma(-j * alpha)
        wb1 = -j / onep
        xx = j / onep2
        xdig = xx * float(digamma(-j * alpha))
        wp1 = j * wp_base + xdig
        xtrig = float(polygamma(1, -j * alpha)) * xx * xx
        wp2 = j * wp2_base + 2.0 * xdig / onep - xtrig
        wpp = j / onep2
        out = (wb, wb1, wp1, wp2, wpp)
        term_cache[j] = out
        return out

    w = np.zeros(n, dtype=np.float64)
    w1 = np.zeros(n, dtype=np.float64)
    w2 = np.zeros(n, dtype=np.float64)
    w1p = np.zeros(n, dtype=np.float64)
    w2p = np.zeros(n, dtype=np.float64)
    w2pp = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x = y[i] ** (2.0 - p) / (phi * (2.0 - p))
        if math.floor(x) > _C_INT_MAX - 2:
            failed = True
            break
        j_max = int(math.floor(x))
        if x - j_max > 0.5 or j_max < 1:
            j_max += 1
        if abs(j_max - x) > 1.0:
            failed = True
            break
        j_max -= j0

        j = j_max + j0
        jalogy = j * alogy[i]
        wdW2d2W = wdlogwdp = dWpp = 0.0
        wi = w1i = w2i = 0.0
        if j_max > j_hi:
            j_max = j_hi
        if j_max < j_lo:
            j_max = j_lo
        wmax = _terms(j_max + j0)[0] - jalogy
        wmin = wmax + log_eps

        def _accumulate(j, jb, i=i, wmax=wmax, wmin=wmin):
            """One series term at true index j; returns (converged, wj)."""
            nonlocal wi, w1i, w2i, wdlogwdp, wdW2d2W, dWpp
            wb, wb1, wp1, wp2, wpp = _terms(j)
            jalogy = j * alogy[i]
            wj = wb - jalogy
            w1j = wb1
            wp1j = wp1 - j * logy1p2[i]
            wp2j = wp2 - 2.0 * j * logy1p3[i]
            wp2j = wp1j * dpth2 + wp2j * dpth1 * dpth1
            wp1j = wp1j * dpth1
            wppj = wpp * dpth1

            wj_scaled = math.exp(wj - wmax)
            wi += wj_scaled
            w1i += wj_scaled * w1j
            w2i += wj_scaled * w1j * w1j
            wdlogwdp += wj_scaled * wp1j
            wdW2d2W += wj_scaled * (wp1j * wp1j + wp2j)
            dWpp += wj_scaled * (wp1j * j / onep + wppj)
            return wj < wmin

        # upsweep to convergence or end of available buffered values
        ok = False
        jb = j_max
        j = j_max + j0
        while jb <= j_hi:
            if _accumulate(j, jb):
                ok = True
                break
            jb += 1
            j += 1

        while not ok:
            while jb < jal:
                if _accumulate(j, jb):
                    ok = True
                    break
                jb += 1
                j += 1
            j_hi = jb if jb <= jal - 1 else jal - 1
            if not ok:
                if jal < jal_lim:
                    jal += 1000  # forward buffer expansion
                else:
                    ok = buffer_run_out = True

        # downsweep to convergence or start of available buffered values
        ok = False
        jb = j_max - 1
        j = j_max - 1 + j0
        while jb >= j_lo:
            if _accumulate(j, jb):
                ok = True
                break
            jb -= 1
            j -= 1
        if j <= 1 and j_lo == 0:
            ok = True

        while not ok:
            jb = j_lo - 1
            while jb >= 0:
                if _accumulate(j, jb):
                    ok = True
                    break
                jb -= 1
                j -= 1
            if j <= 1:
                ok = True
            j_lo = jb if jb >= 0 else 0
            if not ok:
                if jal < jal_lim:
                    # backward buffer expansion by up to 1000 (or to j=1)
                    nback = 1000
                    if nback > j0 - 1:
                        nback = j0 - 1
                    if nback == 0:
                        ok = True
                    else:
                        jal += nback
                        j_lo += nback
                        j_hi += nback
                        j0 -= nback
                else:
                    ok = buffer_run_out = True

        w[i] = wmax + math.log(wi)
        w2[i] = w2i / wi - (w1i / wi) * (w1i / wi)
        w2p[i] = wdW2d2W / wi - (wdlogwdp / wi) * (wdlogwdp / wi)
        w2pp[i] = (w1i / wi) * (wdlogwdp / wi) + dWpp / wi
        w1[i] = -w1i / wi
        w1p[i] = wdlogwdp / wi

    eps_out = eps
    if buffer_run_out:
        eps_out = -1.0
    if failed:
        eps_out = -2.0
    return w, w1, w2, w1p, w2p, w2pp, eps_out


def _tweedious2(y, eps, th, rho, a, b):
    """Port of ``mgcv/src/misc.c::tweedious2``.

    ``tweedious`` reuses its expensive gamma-function terms when ``rho`` and
    ``theta`` are scalar.  ``tweedious2`` is upstream's row-wise path for
    vector-valued working parameters; it deliberately uses the same
    up-sweep/down-sweep order as the C routine and does not share a buffer.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    th = np.broadcast_to(np.asarray(th, dtype=np.float64), y.shape)
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), y.shape)
    log_eps = math.log(eps)
    out = [np.zeros(y.size, dtype=np.float64) for _ in range(6)]
    series_too_long = False
    failed = False

    for i, yi in enumerate(y):
        thi = float(th[i])
        rhoi = float(rho[i])
        if thi > 0.0:
            exp_th = math.exp(-thi)
            x = 1.0 + exp_th
            p = (b + a * exp_th) / x
            x1 = x * x
            dpth1 = exp_th * (b - a) / x1
            dpth2 = ((a - b) * exp_th + (b - a) * exp_th * exp_th) / (x1 * x)
        else:
            exp_th = math.exp(thi)
            x = exp_th + 1.0
            p = (b * exp_th + a) / x
            x1 = x * x
            dpth1 = exp_th * (b - a) / x1
            dpth2 = ((a - b) * exp_th * exp_th + (b - a) * exp_th) / (x * x1)

        phi = math.exp(rhoi)
        x = yi ** (2.0 - p) / (phi * (2.0 - p))
        j_max = int(math.floor(x))
        if x - j_max > 0.5 or j_max < 1:
            j_max += 1
        if abs(j_max - x) > 1.0:
            failed = True
            break

        onep = 1.0 - p
        onep2 = onep * onep
        twop = 2.0 - p
        alpha = twop / onep
        log_y = math.log(yi)
        log_y_1p2 = log_y / onep2
        log_y_1p3 = log_y_1p2 / onep
        alpha_log_y = alpha * log_y

        w_base = alpha * math.log(-onep) + rhoi / onep - math.log(twop)
        wp_base = (math.log(-onep) + rhoi) / onep2 - alpha / onep + 1.0 / twop
        wp2_base = (
            2.0 * (math.log(-onep) + rhoi) / (onep2 * onep)
            - (3.0 * alpha - 2.0) / onep2
            + 1.0 / (twop * twop)
        )

        j = j_max
        lgamma_j1 = math.lgamma(j + 1.0)
        wmax = j * w_base - lgamma_j1 - math.lgamma(-j * alpha) - j * alpha_log_y
        wmin = wmax + log_eps

        wi = w1i = w2i = wdlogwdp = wdW2d2W = dWpp = 0.0
        incr = 1
        ok = False
        for _ in range(50000000):
            wb = j * w_base - lgamma_j1 - math.lgamma(-j * alpha)
            xx = j / onep2
            xdig = xx * float(digamma(-j * alpha))
            wp1 = j * wp_base + xdig
            xtrig = float(polygamma(1, -j * alpha)) * xx * xx
            wp2 = j * wp2_base + 2.0 * xdig / onep - xtrig
            wpp = j / onep2

            wj = wb - j * alpha_log_y
            wp1j = wp1 - j * log_y_1p2
            wp2j = wp2 - 2.0 * j * log_y_1p3
            wp2j = wp1j * dpth2 + wp2j * dpth1 * dpth1
            wp1j *= dpth1
            wppj = wpp * dpth1

            wj_scaled = math.exp(wj - wmax)
            wi += wj_scaled
            w1j = -j / onep
            w1i += wj_scaled * w1j
            w2i += wj_scaled * w1j * w1j
            wdlogwdp += wj_scaled * wp1j
            wdW2d2W += wj_scaled * (wp1j * wp1j + wp2j)
            dWpp += wj_scaled * (wp1j * j / onep + wppj)

            j += incr
            if incr > 0:
                lgamma_j1 += math.log(j)
                if wj < wmin:
                    j = j_max - 1
                    incr = -1
                    if j == 0:
                        ok = True
                        break
                    lgamma_j1 = math.lgamma(j + 1.0)
            else:
                lgamma_j1 += -math.log(j + 1.0)
                if wj < wmin or j < 1:
                    ok = True
                    break
        if not ok:
            series_too_long = True

        out[0][i] = wmax + math.log(wi)
        out[2][i] = w2i / wi - (w1i / wi) * (w1i / wi)
        out[4][i] = wdW2d2W / wi - (wdlogwdp / wi) * (wdlogwdp / wi)
        out[5][i] = (w1i / wi) * (wdlogwdp / wi) + dWpp / wi
        out[1][i] = -w1i / wi
        out[3][i] = wdlogwdp / wi

    eps_out = -2.0 if failed else (-1.0 if series_too_long else eps)
    return (*out, eps_out)


def ldTweedie(
    y,
    mu=None,
    p=1.5,
    phi=1.0,
    rho=None,
    theta=None,
    a=1.001,
    b=1.999,
    all_derivs=False,
):
    """Port of `mgcv/R/gam.fit3.r::ldTweedie` (scalar rho/theta variant).

    Returns an (n, 6) array with columns (l, rho, rho.2, th, th.2, th.rho)
    in the working parameterization when (rho, theta) are supplied, or
    (l, phi, phi.2, p, p.2, phi.p) derivatives when (p, phi) are supplied.
    With ``all_derivs=True`` (working parameterization only) four extra
    columns (mu, mu.2, mu.theta, mu.rho) are appended.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    mu = y.copy() if mu is None else np.asarray(mu, dtype=np.float64).ravel()

    if rho is not None and theta is not None:
        if a >= b or a <= 1.0 or b >= 2.0:
            raise ValueError("1<a<b<2 (strict) required")
        work_param = True
        theta_arr = np.broadcast_to(np.asarray(theta, dtype=np.float64), (n,)).astype(
            np.float64
        )
        rho_arr = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,)).astype(
            np.float64
        )
        phi = np.exp(rho_arr)
        p = _tw_p_from_theta(theta_arr, a, b)
        dpth1, dpth2 = _tw_dp_dth(theta_arr, a, b)
    else:
        work_param = False
        if all_derivs:
            warnings.warn(
                "all.derivs only available in rho, theta parameterization",
                stacklevel=2,
            )
        p = np.broadcast_to(np.asarray(p, dtype=np.float64), (n,)).astype(np.float64)
        phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), (n,)).astype(
            np.float64
        )
        rho_arr = np.log(phi)
        theta_arr = np.zeros(n, dtype=np.float64)
        dthp1 = np.zeros(n, dtype=np.float64)
        dthp2 = np.zeros(n, dtype=np.float64)
        if float(np.min(p)) >= 1.0 and float(np.max(p)) <= 2.0:
            ind = (p > 1.0) & (p < 2.0)
            if np.sum(ind):
                p_ind = p[ind]
                if float(np.min(p_ind)) <= a:
                    a = (1.0 + float(np.min(p_ind))) / 2.0
                if float(np.max(p_ind)) >= b:
                    b = (2.0 + float(np.max(p_ind))) / 2.0
                pabp = (p_ind - a) / (b - p_ind)
                theta_arr[ind] = np.log((p_ind - a) / (b - p_ind))
                dthp1[ind] = (1.0 + pabp) / (p_ind - a)
                dthp2[ind] = (pabp + 1.0) / ((p_ind - a) * (b - p_ind)) - (
                    pabp + 1.0
                ) / (p_ind - a) ** 2

    if float(np.min(p)) < 1.0 or float(np.max(p)) > 2.0:
        raise ValueError("p must be in [1,2]")

    ncols = 10 if (work_param and all_derivs) else 6
    ld = np.zeros((n, ncols), dtype=np.float64)
    ld[:, 3:6] = np.nan

    ind = p == 2.0
    if np.sum(ind):  # It's Gamma
        if np.sum(y[ind] <= 0.0):
            raise ValueError("y must be strictly positive for a Gamma density")
        shape = 1.0 / phi[ind]
        rate = 1.0 / (phi[ind] * mu[ind])
        ld[ind, 0] = (
            shape * np.log(rate)
            - gammaln(shape)
            + (shape - 1.0) * np.log(y[ind])
            - rate * y[ind]
        )
        ld[ind, 1] = (
            digamma(1.0 / phi[ind])
            + np.log(phi[ind])
            - 1.0
            + y[ind] / mu[ind]
            - np.log(y[ind] / mu[ind])
        ) / (phi[ind] * phi[ind])
        ld[ind, 2] = -2.0 * ld[ind, 1] / phi[ind] + (
            1.0 - polygamma(1, 1.0 / phi[ind]) / phi[ind]
        ) / (phi[ind] ** 3)

    ind = p == 1.0
    if np.sum(ind):  # It's Poisson like
        ratio = y[ind] / phi[ind]
        if not np.allclose(ratio, np.round(ratio)):
            raise ValueError("y must be an integer multiple of phi for Tweedie(p=1)")
        indi = (y[ind] != 0.0) | (mu[ind] != 0.0)
        bkt = np.zeros_like(y[ind])
        bkt[indi] = (
            (y[ind])[indi] * np.log((mu[ind] / phi[ind])[indi]) - (mu[ind])[indi]
        )
        dig = digamma(y[ind] / phi[ind] + 1.0)
        trig = polygamma(1, y[ind] / phi[ind] + 1.0)
        ld[ind, 0] = bkt / phi[ind] - gammaln(y[ind] / phi[ind] + 1.0)
        ld[ind, 1] = (-bkt - y[ind] + dig * y[ind]) / (phi[ind] ** 2)
        ld[ind, 2] = (
            2.0 * bkt
            + 3.0 * y[ind]
            - 2.0 * dig * y[ind]
            - trig * y[ind] ** 2 / phi[ind]
        ) / (phi[ind] ** 3)

    # zeros for 1<p<2
    ind = (y == 0.0) & (p > 1.0) & (p < 2.0)
    ld[ind, :] = 0.0
    ind = ind & (mu > 0.0)
    if np.sum(ind):
        mu_ind = mu[ind]
        p_ind = p[ind]
        phii = phi[ind]
        ld[ind, 0] = -(mu_ind ** (2.0 - p_ind)) / (phii * (2.0 - p_ind))
        ld[ind, 1] = -ld[ind, 0] / phii
        ld[ind, 2] = -2.0 * ld[ind, 1] / phii
        ld[ind, 3] = -ld[ind, 0] * (np.log(mu_ind) - 1.0 / (2.0 - p_ind))
        ld[ind, 4] = 2.0 * ld[ind, 3] / (2.0 - p_ind) + ld[ind, 0] * np.log(mu_ind) ** 2
        ld[ind, 5] = -ld[ind, 3] / phii
        if work_param and all_derivs:
            mup = mu_ind**p_ind
            ld[ind, 6] = -mu_ind / (mup * phii)
            ld[ind, 7] = -(1.0 - p_ind) / (mup * phii)
            ld[ind, 8] = np.log(mu_ind) * mu_ind / (mup * phii)
            ld[ind, 9] = -ld[ind, 6] / phii
    # Upstream early return: `if (sum(!ind)==0) return(ld)` with the
    # mu>0-refined zero index — note this skips the work.param transform.
    if int(np.sum(~ind)) == 0:
        return ld

    # now the non-zeros
    ind = np.flatnonzero((y > 0.0) & (p > 1.0) & (p < 2.0))
    series = None
    if ind.size > 0:
        y_i = y[ind]
        mu_i = mu[ind]
        p_i = p[ind]
        eps_in = float(np.finfo(np.float64).eps ** 2)
        if np.unique(theta_arr[ind]).size == 1 and np.unique(rho_arr[ind]).size == 1:
            ow, ow1, ow2, ow1p, ow2p, ow2pp, eps_out = _tweedious(
                y_i,
                eps_in,
                float(theta_arr[ind[0]]),
                float(rho_arr[ind[0]]),
                float(a),
                float(b),
            )
        else:
            ow, ow1, ow2, ow1p, ow2p, ow2pp, eps_out = _tweedious2(
                y_i,
                eps_in,
                theta_arr[ind],
                rho_arr[ind],
                float(a),
                float(b),
            )
        if eps_out < -0.5:
            if eps_out < -1.5:
                ow1 = ow2 = ow1p = ow2p = ow2pp = np.full(ind.size, np.nan)
            else:
                warnings.warn(
                    "Tweedie density may be unreliable - series not fully converged",
                    stacklevel=2,
                )
        phii = phi[ind]
        if not work_param:  # transform working param derivs to p/phi derivs
            dthp1i = dthp1[ind]
            ow2 = ow2 / phii**2 - ow1 / phii**2
            ow1 = ow1 / phii
            ow2p = ow2p * dthp1i**2 + dthp2[ind] * ow1p
            ow1p = ow1p * dthp1i
            ow2pp = ow2pp * dthp1i / phii

        log_mu = np.log(mu_i)
        onep = 1.0 - p_i
        twop = 2.0 - p_i
        mu1p = mu_i**onep
        k_theta = mu_i * mu1p / twop
        theta_i = mu1p / onep
        a1 = y_i / onep - mu_i / twop
        l_base = mu1p * a1 / phii
        ld[ind, 0] = l_base - np.log(y_i)
        ld[ind, 1] = -l_base / phii
        ld[ind, 2] = 2.0 * l_base / (phii**2)
        xterm = (
            theta_i * y_i * (1.0 / onep - log_mu) / phii
            + k_theta * (log_mu - 1.0 / twop) / phii
        )
        ld[ind, 3] = xterm
        ld[ind, 4] = (
            theta_i * y_i * (log_mu**2 - 2.0 * log_mu / onep + 2.0 / onep**2) / phii
            - k_theta * (log_mu**2 - 2.0 * log_mu / twop + 2.0 / twop**2) / phii
        )
        ld[ind, 5] = -xterm / phii
        series = (ow, ow1, ow2, ow1p, ow2p, ow2pp)

    if work_param:  # transform derivs to derivs wrt working
        ld[:, 2] = ld[:, 2] * phi**2 + ld[:, 1] * phi
        ld[:, 1] = ld[:, 1] * phi
        ld[:, 4] = ld[:, 4] * dpth1**2 + ld[:, 3] * dpth2
        ld[:, 3] = ld[:, 3] * dpth1
        ld[:, 5] = ld[:, 5] * dpth1 * phi

    if work_param and all_derivs and ind.size > 0:
        phii = phi[ind]
        log_mu = np.log(mu[ind])
        p_i = p[ind]
        onep = 1.0 - p_i
        twop = 2.0 - p_i
        mu_i = mu[ind]
        y_i = y[ind]
        mu1p = mu_i**onep
        a1 = y_i / onep - mu_i / twop
        a2 = mu1p / (mu_i * phii)
        ld[ind, 6] = a2 * (onep * a1 - mu_i / twop)
        ld[ind, 7] = -a2 * (onep * p_i * a1 / mu_i + 2.0 * onep / twop)
        ld[ind, 8] = a2 * (
            -log_mu * onep * a1
            - a1
            + onep * (y_i / onep**2 - mu_i / twop**2)
            + mu_i * log_mu / twop
            - mu_i / twop**2
        )
        ld[ind, 9] = a2 * (mu_i / (phii * twop) - onep * a1 / phii)
        ld[:, 9] = ld[:, 9] * phi
        ld[:, 8] = ld[:, 8] * dpth1

    if ind.size > 0:
        ow, ow1, ow2, ow1p, ow2p, ow2pp = series
        ld[ind, 0] = ld[ind, 0] + ow
        ld[ind, 1] = ld[ind, 1] + ow1
        ld[ind, 2] = ld[ind, 2] + ow2
        ld[ind, 3] = ld[ind, 3] + ow1p
        ld[ind, 4] = ld[ind, 4] + ow2p
        ld[ind, 5] = ld[ind, 5] + ow2pp

    return ld


class TweedieTwFamily(ExtendedFamily):
    """Tweedie family with estimated power. Matches `mgcv::tw()`.

    `p = (a+b*exp(theta))/(1+exp(theta))`, i.e. a < p < b, with `theta`
    the unbounded working parameter estimated as part of REML/ML
    optimization. Upstream semantics for the `theta` argument: `None`/0
    estimates p starting from working theta 0; `theta>0` fixes
    p = theta; `theta<0` estimates p starting from -theta.
    """

    name = "tw"
    link_name = "log"
    canonical_link = False

    supports_closed_form_solve = False
    supports_pirls = True

    supports_gcv = False
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.TWEEDIE
    # ``mgcv::tw`` does not expose ``dvar``; gam.fit3 therefore uses the raw
    # Pearson scale estimate and skips its Fletcher correction.
    use_fletcher_scale_estimate = False

    known_scale = None
    max_derivative_order = 1

    def __init__(
        self,
        theta=None,
        link: str = "log",
        a: float = 1.01,
        b: float = 1.99,
        eps: float = FAMILY_EPS,
    ):
        super().__init__(eps=eps)
        link_key = str(link).lower()
        if link_key not in {"log", "identity", "sqrt", "inverse"}:
            raise ValueError(
                "tw link must be one of 'log', 'identity', 'sqrt', or 'inverse'."
            )
        self._link_key = link_key
        self.link_name = link_key
        self.link = LINK_REGISTRY[self._link_key](eps=self.eps)
        self.a = float(a)
        self.b = float(b)
        if not (1.0 < self.a < self.b < 2.0):
            raise ValueError("tw requires 1 < a < b < 2 (strict).")

        self.n_theta = 1
        if theta is not None and float(theta) != 0.0:
            theta = float(theta)
            if abs(theta) <= self.a or abs(theta) >= self.b:
                raise ValueError("Tweedie p must be in interval (a,b)")
            if theta > 0:  # fixed theta supplied
                ini_theta = float(np.log((theta - self.a) / (self.b - theta)))
                self.n_theta = 0
            else:  # initial theta supplied
                ini_theta = float(np.log((-theta - self.a) / (self.b + theta)))
        else:
            ini_theta = 0.0
        self.ini_theta = float(ini_theta)
        self._theta_working = float(ini_theta)
        self.variance = TweedieVariance(eps=self.eps, family=self)

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    @property
    def p(self):
        """Current Tweedie power on the natural scale."""
        return float(self.getTheta(trans=True))

    def getTheta(self, trans=False):
        th = float(self._theta_working)
        if trans:
            a, b = self.a, self.b
            if th > 0:
                return float((b + a * np.exp(-th)) / (1.0 + np.exp(-th)))
            return float((b * np.exp(th) + a) / (np.exp(th) + 1.0))
        return th

    def putTheta(self, theta):
        theta = float(theta)
        if not np.isfinite(theta):
            raise ValueError("tw requires a finite working theta.")
        self._theta_working = theta

    def _p_from_theta(self, theta=None):
        th = float(self._theta_working if theta is None else theta)
        a, b = self.a, self.b
        if th > 0:
            return float((b + a * np.exp(-th)) / (1.0 + np.exp(-th)))
        return float((b * np.exp(th) + a) / (np.exp(th) + 1.0))

    def _check_weights(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            return np.ones_like(y, dtype=np.float64)
        return np.asarray(weights, dtype=np.float64)

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def dvar(self, mu):
        return self.variance.d1(mu)

    def d2var(self, mu):
        return self.variance.d2(mu)

    def d3var(self, mu):
        return self.variance.d3(mu)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0.0):
            raise ValueError("TweedieTwFamily requires non-negative targets.")
        return y

    def initialize_mu(self, y):
        # mgcv::tw initialize: mustart <- y + (y == 0)*.1
        y = np.asarray(y, dtype=np.float64)
        return y + (y == 0.0).astype(np.float64) * 0.1

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=np.float64)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0.0))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        if self.link_name in {"sqrt", "inverse"}:
            return bool(np.all(np.isfinite(eta)) and np.all(eta > 0.0))
        return bool(np.all(np.isfinite(eta)))

    def deviance_obs(self, y, mu, weights=None, theta=None):
        """Port of `mgcv::tw()$dev.resids`."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        wt = self._check_weights(y, weights)
        p = self._p_from_theta(theta)
        y1 = y + (y == 0.0).astype(np.float64)
        if p == 1.0:
            theta_term = np.log(y1 / mu)
        else:
            theta_term = (y1 ** (1.0 - p) - mu ** (1.0 - p)) / (1.0 - p)
        if p == 2.0:
            kappa = np.log(y1 / mu)
        else:
            kappa = (y ** (2.0 - p) - mu ** (2.0 - p)) / (2.0 - p)
        return np.maximum(2.0 * (y * theta_term - kappa) * wt, 0.0)

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights=weights)))

    def loglik_obs(self, y, mu, scale=1.0):
        p = self._p_from_theta()
        return ldTweedie(y, mu, p=p, phi=float(scale), a=self.a, b=self.b)[:, 0]

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def aic(
        self,
        y,
        mu,
        theta=None,
        wt=None,
        dev=None,
        *,
        edf=0.0,
        scale=1.0,
        weights=None,
    ):
        """Port of `mgcv::tw()$aic` (family AIC kernel, not the +2*edf form)."""
        del edf, scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        if weights is not None:
            wt = weights
        wt = self._check_weights(y, wt)
        p = self._p_from_theta(theta)
        if dev is None:
            dev = float(np.sum(self.deviance_obs(y, mu, weights=wt)))
        scale_val = float(dev) / float(np.sum(wt))
        ld = ldTweedie(y, mu, p=p, phi=scale_val, a=self.a, b=self.b)
        return float(-2.0 * np.sum(ld[:, 0] * wt) + 2.0)

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        """Port of `mgcv::tw()$Dd`. `theta` is the working parameter."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        wt = self._check_weights(y, wt)
        th = float(self._theta_working if theta is None else theta)
        a, b = self.a, self.b
        if th > 0:
            p = (b + a * np.exp(-th)) / (1.0 + np.exp(-th))
            dpth1 = np.exp(-th) * (b - a) / (1.0 + np.exp(-th)) ** 2
            dpth2 = ((a - b) * np.exp(-th) + (b - a) * np.exp(-2.0 * th)) / (
                np.exp(-th) + 1.0
            ) ** 3
        else:
            p = (b * np.exp(th) + a) / (np.exp(th) + 1.0)
            dpth1 = np.exp(th) * (b - a) / (np.exp(th) + 1.0) ** 2
            dpth2 = ((a - b) * np.exp(2.0 * th) + (b - a) * np.exp(th)) / (
                np.exp(th) + 1.0
            ) ** 3
        p = float(p)
        dpth1 = float(dpth1)
        dpth2 = float(dpth2)

        mu1p = mu ** (1.0 - p)
        mup = mu**p
        r = {}
        ymupi = y / mup
        r["Dmu"] = 2.0 * wt * (mu1p - ymupi)
        r["Dmu2"] = 2.0 * wt * (mu ** (-1.0 - p) * p * y + (1.0 - p) / mup)
        r["EDmu2"] = (2.0 * wt) / mup

        if level > 0:
            i1p = 1.0 / (1.0 - p)
            y1 = y + (y == 0.0).astype(np.float64)
            logmu = np.log(mu)
            mu2p = mu * mu1p
            r["Dth"] = (
                2.0
                * wt
                * (
                    (y ** (2.0 - p) * np.log(y1) - mu2p * logmu) / (2.0 - p)
                    + (y * mu1p * logmu - y ** (2.0 - p) * np.log(y1)) / (1.0 - p)
                    - (y ** (2.0 - p) - mu2p) / (2.0 - p) ** 2
                    + (y ** (2.0 - p) - y * mu1p) * i1p**2
                )
                * dpth1
            )
            r["Dmuth"] = 2.0 * wt * logmu * (ymupi - mu1p) * dpth1
            mup1 = mu ** (-p - 1.0)
            r["Dmu3"] = -2.0 * wt * mup1 * p * (y / mu * (p + 1.0) + 1.0 - p)
            r["Dmu2th"] = (
                2.0
                * wt
                * (mup1 * y * (1.0 - p * logmu) - (logmu * (1.0 - p) + 1.0) / mup)
                * dpth1
            )
            r["EDmu3"] = -2.0 * wt * p * mup1
            r["EDmu2th"] = -2.0 * wt * logmu / mup * dpth1
        if level > 1:
            mup2 = mup1 / mu
            r["Dmu4"] = 2.0 * wt * mup2 * p * (p + 1.0) * (y * (p + 2.0) / mu + 1.0 - p)
            y2plogy = y ** (2.0 - p) * np.log(y1)
            y2plog2y = y2plogy * np.log(y1)
            r["Dth2"] = (
                2.0
                * wt
                * (
                    (
                        (mu2p * logmu**2 - y2plog2y) / (2.0 - p)
                        + (y2plog2y - y * mu1p * logmu**2) / (1.0 - p)
                        + 2.0 * (y2plogy - mu2p * logmu) / (2.0 - p) ** 2
                        + 2.0 * (y * mu1p * logmu - y2plogy) / (1.0 - p) ** 2
                        + 2.0 * (mu2p - y ** (2.0 - p)) / (2.0 - p) ** 3
                        + 2.0 * (y ** (2.0 - p) - y * mu ** (1.0 - p)) / (1.0 - p) ** 3
                    )
                    * dpth1**2
                )
            ) + r["Dth"] * dpth2 / dpth1
            r["Dmuth2"] = (
                2.0 * wt * ((mu1p * logmu**2 - logmu**2 * ymupi) * dpth1**2)
            ) + r["Dmuth"] * dpth2 / dpth1
            r["Dmu2th2"] = (
                2.0
                * wt
                * (
                    (
                        mup1 * logmu * y * (logmu * p - 2.0)
                        + logmu / mup * (logmu * (1.0 - p) + 2.0)
                    )
                    * dpth1**2
                )
            ) + r["Dmu2th"] * dpth2 / dpth1
            r["Dmu3th"] = (
                2.0
                * wt
                * mup1
                * (
                    y / mu * (logmu * (1.0 + p) * p - p - p - 1.0)
                    + logmu * (1.0 - p) * p
                    + p
                    - 1.0
                    + p
                )
                * dpth1
            )
        return r

    def ls(self, y, w, theta=None, scale=1.0):
        """Port of `mgcv::tw()$ls`.

        Returns the saturated log likelihood plus derivatives w.r.t. the
        working theta and log(scale): `lsth1` is length-2 (theta, log scale)
        and `lsth2` is the corresponding 2x2 matrix.
        """
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, w)
        th = float(self._theta_working if theta is None else theta)
        scale = float(scale)
        Ls = w[:, None] * ldTweedie(
            y, y, rho=float(np.log(scale)), theta=th, a=self.a, b=self.b
        )
        LS = Ls.sum(axis=0)
        lsth1 = np.array([LS[3], LS[1]], dtype=np.float64)
        lsth2 = np.array([[LS[4], LS[5]], [LS[5], LS[2]]], dtype=np.float64)
        return {
            "ls": float(LS[0]),
            "lsth1": lsth1,
            "LSTH1": np.asarray(Ls[:, [3, 1]], dtype=np.float64),
            "lsth2": lsth2,
        }

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del n
        y = np.asarray(y, dtype=np.float64)
        w = self._check_weights(y, weights)
        return float(self.ls(y, w, scale=scale)["ls"])

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        """mgcv-style extended-family scale estimate used before the outer
        optimizer supplies exp(log phi): Pearson-based fallback."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        w = self._check_weights(y, weights)
        var = np.asarray(self.variance(mu), dtype=np.float64)
        n_eff = float(np.sum(w > 0.0))
        edf_val = 0.0 if edf is None else float(edf)
        denom = max(n_eff - edf_val, 1.0)
        return float(np.sum(w * (y - mu) ** 2 / np.clip(var, self.eps, None)) / denom)
