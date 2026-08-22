from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import ncx2

__all__ = ["DaviesAlgorithm", "liu2", "psum_chisq"]


class DaviesAlgorithm:
    def __init__(self):
        self._count = 0

    def _counter(self, clear=False):
        if clear:
            a = self._count
            self._count = 0
            return a
        else:
            self._count += 1
            return self._count

    def _ln1(self, x, first):
        if first:
            return math.log1p(x)
        return _log1pmx(x)

    def _errbd(self, u, sigsq, r, n, lb, nc):
        self._counter(False)
        cx = u * sigsq
        sum1 = u * cx
        u2 = u * 2.0

        for j in range(r - 1, -1, -1):
            nj, lj, ncj = n[j], lb[j], nc[j]
            x = u2 * lj
            y = 1.0 - x
            cx += lj * (ncj / y + nj) / y
            xy = x / y
            sum1 += ncj * xy * xy + nj * (x * xy + self._ln1(-x, False))

        return math.exp(-0.5 * sum1), cx

    def _ctff(self, accx, upn, mean, lmin, lmax, sigsq, r, n, lb, nc):
        u2 = upn
        u1 = 0.0
        c1 = mean
        rb = 2.0 * lmax if u2 > 0 else 2.0 * lmin

        while True:
            err, c2 = self._errbd(u2 / (1.0 + u2 * rb), sigsq, r, n, lb, nc)
            if err <= accx:
                break
            u1 = u2
            c1 = c2
            u2 *= 2.0

        while True:
            if (c2 - mean) == 0 or abs((c1 - mean) / (c2 - mean)) >= 0.9:
                break
            u = (u1 + u2) * 0.5
            err, cst = self._errbd(u / (1.0 + u * rb), sigsq, r, n, lb, nc)
            if err > accx:
                u1, c1 = u, cst
            else:
                u2, c2 = u, cst

        return c2, u2

    def _truncation(self, u, tausq, sigsq, r, n, lb, nc):
        self._counter(False)
        pi = math.pi
        sum1 = prod2 = prod3 = 0.0
        s = 0
        sum2 = (sigsq + tausq) * u * u
        prod1 = 2.0 * sum2
        u2 = 2.0 * u

        for j in range(r):
            lj, ncj, nj = lb[j], nc[j], n[j]
            x = u2 * lj
            x2 = x * x
            sum1 += ncj * x2 / (1.0 + x2)
            if x2 > 1.0:
                prod2 += nj * math.log(x2)
                prod3 += nj * self._ln1(x2, True)
                s += nj
            else:
                prod1 += nj * self._ln1(x2, True)

        sum1 *= 0.5
        prod2 += prod1
        prod3 += prod1

        x_val = math.exp(-sum1 - 0.25 * prod2) / pi
        y_val = math.exp(-sum1 - 0.25 * prod3) / pi

        err1 = 1.0 if s == 0 else 2.0 * x_val / s
        err2 = 2.5 * y_val if prod3 > 1.0 else 1.0
        if err2 < err1:
            err1 = err2

        x_val = 0.5 * sum2
        err2 = 1.0 if x_val <= y_val else y_val / x_val
        return min(err1, err2)

    def _findu(self, utx, accx, sigsq, r, n, lb, nc):
        a = [2.0, 1.4, 1.2, 1.1]
        ut = utx
        u = ut * 0.25

        if self._truncation(u, 0, sigsq, r, n, lb, nc) > accx:
            while self._truncation(ut, 0, sigsq, r, n, lb, nc) > accx:
                ut *= 4.0
        else:
            ut = u
            u /= 4.0
            while self._truncation(u, 0, sigsq, r, n, lb, nc) <= accx:
                ut = u
                u /= 4.0

        for factor in a:
            u = ut / factor
            if self._truncation(u, 0, sigsq, r, n, lb, nc) <= accx:
                ut = u
        return ut

    def _integrate(
        self, nterm, interv, tausq, main, c, acc, intl, ersm, sigsq, r, n, lb, nc
    ):
        pi = math.pi
        inpi = interv / pi

        for k in range(nterm, -1, -1):
            u = (k + 0.5) * interv
            sum1 = -2.0 * u * c
            sum3 = -0.5 * sigsq * u * u

            for j in range(r - 1, -1, -1):
                nj = n[j]
                x = 2.0 * lb[j] * u
                y = x * x
                sum3 -= 0.25 * nj * self._ln1(y, True)
                y_val = nc[j] * x / (1.0 + y)
                z = nj * math.atan(x) + y_val
                sum1 += z
                sum3 -= 0.5 * x * y_val

            term_sum2 = abs(-2.0 * u * c)
            for j in range(r):
                x = 2.0 * lb[j] * u
                term_sum2 += abs(n[j] * math.atan(x) + (nc[j] * x / (1.0 + x * x)))

            x_factor = inpi * math.exp(sum3) / u
            if not main:
                x_factor *= 1.0 - math.exp(-0.5 * tausq * u * u)

            intl[0] += math.sin(0.5 * sum1) * x_factor
            ersm[0] += 0.5 * term_sum2 * x_factor

    def _cfe(self, x, th, ln28, r, n, lb, nc):
        self._counter(False)
        pi = math.pi
        axl = abs(x)
        sxl = -1 if x < 0 else 1
        sum1 = 0.0

        for j in range(r - 1, -1, -1):
            t = th[j]
            if lb[t] * sxl > 0.0:
                lj = abs(lb[t])
                axl1 = axl - lj * (n[t] + nc[t])
                axl2 = lj / ln28
                if axl1 > axl2:
                    axl = axl1
                else:
                    if axl > axl2:
                        axl = axl2
                    sum1 = (axl - axl1) / lj
                    for k in range(j - 1, -1, -1):
                        sum1 += n[th[k]] + nc[th[k]]
                    break

        if sum1 > 100.0:
            return 1.0, True
        else:
            res = math.pow(2.0, sum1 * 0.25) / (pi * axl * axl)
            return res, False

    def _c_round_int(self, x):
        base = math.floor(x)
        if x - base > 0.5:
            base += 1
        return int(base)

    def davies(self, lb, nc, n, r, sigma, c_val, lim, acc):
        """
        Main entry point.
        Returns (probability, trace, ifault)
        """
        self._counter(True)
        ln28 = math.log(2.0) / 8.0
        pi = math.pi
        trace = [0.0] * 7
        ifault = 0
        intl, ersm = [0.0], [0.0]
        acc1 = acc

        lb = np.array(lb, dtype=float)
        nc = np.array(nc, dtype=float)
        n = np.array(n, dtype=int)

        th = np.argsort(np.abs(lb))[::-1]

        sd = sigma * sigma
        sigsq = sd
        lmax = lmin = 0.0
        mean = 0.0

        for j in range(r):
            nj, lj, ncj = n[j], lb[j], nc[j]
            if nj < 0 or ncj < 0:
                return -1.0, trace, 3
            sd += lj * lj * (2.0 * nj + 4.0 * ncj)
            mean += lj * (nj + ncj)
            if lmax < lj:
                lmax = lj
            elif lmin > lj:
                lmin = lj

        if sd == 0.0:
            return (1.0 if c_val > 0.0 else 0.0), trace, 0
        if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
            return -1.0, trace, 3

        sd = math.sqrt(sd)
        almx = max(lmax, -lmin)

        utx = 16.0 / sd
        up, un = 4.5 / sd, -4.5 / sd

        utx = self._findu(utx, 0.5 * acc1, sigsq, r, n, lb, nc)

        if c_val != 0.0 and almx > 0.07 * sd:
            val_cfe, fail = self._cfe(c_val, th, ln28, r, n, lb, nc)
            if not fail:
                tausq = 0.25 * acc1 / val_cfe
                if self._truncation(utx, tausq, sigsq, r, n, lb, nc) < 0.2 * acc1:
                    sigsq += tausq
                    utx = self._findu(utx, 0.25 * acc1, sigsq, r, n, lb, nc)
                    trace[5] = math.sqrt(tausq)

        trace[4] = utx
        acc1 *= 0.5

        while True:
            d1_val, up = self._ctff(acc1, up, mean, lmin, lmax, sigsq, r, n, lb, nc)
            d1 = d1_val - c_val
            if d1 < 0.0:
                trace[6] = self._counter(True)
                return 1.0, trace, 0

            d2_val, un = self._ctff(acc1, un, mean, lmin, lmax, sigsq, r, n, lb, nc)
            d2 = c_val - d2_val
            if d2 < 0.0:
                trace[6] = self._counter(True)
                return 0.0, trace, 0

            intv = (2.0 * pi / d1) if d1 > d2 else (2.0 * pi / d2)

            x = utx / intv
            nt = self._c_round_int(x)
            x = 3.0 / math.sqrt(acc1)
            ntm = self._c_round_int(x)

            if nt > ntm * 1.5:
                intv1 = utx / ntm
                x = 2.0 * pi / intv1
                if x <= abs(c_val):
                    break

                cfe1, fail1 = self._cfe(c_val - x, th, ln28, r, n, lb, nc)
                cfe2, fail2 = self._cfe(c_val + x, th, ln28, r, n, lb, nc)

                if fail1 or fail2:
                    break

                tausq = 0.33 * acc1 / (1.1 * (cfe1 + cfe2))
                acc1 *= 0.67
                if ntm > lim:
                    trace[6] = self._counter(True)
                    return -1.0, trace, 0

                self._integrate(
                    ntm,
                    intv1,
                    tausq,
                    False,
                    c_val,
                    acc,
                    intl,
                    ersm,
                    sigsq,
                    r,
                    n,
                    lb,
                    nc,
                )
                lim -= ntm
                sigsq += tausq
                trace[2] += 1.0
                trace[1] += ntm + 1
                utx = self._findu(utx, 0.25 * acc1, sigsq, r, n, lb, nc)
                acc1 *= 0.75
            else:
                break

        trace[3] = intv
        if nt > lim:
            trace[6] = self._counter(True)
            return -1.0, trace, 1

        self._integrate(
            nt, intv, 0.0, True, c_val, acc, intl, ersm, sigsq, r, n, lb, nc
        )
        trace[2] += 1
        trace[1] += nt + 1
        result_c = 0.5 - intl[0]
        trace[0] = ersm[0]

        x_err = ersm[0] + acc / 10.0
        j_fact = 1
        for _i in range(4):
            if float(j_fact * x_err) == float(j_fact * ersm[0]):
                ifault = 2
            j_fact *= 2

        trace[6] = self._counter(True)
        return result_c, trace, ifault


def _log1pmx(x: float) -> float:
    """Stable scalar `log1p(x) - x`, matching mgcv/src/davies.c::ln1()."""
    if x == 0.0:
        return -0.0
    if abs(x) < 0.5:
        term = -0.5 * x * x
        total = term
        power = x * x
        sign = 1.0
        for k in range(3, 1000):
            power *= x
            term = sign * power / k
            total_next = total + term
            if total_next == total or abs(term) <= np.finfo(float).eps * max(
                1.0, abs(total_next)
            ):
                return total_next
            total = total_next
            sign = -sign
        return total
    return math.log1p(x) - x


def liu2(
    x: float | np.ndarray,
    lb: np.ndarray,
    *,
    df: np.ndarray | None = None,
    lower_tail: bool = False,
) -> float | np.ndarray:
    """
    Mirror mgcv/R/mgcv.r::liu2() for central chi-square mixtures.
    """
    q = np.asarray(x, dtype=np.float64)
    scalar = q.ndim == 0
    q = q.reshape(1) if scalar else q.copy()

    lb = np.asarray(lb, dtype=np.float64).ravel()
    if df is None:
        h = np.ones(lb.size, dtype=np.float64)
    else:
        h = np.asarray(df, dtype=np.float64).ravel()
        if h.size == 1:
            h = np.repeat(h, lb.size)
    if h.size != lb.size:
        raise ValueError("lambda and h should have the same length.")

    lh = lb * h
    mu_q = float(np.sum(lh))

    lh = lh * lb
    c2 = float(np.sum(lh))

    lh = lh * lb
    c3 = float(np.sum(lh))

    xpos = q > 0.0
    out = np.ones_like(q, dtype=np.float64)
    if (not np.any(xpos)) or c2 <= 0.0:
        return float(out[0]) if scalar else out

    s1 = c3 / np.power(c2, 1.5)
    s2 = float(np.sum(lh * lb)) / (c2 * c2)
    sig_q = np.sqrt(2.0 * c2)
    t = (q[xpos] - mu_q) / sig_q

    if s1 * s1 > s2:
        a = 1.0 / (s1 - np.sqrt(s1 * s1 - s2))
        delta = s1 * a * a * a - a * a
        l_df = a * a - 2.0 * delta
    else:
        if c3 == 0.0:
            return float(out[0]) if scalar else out
        a = 1.0 / s1
        delta = 0.0
        l_df = (c2 * c2 * c2) / (c3 * c3)

    mu_x = l_df + delta
    sig_x = np.sqrt(2.0) * a
    z = t * sig_x + mu_x
    if lower_tail:
        out[xpos] = ncx2.cdf(z, df=l_df, nc=delta)
    else:
        out[xpos] = ncx2.sf(z, df=l_df, nc=delta)
    return float(out[0]) if scalar else out


def psum_chisq(
    q: float | np.ndarray,
    lb: np.ndarray,
    *,
    df: np.ndarray | None = None,
    nc: np.ndarray | None = None,
    sigz: float = 0.0,
    lower_tail: bool = False,
    tol: float = 2e-5,
    nlim: int = 100000,
) -> float | np.ndarray:
    """
    Mirror mgcv/R/mgcv.r::psum.chisq() using the local Davies port of
    mgcv/src/davies.c together with the Liu fallback from mgcv/R/mgcv.r.
    """
    x = np.asarray(q, dtype=np.float64)
    scalar = x.ndim == 0
    x = x.reshape(1) if scalar else x.copy()

    lb = np.asarray(lb, dtype=np.float64).ravel()
    r = int(lb.size)
    if r <= 0 or np.all(lb == 0.0):
        raise ValueError("at least one element of lb must be non-zero")

    if df is None:
        h = np.ones(r, dtype=np.int64)
    else:
        h = np.rint(np.asarray(df, dtype=np.float64)).astype(np.int64).ravel()
        if h.size == 1:
            h = np.repeat(h, r)
    if nc is None:
        delta = np.zeros(r, dtype=np.float64)
    else:
        delta = np.asarray(nc, dtype=np.float64).ravel()
        if delta.size == 1:
            delta = np.repeat(delta, r)
    if h.size != r or delta.size != r:
        raise ValueError("lengths of lb, df and nc must match")
    if np.any(h < 1):
        raise ValueError("df must be positive integers")

    solver = DaviesAlgorithm()
    out = np.empty_like(x, dtype=np.float64)
    sigz = max(float(sigz), 0.0)
    central = np.all(delta == 0.0)

    for i, qi in enumerate(x):
        cprob, _trace, ifault = solver.davies(
            lb=lb,
            nc=delta,
            n=h,
            r=r,
            sigma=sigz,
            c_val=float(qi),
            lim=int(nlim),
            acc=float(tol),
        )
        if ifault == 0:
            out[i] = float(cprob if lower_tail else 1.0 - cprob)
        elif ifault == 2:
            warnings.warn("danger of round-off error", RuntimeWarning, stacklevel=2)
            out[i] = float(cprob if lower_tail else 1.0 - cprob)
        elif central:
            warnings.warn(
                "failure of Davies method, falling back on Liu et al approximtion",
                RuntimeWarning,
                stacklevel=2,
            )
            out[i] = float(liu2(qi, lb, df=h))
        else:
            warnings.warn(
                "failure of Davies method, falling back on Liu et al approximtion",
                RuntimeWarning,
                stacklevel=2,
            )
            out[i] = np.nan

    return float(out[0]) if scalar else out
