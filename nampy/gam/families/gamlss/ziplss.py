from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo


def _l1ee(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(-exp(x))).  Mirrors mgcv ``l1ee``."""
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(np.minimum(x, 500.0))
    # lower tail: log(1-exp(-f)) ≈ log(f - f^2/2 + f^3/6)
    low = x < np.log(np.finfo(np.float64).eps) / 3.0
    very_low = x < -np.log(np.finfo(np.float64).max)
    l = np.log1p(-np.exp(-ex))
    exi = ex[low]
    l[low] = np.log(exi - exi**2 / 2.0 + exi**3 / 6.0)
    l[very_low] = x[very_low]
    return l


def _lee1(x: np.ndarray) -> np.ndarray:
    """log(exp(exp(x)) - 1).  Mirrors mgcv ``lee1``."""
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(np.minimum(x, 500.0))
    low = x < np.log(np.finfo(np.float64).eps) / 3.0
    very_low = x < -np.log(np.finfo(np.float64).max)
    high = x > np.log(np.log(np.finfo(np.float64).max))
    l = np.log(np.expm1(ex))
    exi = ex[low]
    l[low] = np.log(exi + exi**2 / 2.0 + exi**3 / 6.0)
    l[very_low] = x[very_low]
    l[high] = ex[high]
    return l


def _ldg(g: np.ndarray, deriv: int = 4) -> dict:
    """
    Derivatives of ZIP log-lik w.r.t. g (log Poisson mean) for y>0 observations.

    Mirrors mgcv ``ldg(g, deriv)``.
    Returns dict with l1, l2, (l3, l4 if deriv>1/2).
    """
    g = np.asarray(g, dtype=np.float64)

    def alpha(g_v):
        eg = np.exp(np.minimum(g_v, 500.0))
        low = g_v < np.log(np.finfo(np.float64).eps) / 3.0
        a = eg / (1.0 - np.exp(-eg))
        a[low] = 1.0 + eg[low] / 2.0 + eg[low] ** 2 / 12.0
        return a

    ghi_cap = np.log(np.log(np.finfo(np.float64).max)) + 1.0
    ghi_max = np.log(np.finfo(np.float64).max) / 5.0

    low = g < np.log(np.finfo(np.float64).eps) / 3.0
    high = g > ghi_cap
    a = alpha(g)
    eg = np.exp(np.minimum(g, 500.0))

    l2 = a * (a - eg - 1.0)
    egi = eg[low]
    b = egi * (1.0 + egi / 6.0) / 2.0
    l2[low] = a[low] * (b - egi)
    l2[high] = -eg[high]

    l3 = l4 = None
    if deriv > 1:
        l3 = a * (a * (-2.0 * a + 3.0 * (eg + 1.0)) - 3.0 * eg - eg**2 - 1.0)
        l3[low] = a[low] * (-b - 2.0 * b**2 + 3.0 * b * egi - egi**2)
        l3[high] = -eg[high]

    if deriv > 2:
        l4 = a * (
            6.0 * a**3
            - 12.0 * (eg + 1.0) * a**2
            + 4.0 * eg * a
            + 7.0 * (eg + 1.0) ** 2 * a
            - (4.0 + 3.0 * eg) * eg
            - (eg + 1.0) ** 3
        )
        b_l4 = egi * (1.0 + egi / 6.0) / 2.0
        l4[low] = a[low] * (
            6.0 * b_l4 * (3.0 + 3.0 * b_l4 + b_l4**2)
            - 12.0 * egi * (1.0 + 2.0 * b_l4 + b_l4**2)
            - 12.0 * b_l4 * (2.0 - b_l4)
            + 4.0 * egi * (1.0 + b_l4)
            + 7.0 * (egi**2 + 2.0 * egi + b_l4 * egi**2 + 2.0 * b_l4 * egi + b_l4)
            - (4.0 + 3.0 * egi) * egi
            - egi * (3.0 + 3.0 * egi + egi**2)
        )
        l4[high] = -eg[high]

    l1 = -a
    # clamp extreme g
    ii = g > ghi_max
    if np.any(ii):
        cap = -np.exp(min(ghi_max, 500.0))
        l1[ii] = cap
        l2[ii] = cap
        if l3 is not None:
            l3[ii] = cap
        if l4 is not None:
            l4[ii] = cap

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


def _lde(eta: np.ndarray, deriv: int = 4) -> dict:
    """
    Derivatives of log(1-exp(-exp(eta))) w.r.t. eta (for y>0 observations).

    Mirrors mgcv ``lde(eta, deriv)``.
    """
    eta = np.asarray(eta, dtype=np.float64)

    eps_log = np.log(np.finfo(np.float64).eps) / 3.0
    max_log = np.log(np.finfo(np.float64).max)

    low = eta < eps_log
    high = eta > max_log

    et = np.exp(np.minimum(eta, 500.0))  # exp(eta)
    eti = et[low]

    # l1 = exp(eta)/(exp(exp(eta))-1) = f/(exp(f)-1) where f=exp(eta)
    l1 = et.copy()
    safe_high = ~low & ~high
    ef = np.exp(np.minimum(et[safe_high], 500.0))
    l1[safe_high] = et[safe_high] / (ef - 1.0)
    b = -eti * (1.0 + eti / 6.0) / 2.0
    l1[low] = 1.0 + b
    l1[high] = 0.0

    # l2 = l1*(1-et-l1)
    l2 = l1 * ((1.0 - et) - l1)
    l2[low] = -b * (1.0 + eti + b) - eti
    l2[high] = 0.0

    l3 = l4 = None
    if deriv > 1:
        high3 = eta > max_log / 2.0
        l3 = l1 * ((1.0 - et) ** 2 - et - 3.0 * (1.0 - et) * l1 + 2.0 * l1**2)
        l3[low] = l1[low] * (
            -3.0 * eti + eti**2 - 3.0 * (-eti + b - eti * b) + 2.0 * b * (2.0 + b)
        )
        l3[high3] = 0.0

    if deriv > 2:
        high4 = eta > max_log / 3.0
        l4 = l1 * (
            (3.0 * et - 4.0) * et
            + 4.0 * et * l1
            + (1.0 - et) ** 3
            - 7.0 * (1.0 - et) ** 2 * l1
            + 12.0 * (1.0 - et) * l1**2
            - 6.0 * l1**3
        )
        b_l = b  # same b used in low region
        l4[low] = l1[low] * (
            4.0 * l1[low] * eti
            - eti**3
            - b_l
            - 7.0 * b_l * eti**2
            - eti**2
            - 5.0 * eti
            - 10.0 * b_l * eti
            - 12.0 * eti * b_l**2
            - 6.0 * b_l**2
            - 6.0 * b_l**3
        )
        l4[high4] = 0.0

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


def _zipll(y: np.ndarray, g: np.ndarray, eta: np.ndarray, deriv: int = 0) -> dict:
    """
    ZIP log-likelihood and derivatives w.r.t. g and eta.

    Parameters
    ----------
    y    : observed counts (non-negative integers)
    g    : first predictor (log Poisson mean, identity link applied)
    eta  : second predictor (loglog presence: 1-P(y=0) = 1-exp(-exp(eta)))
    deriv: 0=ll only, 1=grad+Hess, 2=+3rd derivs, 4=+4th derivs

    Mirrors mgcv ``zipll(y, g, eta, deriv)``.
    """
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    n = len(y)

    zind = y == 0
    yp = y[~zind]

    et = np.exp(np.minimum(eta, 500.0))  # exp(eta)
    l = et.copy()  # start with zeros shaped like et
    l[zind] = -et[zind]  # log P(y=0) = log(exp(-exp(eta))) = -exp(eta)
    l[~zind] = _l1ee(eta[~zind]) + yp * g[~zind] - _lee1(g[~zind]) - gammaln(yp + 1.0)

    l1 = l2 = l3 = l4 = None
    if deriv:
        l1 = np.zeros((n, 2), dtype=np.float64)
        le = _lde(eta, deriv)
        lg = _ldg(g, deriv)

        l1[~zind, 0] = yp + lg["l1"][~zind]  # l_g, y>0
        l1[zind, 1] = l[zind]  # l_eta, y=0 = -exp(eta)
        l1[~zind, 1] = le["l1"][~zind]  # l_eta, y>0

        l2 = np.zeros((n, 3), dtype=np.float64)
        # order: gg, ge, ee
        l2[~zind, 0] = lg["l2"][~zind]  # l_gg, y>0
        l2[~zind, 2] = le["l2"][~zind]  # l_ee, y>0
        l2[zind, 2] = l[zind]  # l_ee, y=0

    if deriv > 1:
        l3 = np.zeros((n, 4), dtype=np.float64)
        # order: ggg, gge, gee, eee
        l3[~zind, 0] = lg["l3"][~zind]
        l3[~zind, 3] = le["l3"][~zind]
        l3[zind, 3] = l[zind]

    if deriv > 3:
        l4 = np.zeros((n, 5), dtype=np.float64)
        # order: gggg, ggge, ggee, geee, eeee
        l4[~zind, 0] = lg["l4"][~zind]
        l4[~zind, 4] = le["l4"][~zind]
        l4[zind, 4] = l[zind]

    return {"l": l, "l1": l1, "l2": l2, "l3": l3, "l4": l4}


_ZIPLSS_SATURATED_LAMBDA = np.array(
    [
        1.593624,
        2.821439,
        3.920690,
        4.965114,
        5.984901,
        6.993576,
        7.997309,
        8.998888,
        9.999546,
        10.999816,
        11.999926,
        12.999971,
        13.999988,
        14.999995,
        15.999998,
        16.999999,
    ],
    dtype=np.float64,
)


def _ziplss_saturated_loglik(y: np.ndarray) -> np.ndarray:
    """Saturated log-likelihood for ziplss (mgcv ``zipll(log(g), 1e10)`` analogue)."""
    y = np.asarray(y, dtype=np.float64).ravel().copy()
    l = y.copy()
    if l.size == 0:
        return l

    l[y < 2.0] = 0.0
    ind_mid = (y > 1.0) & (y < 18.0)
    if np.any(ind_mid):
        g = y.copy()
        idx = y[ind_mid].astype(np.int64) - 2
        idx = np.clip(idx, 0, _ZIPLSS_SATURATED_LAMBDA.size - 1)
        g[ind_mid] = _ZIPLSS_SATURATED_LAMBDA[idx]
    else:
        g = y.copy()

    ind = y > 1.0
    if np.any(ind):
        l[ind] = _zipll(
            y[ind],
            np.log(np.asarray(g, dtype=np.float64)[ind]),
            np.full(int(np.sum(ind)), 1.0e10, dtype=np.float64),
            deriv=0,
        )["l"]
    return l


# ---------------------------------------------------------------------------
# ziplss: Zero-inflated Poisson  (mgcv: gamlss.r::ziplss)
# ---------------------------------------------------------------------------


class ZiplssFamily(GamlssFamily):
    """
    Zero-inflated Poisson location-scale family with two linear predictors:
      1. g = log(lambda): log Poisson mean (identity link)
      2. eta: loglog presence parameter [P(y>0) = 1-exp(-exp(eta))]

    Both links are "identity" (in mgcv's terminology — the predictors act
    directly on the internal reparameterised scale, not on λ or p directly).

    Mirrors mgcv ``ziplss(link=list("identity","identity"))``.
    """

    name = "ziplss"
    family_class = "general"
    nlp = 2
    n_linear_predictors = 2

    supports_laml = True
    supports_ml = True
    supports_reml = True
    supports_gcv = False
    supports_ubre = False
    supports_pirls = False
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True
    supports_closed_form_solve = False
    supports_exact_pirls_first_derivatives = False
    supports_exact_pirls_second_derivatives = False

    def __init__(self):
        super().__init__()
        # Both links are identity
        self.linfo = [_IdentityLinkInfo(), _IdentityLinkInfo()]
        self.tri = trind_generator(2)
        self.link_names = ("identity", "identity")
        self.link_name = "(identity, identity)"

    def validate_y(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf")
        if not np.all(y >= 0.0):
            raise ValueError("ziplss requires non-negative response y >= 0.")
        return y

    def ll(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        weights: np.ndarray,
        offset: Any = None,
        deriv: int = 0,
        d1b: Any = 0,
        d2b: Any = 0,
        fh: Any = None,
        D: Any = None,
        **kw,
    ) -> dict[str, Any]:
        """
        Log-likelihood and derivatives for the ZIP model.

        Mirrors mgcv ``ziplss$ll``.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        sandwich = bool(kw.get("sandwich", False))

        off1 = off2 = None
        if offset is not None:
            if isinstance(offset, (list, tuple)):
                off1 = (
                    np.asarray(offset[0], dtype=np.float64)
                    if len(offset) > 0 and offset[0] is not None
                    else None
                )
                off2 = (
                    np.asarray(offset[1], dtype=np.float64)
                    if len(offset) > 1 and offset[1] is not None
                    else None
                )
            else:
                off1 = np.asarray(offset, dtype=np.float64)

        # Linear predictors (both identity links)
        g = X[:, jj[0]] @ coef[jj[0]]  # log Poisson mean
        if off1 is not None:
            g = g + off1
        eta = X[:, jj[1]] @ coef[jj[1]]  # loglog presence
        if off2 is not None:
            eta = eta + off2

        # lambda and p are linkinv(eta_k) = identity = eta_k directly
        lam = self.linfo[0].linkinv(g)  # = g
        p = self.linfo[1].linkinv(eta)  # = eta

        zl = _zipll(y, lam, p, deriv)
        l = float(np.sum(zl["l"]))

        if deriv == 0:
            return {"l": l, "l0": zl["l"]}

        # Link derivatives for chain rule (both identity → trivial)
        ig1 = np.column_stack(
            [
                self.linfo[0].mu_eta(g),
                self.linfo[1].mu_eta(eta),
            ]
        )
        g2 = np.column_stack(
            [
                self.linfo[0].d2link(lam),
                self.linfo[1].d2link(p),
            ]
        )
        g3: Any = 0
        g4: Any = 0
        if deriv > 1:
            g3 = np.column_stack(
                [
                    self.linfo[0].d3link(lam),
                    self.linfo[1].d3link(p),
                ]
            )
        if deriv > 3:
            g4 = np.column_stack(
                [
                    self.linfo[0].d4link(lam),
                    self.linfo[1].d4link(p),
                ]
            )

        i2 = self.tri["i2"]
        i3 = self.tri["i3"]
        i4 = self.tri["i4"]

        de = gamlss_etamu(
            zl["l1"],
            zl["l2"],
            zl["l3"] if zl["l3"] is not None else 0,
            zl["l4"] if zl["l4"] is not None else 0,
            ig1,
            g2,
            g3,
            g4,
            i2,
            i3,
            i4,
            deriv - 1,
        )

        ret = gamlss_gH(
            X,
            jj,
            de["l1"],
            de["l2"],
            i2,
            l3=de["l3"],
            i3=i3,
            l4=de["l4"],
            i4=i4,
            d1b=d1b,
            d2b=d2b,
            deriv=deriv - 1,
            fh=fh,
            D=D,
            sandwich=sandwich,
        )
        ret["l"] = l
        ret["l0"] = zl["l"]
        return ret

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        offset: Any = None,
        weights: Any = None,
        E: Any = None,
    ) -> np.ndarray:
        """
        Initialize coefficients for ziplss.

        Regress binarized y on X2 (presence predictor), then weighted-regress
        log(y + 0.2) on X1 (log-mean predictor) downweighting y=0 with low p.

        Mirrors mgcv ``ziplss$initialize`` regular matrix path.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        start = np.zeros(p, dtype=np.float64)

        # --- Fit presence predictor on binarized y ---
        X2 = X[:, jj[1]]
        yt_bin = (y > 0).astype(np.float64)

        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            XE2 = np.vstack([X2, E2])
            y2e = np.concatenate([yt_bin, np.zeros(E2.shape[0])])
        else:
            XE2 = X2
            y2e = yt_bin

        try:
            start2 = np.linalg.lstsq(XE2, y2e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # Downweight y=0 with low estimated presence
        p_est = X2[:n] @ start2
        w = np.ones(n, dtype=np.float64)
        w[(y == 0) & (p_est < 0.5)] = 0.1

        # --- Fit log-mean predictor ---
        yt_lam = np.log(np.maximum(y + 0.2, 1e-300)) * w
        X1 = X[:, jj[0]]
        Xw1 = X1 * w[:, None]

        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            XE1 = np.vstack([Xw1, E1])
            y1e = np.concatenate([yt_lam, np.zeros(E1.shape[0])])
        else:
            XE1 = Xw1
            y1e = yt_lam

        try:
            start1 = np.linalg.lstsq(XE1, y1e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        return start

    def residuals(
        self,
        y: np.ndarray,
        fitted: np.ndarray,
        rtype: str = "deviance",
        *,
        eta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Response or deviance residuals.  Mirrors mgcv ``ziplss$residuals``."""
        y = np.asarray(y, dtype=np.float64)
        fitted = np.asarray(fitted, dtype=np.float64)
        eta_arr = None if eta is None else np.asarray(eta, dtype=np.float64)

        if eta_arr is not None:
            if eta_arr.ndim == 1:
                eta_arr = eta_arr[:, None]
            if eta_arr.shape[1] >= 2:
                lam_pred = np.asarray(eta_arr[:, 0], dtype=np.float64)
                eta_pred = np.asarray(eta_arr[:, 1], dtype=np.float64)
            else:
                lam_pred = eta_pred = None
        elif fitted.ndim == 2 and fitted.shape[1] >= 2:
            lam_pred = np.asarray(fitted[:, 0], dtype=np.float64)
            eta_pred = np.asarray(fitted[:, 1], dtype=np.float64)
        else:
            lam_pred = eta_pred = None

        if lam_pred is None or eta_pred is None:
            rsd = y - np.asarray(fitted, dtype=np.float64).ravel()
            if rtype == "response":
                return rsd
            raise NotImplementedError(
                "ziplss deviance residuals require fitted eta for both predictors."
            )

        lam = np.exp(lam_pred)
        p = 1.0 - np.exp(-np.exp(eta_pred))  # prob of presence

        small_lam = lam <= np.sqrt(np.finfo(np.float64).eps)
        Ey = p.copy()
        Ey[~small_lam] = (
            p[~small_lam] * lam[~small_lam] / (1.0 - np.exp(-lam[~small_lam]))
        )

        rsd = y - Ey
        if rtype == "response":
            return rsd

        rsd_dev = 2.0 * (
            _ziplss_saturated_loglik(y) - _zipll(y, lam_pred, eta_pred, deriv=0)["l"]
        )
        rsd_dev = np.maximum(0.0, rsd_dev)
        return np.sqrt(rsd_dev) * np.sign(rsd)

    def _predict_response_from_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        gamma = np.asarray(eta[:, 0], dtype=np.float64)
        eta_p = np.asarray(eta[:, 1], dtype=np.float64)
        et = np.exp(eta_p)
        p = 1.0 - np.exp(-et)
        lam = np.exp(gamma)
        mu = p.copy()
        ind = gamma < np.log(np.finfo(np.float64).eps) / 2.0
        mu[~ind] = lam[~ind] / (1.0 - np.exp(-lam[~ind]))
        mu[ind] = 1.0
        return (p * mu).reshape(-1, 1)


def ziplss() -> "ZiplssFamily":
    """
    Zero-inflated Poisson location-scale family.

    Two linear predictors:
      1. identity link on log Poisson mean
      2. identity link on loglog presence parameter

    Returns
    -------
    ZiplssFamily instance.
    """
    return ZiplssFamily()


# ---------------------------------------------------------------------------
# Link: shifted logit  (gevlss shape parameter xi confined to (-1, 0.5))
# ---------------------------------------------------------------------------
