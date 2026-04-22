from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg


def _l1ee(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(-exp(x))).  Mirrors mgcv ``l1ee``."""
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(np.minimum(x, 500.0))
    # lower tail: log(1-exp(-f)) ≈ log(f - f^2/2 + f^3/6)
    low = x < np.log(np.finfo(np.float64).eps) / 3.0
    very_low = x < -np.log(np.finfo(np.float64).max)
    ll = np.log1p(-np.exp(-ex))
    exi = ex[low]
    ll[low] = np.log(exi - exi**2 / 2.0 + exi**3 / 6.0)
    ll[very_low] = x[very_low]
    return ll


def _lee1(x: np.ndarray) -> np.ndarray:
    """log(exp(exp(x)) - 1).  Mirrors mgcv ``lee1``."""
    x = np.asarray(x, dtype=np.float64)
    ex = np.exp(np.minimum(x, 500.0))
    low = x < np.log(np.finfo(np.float64).eps) / 3.0
    very_low = x < -np.log(np.finfo(np.float64).max)
    high = x > np.log(np.log(np.finfo(np.float64).max))
    ll = np.log(np.expm1(ex))
    exi = ex[low]
    ll[low] = np.log(exi + exi**2 / 2.0 + exi**3 / 6.0)
    ll[very_low] = x[very_low]
    ll[high] = ex[high]
    return ll


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
    ll = et.copy()  # start with zeros shaped like et
    ll[zind] = -et[zind]  # log P(y=0) = log(exp(-exp(eta))) = -exp(eta)
    ll[~zind] = _l1ee(eta[~zind]) + yp * g[~zind] - _lee1(g[~zind]) - gammaln(yp + 1.0)

    l1 = l2 = l3 = l4 = None
    if deriv:
        l1 = np.zeros((n, 2), dtype=np.float64)
        le = _lde(eta, deriv)
        lg = _ldg(g, deriv)

        l1[~zind, 0] = yp + lg["l1"][~zind]  # l_g, y>0
        l1[zind, 1] = ll[zind]  # l_eta, y=0 = -exp(eta)
        l1[~zind, 1] = le["l1"][~zind]  # l_eta, y>0

        l2 = np.zeros((n, 3), dtype=np.float64)
        # order: gg, ge, ee
        l2[~zind, 0] = lg["l2"][~zind]  # l_gg, y>0
        l2[~zind, 2] = le["l2"][~zind]  # l_ee, y>0
        l2[zind, 2] = ll[zind]  # l_ee, y=0

    if deriv > 1:
        l3 = np.zeros((n, 4), dtype=np.float64)
        # order: ggg, gge, gee, eee
        l3[~zind, 0] = lg["l3"][~zind]
        l3[~zind, 3] = le["l3"][~zind]
        l3[zind, 3] = ll[zind]

    if deriv > 3:
        l4 = np.zeros((n, 5), dtype=np.float64)
        # order: gggg, ggge, ggee, geee, eeee
        l4[~zind, 0] = lg["l4"][~zind]
        l4[~zind, 4] = le["l4"][~zind]
        l4[zind, 4] = ll[zind]

    return {"l": ll, "l1": l1, "l2": l2, "l3": l3, "l4": l4}


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
    ll = y.copy()
    if ll.size == 0:
        return ll

    ll[y < 2.0] = 0.0
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
        ll[ind] = _zipll(
            y[ind],
            np.log(np.asarray(g, dtype=np.float64)[ind]),
            np.full(int(np.sum(ind)), 1.0e10, dtype=np.float64),
            deriv=0,
        )["l"]
    return ll


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

        eta_mat = self._eta_matrix_from_inputs(
            X,
            jj,
            coef,
            offset=offset,
            eta=kw.get("eta", None),
        )
        g = np.asarray(eta_mat[:, 0], dtype=np.float64)
        eta = np.asarray(eta_mat[:, 1], dtype=np.float64)

        # lambda and p are linkinv(eta_k) = identity = eta_k directly
        lam = self.linfo[0].linkinv(g)  # = g
        p = self.linfo[1].linkinv(eta)  # = eta

        zl = _zipll(y, lam, p, deriv)
        ll = float(np.sum(zl["l"]))

        if deriv == 0:
            return {"l": ll, "l0": zl["l"]}

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
        if bool(kw.get("ncv", False)):
            ret["l1"] = np.asarray(de["l1"], dtype=np.float64)
            ret["l2"] = np.asarray(de["l2"], dtype=np.float64)
            ret["l3"] = de["l3"]
        ret["l"] = ll
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
            start2 = _pen_reg(X2, E2, yt_bin)
        else:
            try:
                start2 = np.linalg.lstsq(X2, yt_bin, rcond=None)[0]
            except np.linalg.LinAlgError:
                start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # Downweight y=0 with low estimated presence
        p_est = X2[:n] @ start2
        w = np.ones(n, dtype=np.float64)
        w[(y == 0) & (p_est < 0.5)] = 0.1

        # --- Fit log-mean predictor ---
        yt_lam = self.linfo[0].linkfun(np.log(np.abs(y) + (y == 0.0) * 0.2)) * w
        X1 = X[:, jj[0]]
        Xw1 = X1 * w[:, None]

        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            start1 = _pen_reg(Xw1, E1, yt_lam)
        else:
            try:
                start1 = np.linalg.lstsq(Xw1, yt_lam, rcond=None)[0]
            except np.linalg.LinAlgError:
                start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        return start

    def Dd(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        theta,
        wt=None,
        level: int = 0,
    ) -> dict[str, Any]:
        del theta
        y = np.asarray(y, dtype=np.float64).ravel()
        mu = np.asarray(mu, dtype=np.float64)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        if mu.ndim != 2 or mu.shape[1] < 2:
            raise ValueError("ziplss Dd expects mu with at least two predictor columns.")
        if mu.shape[0] != y.size:
            raise ValueError(
                f"ziplss Dd received {mu.shape[0]} rows, expected {y.size}."
            )
        if wt is None:
            w = np.ones(y.size, dtype=np.float64)
        else:
            w = np.asarray(wt, dtype=np.float64).ravel()
            if w.size != y.size:
                raise ValueError(
                    f"ziplss Dd received weights of length {w.size}, expected {y.size}."
                )

        deriv = int(level)
        zz = _zipll(y, mu[:, 0], mu[:, 1], deriv=max(deriv + 1, 1))
        out: dict[str, Any] = {
            "Dmu": np.asarray(zz["l1"], dtype=np.float64) * w[:, None],
        }
        if deriv >= 1:
            out["Dmu2"] = np.asarray(zz["l2"], dtype=np.float64) * w[:, None]
        if deriv >= 2:
            out["Dmu3"] = np.asarray(zz["l3"], dtype=np.float64) * w[:, None]
        if deriv >= 3:
            out["Dmu4"] = np.asarray(zz["l4"], dtype=np.float64) * w[:, None]
        return out

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

    def predict(
        self,
        *,
        eta: np.ndarray | None = None,
        X: np.ndarray | None = None,
        jj: list[np.ndarray] | None = None,
        coef: np.ndarray | None = None,
        offset: Any = None,
        se: bool = False,
        Vb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Response with ZIPLSS-specific uncertainty propagation.

        This explicit Jacobian path accounts for cross-predictor covariance terms
        that are otherwise dropped in the generic linear-response approximation.
        """
        if eta is None:
            if X is None or jj is None or coef is None:
                raise ValueError("Provide either eta or X/jj/coef for prediction.")
            eta = self._stacked_eta(X, jj, coef, offset=offset)
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim == 1:
            eta = eta[:, None]
        if eta.ndim != 2 or eta.shape[1] < 2:
            raise ValueError("ziplss predict expects two linear predictors.")
        eta = eta[:, :2]

        fit = np.asarray(self._predict_response_from_eta(eta), dtype=np.float64)
        if not se:
            return fit

        if Vb is None:
            raise ValueError("Vb is required when se=True.")
        if X is None or jj is None:
            raise ValueError("X and jj are required when se=True.")
        if len(jj) < 2:
            raise ValueError("ziplss predict requires two predictor index blocks when se=True.")

        X = np.asarray(X, dtype=np.float64)
        V = np.asarray(Vb, dtype=np.float64)
        jj0 = np.asarray(jj[0], dtype=int)
        jj1 = np.asarray(jj[1], dtype=int)
        if jj0.size == 0 or jj1.size == 0:
            raise ValueError("ziplss predict requires both predictor index blocks.")

        g = eta[:, 0]
        eta_p = eta[:, 1]
        lam = np.exp(np.clip(g, -700.0, 700.0))
        et = np.exp(np.clip(eta_p, -700.0, 700.0))
        p = 1.0 - np.exp(-et)

        q = 1.0 - np.exp(-lam)
        q_safe = np.maximum(q, np.finfo(np.float64).eps)
        mu_term = lam / q_safe
        tiny = np.sqrt(np.finfo(np.float64).eps)
        mu_term[lam <= tiny] = 1.0

        dgd = p * lam * (q - lam * np.exp(-lam)) / (q_safe**2)
        dgd[lam <= tiny] = 0.5 * p[lam <= tiny] * lam[lam <= tiny]
        dpe = mu_term * (et * np.exp(-et))

        Xg = X[:, jj0]
        Xe = X[:, jj1]
        Vgg = V[np.ix_(jj0, jj0)]
        Vee = V[np.ix_(jj1, jj1)]
        Vge = V[np.ix_(jj0, jj1)]

        var = np.einsum("ij,jk,ik->i", Xg, Vgg, Xg)
        var = (dgd**2) * var
        var += (dpe**2) * np.einsum("ij,jk,ik->i", Xe, Vee, Xe)
        var += 2.0 * dgd * dpe * np.einsum("ij,jk,ik->i", Xg, Vge, Xe)

        return fit, np.sqrt(np.maximum(var, 0.0))[:, None]

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
