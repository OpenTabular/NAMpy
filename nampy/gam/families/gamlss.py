"""
GAMLSS (General Additive Models for Location Scale and Shape) family implementations.

Families implement the ``ll`` interface expected by ``gam_fit5``:

    ll(y, X, jj, coef, weights, offset, deriv, d1b, d2b, fh, D) → dict

where ``jj`` is a list of column-index arrays (one per linear predictor),
mirroring mgcv's ``attr(X, "lpi")``.

Currently implemented
---------------------
``gaulss``
    Gaussian location-scale.  Two linear predictors: mean (μ) and precision
    (τ = 1/σ) via a "logb" link.  Mirrors mgcv ``gaulss``.

``gammals``
    Gamma location-scale.  Two linear predictors: log mean (identity link on
    log scale) and log scale/sigma (softplus-b link).  Mirrors mgcv ``gammals``.

``ziplss``
    Zero-inflated Poisson location-scale.  Two linear predictors: log Poisson
    mean (identity link) and loglog presence parameter.  Mirrors mgcv ``ziplss``.

``gevlss``
    Generalized Extreme Value location-scale-shape.  Three linear predictors:
    location μ (identity or log link), log-scale ρ (identity link), and shape
    ξ ∈ (-1, 0.5) (shifted-logit link).  Mirrors mgcv ``gevlss``.

``shashlss``
    Sinh-arcsinh location-scale-skewness-kurtosis (Fasiolo 2020).  Four linear
    predictors: location μ (identity), log-scale τ (logeb link), skewness ε
    (identity), log-kurtosis φ (identity).  Mirrors mgcv ``shash``.

Mirrors: mgcv/R/gamlss.r
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .._mgcv_constants import FAMILY_EPS
from ..fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .family_base import GeneralFamily

# ---------------------------------------------------------------------------
# Link helpers for gaulss
# ---------------------------------------------------------------------------


class _IdentityLinkInfo:
    """Identity link: mu = eta.  Mirrors mgcv make.link("identity") + fix.family.link."""

    name = "identity"

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return np.asarray(mu, dtype=np.float64)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=np.float64)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(eta).shape, dtype=np.float64)

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)


class _LogBLinkInfo:
    """
    'logb' link for gaulss precision parameter tau:
        eta = log(1/tau - b)   →   tau = 1/(exp(eta) + b)

    Mirrors mgcv stats[[2]] constructed inside ``gaulss(b=0.01)``.
    """

    name = "logb"

    def __init__(self, b: float = 0.01):
        self.b = float(b)

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(np.clip(1.0 / mu - self.b, 1e-300, None))

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (np.exp(np.clip(eta, -500.0, 500.0)) + self.b)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        ee = np.exp(np.clip(eta, -500.0, 500.0))
        return -ee / (ee + self.b) ** 2

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        mub = np.maximum(1.0 - mu * self.b, np.finfo(np.float64).eps)
        return (2.0 * mub - 1.0) / (mub * mu) ** 2

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        mub = np.maximum(1.0 - mu * self.b, np.finfo(np.float64).eps)
        return ((1.0 - mub) * mub * 6.0 - 2.0) / (mub * mu) ** 3

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        mub = np.maximum(1.0 - mu * self.b, np.finfo(np.float64).eps)
        return (((24.0 * mub - 36.0) * mub + 24.0) * mub - 6.0) / (mub * mu) ** 4


# ---------------------------------------------------------------------------
# Link helpers for shashlss
# ---------------------------------------------------------------------------


class _LogEBLinkInfo:
    """
    'logeb' link for shash log-scale parameter tau:
        linkinv(eta) = log(exp(eta) + b)   →   tau = log(exp(eta) + b)
        linkfun(tau) = log(exp(tau) - b)

    Mirrors the logeb link constructed inside mgcv ``shash(b=0.01)``.
    """

    name = "logeb"

    def __init__(self, b: float = 0.01):
        self.b = float(b)

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(np.maximum(np.exp(mu) - self.b, 1e-300))

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        return np.log(np.exp(np.minimum(eta, 500.0)) + self.b)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        ee = np.exp(np.minimum(eta, 500.0))
        return ee / (ee + self.b)

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        # d^2 eta / d mu^2 = fr*(1-fr) where fr = exp(mu)/(exp(mu)-b)
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(np.minimum(mu, 500.0))
        fr = em / np.maximum(em - self.b, 1e-300)
        return fr * (1.0 - fr)

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        # d^3 eta / d mu^3 = oo - 2*oo*fr  (oo = fr*(1-fr))
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(np.minimum(mu, 500.0))
        fr = em / np.maximum(em - self.b, 1e-300)
        oo = fr * (1.0 - fr)
        return oo - 2.0 * oo * fr

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        # -b*em*(b^2 + 4*b*em + em^2) / (em - b)^4
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(np.minimum(mu, 500.0))
        denom = np.maximum(em - self.b, 1e-300) ** 4
        return -self.b * em * (self.b**2 + 4.0 * self.b * em + em**2) / denom


# ---------------------------------------------------------------------------
# Base class for GAMLSS / general families
# ---------------------------------------------------------------------------


class GamlssFamily(GeneralFamily):
    """
    Base class for multi-predictor GAMLSS families.

    Concrete subclasses must set:
      ``nlp``       — number of linear predictors
      ``linfo``     — list of link-info objects (one per predictor)
      ``tri``       — dict from ``trind_generator(nlp)``
      ``name``      — family name string

    And implement:
      ``ll(y, X, jj, coef, weights, offset, deriv, **kw)``
      ``initialize(y, X, offset, weights)``
    """

    family_class = "general"
    nlp: int = 1
    linfo: list = []
    tri: dict = {}

    supports_laml = True
    supports_ml = True
    supports_reml = True
    supports_analytic_outer_derivatives = False
    supports_analytic_outer_gradient = False
    supports_analytic_outer_hessian = False

    n_linear_predictors: int = 1

    def validate_y(self, y):
        return np.asarray(y, dtype=np.float64).ravel()

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
        raise NotImplementedError

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        offset: Any = None,
        weights: Any = None,
        E: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _stacked_eta(
        self,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        offset: Any = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        eta_cols = []
        off_list: list[Any] | None = None
        if offset is not None:
            off_list = list(offset) if isinstance(offset, (list, tuple)) else [offset]
        for k, cols in enumerate(jj):
            eta_k = X[:, cols] @ coef[cols]
            if off_list is not None and k < len(off_list) and off_list[k] is not None:
                eta_k = eta_k + np.asarray(off_list[k], dtype=np.float64)
            eta_cols.append(np.asarray(eta_k, dtype=np.float64))
        return (
            np.column_stack(eta_cols)
            if eta_cols
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )

    def _predict_response_from_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim == 1:
            eta = eta[:, None]
        cols = []
        for k in range(eta.shape[1]):
            cols.append(np.asarray(self.linfo[k].linkinv(eta[:, k]), dtype=np.float64))
        return np.column_stack(cols) if cols else np.empty((eta.shape[0], 0))

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
        del Vb
        if se:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet implement predictive standard errors."
            )
        if eta is None:
            if X is None or jj is None or coef is None:
                raise ValueError("Provide either eta or X/jj/coef for prediction.")
            eta = self._stacked_eta(X, jj, coef, offset=offset)
        return np.asarray(self._predict_response_from_eta(eta), dtype=np.float64)

    def predict_fitted(
        self,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        offset: Any = None,
    ) -> np.ndarray:
        return self.predict(X=X, jj=jj, coef=coef, offset=offset)

    def sandwich(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        weights: np.ndarray | None,
        *,
        offset: Any = None,
    ) -> np.ndarray:
        ll = self.ll(
            np.asarray(y, dtype=np.float64),
            np.asarray(X, dtype=np.float64),
            jj,
            np.asarray(coef, dtype=np.float64),
            (
                np.ones(len(np.asarray(y, dtype=np.float64).ravel()), dtype=np.float64)
                if weights is None
                else np.asarray(weights, dtype=np.float64)
            ),
            offset=offset,
            deriv=1,
            sandwich=True,
        )
        return np.asarray(ll["lbb"], dtype=np.float64)


# ---------------------------------------------------------------------------
# gaulss: Gaussian location-scale  (mgcv: gamlss.r::gaulss)
# ---------------------------------------------------------------------------


class GaulssFamily(GamlssFamily):
    """
    Gaussian location-scale family with two linear predictors:
      1. mean μ (identity link by default)
      2. precision τ = 1/σ (logb link)

    Mirrors mgcv ``gaulss(link=list("identity","logb"), b=0.01)``.
    """

    name = "gaulss"
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

    def __init__(self, link=("identity", "logb"), b: float = 0.01):
        super().__init__()
        self.b = float(b)

        ok1 = ("identity", "log", "inverse", "sqrt")
        ok2 = ("logb",)

        link1_name = link[0] if isinstance(link[0], str) else "identity"
        link2_name = link[1] if isinstance(link[1], str) else "logb"

        if link1_name not in ok1:
            raise ValueError(f"Link {link1_name!r} not available for mu of gaulss.")
        if link2_name not in ok2:
            raise ValueError(
                f"Link {link2_name!r} not available for precision of gaulss."
            )

        # Build link info objects
        if link1_name == "identity":
            linfo1 = _IdentityLinkInfo()
        elif link1_name == "log":
            from ._function_maps import LogLink

            _lobj = LogLink(eps=FAMILY_EPS)
            linfo1 = _AdaptedLinkInfo(_lobj, link1_name)
        elif link1_name == "inverse":
            from ._function_maps import InverseLink

            _lobj = InverseLink(eps=FAMILY_EPS)
            linfo1 = _AdaptedLinkInfo(_lobj, link1_name)
        else:
            raise ValueError(f"Unsupported link {link1_name!r}")

        linfo2 = _LogBLinkInfo(b=b)

        self.linfo = [linfo1, linfo2]
        self.tri = trind_generator(2)
        self.link_names = (link1_name, link2_name)
        self.link_name = f"({link1_name}, {link2_name})"

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
        Log-likelihood and derivatives for the Gaussian location-scale model.

        Mirrors mgcv ``gaulss$ll``.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        n = len(y)
        sandwich = bool(kw.get("sandwich", False))

        # offset handling: offset[[1]], offset[[2]], offset[[3]] in R
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

        # Linear predictors
        eta = X[:, jj[0]] @ coef[jj[0]]
        if off1 is not None:
            eta = eta + off1
        eta1 = X[:, jj[1]] @ coef[jj[1]]
        if off2 is not None:
            eta1 = eta1 + off2

        mu = self.linfo[0].linkinv(eta)  # mean
        tau = self.linfo[1].linkinv(eta1)  # precision 1/sigma

        ymu = y - mu
        ymu2 = ymu**2
        tau2 = tau**2

        # log-likelihood: N(mu, sigma^2) with sigma = 1/tau
        # l = -0.5*(y-mu)^2 * tau^2 - 0.5*log(2pi) + log(tau)
        l0 = (
            -0.5 * ymu2 * tau2
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(np.maximum(tau, 1e-300))
        )
        l = float(np.sum(l0))

        if deriv == 0:
            return {
                "l": l,
                "l0": l0,
                "lb": None,
                "lbb": None,
                "d1H": None,
                "trHid2H": None,
            }

        # First derivatives w.r.t. mu and tau  (mgcv lines 957-966)
        l1 = np.empty((n, 2), dtype=np.float64)
        l1[:, 0] = tau2 * ymu  # dl/dmu
        l1[:, 1] = 1.0 / tau - tau * ymu2  # dl/dtau

        # Second derivatives: (mm, mt, tt)  →  packed (0,1,2)
        l2 = np.column_stack(
            [
                -tau2,  # d2l/dmu2
                2.0 * l1[:, 0] / tau,  # d2l/dmu dtau
                -ymu2 - 1.0 / tau2,  # d2l/dtau2
            ]
        )

        # Link derivatives for chain rule
        ig1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),  # d mu / d eta
                self.linfo[1].mu_eta(eta1),  # d tau / d eta1
            ]
        )
        g2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(tau),
            ]
        )

        l3_val: Any = 0
        l4_val: Any = 0
        g3: Any = 0
        g4: Any = 0

        if deriv > 1:
            # Third derivatives: (mmm, mmt, mtt, ttt)  →  packed (0,1,2,3)
            # mgcv lines 975-980
            l3_val = np.column_stack(
                [
                    np.zeros(n, dtype=np.float64),  # d3l/dmu3 = 0
                    -2.0 * tau,  # d3l/dmu2 dtau
                    2.0 * ymu,  # d3l/dmu dtau2
                    2.0 / tau**3,  # d3l/dtau3
                ]
            )
            g3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(tau),
                ]
            )

        if deriv > 3:
            # Fourth derivatives: (mmmm, mmmt, mmtt, mttt, tttt)  →  packed (0,1,2,3,4)
            # mgcv lines 987-992
            l4_val = np.column_stack(
                [
                    np.zeros(n, dtype=np.float64),  # d4l/dmu4 = 0
                    np.zeros(n, dtype=np.float64),  # = 0
                    np.full(n, -2.0, dtype=np.float64),  # d4l/dmu2 dtau2
                    np.zeros(n, dtype=np.float64),  # = 0
                    -6.0 / tau2**2,  # d4l/dtau4
                ]
            )
            g4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(tau),
                ]
            )

        i2 = self.tri["i2"]
        i3 = self.tri["i3"]
        i4 = self.tri["i4"]

        # Transform mu-derivatives to eta-derivatives  (mgcv: gamlss.etamu)
        de = gamlss_etamu(
            l1, l2, l3_val, l4_val, ig1, g2, g3, g4, i2, i3, i4, deriv - 1
        )

        # Gradient and Hessian w.r.t. coefficients  (mgcv: gamlss.gH)
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
        ret["l0"] = l0
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
        Initialize coefficients for gaulss by two penalized LS fits.

        First regress y on the mean design block.
        Then regress log|residuals| on the precision design block.

        Mirrors mgcv ``gaulss$initialize`` expression (regular matrix path).
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        start = np.zeros(p, dtype=np.float64)

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

        # --- Fit mean predictor ---
        X1 = X[:, jj[0]]
        yt1 = y.copy()
        if self.linfo[0].name != "identity":
            yt1 = self.linfo[0].linkfun(np.abs(y) + np.max(np.abs(y)) * 1e-7)
        if off1 is not None:
            yt1 = yt1 - off1

        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            XE1 = np.vstack([X1, E1])
            y1e = np.concatenate([yt1, np.zeros(E1.shape[0])])
        else:
            XE1 = X1
            y1e = yt1

        try:
            start1 = np.linalg.lstsq(XE1, y1e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # --- Fit precision predictor ---
        mu_init = self.linfo[0].linkinv(
            X1 @ start1 + (off1 if off1 is not None else 0.0)
        )
        lres1 = np.log(np.maximum(np.abs(y - mu_init), 1e-300))
        if off2 is not None:
            lres1 = lres1 - off2

        X2 = X[:, jj[1]]
        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            XE2 = np.vstack([X2, E2])
            y2e = np.concatenate([lres1, np.zeros(E2.shape[0])])
        else:
            XE2 = X2
            y2e = lres1

        try:
            start2 = np.linalg.lstsq(XE2, y2e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        return start

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Standardized residuals (y - mu) / sigma = (y - mu) * tau.

        Mirrors mgcv ``gaulss$residuals``.
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        tau = np.asarray(fitted[:, 1], dtype=np.float64)
        rsd = y - mu
        if rtype == "response":
            return rsd
        return rsd * tau

    def predict_fitted(
        self, X: np.ndarray, jj: list[np.ndarray], coef: np.ndarray, offset: Any = None
    ) -> np.ndarray:
        """Return (n, 2) fitted values: column 0 = mu, column 1 = tau."""
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        off1 = off2 = None
        if offset is not None:
            if isinstance(offset, (list, tuple)):
                off1 = offset[0]
                off2 = offset[1]
            else:
                off1 = offset

        eta = X[:, jj[0]] @ coef[jj[0]]
        if off1 is not None:
            eta = eta + np.asarray(off1, dtype=np.float64)
        eta1 = X[:, jj[1]] @ coef[jj[1]]
        if off2 is not None:
            eta1 = eta1 + np.asarray(off2, dtype=np.float64)

        mu = self.linfo[0].linkinv(eta)
        tau = self.linfo[1].linkinv(eta1)
        return np.column_stack([mu, tau])


def gaulss(link=("identity", "logb"), b: float = 0.01) -> GaulssFamily:
    """
    Gaussian location-scale family.

    Parameters
    ----------
    link : tuple of str
        Links for (mean, precision).  Mean link: one of
        ``"identity"`` (default), ``"log"``, ``"inverse"``.
        Precision link: ``"logb"`` (only option).
    b : float
        Small positive offset in logb link: tau = 1/(exp(eta) + b).
        Mirrors mgcv ``gaulss(b=0.01)``.

    Returns
    -------
    GaulssFamily instance.
    """
    return GaulssFamily(link=link, b=b)


# ---------------------------------------------------------------------------
# Adapter for existing LinkFunction objects
# ---------------------------------------------------------------------------


class _AdaptedLinkInfo:
    """Wraps a _function_maps.LinkFunction into the linfo interface."""

    def __init__(self, lobj: Any, name: str):
        self._lobj = lobj
        self.name = name

    def linkfun(self, mu):
        return self._lobj(mu)

    def linkinv(self, eta):
        return self._lobj.inverse(eta)

    def mu_eta(self, eta):
        return self._lobj.mu_eta(eta)

    def d2link(self, mu):
        return self._lobj.d2(mu)

    def d3link(self, mu):
        return self._lobj.d3(mu)

    def d4link(self, mu):
        return self._lobj.d4(mu)


# ---------------------------------------------------------------------------
# Link: softplus-b  (used by gammals for log-sigma predictor)
# ---------------------------------------------------------------------------


class _SoftplusBLinkInfo:
    """
    Softplus-b link for gammals log-sigma parameter:
        linkinv(eta) = b + log(1 + exp(eta - b))   [softplus shifted by b]
        linkfun(mu)  = b + log(exp(mu - b) - 1)    [inverse softplus shifted]

    The lower bound b (default -7) ensures log(sigma) > b.
    Numerically stable via sign-split.

    Mirrors mgcv ``gammals(link=list("identity","log"))`` stats[[2]] construction.
    """

    name = "softplusb"

    def __init__(self, b: float = -7.0):
        self.b = float(b)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        x = eta - self.b
        # for x > 500 the log1p(exp(x)) ≈ x, so linkinv ≈ eta
        return np.where(x > 500.0, eta, self.b + np.log1p(np.exp(np.minimum(x, 500.0))))

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        mub = mu - self.b
        # log(exp(mub) - 1) + b, with edge clamps
        eps = np.finfo(np.float64).eps
        eta = np.where(
            mub < eps,
            np.log(eps) + self.b,
            np.where(
                mub > -np.log(eps),
                mub + self.b,
                np.log(np.expm1(np.clip(mub, eps, 500.0))) + self.b,
            ),
        )
        return eta

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        """d linkinv / d eta = sigmoid(eta - b)."""
        eta = np.asarray(eta, dtype=np.float64)
        x = eta - self.b
        # numerically stable sigmoid
        pos = x >= 0.0
        ex = np.exp(-np.abs(x))
        result = np.where(pos, 1.0 / (1.0 + ex), ex / (1.0 + ex))
        return result

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        """d^2 eta / d mu^2.  Mirrors mgcv d2link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub = mu - self.b
        mub_v = np.exp(-np.abs(mub) * np.sign(mub))  # exp(-|mu-b|*sign) = exp(-|mu-b|)
        return -mub_v / (mub_v - 1.0) ** 2

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        """d^3 eta / d mu^3.  Mirrors mgcv d3link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub_raw = mu - self.b
        sm = -np.sign(mub_raw)
        mub_v = np.exp(mub_raw * sm)  # = exp(-|mu-b|)
        return sm * (mub_v + mub_v**2) / (mub_v - 1.0) ** 3

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        """d^4 eta / d mu^4.  Mirrors mgcv d4link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub_raw = mu - self.b
        sm = -np.sign(mub_raw)
        mub_v = np.exp(mub_raw * sm)
        return sm * (mub_v + 4.0 * mub_v**2 + mub_v**3) / (mub_v - 1.0) ** 4


# ---------------------------------------------------------------------------
# gammals: Gamma location-scale  (mgcv: gamlss.r::gammals)
# ---------------------------------------------------------------------------


class GammalsFamily(GamlssFamily):
    """
    Gamma location-scale family with two linear predictors:
      1. log mean (identity link on log-mean = mu = log(E[Y]))
      2. log sigma (softplus-b link, lower-bounded at b=-7)

    Parameterisation: Y ~ Gamma(shape=1/sigma, scale=mean*sigma).

    Mirrors mgcv ``gammals(link=list("identity","log"), b=-7)``.
    """

    name = "gammals"
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

    def __init__(self, link=("identity", "log"), b: float = -7.0):
        super().__init__()
        self.b = float(b)

        ok1 = ("identity",)
        ok2 = ("identity", "log")

        link1_name = link[0] if isinstance(link[0], str) else "identity"
        link2_name = link[1] if isinstance(link[1], str) else "log"

        if link1_name not in ok1:
            raise ValueError(f"Link {link1_name!r} not available for mu of gammals.")
        if link2_name not in ok2:
            raise ValueError(f"Link {link2_name!r} not available for sigma of gammals.")

        linfo1 = _IdentityLinkInfo()
        if link2_name == "log":
            linfo2 = _SoftplusBLinkInfo(b=b)
        else:
            linfo2 = _IdentityLinkInfo()

        self.linfo = [linfo1, linfo2]
        self.tri = trind_generator(2)
        self.link_names = (link1_name, link2_name)
        self.link_name = f"({link1_name}, {link2_name})"

    def validate_y(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf")
        if np.any(y <= 0.0):
            raise ValueError("gammals requires strictly positive response y > 0.")
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
        Log-likelihood and derivatives for the gamma location-scale model.

        Parameterisation:
          mu  = eta1        (log of mean, via identity link)
          th  = linkinv(eta2)   (log of sigma)
          eth = exp(-th)    (shape = 1/sigma)
          Y ~ Gamma(shape=eth, scale=exp(mu+th))

        Mirrors mgcv ``gammals$ll``.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        n = len(y)
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

        # Linear predictors
        eta = X[:, jj[0]] @ coef[jj[0]]
        if off1 is not None:
            eta = eta + off1
        etat = X[:, jj[1]] @ coef[jj[1]]
        if off2 is not None:
            etat = etat + off2

        mu = self.linfo[0].linkinv(eta)  # log(mean)
        th = self.linfo[1].linkinv(etat)  # log(sigma)

        eth = np.exp(-th)  # shape = 1/sigma
        logy = np.log(np.maximum(y, 1e-300))
        ethmu = np.exp(-th - mu)  # eth * exp(-mu) = rate
        ethmuy = ethmu * y  # rate * y
        etlymt = eth * (logy - mu - th)  # eth*(log(y) - log(mean) - log(sigma))

        # log-lik per obs
        l0 = etlymt - logy - ethmuy - gammaln(eth)
        if not np.isfinite(np.sum(l0)):
            return {"l": float(np.sum(l0)), "l0": l0}
        l = float(np.sum(l0))

        if deriv == 0:
            return {"l": l, "l0": l0}

        # First derivatives w.r.t. mu and th
        l1 = np.empty((n, 2), dtype=np.float64)
        l1[:, 0] = ethmuy - eth  # dl/d(mu)
        digeth = digamma(eth)
        l1[:, 1] = -etlymt + ethmuy + eth * digeth - eth  # dl/d(th)

        # Second derivatives (packed: mm, mt, tt)
        l2 = np.empty((n, 3), dtype=np.float64)
        l2[:, 0] = -ethmuy  # lmm
        l2[:, 1] = eth - ethmuy  # lmt
        eth2 = eth**2
        treth = polygamma(1, eth)  # trigamma
        l2[:, 2] = etlymt - ethmuy - treth * eth2 - eth * digeth + 2.0 * eth  # ltt

        # Link derivatives for chain rule
        ig1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),
                self.linfo[1].mu_eta(etat),
            ]
        )
        g2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(th),
            ]
        )

        l3_val: Any = 0
        l4_val: Any = 0
        g3: Any = 0
        g4: Any = 0

        if deriv > 1:
            # Third derivatives (packed: mmm, mmt, mtt, ttt)
            l3_val = np.empty((n, 4), dtype=np.float64)
            l3_val[:, 0] = ethmuy  # lmmm
            l3_val[:, 1] = ethmuy  # lmmt
            l3_val[:, 2] = ethmuy - eth  # lmtt
            eth3 = eth2 * eth
            g3eth = polygamma(2, eth)  # tetragamma
            l3_val[:, 3] = (
                -etlymt
                + ethmuy
                + g3eth * eth3
                + 3.0 * treth * eth2
                + eth * digeth
                - 3.0 * eth
            )  # lttt
            g3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(th),
                ]
            )

        if deriv > 3:
            # Fourth derivatives (packed: mmmm, mmmt, mmtt, mttt, tttt)
            l4_val = np.empty((n, 5), dtype=np.float64)
            l4_val[:, 0] = -ethmuy  # lmmmm
            l4_val[:, 1] = -ethmuy  # lmmmt
            l4_val[:, 2] = -ethmuy  # lmmtt
            l4_val[:, 3] = eth - ethmuy  # lmttt
            eth4 = eth3 * eth
            l4_val[:, 4] = (
                etlymt
                - ethmuy
                - polygamma(3, eth) * eth4
                - 6.0 * g3eth * eth3
                - 7.0 * treth * eth2
                - eth * digeth
                + 4.0 * eth
            )  # ltttt
            g4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(th),
                ]
            )

        i2 = self.tri["i2"]
        i3 = self.tri["i3"]
        i4 = self.tri["i4"]

        de = gamlss_etamu(
            l1, l2, l3_val, l4_val, ig1, g2, g3, g4, i2, i3, i4, deriv - 1
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
        ret["l0"] = l0
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
        Initialize coefficients for gammals.

        Regresses X1 on log(y) for the log-mean predictor.
        Then regresses X2 on log|residuals| transformed through link2
        for the log-sigma predictor.

        Mirrors mgcv ``gammals$initialize`` regular matrix path.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        start = np.zeros(p, dtype=np.float64)

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

        eps = np.max(y) * np.finfo(np.float64).eps ** 0.75

        # --- Fit log-mean predictor on log(y) ---
        yt1 = np.log(np.maximum(y + eps, 1e-300))
        if off1 is not None:
            yt1 = yt1 - off1

        X1 = X[:, jj[0]]
        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            XE1 = np.vstack([X1, E1])
            y1e = np.concatenate([yt1, np.zeros(E1.shape[0])])
        else:
            XE1 = X1
            y1e = yt1

        try:
            start1 = np.linalg.lstsq(XE1, y1e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # --- Fit log-sigma predictor on transformed residuals ---
        mu_init = self.linfo[0].linkinv(
            X1 @ start1 + (off1 if off1 is not None else 0.0)
        )
        # residuals from fitted log-mean: y/exp(mu_init) - 1
        res = np.log(np.maximum(np.abs(y - np.exp(mu_init)), 1e-300))
        lres1 = self.linfo[1].linkfun(res)
        if off2 is not None:
            lres1 = lres1 - off2

        X2 = X[:, jj[1]]
        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            XE2 = np.vstack([X2, E2])
            y2e = np.concatenate([lres1, np.zeros(E2.shape[0])])
        else:
            XE2 = X2
            y2e = lres1

        try:
            start2 = np.linalg.lstsq(XE2, y2e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        return start

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Residuals for gammals.  Mirrors mgcv ``gammals$residuals``."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.exp(np.asarray(fitted[:, 0], dtype=np.float64))  # actual mean
        rho = np.asarray(fitted[:, 1], dtype=np.float64)  # log sigma
        if rtype == "deviance":
            rsd = 2.0 * ((y - mu) / mu - np.log(y / mu)) * np.exp(-rho)
            return np.sqrt(np.maximum(0.0, rsd)) * np.sign(y - mu)
        elif rtype == "pearson":
            return (y - mu) / (np.exp(rho * 0.5) * mu)
        else:
            return y - mu

    def _predict_response_from_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        out = np.empty((eta.shape[0], 2), dtype=np.float64)
        out[:, 0] = np.exp(eta[:, 0])
        out[:, 1] = np.asarray(self.linfo[1].linkinv(eta[:, 1]), dtype=np.float64)
        return out


def gammals(link=("identity", "log"), b: float = -7.0) -> GammalsFamily:
    """
    Gamma location-scale family.

    Parameters
    ----------
    link : tuple of str
        Links for (log-mean, log-sigma).  Log-mean: ``"identity"`` only.
        Log-sigma: ``"identity"`` or ``"log"`` (default, softplus-b).
    b : float
        Lower bound for log-sigma predictor when ``link[1]="log"``.
        Mirrors mgcv ``gammals(b=-7)``.

    Returns
    -------
    GammalsFamily instance.
    """
    return GammalsFamily(link=link, b=b)


# ---------------------------------------------------------------------------
# ZIP helpers  (mirrors mgcv/R/gamlss.r: l1ee, lee1, ldg, lde, zipll)
# ---------------------------------------------------------------------------


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
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Response or deviance residuals.  Mirrors mgcv ``ziplss$residuals``."""
        y = np.asarray(y, dtype=np.float64)
        lam_pred = np.asarray(fitted[:, 0], dtype=np.float64)  # log lambda
        p_pred = np.asarray(fitted[:, 1], dtype=np.float64)  # loglog p

        lam = np.exp(lam_pred)
        p = 1.0 - np.exp(-np.exp(p_pred))  # prob of presence

        small_lam = lam <= np.sqrt(np.finfo(np.float64).eps)
        Ey = p.copy()
        Ey[~small_lam] = (
            p[~small_lam] * lam[~small_lam] / (1.0 - np.exp(-lam[~small_lam]))
        )

        rsd = y - Ey
        if rtype == "response":
            return rsd
        # deviance residuals omitted (requires saturated log-lik computation)
        return rsd

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


class _ShiftedLogitLinkInfo:
    """
    Shifted logit link confining xi to (-1, 0.5):
        linkinv(eta) = 1.5 * sigmoid(eta) - 1
        linkfun(xi)  = logit((xi + 1) / 1.5)
        mu_eta(eta)  = 1.5 * sigmoid(eta) * (1 - sigmoid(eta))

    Mirrors mgcv gevlss stats[[3]] with link="logit".
    """

    name = "shifted_logit"

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return 1.5 * self._sigmoid(np.asarray(eta, dtype=np.float64)) - 1.0

    def linkfun(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        p = np.clip((xi + 1.0) / 1.5, 1e-15, 1.0 - 1e-15)
        return np.log(p / (1.0 - p))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        s = self._sigmoid(np.asarray(eta, dtype=np.float64))
        return 1.5 * s * (1.0 - s)

    def d2link(self, xi: np.ndarray) -> np.ndarray:
        """d^2 eta / d xi^2.  Mirrors mgcv d2link for shifted logit."""
        xi = np.asarray(xi, dtype=np.float64)
        mu = np.clip((xi + 1.0) / 1.5, 1e-15, 1.0 - 1e-15)
        return (1.0 / (1.0 - mu) ** 2 - 1.0 / mu**2) / 1.5**2

    def d3link(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        mu = np.clip((xi + 1.0) / 1.5, 1e-15, 1.0 - 1e-15)
        return (2.0 / (1.0 - mu) ** 3 + 2.0 / mu**3) / 1.5**3

    def d4link(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        mu = np.clip((xi + 1.0) / 1.5, 1e-15, 1.0 - 1e-15)
        return (6.0 / (1.0 - mu) ** 4 - 6.0 / mu**4) / 1.5**4


# ---------------------------------------------------------------------------
# gevlss: GEV location-scale-shape  (mgcv: gamlss.r::gevlss)
# ---------------------------------------------------------------------------


class GevlssFamily(GamlssFamily):
    """
    GEV location-scale-shape family with three linear predictors:
      1. location μ (identity or log link)
      2. log scale ρ = log(σ) (identity link)
      3. shape ξ ∈ (-1, 0.5) (shifted-logit link)

    GEV distribution: F(y) = exp(-(1 + ξ*(y-μ)/σ)^(-1/ξ))

    Mirrors mgcv ``gevlss(link=list("identity","identity","logit"))``.

    Note: family now exposes analytic outer gradient and Hessian terms through
    ``gam.fit5``.
    """

    name = "gevlss"
    family_class = "general"
    nlp = 3
    n_linear_predictors = 3

    supports_laml = True
    supports_ml = True
    supports_reml = True
    supports_gcv = False
    supports_ubre = False
    supports_pirls = False
    supports_closed_form_solve = False
    supports_exact_pirls_first_derivatives = False
    supports_exact_pirls_second_derivatives = False
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, link=("identity", "identity", "logit")):
        super().__init__()

        ok1 = ("identity", "log")
        ok2 = ("identity",)
        ok3 = ("identity", "logit")

        link1_name = link[0] if isinstance(link[0], str) else "identity"
        link2_name = link[1] if isinstance(link[1], str) else "identity"
        link3_name = link[2] if isinstance(link[2], str) else "logit"

        if link1_name not in ok1:
            raise ValueError(f"Link {link1_name!r} not available for mu of gevlss.")
        if link2_name not in ok2:
            raise ValueError(
                f"Link {link2_name!r} not available for log-sigma of gevlss."
            )
        if link3_name not in ok3:
            raise ValueError(f"Link {link3_name!r} not available for xi of gevlss.")

        if link1_name == "identity":
            linfo1 = _IdentityLinkInfo()
        else:
            from ._function_maps import LogLink

            linfo1 = _AdaptedLinkInfo(LogLink(eps=FAMILY_EPS), link1_name)

        linfo2 = _IdentityLinkInfo()

        if link3_name == "logit":
            linfo3 = _ShiftedLogitLinkInfo()
        else:
            linfo3 = _IdentityLinkInfo()

        self.linfo = [linfo1, linfo2, linfo3]
        self.tri = trind_generator(3)
        self.link_names = (link1_name, link2_name, link3_name)
        self.link_name = f"({link1_name}, {link2_name}, {link3_name})"

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
        Log-likelihood and derivatives for the GEV model.

        Mirrors mgcv ``gevlss$ll`` (auto-generated derivative expressions).
        Third/fourth derivative orders (deriv>1) return zeros.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        n = len(y)
        sandwich = bool(kw.get("sandwich", False))

        off1 = off2 = off3 = None
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
                off3 = (
                    np.asarray(offset[2], dtype=np.float64)
                    if len(offset) > 2 and offset[2] is not None
                    else None
                )
            else:
                off1 = np.asarray(offset, dtype=np.float64)

        eta = X[:, jj[0]] @ coef[jj[0]]
        if off1 is not None:
            eta = eta + off1
        etar = X[:, jj[1]] @ coef[jj[1]]
        if off2 is not None:
            etar = etar + off2
        etax = X[:, jj[2]] @ coef[jj[2]]
        if off3 is not None:
            etax = etax + off3

        mu = self.linfo[0].linkinv(eta)  # location
        rho = self.linfo[1].linkinv(etar)  # log scale
        xi = self.linfo[2].linkinv(etax)  # shape

        # Clamp xi away from zero (avoids xi=0 branch)
        eps_xi = 1e-7
        xi = np.where((xi >= 0.0) & (xi < eps_xi), eps_xi, xi)
        xi = np.where((xi < 0.0) & (xi > -eps_xi), -eps_xi, xi)

        ymu = y - mu
        sigma_inv = np.exp(-rho)  # = bb1 = 1/sigma
        aa0 = xi * ymu * sigma_inv  # = xi*(y-mu)/exp(rho)
        log_aa1 = np.log1p(aa0)
        aa1 = 1.0 + aa0  # = cc3 in R
        aa2 = 1.0 / xi  # = 1/xi

        # Check support: need aa1 > 0
        valid = aa1 > 0.0
        if not np.all(valid):
            # Return -inf for out-of-support observations
            l0 = np.where(valid, 0.0, -np.inf)
            l0[valid] = (
                -(aa2[valid] * (1.0 + xi[valid]) * log_aa1[valid])
                - aa1[valid] ** (-aa2[valid])
                - rho[valid]
            )
            return {"l": float(np.sum(l0)), "l0": l0}

        l0 = -(aa2 * (1.0 + xi) * log_aa1) - aa1 ** (-aa2) - rho
        l = float(np.sum(l0))

        if not np.isfinite(l):
            return {"l": l, "l0": l0}

        if deriv == 0:
            return {"l": l, "l0": l0}

        # ---- First derivatives: dm, dr, dx ---
        # Precompute reused quantities (mirroring mgcv variable names)
        bb1 = sigma_inv
        bb2 = aa1  # bb1*xi*ymu+1 = aa1
        cc2 = ymu
        cc0 = bb1 * xi * cc2  # = aa0
        log_cc3 = log_aa1
        cc3 = aa1
        dd3 = xi + 1.0
        dd6 = 1.0 / cc3
        dd7 = log_cc3
        dd8 = 1.0 / xi**2

        l1 = np.empty((n, 3), dtype=np.float64)
        # dl/dmu  (chain: dm wrt eta = 1 for identity)
        l1[:, 0] = (bb1 * dd3) / bb2 - bb1 * bb2 ** ((-1.0 / xi) - 1.0)
        # dl/drho
        l1[:, 1] = (
            (-bb1 * cc2 * cc3 ** ((-1.0 / xi) - 1.0)) + (bb1 * dd3 * cc2) / cc3 - 1.0
        )
        # dl/dxi
        aa2_v = aa2  # = 1/xi
        l1[:, 2] = (
            -(dd8 * dd7 - bb1 * aa2_v * cc2 * dd6) / cc3**aa2_v
            + dd8 * dd3 * dd7
            - aa2_v * dd7
            - bb1 * aa2_v * dd3 * cc2 * dd6
        )

        # ---- Second derivatives (6 packed: mm, mr, mx, rr, rx, xx) ----
        ee1 = np.exp(-2.0 * rho)
        ee3 = -aa2  # = -1/xi
        ff7 = ee3 - 1.0
        gg7 = -aa2
        hh4 = cc2**2
        jj08 = 1.0 / cc3**2
        jj12 = 1.0 / xi**3
        jj13 = 1.0 / cc3**aa2

        l2 = np.empty((n, 6), dtype=np.float64)
        l2[:, 0] = (
            ee1 * (ee3 - 1.0) * xi * aa1 ** (ee3 - 2.0) + (ee1 * xi * dd3) / aa1**2
        )
        l2[:, 1] = (
            bb1 * cc3**ff7
            + ee1 * ff7 * xi * cc2 * cc3 ** (ee3 - 2.0)
            - (bb1 * dd3) / cc3
            + (ee1 * xi * dd3 * cc2) / cc3**2
        )
        l2[:, 2] = (
            -bb1 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
            + ee1 * cc2 * cc3 ** (gg7 - 2.0)
            + bb1 * dd6
            - (ee1 * dd3 * cc2) / cc3**2
        )
        l2[:, 3] = (
            bb1 * cc2 * cc3**ff7
            + ee1 * ff7 * xi * hh4 * cc3 ** (ee3 - 2.0)
            - (bb1 * dd3 * cc2) / cc3
            + (ee1 * xi * dd3 * hh4) / cc3**2
        )
        l2[:, 4] = (
            -bb1 * cc2 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
            + ee1 * hh4 * cc3 ** (gg7 - 2.0)
            + bb1 * cc2 * dd6
            - (ee1 * dd3 * hh4) / cc3**2
        )
        l2[:, 5] = (
            -jj13 * (dd8 * dd7 - bb1 * aa2 * cc2 * dd6) ** 2
            - jj13
            * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
            - 2.0 * jj12 * dd3 * dd7
            + 2.0 * dd8 * dd7
            + 2.0 * bb1 * dd8 * dd3 * cc2 * dd6
            - 2.0 * bb1 * aa2 * cc2 * dd6
            + ee1 * aa2 * dd3 * hh4 * jj08
        )

        # Link derivatives for chain rule
        ig1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),
                self.linfo[1].mu_eta(etar),
                self.linfo[2].mu_eta(etax),
            ]
        )
        g2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(rho),
                self.linfo[2].d2link(xi),
            ]
        )

        l3_val: Any = 0
        l4_val: Any = 0
        g3: Any = 0
        g4: Any = 0

        if deriv > 1:
            # Third derivatives: 10 components, order mmm mmr mmx mrr mrx mxx rrr rrx rxx xxx
            # Mirrors mgcv gamlss.r gevlss$ll deriv>1 block (auto-generated expressions).
            kk1 = np.exp(-3.0 * rho)  # 1/exp(3*rho)
            kk2 = xi**2
            ll8 = ee3 - 2.0  # -1/xi - 2
            mm11 = gg7 - 2.0  # -1/xi - 2  (same as ll8)
            mm12 = cc3**mm11  # cc3^(-1/xi - 2)
            mm10 = cc3 ** (gg7 - 3.0)  # cc3^(-1/xi - 3)
            oo10 = ff7  # gg7 - 1 = -1/xi - 1
            oo13 = log_cc3 / xi**2  # log(cc3)/xi^2
            pp08 = cc3**ff7  # cc3^(-1/xi - 1)
            qq05 = cc2**3
            rr17 = log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6
            tt08 = 1.0 / cc3**3
            tt16 = 1.0 / xi**4
            tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6

            l3_val = np.empty((n, 10), dtype=np.float64)
            # mmm
            l3_val[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (
                ee3 - 1.0
            ) * kk2 * cc3 ** (ee3 - 3.0)
            # mmr
            l3_val[:, 1] = (
                -2.0 * ee1 * ff7 * xi * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0)
                - (2.0 * ee1 * xi * dd3) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
            )
            # mmx
            l3_val[:, 2] = (
                ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
                - ee1 * mm12
                - kk1 * mm11 * xi * cc2 * mm10
                + kk1 * cc2 * mm10
                + ee1 * dd3 * jj08
                + ee1 * xi * jj08
                - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
            )
            # mrr
            l3_val[:, 3] = (
                -bb1 * cc3**ff7
                - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0)
                + (bb1 * dd3) / cc3
                - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
            )
            # mrx
            l3_val[:, 4] = (
                bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13)
                + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                + ee1 * aa2 * cc2 * mm12
                + ee1 * oo10 * cc2 * mm12
                - bb1 * dd6
                + 2.0 * ee1 * dd3 * cc2 * jj08
                + ee1 * xi * cc2 * jj08
                - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
            )
            # mxx
            l3_val[:, 5] = (
                -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
                - bb1
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 / xi**3
                )
                - 2.0 * ee1 * cc2 * jj08
                + 2.0 * dd3 * hh4 * kk1 / cc3**3
            )
            # rrr
            l3_val[:, 6] = (
                -bb1 * cc2 * cc3**ff7
                - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0)
                + (bb1 * dd3 * cc2) / cc3
                - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
            )
            # rrx
            l3_val[:, 7] = (
                bb1 * cc2 * cc3**oo10 * rr17
                + ee1 * oo10 * xi * hh4 * mm12 * rr17
                - 2.0 * ee1 * hh4 * mm12
                - kk1 * mm11 * xi * qq05 * mm10
                + kk1 * qq05 * mm10
                - bb1 * cc2 * dd6
                + 2.0 * ee1 * dd3 * hh4 * jj08
                + ee1 * xi * hh4 * jj08
                - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
            )
            # rxx
            l3_val[:, 8] = (
                -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
                - bb1
                * cc2
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 / xi**3
                )
                - 2.0 * ee1 * hh4 * jj08
                + 2.0 * dd3 * qq05 * kk1 / cc3**3
            )
            # xxx
            l3_val[:, 9] = (
                -jj13 * tt18**3
                - 3.0
                * jj13
                * (
                    ee1 * aa2 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * tt18
                - jj13
                * (
                    -2.0 * kk1 * aa2 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * tt16 * dd7
                )
                + 6.0 * tt16 * dd3 * dd7
                - 6.0 * jj12 * dd7
                - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6
                + 6.0 * bb1 * dd8 * cc2 * dd6
                - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08
                + 3.0 * ee1 * aa2 * hh4 * jj08
                - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08
            )
            g3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(rho),
                    self.linfo[2].d3link(xi),
                ]
            )

        if deriv > 3:
            # Fourth derivatives: 15 components, order mmmm mmmr mmmx mmrr mmrx mmxx
            #                                           mrrr mrrx mrxx mxxx rrrr rrrx
            #                                           rrxx rxxx xxxx
            # Mirrors mgcv gamlss.r gevlss$ll deriv>3 block.
            # All l3 temporaries (kk1,kk2,ll8,mm11,mm12,qq05,pp08,rr17,tt08,tt16,tt18,oo13,ff7)
            # are in scope from the deriv>1 block above.
            uu1 = np.exp(-4.0 * rho)  # 1/exp(4*rho)
            uu2 = xi**3
            vv09 = ee3 - 3.0  # -1/xi - 3
            ww11 = gg7 - 3.0  # -1/xi - 3  (= vv09)
            ww12 = cc3 ** (gg7 - 4.0)  # cc3^(-1/xi - 4)
            ww15 = cc3**ww11  # cc3^(-1/xi - 3)
            # intermediate scalars reused across several l4 expressions
            ad17 = 2.0 * bb1 * dd8 * cc2 * dd6
            ad19 = -2.0 * jj12 * dd7  # -(2*dd7)/xi^3
            ad20 = pp08  # cc3^ff7
            ad21 = dd8 * dd7
            ad22 = ad21 + bb1 * mm11 * cc2 * dd6
            ae16 = dd8 * dd7 + bb1 * ff7 * cc2 * dd6
            af05 = cc2**4
            ah24 = ad19 + ad17 + ee1 * aa2 * hh4 * jj08
            aj08 = 1.0 / cc3**4
            aj20 = 1.0 / xi**5

            l4_val = np.empty((n, 15), dtype=np.float64)
            # mmmm
            l4_val[:, 0] = (
                uu1 * (ee3 - 3.0) * (ee3 - 2.0) * (ee3 - 1.0) * uu2 * cc3 ** (ee3 - 4.0)
                + (6.0 * uu1 * uu2 * dd3) / cc3**4
            )
            # mmmr
            l4_val[:, 1] = (
                3.0 * kk1 * ll8 * ff7 * kk2 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * cc2 * cc3 ** (ee3 - 4.0)
                - (6.0 * kk1 * kk2 * dd3) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * cc2) / cc3**4
            )
            # mmmx
            l4_val[:, 2] = (
                -kk1 * mm11 * ff7 * kk2 * ww15 * rr17
                + 2.0 * kk1 * mm11 * xi * ww15
                - kk1 * ww15
                + uu1 * ww11 * mm11 * kk2 * cc2 * ww12
                - uu1 * ff7 * xi * cc2 * ww12
                - uu1 * ww11 * xi * cc2 * ww12
                + 2.0 * kk1 * kk2 * tt08
                + 4.0 * kk1 * xi * dd3 * tt08
                - (6.0 * uu1 * kk2 * dd3 * cc2) / cc3**4
            )
            # mmrr
            l4_val[:, 3] = (
                4.0 * ee1 * ff7 * xi * cc3**ll8
                + 5.0 * kk1 * ll8 * ff7 * kk2 * cc2 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * hh4 * cc3 ** (ee3 - 4.0)
                + (4.0 * ee1 * xi * dd3) / cc3**2
                - (10.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * hh4) / cc3**4
            )
            # mmrx
            l4_val[:, 4] = (
                -2.0 * ee1 * ff7 * xi * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                - kk1 * mm11 * ff7 * kk2 * cc2 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
                - 2.0 * ee1 * aa2 * mm12
                - 2.0 * ee1 * ff7 * mm12
                - 2.0 * kk1 * mm11 * ff7 * xi * cc2 * ww15
                - kk1 * ff7 * cc2 * ww15
                - kk1 * mm11 * cc2 * ww15
                - 2.0 * ee1 * dd3 * jj08
                - 2.0 * ee1 * xi * jj08
                + 2.0 * kk1 * kk2 * cc2 * tt08
                + 8.0 * kk1 * xi * dd3 * cc2 * tt08
                - 6.0 * kk2 * dd3 * hh4 * uu1 / cc3**4
            )
            # mmxx
            l4_val[:, 5] = (
                ee1 * ff7 * xi * mm12 * tt18**2
                - 2.0 * ee1 * mm12 * tt18
                - 2.0 * kk1 * mm11 * xi * cc2 * ww15 * tt18
                + 2.0 * kk1 * cc2 * ww15 * tt18
                + ee1
                * ff7
                * xi
                * mm12
                * (
                    ee1 * aa2 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 * jj12
                )
                + 4.0 * kk1 * cc2 * ww15
                + 2.0 * uu1 * ww11 * xi * hh4 * ww12
                - 4.0 * uu1 * hh4 * ww12
                + 2.0 * ee1 * jj08
                - 4.0 * kk1 * dd3 * cc2 * tt08
                - 4.0 * kk1 * xi * cc2 * tt08
                + (6.0 * uu1 * xi * dd3 * hh4) / cc3**4
            )
            # mrrr
            l4_val[:, 6] = (
                bb1 * cc3**ff7
                + 7.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
                + 6.0 * kk1 * ll8 * ff7 * kk2 * hh4 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * qq05 * cc3 ** (ee3 - 4.0)
                - (bb1 * dd3) / cc3
                + (7.0 * ee1 * xi * dd3 * cc2) / cc3**2
                - (12.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * qq05) / cc3**4
            )
            # mrrx
            l4_val[:, 7] = (
                -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + oo13)
                - 3.0 * ee1 * ff7 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                - kk1 * mm11 * ff7 * kk2 * hh4 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
                - 3.0 * ee1 * aa2 * cc2 * mm12
                - 3.0 * ee1 * ff7 * cc2 * mm12
                - 2.0 * kk1 * mm11 * ff7 * xi * hh4 * ww15
                - kk1 * ff7 * hh4 * ww15
                - kk1 * mm11 * hh4 * ww15
                + bb1 * dd6
                - 4.0 * ee1 * dd3 * cc2 * jj08
                - 3.0 * ee1 * xi * cc2 * jj08
                + 2.0 * kk1 * kk2 * hh4 * tt08
                + 10.0 * kk1 * xi * dd3 * hh4 * tt08
                - 6.0 * kk2 * dd3 * qq05 * uu1 / cc3**4
            )
            # mrxx
            l4_val[:, 8] = (
                bb1 * ad20 * (bb1 * ff7 * cc2 * dd6 + ad21) ** 2
                + ee1 * ff7 * xi * cc2 * mm12 * ad22**2
                + 2.0 * ee1 * aa2 * cc2 * mm12 * ad22
                + 2.0 * ee1 * ff7 * cc2 * mm12 * ad22
                + bb1 * ad20 * (-ee1 * ff7 * hh4 * jj08 + ad17 + ad19)
                + ee1 * ff7 * xi * cc2 * mm12 * (-ee1 * mm11 * hh4 * jj08 + ad17 + ad19)
                + 4.0 * ee1 * cc2 * jj08
                - 6.0 * kk1 * dd3 * hh4 * tt08
                - 4.0 * kk1 * xi * hh4 * tt08
                + 6.0 * xi * dd3 * qq05 * uu1 / cc3**4
            )
            # mxxx
            l4_val[:, 9] = (
                -bb1 * pp08 * ae16**3
                - 3.0
                * bb1
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * ae16
                - bb1
                * pp08
                * (
                    2.0 * kk1 * ff7 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * dd7 * tt16
                )
                + 6.0 * kk1 * hh4 * tt08
                - 6.0 * dd3 * qq05 * uu1 / cc3**4
            )
            # rrrr
            l4_val[:, 10] = (
                bb1 * cc2 * cc3**ff7
                + 7.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
                + 6.0 * kk1 * ll8 * ff7 * kk2 * qq05 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * af05 * cc3 ** (ee3 - 4.0)
                - (bb1 * dd3 * cc2) / cc3
                + (7.0 * ee1 * xi * dd3 * hh4) / cc3**2
                - (12.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * af05) / cc3**4
            )
            # rrrx
            l4_val[:, 11] = (
                -bb1 * cc2 * pp08 * rr17
                - 3.0 * ee1 * ff7 * xi * hh4 * mm12 * rr17
                - kk1 * mm11 * ff7 * kk2 * qq05 * ww15 * rr17
                + 4.0 * ee1 * hh4 * mm12
                + 5.0 * kk1 * mm11 * xi * qq05 * ww15
                - 4.0 * kk1 * qq05 * ww15
                + uu1 * ww11 * mm11 * kk2 * af05 * ww12
                - uu1 * ff7 * xi * af05 * ww12
                - uu1 * ww11 * xi * af05 * ww12
                + bb1 * cc2 * dd6
                - 4.0 * ee1 * dd3 * hh4 * jj08
                - 3.0 * ee1 * xi * hh4 * jj08
                + 2.0 * kk1 * kk2 * qq05 * tt08
                + 10.0 * kk1 * xi * dd3 * qq05 * tt08
                - 6.0 * uu1 * kk2 * dd3 * af05 / cc3**4
            )
            # rrxx
            l4_val[:, 12] = (
                bb1 * cc2 * ad20 * tt18**2
                + ee1 * ff7 * xi * hh4 * mm12 * tt18**2
                - 4.0 * ee1 * hh4 * mm12 * tt18
                - 2.0 * kk1 * mm11 * xi * qq05 * ww15 * tt18
                + 2.0 * kk1 * qq05 * ww15 * tt18
                + bb1 * cc2 * ad20 * ah24
                + ee1 * ff7 * xi * hh4 * mm12 * ah24
                + 6.0 * kk1 * qq05 * ww15
                + 2.0 * uu1 * ww11 * xi * af05 * ww12
                - 4.0 * uu1 * af05 * ww12
                + 4.0 * ee1 * hh4 * jj08
                - 6.0 * kk1 * dd3 * qq05 * tt08
                - 4.0 * kk1 * xi * qq05 * tt08
                + 6.0 * uu1 * xi * dd3 * af05 / cc3**4
            )
            # rxxx
            l4_val[:, 13] = (
                -bb1 * cc2 * pp08 * ae16**3
                - 3.0
                * bb1
                * cc2
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * ae16
                - bb1
                * cc2
                * pp08
                * (
                    2.0 * kk1 * ff7 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * dd7 * tt16
                )
                + 6.0 * kk1 * qq05 * tt08
                - 6.0 * dd3 * af05 * uu1 / cc3**4
            )
            # xxxx
            l4_val[:, 14] = (
                -jj13 * tt18**4
                - 6.0 * jj13 * ah24 * tt18**2
                - 3.0 * jj13 * ah24**2
                - 4.0
                * jj13
                * (
                    -2.0 * kk1 * aa2 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * tt16 * dd7
                )
                * tt18
                - jj13
                * (
                    6.0 * uu1 * aa2 * af05 * aj08
                    + 8.0 * kk1 * dd8 * qq05 * tt08
                    + 12.0 * ee1 * jj12 * hh4 * jj08
                    + 24.0 * bb1 * tt16 * cc2 * dd6
                    - 24.0 * aj20 * dd7
                )
                - 24.0 * aj20 * dd3 * dd7
                + 24.0 * tt16 * dd7
                + 24.0 * bb1 * tt16 * dd3 * cc2 * dd6
                - 24.0 * bb1 * jj12 * cc2 * dd6
                + 12.0 * ee1 * jj12 * dd3 * hh4 * jj08
                - 12.0 * ee1 * dd8 * hh4 * jj08
                + 8.0 * kk1 * dd8 * dd3 * qq05 * tt08
                - 8.0 * kk1 * aa2 * qq05 * tt08
                + 6.0 * uu1 * aa2 * dd3 * af05 * aj08
            )
            g4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(rho),
                    self.linfo[2].d4link(xi),
                ]
            )

        i2 = self.tri["i2"]
        i3 = self.tri["i3"]
        i4 = self.tri["i4"]

        de = gamlss_etamu(
            l1, l2, l3_val, l4_val, ig1, g2, g3, g4, i2, i3, i4, deriv - 1
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
        ret["l0"] = l0
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
        Initialize coefficients for gevlss.

        Regress X1 on g(y) for location, X2 on log|residuals| for log-scale,
        then initialize xi near 0 (xi=1e-3 in link scale).

        Mirrors mgcv ``gevlss$initialize`` non-discrete path.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        start = np.zeros(p, dtype=np.float64)

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

        # --- Fit location predictor ---
        if self.link_names[0] == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.linfo[0].linkfun(np.abs(y) + np.max(np.abs(y)) * 1e-7)
        if off1 is not None:
            yt1 = yt1 - off1

        X1 = X[:, jj[0]]
        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            XE1 = np.vstack([X1, E1])
            y1e = np.concatenate([yt1, np.zeros(E1.shape[0])])
        else:
            XE1 = X1
            y1e = yt1

        try:
            start1 = np.linalg.lstsq(XE1, y1e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # --- Fit log-scale predictor on log|residuals| ---
        mu_init = self.linfo[0].linkinv(
            X1 @ start1 + (off1 if off1 is not None else 0.0)
        )
        lres1 = np.log(np.maximum(np.abs(y - mu_init), 1e-300))
        if off2 is not None:
            lres1 = lres1 - off2

        X2 = X[:, jj[1]]
        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            XE2 = np.vstack([X2, E2])
            y2e = np.concatenate([lres1, np.zeros(E2.shape[0])])
        else:
            XE2 = X2
            y2e = lres1

        try:
            start2 = np.linalg.lstsq(XE2, y2e, rcond=None)[0]
        except np.linalg.LinAlgError:
            start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # --- Initialize xi near 0 in link space ---
        xi_init_val = 1e-3
        eta_xi0 = self.linfo[2].linkfun(np.full(1, xi_init_val))[0]
        X3 = X[:, jj[2]]
        yt3 = np.full(n, eta_xi0)
        try:
            start3 = np.linalg.lstsq(X3, yt3, rcond=None)[0]
        except np.linalg.LinAlgError:
            start3 = np.zeros(X3.shape[1], dtype=np.float64)
        start3 = np.where(np.isfinite(start3), start3, 0.0)
        start[jj[2]] = start3

        return start

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Residuals for gevlss.  Mirrors mgcv ``gevlss$residuals``."""
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        rho = np.asarray(fitted[:, 1], dtype=np.float64)
        xi = np.asarray(fitted[:, 2], dtype=np.float64)
        sigma = np.exp(rho)
        # GEV mean: mu + sigma*(Gamma(1-xi)-1)/xi for xi != 0
        from scipy.special import gamma as gamma_fn

        eps_xi = 1e-7
        xi_safe = np.where(np.abs(xi) < eps_xi, eps_xi, xi)
        fv = mu + sigma * (gamma_fn(1.0 - xi_safe) - 1.0) / xi_safe
        rsd = y - fv
        if rtype == "response":
            return rsd
        if rtype == "pearson":
            var = np.empty_like(rsd)
            near_zero = np.abs(xi) < eps_xi
            var[near_zero] = sigma[near_zero] ** 2 * (np.pi**2 / 6.0)
            if np.any(~near_zero):
                xi_nz = xi[~near_zero]
                var[~near_zero] = (
                    sigma[~near_zero] ** 2
                    * (
                        gamma_fn(1.0 - 2.0 * xi_nz)
                        - gamma_fn(1.0 - xi_nz) ** 2
                    )
                    / (xi_nz**2)
                )
            return rsd / np.sqrt(np.maximum(var, 1e-300))
        # deviance residuals
        eps = 1e-7
        xi2 = xi.copy()
        xi2[(xi2 >= 0) & (xi2 < eps)] = eps
        xi2[(xi2 < 0) & (xi2 > -eps)] = -eps
        aa = np.maximum(1.0 + (y - mu) * np.exp(-rho) * xi2, 1e-300)
        rsd_dev = (
            (xi2 + 1.0) / xi2 * np.log(aa)
            + aa ** (-1.0 / xi2)
            + (1.0 + xi2) * np.log(1.0 + xi2)
            - (1.0 + xi2)
        )
        return np.sqrt(np.maximum(0.0, rsd_dev)) * np.sign(y - fv)


def gevlss(link=("identity", "identity", "logit")) -> GevlssFamily:
    """
    Generalized Extreme Value location-scale-shape family.

    Parameters
    ----------
    link : tuple of str
        Links for (location, log-scale, shape).
        Location: ``"identity"`` (default) or ``"log"``.
        Log-scale: ``"identity"`` only.
        Shape xi ∈ (-1, 0.5): ``"logit"`` (default, shifted) or ``"identity"``.

    Returns
    -------
    GevlssFamily instance.
    """
    return GevlssFamily(link=link)


# ---------------------------------------------------------------------------
# shashlss: Sinh-arcsinh location-scale-skewness-kurtosis  (mgcv: shash)
# ---------------------------------------------------------------------------


class ShashlssFamily(GamlssFamily):
    """
    Sinh-arcsinh (shash) family with four linear predictors:
      1. location μ        (identity link)
      2. log-scale τ       (logeb link, lower-bounded)
      3. skewness ε        (identity link)
      4. log-kurtosis φ    (identity link)

    Derived quantities: sig = exp(τ), del = exp(φ).
    Observation model: Y = μ + sig*del*sinh((1/del)*asinh(Z) + ε/del)
    where Z ~ N(0,1).

    Mirrors mgcv ``shash`` (Fasiolo 2020) with phiPen=1e-3.

    Note: raw ``ll(..., deriv=4)`` blocks now support analytic outer gradient
    and Hessian terms through ``gam.fit5``.
    """

    name = "shashlss"
    nlp = 4
    n_linear_predictors = 4
    supports_analytic_outer_derivatives = False
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, b: float = 0.01, phi_pen: float = 1e-3):
        self.b = float(b)
        self.phi_pen = float(phi_pen)
        self.linfo = [
            _IdentityLinkInfo(),  # mu
            _LogEBLinkInfo(b=b),  # tau = log(sigma)
            _IdentityLinkInfo(),  # eps (skewness)
            _IdentityLinkInfo(),  # phi = log(delta) (kurtosis)
        ]
        self.tri = trind_generator(4)

    @staticmethod
    def _sqrtX2pm(x: np.ndarray, m: float) -> np.ndarray:
        """sqrt(x^2 + m), stable for large |x|.  Mirrors mgcv .sqrtX2pm."""
        x = np.abs(x)
        out = x.copy()
        small = x < 1e8
        out[small] = np.sqrt(x[small] ** 2 + m)
        return out

    @staticmethod
    def _ax2m1DivX2m2SQ(
        x: np.ndarray, m1: float, m2: float, a: float = 1.0
    ) -> np.ndarray:
        """(a*x^2 + m1) / (x^2 + m2)^2, stable for large |x|.

        Mirrors mgcv .ax2m1DivX2m2SQ.  Called with m1=-1, m2=1, a=1 to
        compute (z^2-1)/(z^2+1)^2 in Dmm.
        """
        x = np.abs(x)
        out = np.zeros_like(x, dtype=np.float64)
        neg = (a * x**2 + m1) < 0.0
        if np.any(neg):
            xn = x[neg]
            out[neg] = (a * xn**2 + m1) / (xn**2 + m2) ** 2
        pos = ~neg
        if np.any(pos):
            xp = x[pos]
            sq_num = np.sqrt(np.maximum(a * xp**2 + m1, 0.0))
            sq_den = np.sqrt(xp**2 + m2)
            out[pos] = (sq_num / sq_den / sq_den) ** 2
        return out

    def ll(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list,
        coef: np.ndarray,
        weights: np.ndarray,
        offset=None,
        deriv: int = 0,
        d1b=0,
        d2b=0,
        fh=None,
        D=None,
        **kw,
    ) -> dict:
        """Log-likelihood and derivatives for shash.  Mirrors mgcv shash$ll."""
        y = np.asarray(y, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        sandwich = bool(kw.get("sandwich", False))
        if weights is None:
            weights = np.ones(len(y), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        eta = X[:, jj[0]] @ coef[jj[0]]
        eta1 = X[:, jj[1]] @ coef[jj[1]]
        eta2 = X[:, jj[2]] @ coef[jj[2]]
        eta3 = X[:, jj[3]] @ coef[jj[3]]

        mu = self.linfo[0].linkinv(eta)
        tau = self.linfo[1].linkinv(eta1)
        eps = self.linfo[2].linkinv(eta2)
        phi = self.linfo[3].linkinv(eta3)

        sig = np.exp(tau)
        delta = np.exp(phi)

        z = (y - mu) / (sig * delta)
        dTasMe = delta * np.arcsinh(z) - eps
        g = -dTasMe
        CC = np.cosh(dTasMe)
        SS = np.sinh(dTasMe)

        # Numerically stable -0.5*log(1+z^2) = -0.5*log1p(z^2)
        log1pz2 = np.log1p(z**2)
        l0 = (
            -tau
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(np.maximum(CC, 1e-300))
            - 0.5 * log1pz2
            - 0.5 * SS**2
            - self.phi_pen * phi**2
        )
        l = float(np.sum(l0 * weights))

        if not deriv:
            return {"l": l, "l0": l0}

        # ----------------------------------------------------------------
        # First derivatives w.r.t. distribution parameters (mu,tau,eps,phi)
        # ----------------------------------------------------------------
        zsd = y - mu  # = z * sig * delta
        sSp1 = self._sqrtX2pm(z, 1.0)  # sqrt(z^2 + 1)
        asinhZ = np.arcsinh(z)

        tanh_g = np.tanh(g)
        sinh_2g = np.sinh(2.0 * g)
        sech_g_sq = 1.0 / np.cosh(g) ** 2

        De = tanh_g - 0.5 * sinh_2g
        Dm = (delta * De + z / sSp1) / (delta * sig * sSp1)
        Dt = zsd * Dm - 1.0
        Dp = Dt + 1.0 - delta * asinhZ * De - 2.0 * self.phi_pen * phi

        # shape: (n, 4) — L1 in mgcv
        L1 = np.column_stack([Dm, Dt, De, Dp])

        # ----------------------------------------------------------------
        # Second derivatives
        # ----------------------------------------------------------------
        cosh_2g = np.cosh(2.0 * g)

        Dme = (sech_g_sq - cosh_2g) / (sig * sSp1)
        Dte = zsd * Dme
        Dmm = (
            Dme / (sig * sSp1)
            + z * De / (sig**2 * delta * sSp1**3)
            + self._ax2m1DivX2m2SQ(z, -1.0, 1.0) / (delta * sig) ** 2
        )
        Dmt = zsd * Dmm - Dm
        Dee = -2.0 * np.cosh(g) ** 2 + sech_g_sq + 1.0
        Dtt = zsd * Dmt
        Dep = Dte - delta * asinhZ * Dee
        Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
        Dtp = zsd * Dmp
        Dpp = (
            Dtp
            - delta * asinhZ * Dep
            + delta * (z / sSp1 - asinhZ) * De
            - 2.0 * self.phi_pen
        )

        # shape: (n, 10) — L2 in mgcv (packed upper-triangular)
        L2 = np.column_stack([Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee, Dep, Dpp])

        # Link derivatives for chain rule
        IG1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),
                self.linfo[1].mu_eta(eta1),
                self.linfo[2].mu_eta(eta2),
                self.linfo[3].mu_eta(eta3),
            ]
        )
        G2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(tau),
                self.linfo[2].d2link(eps),
                self.linfo[3].d2link(phi),
            ]
        )

        L3 = G3 = L4 = G4 = 0

        if deriv > 1:
            # Third derivatives: 20 components for K=4 predictors.
            # Order: Dmmm Dmmt Dmme Dmmp Dmtt Dmte Dmtp Dmee Dmep Dmpp
            #        Dttt Dtte Dttp Dtee Dtep Dtpp Deee Deep Depp Dppp
            # Mirrors mgcv gamlss.r shash$ll deriv>1 block.
            Deee = -2.0 * (sinh_2g + sech_g_sq * tanh_g)
            Dmee = Deee / (sig * sSp1)
            Dmme = Dmee / (sig * sSp1) + z * Dee / (sig**2 * delta * sSp1**3)
            Dmmm = (
                2.0 * z * Dme / (sig**2 * delta * sSp1**3)
                + Dmme / (sig * sSp1)
                + self._ax2m1DivX2m2SQ(z, -1.0, 1.0, 2.0)
                * De
                / (sig**3 * delta**2 * sSp1)
                + 2.0
                * (z / sSp1)
                * self._ax2m1DivX2m2SQ(z, -3.0, 1.0)
                / ((sig * delta) ** 3 * sSp1)
            )
            Dmmt = zsd * Dmmm - 2.0 * Dmm
            Dtee = zsd * Dmee
            Dmte = zsd * Dmme - Dme
            Dtte = zsd * Dmte
            Dmtt = zsd * Dmmt - Dmt
            Dttt = zsd * Dmtt
            Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
            Dtep = zsd * Dmep
            Deep = Dtee - delta * asinhZ * Deee
            Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
            Dmmp = (
                Dmmt
                + 2.0 * Dme / (sig * sSp1)
                + z * De / (delta * sig**2 * sSp1**3)
                - delta * asinhZ * Dmme
            )
            Dmtp = zsd * Dmmp - Dmp
            Dttp = zsd * Dmtp
            Dmpp = (
                Dmtp
                + Dep / (sig * sSp1)
                + z**2 * De / (sig * sSp1**3)
                - delta * asinhZ * Dmep
                + delta * Dme * (z / sSp1 - asinhZ)
            )
            Dtpp = zsd * Dmpp
            Dppp = (
                Dtpp
                - delta * asinhZ * Depp
                + delta * (z / sSp1 - asinhZ) * (2.0 * Dep + De)
                + delta * (z / sSp1) ** 3 * De
            )
            L3 = np.column_stack(
                [
                    Dmmm,
                    Dmmt,
                    Dmme,
                    Dmmp,
                    Dmtt,
                    Dmte,
                    Dmtp,
                    Dmee,
                    Dmep,
                    Dmpp,
                    Dttt,
                    Dtte,
                    Dttp,
                    Dtee,
                    Dtep,
                    Dtpp,
                    Deee,
                    Deep,
                    Depp,
                    Dppp,
                ]
            )
            G3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(tau),
                    self.linfo[2].d3link(eps),
                    self.linfo[3].d3link(phi),
                ]
            )

        if deriv > 3:
            # 35 fourth-order derivatives from mgcv gamlss.r shashlss$ll (deriv>3).
            # R aliases: m→mu, t→tau, p→phi, e→eps, aaa2→zsd; exp1^x→np.exp(x); ^→**.
            abb8 = CC
            abb9 = SS
            abb1 = np.exp(-2.0 * tau - 2.0 * phi)
            abb3 = zsd**2
            abb4 = np.exp(-tau)
            abb5 = -tau - phi
            abb7 = np.exp(2.0 * abb5) * abb3 + 1.0
            abb6 = 1.0 / np.sqrt(abb7)
            aee5 = dTasMe + eps
            aff04 = abb1 * abb3 + 1.0
            aff05 = abb4**2
            aff08 = 2.0 * abb5
            aff10 = 1.0 / abb7
            aff13 = CC**2
            aff14 = np.exp(-tau + aff08)
            aff15 = abb6**3
            aff17 = SS**2
            agg15 = 1.0 / abb6
            agg17 = 1.0 / CC
            aii11 = dTasMe + eps
            aii12 = aii11 - abb4 * zsd * abb6
            aii17 = abb6**3
            ajj15 = zsd**3
            ann05 = np.exp(phi)
            ann06 = np.arcsinh(np.exp(abb5) * zsd)
            aoo09 = -zsd / (np.exp(tau) * agg15)
            app04 = np.exp(-2.0 * tau - 2.0 * phi) * abb3 + 1.0
            app08 = np.exp(-2.0 * tau + aff08)
            app10 = 1.0 / abb7**2
            app14 = np.exp(-tau + 4.0 * abb5)
            app16 = 1.0 / agg15**5
            app21 = 1.0 / np.exp(3.0 * tau)
            aqq03 = np.exp(-2.0 * tau - 2.0 * phi)
            aqq05 = aqq03 * abb3 + 1.0
            aqq27 = 1.0 / aff13
            arr06 = np.exp(aff08) * zsd**2 + 1.0
            arr07 = 1.0 / np.sqrt(arr06) ** 3
            arr12 = 1.0 / arr06
            ass16 = aii11 - zsd / (np.exp(tau) * agg15)
            ass23 = 1.0 / CC
            ass28 = 1.0 / aff13
            att19 = zsd**4
            avv19 = aii11 - abb4 * zsd * abb6
            ayy14 = -abb4 * zsd * abb6
            ayy16 = aii11 + ayy14
            ayy17 = aii11 + ayy14 - aff14 * ajj15 * aii17
            ayy24 = ayy16**2
            azz19 = zsd**5
            bdd07 = np.sqrt(np.exp(aff08) * zsd**2 + 1.0)
            bdd08 = 1.0 / bdd07**3
            bdd14 = 1.0 / bdd07
            bdd15 = aii11 - abb4 * zsd * bdd14
            bgg4 = aee5 - zsd / (
                np.exp(tau) * np.sqrt(np.exp(2.0 * abb5) * zsd**2 + 1.0)
            )
            bhh13 = -abb4 * zsd * bdd14
            bhh14 = ann05 * ann06
            bii11 = aii11 + aoo09
            bii15 = aii11 + aoo09 - aff14 * ajj15 * aii17
            bjj07 = 4.0 * abb5
            bjj08 = np.exp(-2.0 * tau + bjj07)
            bjj11 = 1.0 / abb7**3
            bjj14 = 1.0 / np.exp(4.0 * tau)
            bjj18 = np.exp(-tau + 6.0 * abb5)
            bjj21 = 1.0 / agg15**7
            bjj24 = np.exp(aff08 - 3.0 * tau)
            bjj26 = np.exp(-tau + bjj07)
            bkk33 = 1.0 / CC**3
            bkk34 = SS**3
            bll16 = np.exp(aff08 - 2.0 * tau)
            bmm34 = 1.0 / CC**3
            bmm35 = SS**3
            bss21 = 2.0 * aff14 * abb3 * aff15 - 3.0 * bjj26 * att19 * app16
            bss23 = -abb4 * zsd * abb6
            bss25 = aii11 + bss23
            bss26 = aii11 + bss23 - aff14 * ajj15 * aff15
            bss29 = bss25**2
            bss33 = (
                -4.0 * aff14 * zsd * aff15
                + 18.0 * bjj26 * ajj15 * app16
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**5 / agg15**7
            )
            btt24 = zsd**6
            byy24 = 2.0 * aff14 * ajj15 * aff15 - 3.0 * bjj26 * azz19 * app16
            byy35 = (
                -6.0 * aff14 * abb3 * aff15
                + 21.0 * bjj26 * att19 * app16
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**6 / agg15**7
            )
            bzz7 = CC**2
            bzz9 = SS**2
            cbb09 = 1.0 / agg15**5
            cbb18 = 2.0 * aff14 * abb3 * aii17 - 3.0 * app14 * att19 * cbb09
            cbb24 = aii11 + ayy14 - aff14 * zsd**3 * aii17
            cdd24 = zsd**7
            cll08 = 1.0 / bdd07**5
            cll16 = aii11 + bhh13
            cll17 = cll16**2
            cll18 = 2.0 * aff14 * ajj15 * bdd08 - 3.0 * app14 * azz19 * cll08
            cll24 = aii11 + bhh13 - aff14 * ajj15 * bdd08
            cmm12 = -3.0 * app14 * azz19 * cbb09
            cmm16 = 2.0 * aff14 * ajj15 * aii17 + cmm12
            cmm23 = aii11 + ayy14 + aff14 * ajj15 * aii17 + cmm12
            cmm28 = (
                -4.0 * aff14 * ajj15 * aii17
                + 18.0 * app14 * azz19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**7 / agg15**7
            )
            cpp06 = -zsd / (np.exp(tau) * bdd07)
            cpp08 = (cpp06 + aii11) ** 2
            cpp12 = aii11 + cpp06 - np.exp(-tau + aff08) * zsd**3 / bdd07**3
            cqq12 = -aff14 * ajj15 * bdd08
            cqq19 = bhh14 + bhh13
            cqq20 = cqq19**3
            cqq21 = bhh14 + bhh13 + aff14 * ajj15 * bdd08 - 3.0 * app14 * azz19 * cll08
            cqq25 = bhh14 + bhh13 + cqq12
            cqq28 = 1.0 / aff13
            crr18 = aii11 + aoo09 + aff14 * ajj15 * aii17 - 3.0 * app14 * azz19 * cbb09
            crr19 = bii11**4
            crr21 = bii15**2
            crr25 = (
                aii11
                + aoo09
                - 3.0 * aff14 * ajj15 * aii17
                + 15.0 * app14 * azz19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**7 / agg15**7
            )
            crr28 = bii11**2
            ccc23 = aii11 + ayy14 + aff14 * ajj15 * aii17 - 3.0 * app14 * azz19 * cbb09
            ccc24 = ayy16**3
            ccc28 = (
                -4.0 * aff14 * abb3 * aii17
                + 18.0 * app14 * att19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**6 / agg15**7
            )
            cnn3 = CC**2
            cnn5 = SS**2
            coo7 = CC**2
            coo9 = SS**2

            # j2 (mmmm)
            j2 = (
                -(6.0 * bjj14 * app10 * abb9**4) / abb8**4
                - (12.0 * bjj24 * zsd * app16 * abb9**3) / abb8**3
                + 8.0 * bjj14 * app10 * aqq27 * aff17
                + 4.0 * app08 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * abb3 * bjj11 * aqq27 * aff17
                - 4.0 * bjj14 * app10 * aff17
                + 4.0 * app08 * app10 * aff17
                - 15.0 * bjj08 * abb3 * bjj11 * aff17
                - 9.0 * bjj26 * zsd * app16 * abb8 * abb9
                + 24.0 * bjj24 * zsd * app16 * abb8 * abb9
                + 15.0 * bjj18 * ajj15 * bjj21 * abb8 * abb9
                + 9.0 * bjj26 * zsd * app16 * agg17 * abb9
                + 12.0 * bjj24 * zsd * app16 * agg17 * abb9
                - 15.0 * bjj18 * ajj15 * bjj21 * agg17 * abb9
                - 4.0 * bjj14 * app10 * aff13
                + 4.0 * app08 * app10 * aff13
                - 15.0 * bjj08 * abb3 * bjj11 * aff13
                - 2.0 * bjj14 * app10
                - 4.0 * app08 * app10
                + 15.0 * bjj08 * abb3 * bjj11
                + (6.0 * np.exp((-4.0 * tau) - 4.0 * phi)) / app04**2
                - (48.0 * np.exp((-6.0 * tau) - 6.0 * phi) * abb3) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**4) / app04**4
            )
            # k2 (mmmt)
            k2 = (
                -(6.0 * bjj14 * zsd * app10 * abb9**4) / abb8**4
                + 6.0 * app21 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * abb3 * app16 * bkk33 * bkk34
                + 8.0 * bjj14 * zsd * app10 * aqq27 * aff17
                + 13.0 * app08 * zsd * app10 * aqq27 * aff17
                - 15.0 * bjj08 * ajj15 * bjj11 * aqq27 * aff17
                - 4.0 * bjj14 * zsd * app10 * aff17
                + 13.0 * app08 * zsd * app10 * aff17
                - 15.0 * bjj08 * ajj15 * bjj11 * aff17
                - 12.0 * app21 * aff15 * abb8 * abb9
                + 3.0 * aff14 * aff15 * abb8 * abb9
                - 18.0 * bjj26 * abb3 * app16 * abb8 * abb9
                + 24.0 * bjj24 * abb3 * app16 * abb8 * abb9
                + 15.0 * bjj18 * att19 * bjj21 * abb8 * abb9
                - 6.0 * app21 * aff15 * agg17 * abb9
                - 3.0 * aff14 * aff15 * agg17 * abb9
                + 18.0 * bjj26 * abb3 * app16 * agg17 * abb9
                + 12.0 * bjj24 * abb3 * app16 * agg17 * abb9
                - 15.0 * bjj18 * att19 * bjj21 * agg17 * abb9
                - 4.0 * bjj14 * zsd * app10 * aff13
                + 13.0 * app08 * zsd * app10 * aff13
                - 15.0 * bjj08 * ajj15 * bjj11 * aff13
                - 2.0 * bjj14 * zsd * app10
                - 13.0 * app08 * zsd * app10
                + 15.0 * bjj08 * ajj15 * bjj11
                + (24.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd) / app04**2
                - (72.0 * np.exp((-6.0 * tau) - 6.0 * phi) * ajj15) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**5) / app04**4
            )
            # l2 (mmme)
            l2 = (
                -(6.0 * app21 * aff15 * abb9**4) / abb8**4
                - (6.0 * bll16 * zsd * app10 * abb9**3) / abb8**3
                + 8.0 * app21 * aff15 * aqq27 * aff17
                + aff14 * aff15 * aqq27 * aff17
                - 3.0 * app14 * abb3 * app16 * aqq27 * aff17
                - 4.0 * app21 * aff15 * aff17
                + aff14 * aff15 * aff17
                - 3.0 * app14 * abb3 * app16 * aff17
                + 12.0 * bll16 * zsd * app10 * abb8 * abb9
                + (6.0 * bll16 * zsd * app10 * abb9) / abb8
                - 4.0 * app21 * aff15 * aff13
                + aff14 * aff15 * aff13
                - 3.0 * app14 * abb3 * app16 * aff13
                - 2.0 * app21 * aff15
                - aff14 * aff15
                + 3.0 * app14 * abb3 * app16
            )
            # m2 (mmmp)
            m2 = (
                (6.0 * app21 * aff15 * ass16 * abb9**4) / abb8**4
                + 6.0 * app08 * zsd * app10 * ass16 * bmm34 * bmm35
                - 6.0 * bjj24 * abb3 * app16 * bmm34 * bmm35
                - 8.0 * app21 * aff15 * ass16 * ass28 * aff17
                - aff14 * aff15 * ass16 * ass28 * aff17
                + 3.0 * bjj26 * abb3 * app16 * ass16 * ass28 * aff17
                + 6.0 * app08 * zsd * app10 * ass28 * aff17
                - 12.0 * bjj08 * ajj15 * bjj11 * ass28 * aff17
                + 4.0 * app21 * aff15 * ass16 * aff17
                - aff14 * aff15 * ass16 * aff17
                + 3.0 * bjj26 * abb3 * app16 * ass16 * aff17
                + 6.0 * app08 * zsd * app10 * aff17
                - 12.0 * bjj08 * ajj15 * bjj11 * aff17
                - 12.0 * app08 * zsd * app10 * ass16 * abb8 * abb9
                + 2.0 * aff14 * aff15 * abb8 * abb9
                - 15.0 * bjj26 * abb3 * app16 * abb8 * abb9
                + 12.0 * bjj24 * abb3 * app16 * abb8 * abb9
                + 15.0 * bjj18 * att19 * bjj21 * abb8 * abb9
                - 6.0 * app08 * zsd * app10 * ass16 * ass23 * abb9
                - 2.0 * aff14 * aff15 * ass23 * abb9
                + 15.0 * bjj26 * abb3 * app16 * ass23 * abb9
                + 6.0 * bjj24 * abb3 * app16 * ass23 * abb9
                - 15.0 * bjj18 * att19 * bjj21 * ass23 * abb9
                + 4.0 * app21 * aff15 * ass16 * aff13
                - aff14 * aff15 * ass16 * aff13
                + 3.0 * bjj26 * abb3 * app16 * ass16 * aff13
                + 6.0 * app08 * zsd * app10 * aff13
                - 12.0 * bjj08 * ajj15 * bjj11 * aff13
                + 2.0 * app21 * aff15 * ass16
                + aff14 * aff15 * ass16
                - 3.0 * bjj26 * abb3 * app16 * ass16
                - 6.0 * app08 * zsd * app10
                + 12.0 * bjj08 * ajj15 * bjj11
                + (24.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd) / app04**2
                - (72.0 * np.exp((-6.0 * tau) - 6.0 * phi) * ajj15) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**5) / app04**4
            )
            # n2 (mmtt)
            n2 = (
                -(6.0 * bjj14 * abb3 * app10 * abb9**4) / abb8**4
                + 10.0 * app21 * zsd * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * ajj15 * app16 * bkk33 * bkk34
                - 4.0 * aff05 * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * abb3 * app10 * aqq27 * aff17
                + 19.0 * app08 * abb3 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * att19 * bjj11 * aqq27 * aff17
                - 4.0 * aff05 * aff10 * aff17
                - 4.0 * bjj14 * abb3 * app10 * aff17
                + 19.0 * app08 * abb3 * app10 * aff17
                - 15.0 * bjj08 * att19 * bjj11 * aff17
                - 20.0 * app21 * zsd * aff15 * abb8 * abb9
                + 9.0 * aff14 * zsd * aff15 * abb8 * abb9
                - 24.0 * bjj26 * ajj15 * app16 * abb8 * abb9
                + 24.0 * bjj24 * ajj15 * app16 * abb8 * abb9
                + 15.0 * bjj18 * azz19 * bjj21 * abb8 * abb9
                - 10.0 * app21 * zsd * aff15 * agg17 * abb9
                - 9.0 * aff14 * zsd * aff15 * agg17 * abb9
                + 24.0 * bjj26 * ajj15 * app16 * agg17 * abb9
                + 12.0 * bjj24 * ajj15 * app16 * agg17 * abb9
                - 15.0 * bjj18 * azz19 * bjj21 * agg17 * abb9
                - 4.0 * aff05 * aff10 * aff13
                - 4.0 * bjj14 * abb3 * app10 * aff13
                + 19.0 * app08 * abb3 * app10 * aff13
                - 15.0 * bjj08 * att19 * bjj11 * aff13
                + 4.0 * aff05 * aff10
                - 2.0 * bjj14 * abb3 * app10
                - 19.0 * app08 * abb3 * app10
                + 15.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # o2 (mmte)
            o2 = (
                -(6.0 * app21 * zsd * aff15 * abb9**4) / abb8**4
                + 4.0 * aff05 * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * abb3 * app10 * bkk33 * bkk34
                + 8.0 * app21 * zsd * aff15 * aqq27 * aff17
                + 3.0 * aff14 * zsd * aff15 * aqq27 * aff17
                - 3.0 * app14 * ajj15 * app16 * aqq27 * aff17
                - 4.0 * app21 * zsd * aff15 * aff17
                + 3.0 * aff14 * zsd * aff15 * aff17
                - 3.0 * app14 * ajj15 * app16 * aff17
                - 8.0 * aff05 * aff10 * abb8 * abb9
                + 12.0 * bll16 * abb3 * app10 * abb8 * abb9
                - 4.0 * aff05 * aff10 * agg17 * abb9
                + 6.0 * bll16 * abb3 * app10 * agg17 * abb9
                - 4.0 * app21 * zsd * aff15 * aff13
                + 3.0 * aff14 * zsd * aff15 * aff13
                - 3.0 * app14 * ajj15 * app16 * aff13
                - 2.0 * app21 * zsd * aff15
                - 3.0 * aff14 * zsd * aff15
                + 3.0 * app14 * ajj15 * app16
            )
            # p2 (mmtp)
            p2 = (
                (6.0 * app21 * zsd * aff15 * ass16 * abb9**4) / abb8**4
                - 4.0 * aff05 * aff10 * ass16 * bmm34 * bmm35
                + 6.0 * app08 * abb3 * app10 * ass16 * bmm34 * bmm35
                - 6.0 * bjj24 * ajj15 * app16 * bmm34 * bmm35
                - 8.0 * app21 * zsd * aff15 * ass16 * ass28 * aff17
                - 3.0 * aff14 * zsd * aff15 * ass16 * ass28 * aff17
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * ass28 * aff17
                + 10.0 * app08 * abb3 * app10 * ass28 * aff17
                - 12.0 * bjj08 * att19 * bjj11 * ass28 * aff17
                + 4.0 * app21 * zsd * aff15 * ass16 * aff17
                - 3.0 * aff14 * zsd * aff15 * ass16 * aff17
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * aff17
                + 10.0 * app08 * abb3 * app10 * aff17
                - 12.0 * bjj08 * att19 * bjj11 * aff17
                + 8.0 * aff05 * aff10 * ass16 * abb8 * abb9
                - 12.0 * app08 * abb3 * app10 * ass16 * abb8 * abb9
                + 6.0 * aff14 * zsd * aff15 * abb8 * abb9
                - 21.0 * bjj26 * ajj15 * app16 * abb8 * abb9
                + 12.0 * bjj24 * ajj15 * app16 * abb8 * abb9
                + 15.0 * bjj18 * azz19 * bjj21 * abb8 * abb9
                + 4.0 * aff05 * aff10 * ass16 * ass23 * abb9
                - 6.0 * app08 * abb3 * app10 * ass16 * ass23 * abb9
                - 6.0 * aff14 * zsd * aff15 * ass23 * abb9
                + 21.0 * bjj26 * ajj15 * app16 * ass23 * abb9
                + 6.0 * bjj24 * ajj15 * app16 * ass23 * abb9
                - 15.0 * bjj18 * azz19 * bjj21 * ass23 * abb9
                + 4.0 * app21 * zsd * aff15 * ass16 * aff13
                - 3.0 * aff14 * zsd * aff15 * ass16 * aff13
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * aff13
                + 10.0 * app08 * abb3 * app10 * aff13
                - 12.0 * bjj08 * att19 * bjj11 * aff13
                + 2.0 * app21 * zsd * aff15 * ass16
                + 3.0 * aff14 * zsd * aff15 * ass16
                - 3.0 * bjj26 * ajj15 * app16 * ass16
                - 10.0 * app08 * abb3 * app10
                + 12.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # q2 (mmee)
            q2 = (
                -(6.0 * aff05 * arr12 * abb9**4) / abb8**4
                - (2.0 * aff14 * zsd * arr07 * abb9**3) / abb8**3
                + (8.0 * aff05 * arr12 * aff17) / aff13
                - 4.0 * aff05 * arr12 * aff17
                + 4.0 * aff14 * zsd * arr07 * abb8 * abb9
                + (2.0 * aff14 * zsd * arr07 * abb9) / abb8
                - 4.0 * aff05 * arr12 * aff13
                - 2.0 * aff05 * arr12
            )
            # r2 (mmep)
            r2 = (
                (6.0 * aff05 * aff10 * ass16 * abb9**4) / abb8**4
                + 2.0 * aff14 * zsd * aff15 * ass16 * bmm34 * bmm35
                - 4.0 * bll16 * abb3 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * aff10 * ass16 * ass28 * aff17
                + 2.0 * aff14 * zsd * aff15 * ass28 * aff17
                - 3.0 * app14 * ajj15 * app16 * ass28 * aff17
                + 4.0 * aff05 * aff10 * ass16 * aff17
                + 2.0 * aff14 * zsd * aff15 * aff17
                - 3.0 * app14 * ajj15 * app16 * aff17
                - 4.0 * aff14 * zsd * aff15 * ass16 * abb8 * abb9
                + 8.0 * bll16 * abb3 * app10 * abb8 * abb9
                - 2.0 * aff14 * zsd * aff15 * ass16 * ass23 * abb9
                + 4.0 * bll16 * abb3 * app10 * ass23 * abb9
                + 4.0 * aff05 * aff10 * ass16 * aff13
                + 2.0 * aff14 * zsd * aff15 * aff13
                - 3.0 * app14 * ajj15 * app16 * aff13
                + 2.0 * aff05 * aff10 * ass16
                - 2.0 * aff14 * zsd * aff15
                + 3.0 * app14 * ajj15 * app16
            )
            # s2 (mmpp)
            s2 = (
                -(6.0 * aff05 * aff10 * bss29 * abb9**4) / abb8**4
                - 2.0 * aff14 * zsd * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * abb3 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * aff10 * bss29 * ass28 * aff17
                + aff14 * zsd * aff15 * bss26 * ass28 * aff17
                - 4.0 * aff14 * zsd * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * ass28 * aff17
                + 2.0 * abb4 * abb6 * bss21 * ass28 * aff17
                - 2.0 * bjj08 * att19 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * aff10 * bss29 * aff17
                + aff14 * zsd * aff15 * bss26 * aff17
                - 4.0 * aff14 * zsd * aff15 * bss25 * aff17
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * aff17
                + 2.0 * abb4 * abb6 * bss21 * aff17
                - 2.0 * bjj08 * att19 * bjj11 * aff17
                + 4.0 * aff14 * zsd * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * abb3 * app10 * bss25 * abb8 * abb9
                - bss33 * abb8 * abb9
                + 2.0 * aff14 * zsd * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * abb3 * app10 * bss25 * ass23 * abb9
                + bss33 * ass23 * abb9
                - 4.0 * aff05 * aff10 * bss29 * aff13
                + aff14 * zsd * aff15 * bss26 * aff13
                - 4.0 * aff14 * zsd * aff15 * bss25 * aff13
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * aff13
                + 2.0 * abb4 * abb6 * bss21 * aff13
                - 2.0 * bjj08 * att19 * bjj11 * aff13
                - 2.0 * aff05 * aff10 * bss29
                - aff14 * zsd * aff15 * bss26
                + 4.0 * aff14 * zsd * aff15 * bss25
                - 6.0 * bjj26 * ajj15 * app16 * bss25
                - 2.0 * abb4 * abb6 * bss21
                + 2.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # t2 (mttt)
            t2 = (
                -(6.0 * bjj14 * ajj15 * app10 * abb9**4) / abb8**4
                + 12.0 * app21 * abb3 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * att19 * app16 * bkk33 * bkk34
                - 7.0 * aff05 * zsd * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * ajj15 * app10 * aqq27 * aff17
                + 22.0 * app08 * ajj15 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * azz19 * bjj11 * aqq27 * aff17
                - 7.0 * aff05 * zsd * aff10 * aff17
                - 4.0 * bjj14 * ajj15 * app10 * aff17
                + 22.0 * app08 * ajj15 * app10 * aff17
                - 15.0 * bjj08 * azz19 * bjj11 * aff17
                - abb4 * abb6 * abb8 * abb9
                - 24.0 * app21 * abb3 * aff15 * abb8 * abb9
                + 13.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 27.0 * bjj26 * att19 * app16 * abb8 * abb9
                + 24.0 * bjj24 * att19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * btt24 * bjj21 * abb8 * abb9
                + abb4 * abb6 * agg17 * abb9
                - 12.0 * app21 * abb3 * aff15 * agg17 * abb9
                - 13.0 * aff14 * abb3 * aff15 * agg17 * abb9
                + 27.0 * bjj26 * att19 * app16 * agg17 * abb9
                + 12.0 * bjj24 * att19 * app16 * agg17 * abb9
                - 15.0 * bjj18 * btt24 * bjj21 * agg17 * abb9
                - 7.0 * aff05 * zsd * aff10 * aff13
                - 4.0 * bjj14 * ajj15 * app10 * aff13
                + 22.0 * app08 * ajj15 * app10 * aff13
                - 15.0 * bjj08 * azz19 * bjj11 * aff13
                + 7.0 * aff05 * zsd * aff10
                - 2.0 * bjj14 * ajj15 * app10
                - 22.0 * app08 * ajj15 * app10
                + 15.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # u2 (mtte)
            u2 = (
                -(6.0 * app21 * abb3 * aff15 * abb9**4) / abb8**4
                + 6.0 * aff05 * zsd * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * ajj15 * app10 * bkk33 * bkk34
                - abb4 * abb6 * aqq27 * aff17
                + 8.0 * app21 * abb3 * aff15 * aqq27 * aff17
                + 4.0 * aff14 * abb3 * aff15 * aqq27 * aff17
                - 3.0 * app14 * att19 * app16 * aqq27 * aff17
                - abb4 * abb6 * aff17
                - 4.0 * app21 * abb3 * aff15 * aff17
                + 4.0 * aff14 * abb3 * aff15 * aff17
                - 3.0 * app14 * att19 * app16 * aff17
                - 12.0 * aff05 * zsd * aff10 * abb8 * abb9
                + 12.0 * bll16 * ajj15 * app10 * abb8 * abb9
                - 6.0 * aff05 * zsd * aff10 * agg17 * abb9
                + 6.0 * bll16 * ajj15 * app10 * agg17 * abb9
                - abb4 * abb6 * aff13
                - 4.0 * app21 * abb3 * aff15 * aff13
                + 4.0 * aff14 * abb3 * aff15 * aff13
                - 3.0 * app14 * att19 * app16 * aff13
                + abb4 * abb6
                - 2.0 * app21 * abb3 * aff15
                - 4.0 * aff14 * abb3 * aff15
                + 3.0 * app14 * att19 * app16
            )
            # v2 (mttp)
            v2 = (
                (6.0 * app21 * abb3 * aff15 * avv19 * abb9**4) / abb8**4
                - 6.0 * aff05 * zsd * aff10 * avv19 * bmm34 * bmm35
                + 6.0 * app08 * ajj15 * app10 * avv19 * bmm34 * bmm35
                - 6.0 * bjj24 * att19 * app16 * bmm34 * bmm35
                + abb4 * abb6 * avv19 * ass28 * aff17
                - 8.0 * app21 * abb3 * aff15 * avv19 * ass28 * aff17
                - 4.0 * aff14 * abb3 * aff15 * avv19 * ass28 * aff17
                + 3.0 * bjj26 * att19 * app16 * avv19 * ass28 * aff17
                + 12.0 * app08 * ajj15 * app10 * ass28 * aff17
                - 12.0 * bjj08 * azz19 * bjj11 * ass28 * aff17
                + abb4 * abb6 * avv19 * aff17
                + 4.0 * app21 * abb3 * aff15 * avv19 * aff17
                - 4.0 * aff14 * abb3 * aff15 * avv19 * aff17
                + 3.0 * bjj26 * att19 * app16 * avv19 * aff17
                + 12.0 * app08 * ajj15 * app10 * aff17
                - 12.0 * bjj08 * azz19 * bjj11 * aff17
                + 12.0 * aff05 * zsd * aff10 * avv19 * abb8 * abb9
                - 12.0 * app08 * ajj15 * app10 * avv19 * abb8 * abb9
                + 9.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 24.0 * bjj26 * att19 * app16 * abb8 * abb9
                + 12.0 * bjj24 * att19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * btt24 * bjj21 * abb8 * abb9
                + 6.0 * aff05 * zsd * aff10 * avv19 * ass23 * abb9
                - 6.0 * app08 * ajj15 * app10 * avv19 * ass23 * abb9
                - 9.0 * aff14 * abb3 * aff15 * ass23 * abb9
                + 24.0 * bjj26 * att19 * app16 * ass23 * abb9
                + 6.0 * bjj24 * att19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * btt24 * bjj21 * ass23 * abb9
                + abb4 * abb6 * avv19 * aff13
                + 4.0 * app21 * abb3 * aff15 * avv19 * aff13
                - 4.0 * aff14 * abb3 * aff15 * avv19 * aff13
                + 3.0 * bjj26 * att19 * app16 * avv19 * aff13
                + 12.0 * app08 * ajj15 * app10 * aff13
                - 12.0 * bjj08 * azz19 * bjj11 * aff13
                - abb4 * abb6 * avv19
                + 2.0 * app21 * abb3 * aff15 * avv19
                + 4.0 * aff14 * abb3 * aff15 * avv19
                - 3.0 * bjj26 * att19 * app16 * avv19
                - 12.0 * app08 * ajj15 * app10
                + 12.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # w2 (mtte... wait, this is mtee)
            w2 = (
                -(6.0 * aff05 * zsd * aff10 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * bkk33 * bkk34
                - 2.0 * aff14 * abb3 * aff15 * bkk33 * bkk34
                + (8.0 * aff05 * zsd * aff10 * aff17) / aff13
                - 4.0 * aff05 * zsd * aff10 * aff17
                - 4.0 * abb4 * abb6 * abb8 * abb9
                + 4.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 2.0 * abb4 * abb6 * agg17 * abb9
                + 2.0 * aff14 * abb3 * aff15 * agg17 * abb9
                - 4.0 * aff05 * zsd * aff10 * aff13
                - 2.0 * aff05 * zsd * aff10
            )
            # x2 (mtep)
            x2 = (
                (6.0 * aff05 * zsd * aff10 * avv19 * abb9**4) / abb8**4
                - 2.0 * abb4 * abb6 * avv19 * bmm34 * bmm35
                + 2.0 * aff14 * abb3 * aff15 * avv19 * bmm34 * bmm35
                - 4.0 * bll16 * ajj15 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * zsd * aff10 * avv19 * ass28 * aff17
                + 3.0 * aff14 * abb3 * aff15 * ass28 * aff17
                - 3.0 * app14 * att19 * app16 * ass28 * aff17
                + 4.0 * aff05 * zsd * aff10 * avv19 * aff17
                + 3.0 * aff14 * abb3 * aff15 * aff17
                - 3.0 * app14 * att19 * app16 * aff17
                + 4.0 * abb4 * abb6 * avv19 * abb8 * abb9
                - 4.0 * aff14 * abb3 * aff15 * avv19 * abb8 * abb9
                + 8.0 * bll16 * ajj15 * app10 * abb8 * abb9
                + 2.0 * abb4 * abb6 * avv19 * ass23 * abb9
                - 2.0 * aff14 * abb3 * aff15 * avv19 * ass23 * abb9
                + 4.0 * bll16 * ajj15 * app10 * ass23 * abb9
                + 4.0 * aff05 * zsd * aff10 * avv19 * aff13
                + 3.0 * aff14 * abb3 * aff15 * aff13
                - 3.0 * app14 * att19 * app16 * aff13
                + 2.0 * aff05 * zsd * aff10 * avv19
                - 3.0 * aff14 * abb3 * aff15
                + 3.0 * app14 * att19 * app16
            )
            # y2 (mtpp)
            y2 = (
                -(6.0 * aff05 * zsd * aff10 * bss29 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * bss29 * bmm34 * bmm35
                - 2.0 * aff14 * abb3 * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * zsd * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * ajj15 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * zsd * aff10 * bss29 * ass28 * aff17
                - abb4 * abb6 * bss26 * ass28 * aff17
                + aff14 * abb3 * aff15 * bss26 * ass28 * aff17
                - 6.0 * aff14 * abb3 * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * att19 * app16 * bss25 * ass28 * aff17
                + abb4 * abb6 * byy24 * ass28 * aff17
                + abb4 * zsd * abb6 * bss21 * ass28 * aff17
                - 2.0 * bjj08 * azz19 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * zsd * aff10 * bss29 * aff17
                - abb4 * abb6 * bss26 * aff17
                + aff14 * abb3 * aff15 * bss26 * aff17
                - 6.0 * aff14 * abb3 * aff15 * bss25 * aff17
                + 6.0 * bjj26 * att19 * app16 * bss25 * aff17
                + abb4 * abb6 * byy24 * aff17
                + abb4 * zsd * abb6 * bss21 * aff17
                - 2.0 * bjj08 * azz19 * bjj11 * aff17
                - 4.0 * abb4 * abb6 * bss29 * abb8 * abb9
                + 4.0 * aff14 * abb3 * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * zsd * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * ajj15 * app10 * bss25 * abb8 * abb9
                - byy35 * abb8 * abb9
                - 2.0 * abb4 * abb6 * bss29 * ass23 * abb9
                + 2.0 * aff14 * abb3 * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * zsd * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * ajj15 * app10 * bss25 * ass23 * abb9
                + byy35 * ass23 * abb9
                - 4.0 * aff05 * zsd * aff10 * bss29 * aff13
                - abb4 * abb6 * bss26 * aff13
                + aff14 * abb3 * aff15 * bss26 * aff13
                - 6.0 * aff14 * abb3 * aff15 * bss25 * aff13
                + 6.0 * bjj26 * att19 * app16 * bss25 * aff13
                + abb4 * abb6 * byy24 * aff13
                + abb4 * zsd * abb6 * bss21 * aff13
                - 2.0 * bjj08 * azz19 * bjj11 * aff13
                - 2.0 * aff05 * zsd * aff10 * bss29
                + abb4 * abb6 * bss26
                - aff14 * abb3 * aff15 * bss26
                + 6.0 * aff14 * abb3 * aff15 * bss25
                - 6.0 * bjj26 * att19 * app16 * bss25
                - abb4 * abb6 * byy24
                - abb4 * zsd * abb6 * bss21
                + 2.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # z2 (meee)
            z2 = (
                -(6.0 * abb4 * abb6 * abb9**4) / abb8**4
                + (8.0 * abb4 * abb6 * bzz9) / bzz7
                - 4.0 * abb4 * abb6 * bzz9
                - 4.0 * abb4 * abb6 * bzz7
                - 2.0 * abb4 * abb6
            )
            # a3 (meep)
            a3 = (
                (6.0 * abb4 * abb6 * aii12 * abb9**4) / abb8**4
                - (2.0 * aff14 * abb3 * aii17 * abb9**3) / abb8**3
                - (8.0 * abb4 * abb6 * aii12 * aff17) / aff13
                + 4.0 * abb4 * abb6 * aii12 * aff17
                + 4.0 * aff14 * abb3 * aii17 * abb8 * abb9
                + (2.0 * aff14 * abb3 * aii17 * abb9) / abb8
                + 4.0 * abb4 * abb6 * aii12 * aff13
                + 2.0 * abb4 * abb6 * aii12
            )
            # b3 (mepp)
            b3 = (
                -(6.0 * abb4 * abb6 * ayy24 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * cbb24 * bmm34 * bmm35
                + 4.0 * aff14 * abb3 * aii17 * ayy16 * bmm34 * bmm35
                + 8.0 * abb4 * abb6 * ayy24 * ass28 * aff17
                + cbb18 * ass28 * aff17
                - 4.0 * abb4 * abb6 * ayy24 * aff17
                + cbb18 * aff17
                - 4.0 * abb4 * abb6 * cbb24 * abb8 * abb9
                - 8.0 * aff14 * abb3 * aii17 * ayy16 * abb8 * abb9
                - 2.0 * abb4 * abb6 * cbb24 * ass23 * abb9
                - 4.0 * aff14 * abb3 * aii17 * ayy16 * ass23 * abb9
                - 4.0 * abb4 * abb6 * ayy24 * aff13
                + cbb18 * aff13
                - 2.0 * abb4 * abb6 * ayy24
                - 2.0 * aff14 * abb3 * aii17
                + 3.0 * app14 * att19 * cbb09
            )
            # c3 (mppp)
            c3 = (
                (6.0 * abb4 * abb6 * ccc24 * abb9**4) / abb8**4
                - 6.0 * aff14 * abb3 * aii17 * ayy24 * bmm34 * bmm35
                - 6.0 * abb4 * abb6 * ayy16 * ayy17 * bmm34 * bmm35
                - 8.0 * abb4 * abb6 * ccc24 * ass28 * aff17
                + abb4 * abb6 * ccc23 * ass28 * aff17
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * ass28 * aff17
                - 3.0 * cbb18 * ayy16 * ass28 * aff17
                + 4.0 * abb4 * abb6 * ccc24 * aff17
                + abb4 * abb6 * ccc23 * aff17
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * aff17
                - 3.0 * cbb18 * ayy16 * aff17
                + 12.0 * aff14 * abb3 * aii17 * ayy24 * abb8 * abb9
                + 12.0 * abb4 * abb6 * ayy16 * ayy17 * abb8 * abb9
                - ccc28 * abb8 * abb9
                + 6.0 * aff14 * abb3 * aii17 * ayy24 * ass23 * abb9
                + 6.0 * abb4 * abb6 * ayy16 * ayy17 * ass23 * abb9
                + ccc28 * ass23 * abb9
                + 4.0 * abb4 * abb6 * ccc24 * aff13
                + abb4 * abb6 * ccc23 * aff13
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * aff13
                - 3.0 * cbb18 * ayy16 * aff13
                + 2.0 * abb4 * abb6 * ccc24
                - abb4 * abb6 * ccc23
                - 3.0 * aff14 * abb3 * aii17 * ayy17
                + 3.0 * cbb18 * ayy16
                - (8.0 * abb1 * zsd) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aff04**4
            )
            # d3 (tttt)
            d3 = (
                -(6.0 * bjj14 * att19 * app10 * abb9**4) / abb8**4
                + 12.0 * app21 * ajj15 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * azz19 * app16 * bkk33 * bkk34
                - 7.0 * aff05 * abb3 * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * att19 * app10 * aqq27 * aff17
                + 22.0 * app08 * att19 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * btt24 * bjj11 * aqq27 * aff17
                - 7.0 * aff05 * abb3 * aff10 * aff17
                - 4.0 * bjj14 * att19 * app10 * aff17
                + 22.0 * app08 * att19 * app10 * aff17
                - 15.0 * bjj08 * btt24 * bjj11 * aff17
                - abb4 * zsd * abb6 * abb8 * abb9
                - 24.0 * app21 * ajj15 * aff15 * abb8 * abb9
                + 13.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 27.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 24.0 * bjj24 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                + abb4 * zsd * abb6 * agg17 * abb9
                - 12.0 * app21 * ajj15 * aff15 * agg17 * abb9
                - 13.0 * aff14 * ajj15 * aff15 * agg17 * abb9
                + 27.0 * bjj26 * azz19 * app16 * agg17 * abb9
                + 12.0 * bjj24 * azz19 * app16 * agg17 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * agg17 * abb9
                - 7.0 * aff05 * abb3 * aff10 * aff13
                - 4.0 * bjj14 * att19 * app10 * aff13
                + 22.0 * app08 * att19 * app10 * aff13
                - 15.0 * bjj08 * btt24 * bjj11 * aff13
                + 7.0 * aff05 * abb3 * aff10
                - 2.0 * bjj14 * att19 * app10
                - 22.0 * app08 * att19 * app10
                + 15.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # e3 (ttte)
            e3 = (
                -(6.0 * app21 * ajj15 * aff15 * abb9**4) / abb8**4
                + 6.0 * aff05 * abb3 * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * att19 * app10 * bkk33 * bkk34
                - abb4 * zsd * abb6 * aqq27 * aff17
                + 8.0 * app21 * ajj15 * aff15 * aqq27 * aff17
                + 4.0 * aff14 * ajj15 * aff15 * aqq27 * aff17
                - 3.0 * app14 * azz19 * app16 * aqq27 * aff17
                - abb4 * zsd * abb6 * aff17
                - 4.0 * app21 * ajj15 * aff15 * aff17
                + 4.0 * aff14 * ajj15 * aff15 * aff17
                - 3.0 * app14 * azz19 * app16 * aff17
                - 12.0 * aff05 * abb3 * aff10 * abb8 * abb9
                + 12.0 * bll16 * att19 * app10 * abb8 * abb9
                - 6.0 * aff05 * abb3 * aff10 * agg17 * abb9
                + 6.0 * bll16 * att19 * app10 * agg17 * abb9
                - abb4 * zsd * abb6 * aff13
                - 4.0 * app21 * ajj15 * aff15 * aff13
                + 4.0 * aff14 * ajj15 * aff15 * aff13
                - 3.0 * app14 * azz19 * app16 * aff13
                + abb4 * zsd * abb6
                - 2.0 * app21 * ajj15 * aff15
                - 4.0 * aff14 * ajj15 * aff15
                + 3.0 * app14 * azz19 * app16
            )
            # f3 (tttp)
            f3 = (
                (6.0 * app21 * ajj15 * aff15 * avv19 * abb9**4) / abb8**4
                - 6.0 * aff05 * abb3 * aff10 * avv19 * bmm34 * bmm35
                + 6.0 * app08 * att19 * app10 * avv19 * bmm34 * bmm35
                - 6.0 * bjj24 * azz19 * app16 * bmm34 * bmm35
                + abb4 * zsd * abb6 * avv19 * ass28 * aff17
                - 8.0 * app21 * ajj15 * aff15 * avv19 * ass28 * aff17
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * ass28 * aff17
                + 3.0 * bjj26 * azz19 * app16 * avv19 * ass28 * aff17
                + 12.0 * app08 * att19 * app10 * ass28 * aff17
                - 12.0 * bjj08 * btt24 * bjj11 * ass28 * aff17
                + abb4 * zsd * abb6 * avv19 * aff17
                + 4.0 * app21 * ajj15 * aff15 * avv19 * aff17
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * aff17
                + 3.0 * bjj26 * azz19 * app16 * avv19 * aff17
                + 12.0 * app08 * att19 * app10 * aff17
                - 12.0 * bjj08 * btt24 * bjj11 * aff17
                + 12.0 * aff05 * abb3 * aff10 * avv19 * abb8 * abb9
                - 12.0 * app08 * att19 * app10 * avv19 * abb8 * abb9
                + 9.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 24.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 12.0 * bjj24 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                + 6.0 * aff05 * abb3 * aff10 * avv19 * ass23 * abb9
                - 6.0 * app08 * att19 * app10 * avv19 * ass23 * abb9
                - 9.0 * aff14 * ajj15 * aff15 * ass23 * abb9
                + 24.0 * bjj26 * azz19 * app16 * ass23 * abb9
                + 6.0 * bjj24 * azz19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * ass23 * abb9
                + abb4 * zsd * abb6 * avv19 * aff13
                + 4.0 * app21 * ajj15 * aff15 * avv19 * aff13
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * aff13
                + 3.0 * bjj26 * azz19 * app16 * avv19 * aff13
                + 12.0 * app08 * att19 * app10 * aff13
                - 12.0 * bjj08 * btt24 * bjj11 * aff13
                - abb4 * zsd * abb6 * avv19
                + 2.0 * app21 * ajj15 * aff15 * avv19
                + 4.0 * aff14 * ajj15 * aff15 * avv19
                - 3.0 * bjj26 * azz19 * app16 * avv19
                - 12.0 * app08 * att19 * app10
                + 12.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # g3 (ttee)
            g3 = (
                -(6.0 * aff05 * abb3 * aff10 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * abb6 * bkk33 * bkk34
                - 2.0 * aff14 * ajj15 * aff15 * bkk33 * bkk34
                + (8.0 * aff05 * abb3 * aff10 * aff17) / aff13
                - 4.0 * aff05 * abb3 * aff10 * aff17
                - 4.0 * abb4 * zsd * abb6 * abb8 * abb9
                + 4.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 2.0 * abb4 * zsd * abb6 * agg17 * abb9
                + 2.0 * aff14 * ajj15 * aff15 * agg17 * abb9
                - 4.0 * aff05 * abb3 * aff10 * aff13
                - 2.0 * aff05 * abb3 * aff10
            )
            # h3 (ttep)
            h3 = (
                (6.0 * aff05 * abb3 * aff10 * avv19 * abb9**4) / abb8**4
                - 2.0 * abb4 * zsd * abb6 * avv19 * bmm34 * bmm35
                + 2.0 * aff14 * ajj15 * aff15 * avv19 * bmm34 * bmm35
                - 4.0 * bll16 * att19 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * abb3 * aff10 * avv19 * ass28 * aff17
                + 3.0 * aff14 * ajj15 * aff15 * ass28 * aff17
                - 3.0 * app14 * azz19 * app16 * ass28 * aff17
                + 4.0 * aff05 * abb3 * aff10 * avv19 * aff17
                + 3.0 * aff14 * ajj15 * aff15 * aff17
                - 3.0 * app14 * azz19 * app16 * aff17
                + 4.0 * abb4 * zsd * abb6 * avv19 * abb8 * abb9
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * abb8 * abb9
                + 8.0 * bll16 * att19 * app10 * abb8 * abb9
                + 2.0 * abb4 * zsd * abb6 * avv19 * ass23 * abb9
                - 2.0 * aff14 * ajj15 * aff15 * avv19 * ass23 * abb9
                + 4.0 * bll16 * att19 * app10 * ass23 * abb9
                + 4.0 * aff05 * abb3 * aff10 * avv19 * aff13
                + 3.0 * aff14 * ajj15 * aff15 * aff13
                - 3.0 * app14 * azz19 * app16 * aff13
                + 2.0 * aff05 * abb3 * aff10 * avv19
                - 3.0 * aff14 * ajj15 * aff15
                + 3.0 * app14 * azz19 * app16
            )
            # i3 (ttpp)
            i3 = (
                -(6.0 * aff05 * abb3 * aff10 * bss29 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * abb6 * bss29 * bmm34 * bmm35
                - 2.0 * aff14 * ajj15 * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * abb3 * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * att19 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * abb3 * aff10 * bss29 * ass28 * aff17
                - abb4 * zsd * abb6 * bss26 * ass28 * aff17
                + aff14 * ajj15 * aff15 * bss26 * ass28 * aff17
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * azz19 * app16 * bss25 * ass28 * aff17
                + 4.0 * app08 * att19 * app10 * ass28 * aff17
                - 8.0 * bjj08 * btt24 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * abb3 * aff10 * bss29 * aff17
                - abb4 * zsd * abb6 * bss26 * aff17
                + aff14 * ajj15 * aff15 * bss26 * aff17
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * aff17
                + 6.0 * bjj26 * azz19 * app16 * bss25 * aff17
                + 4.0 * app08 * att19 * app10 * aff17
                - 8.0 * bjj08 * btt24 * bjj11 * aff17
                - 4.0 * abb4 * zsd * abb6 * bss29 * abb8 * abb9
                + 4.0 * aff14 * ajj15 * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * abb3 * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * att19 * app10 * bss25 * abb8 * abb9
                + 6.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 21.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                - 2.0 * abb4 * zsd * abb6 * bss29 * ass23 * abb9
                + 2.0 * aff14 * ajj15 * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * abb3 * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * att19 * app10 * bss25 * ass23 * abb9
                - 6.0 * aff14 * ajj15 * aff15 * ass23 * abb9
                + 21.0 * bjj26 * azz19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * ass23 * abb9
                - 4.0 * aff05 * abb3 * aff10 * bss29 * aff13
                - abb4 * zsd * abb6 * bss26 * aff13
                + aff14 * ajj15 * aff15 * bss26 * aff13
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * aff13
                + 6.0 * bjj26 * azz19 * app16 * bss25 * aff13
                + 4.0 * app08 * att19 * app10 * aff13
                - 8.0 * bjj08 * btt24 * bjj11 * aff13
                - 2.0 * aff05 * abb3 * aff10 * bss29
                + abb4 * zsd * abb6 * bss26
                - aff14 * ajj15 * aff15 * bss26
                + 6.0 * aff14 * ajj15 * aff15 * bss25
                - 6.0 * bjj26 * azz19 * app16 * bss25
                - 4.0 * app08 * att19 * app10
                + 8.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # j3 (teee)
            j3 = (
                -(6.0 * abb4 * zsd * abb6 * abb9**4) / abb8**4
                + (8.0 * abb4 * zsd * abb6 * bzz9) / bzz7
                - 4.0 * abb4 * zsd * abb6 * bzz9
                - 4.0 * abb4 * zsd * abb6 * bzz7
                - 2.0 * abb4 * zsd * abb6
            )
            # k3 (teep)
            k3 = (
                (6.0 * abb4 * zsd * bdd14 * bdd15 * abb9**4) / abb8**4
                - (2.0 * aff14 * ajj15 * bdd08 * abb9**3) / abb8**3
                - (8.0 * abb4 * zsd * bdd14 * bdd15 * aff17) / aff13
                + 4.0 * abb4 * zsd * bdd14 * bdd15 * aff17
                + 4.0 * aff14 * ajj15 * bdd08 * abb8 * abb9
                + (2.0 * aff14 * ajj15 * bdd08 * abb9) / abb8
                + 4.0 * abb4 * zsd * bdd14 * bdd15 * aff13
                + 2.0 * abb4 * zsd * bdd14 * bdd15
            )
            # l3 (tepp)
            l3 = (
                -(6.0 * abb4 * zsd * bdd14 * cll17 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * bdd14 * cll24 * bmm34 * bmm35
                + 4.0 * aff14 * ajj15 * bdd08 * cll16 * bmm34 * bmm35
                + 8.0 * abb4 * zsd * bdd14 * cll17 * ass28 * aff17
                + cll18 * ass28 * aff17
                - 4.0 * abb4 * zsd * bdd14 * cll17 * aff17
                + cll18 * aff17
                - 4.0 * abb4 * zsd * bdd14 * cll24 * abb8 * abb9
                - 8.0 * aff14 * ajj15 * bdd08 * cll16 * abb8 * abb9
                - 2.0 * abb4 * zsd * bdd14 * cll24 * ass23 * abb9
                - 4.0 * aff14 * ajj15 * bdd08 * cll16 * ass23 * abb9
                - 4.0 * abb4 * zsd * bdd14 * cll17 * aff13
                + cll18 * aff13
                - 2.0 * abb4 * zsd * bdd14 * cll17
                - 2.0 * aff14 * ajj15 * bdd08
                + 3.0 * app14 * azz19 * cll08
            )
            # m3 (tppp)
            m3 = (
                (6.0 * abb4 * zsd * abb6 * ccc24 * abb9**4) / abb8**4
                - 6.0 * aff14 * ajj15 * aii17 * ayy24 * bmm34 * bmm35
                - 6.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * bmm34 * bmm35
                - 8.0 * abb4 * zsd * abb6 * ccc24 * ass28 * aff17
                + abb4 * zsd * abb6 * cmm23 * ass28 * aff17
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * ass28 * aff17
                - 3.0 * cmm16 * ayy16 * ass28 * aff17
                + 4.0 * abb4 * zsd * abb6 * ccc24 * aff17
                + abb4 * zsd * abb6 * cmm23 * aff17
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * aff17
                - 3.0 * cmm16 * ayy16 * aff17
                + 12.0 * aff14 * ajj15 * aii17 * ayy24 * abb8 * abb9
                + 12.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * abb8 * abb9
                - cmm28 * abb8 * abb9
                + 6.0 * aff14 * ajj15 * aii17 * ayy24 * ass23 * abb9
                + 6.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * ass23 * abb9
                + cmm28 * ass23 * abb9
                + 4.0 * abb4 * zsd * abb6 * ccc24 * aff13
                + abb4 * zsd * abb6 * cmm23 * aff13
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * aff13
                - 3.0 * cmm16 * ayy16 * aff13
                + 2.0 * abb4 * zsd * abb6 * ccc24
                - abb4 * zsd * abb6 * cmm23
                - 3.0 * aff14 * ajj15 * aii17 * ayy17
                + 3.0 * cmm16 * ayy16
                - (8.0 * abb1 * abb3) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd**4) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * zsd**6) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aff04**4
            )
            # n3 (eeee)
            n3 = (
                -(6.0 * abb9**4) / abb8**4
                + (8.0 * cnn5) / cnn3
                - 4.0 * cnn5
                - 4.0 * cnn3
                - 2.0
            )
            # o3 (eeep)
            o3 = (
                (6.0 * bgg4 * abb9**4) / abb8**4
                - (8.0 * bgg4 * coo9) / coo7
                + 4.0 * bgg4 * coo9
                + 4.0 * bgg4 * coo7
                + 2.0 * bgg4
            )
            # p3 (eepp)
            p3 = (
                -(6.0 * cpp08 * abb9**4) / abb8**4
                + (2.0 * cpp12 * abb9**3) / abb8**3
                + (8.0 * cpp08 * aff17) / aff13
                - 4.0 * cpp08 * aff17
                - 4.0 * cpp12 * abb8 * abb9
                - (2.0 * cpp12 * abb9) / abb8
                - 4.0 * cpp08 * aff13
                - 2.0 * cpp08
            )
            # q3 (eppp)
            q3 = (
                (6.0 * cqq20 * abb9**4) / abb8**4
                - (6.0 * cqq19 * cqq25 * abb9**3) / abb8**3
                - 8.0 * cqq20 * cqq28 * aff17
                + cqq21 * cqq28 * aff17
                + 4.0 * cqq20 * aff17
                + cqq21 * aff17
                + 12.0 * cqq19 * cqq25 * abb8 * abb9
                + (6.0 * cqq19 * cqq25 * abb9) / abb8
                + 4.0 * cqq20 * aff13
                + cqq21 * aff13
                + 2.0 * cqq20
                - ann05 * ann06
                + abb4 * zsd * bdd14
                + cqq12
                + 3.0 * app14 * azz19 * cll08
            )
            # r3 (pppp)
            r3 = (
                -(6.0 * crr19 * abb9**4) / abb8**4
                + (12.0 * crr28 * bii15 * abb9**3) / abb8**3
                - 3.0 * crr21 * ass28 * aff17
                + 8.0 * crr19 * ass28 * aff17
                - 4.0 * bii11 * crr18 * ass28 * aff17
                - 3.0 * crr21 * aff17
                - 4.0 * crr19 * aff17
                - 4.0 * bii11 * crr18 * aff17
                - 24.0 * crr28 * bii15 * abb8 * abb9
                - crr25 * abb8 * abb9
                - 12.0 * crr28 * bii15 * ass23 * abb9
                + crr25 * ass23 * abb9
                - 3.0 * crr21 * aff13
                - 4.0 * crr19 * aff13
                - 4.0 * bii11 * crr18 * aff13
                + 3.0 * crr21
                - 2.0 * crr19
                + 4.0 * bii11 * crr18
                - (8.0 * abb1 * abb3) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd**4) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * zsd**6) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aff04**4
            )

            L4 = np.column_stack(
                [
                    j2,
                    k2,
                    l2,
                    m2,
                    n2,
                    o2,
                    p2,
                    q2,
                    r2,
                    s2,
                    t2,
                    u2,
                    v2,
                    w2,
                    x2,
                    y2,
                    z2,
                    a3,
                    b3,
                    c3,
                    d3,
                    e3,
                    f3,
                    g3,
                    h3,
                    i3,
                    j3,
                    k3,
                    l3,
                    m3,
                    n3,
                    o3,
                    p3,
                    q3,
                    r3,
                ]
            )
            G4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(tau),
                    self.linfo[2].d4link(eps),
                    self.linfo[3].d4link(phi),
                ]
            )

        I2 = self.tri["i2"]
        I3 = self.tri["i3"]
        I4 = self.tri["i4"]

        de = gamlss_etamu(L1, L2, L3, L4, IG1, G2, G3, G4, I2, I3, I4, deriv - 1)
        ret = gamlss_gH(
            X,
            jj,
            de["l1"],
            de["l2"],
            I2,
            l3=de["l3"],
            i3=I3,
            l4=de["l4"],
            i4=I4,
            d1b=d1b,
            d2b=d2b,
            deriv=deriv - 1,
            fh=fh,
            D=D,
            sandwich=sandwich,
        )
        ret["l"] = l
        ret["l0"] = l0
        return ret

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list,
        offset=None,
        weights=None,
        E=None,
    ) -> np.ndarray:
        """Starting values for shash.  Mirrors mgcv shash$initialize."""
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        p = X.shape[1]
        start = np.zeros(p, dtype=np.float64)

        # 1) Location: regress y on X1
        X1 = X[:, jj[0]]
        try:
            start1 = np.linalg.lstsq(X1, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            start1 = np.zeros(X1.shape[1], dtype=np.float64)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # 2) Log-scale: regress log|residuals| on X2
        mu_hat = X1 @ start1
        res = y - mu_hat
        log_abs_res = np.log(np.maximum(np.abs(res), 1e-7))
        X2 = X[:, jj[1]]
        try:
            start2 = np.linalg.lstsq(X2, log_abs_res, rcond=None)[0]
        except np.linalg.LinAlgError:
            start2 = np.zeros(X2.shape[1], dtype=np.float64)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # 3) Skewness: initialize eps near 0 (linkfun(0) = 0 for identity)
        X3 = X[:, jj[2]]
        yt3 = np.zeros(n, dtype=np.float64)
        try:
            start3 = np.linalg.lstsq(X3, yt3, rcond=None)[0]
        except np.linalg.LinAlgError:
            start3 = np.zeros(X3.shape[1], dtype=np.float64)
        start3 = np.where(np.isfinite(start3), start3, 0.0)
        start[jj[2]] = start3

        # 4) Log-kurtosis: initialize phi near 0 (linkfun(0) = 0 for identity)
        X4 = X[:, jj[3]]
        yt4 = np.zeros(n, dtype=np.float64)
        try:
            start4 = np.linalg.lstsq(X4, yt4, rcond=None)[0]
        except np.linalg.LinAlgError:
            start4 = np.zeros(X4.shape[1], dtype=np.float64)
        start4 = np.where(np.isfinite(start4), start4, 0.0)
        start[jj[3]] = start4

        return start


def shashlss(b: float = 0.01, phi_pen: float = 1e-3) -> ShashlssFamily:
    """
    Sinh-arcsinh location-scale-skewness-kurtosis family.

    Parameters
    ----------
    b : float
        Lower bound parameter for the logeb link on tau. Default 0.01.
    phi_pen : float
        L2 penalty weight on the log-kurtosis parameter phi. Default 1e-3.

    Returns
    -------
    ShashlssFamily instance.
    """
    return ShashlssFamily(b=b, phi_pen=phi_pen)
