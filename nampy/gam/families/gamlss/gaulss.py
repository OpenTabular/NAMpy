from __future__ import annotations

from typing import Any

import numpy as np

from ..._mgcv_constants import FAMILY_EPS
from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .._function_maps import InverseLink, LogLink, SqrtLink
from ._base import (
    GamlssFamily,
    _AdaptedLinkInfo,
    _IdentityLinkInfo,
    _pen_reg,
    _qr_coef_pivoted,
)


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
        return np.log(1.0 / mu - self.b)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        return 1.0 / (np.exp(eta) + self.b)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        ee = np.exp(eta)
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

        if not isinstance(link, (list, tuple)) or len(link) != 2:
            raise ValueError("gaulss link must be a length-2 list/tuple.")
        if not isinstance(link[0], str) or not isinstance(link[1], str):
            raise ValueError("gaulss links must be strings.")
        link1_name = link[0]
        link2_name = link[1]

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
            _lobj = LogLink(eps=FAMILY_EPS)
            linfo1 = _AdaptedLinkInfo(_lobj, link1_name)
        elif link1_name == "inverse":
            _lobj = InverseLink(eps=FAMILY_EPS)
            linfo1 = _AdaptedLinkInfo(_lobj, link1_name)
        elif link1_name == "sqrt":
            _lobj = SqrtLink(eps=FAMILY_EPS)
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

        eta_mat = self._eta_matrix_from_inputs(
            X,
            jj,
            coef,
            offset=offset,
            eta=kw.get("eta", None),
        )
        eta = np.asarray(eta_mat[:, 0], dtype=np.float64)
        eta1 = np.asarray(eta_mat[:, 1], dtype=np.float64)

        mu = self.linfo[0].linkinv(eta)  # mean
        tau = self.linfo[1].linkinv(eta1)  # precision 1/sigma

        ymu = y - mu
        ymu2 = ymu**2
        tau2 = tau**2

        # log-likelihood: N(mu, sigma^2) with sigma = 1/tau
        # l = -0.5*(y-mu)^2 * tau^2 - 0.5*log(2pi) + log(tau)
        l0 = -0.5 * ymu2 * tau2 - 0.5 * np.log(2.0 * np.pi) + np.log(tau)
        ll = float(np.sum(l0))

        if deriv == 0:
            return {
                "l": ll,
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
        ret["l"] = ll
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
        Initialize coefficients for gaulss by two `pen.reg()` solves.

        Mirrors the regular-matrix branch of mgcv `gaulss$initialize` in
        `mgcv/R/gamlss.r`.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        _n, p = X.shape
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

        use_unscaled = bool(E is not None and getattr(E, "use_unscaled", False))
        E_arr = None if E is None else np.asarray(E, dtype=np.float64)

        X1 = np.asarray(X[:, jj[0]], dtype=np.float64)
        yt1 = np.asarray(y, dtype=np.float64).copy()
        if self.linfo[0].name != "identity":
            yt1 = self.linfo[0].linkfun(np.abs(y) + np.max(y) * 1e-7)
        if off1 is not None:
            yt1 = yt1 - off1
        E1 = (
            np.zeros((0, X1.shape[1]), dtype=np.float64)
            if E_arr is None or E_arr.shape[1] == 0
            else np.asarray(E_arr[:, jj[0]], dtype=np.float64)
        )
        if use_unscaled and E1.shape[0] > 0:
            start1 = _qr_coef_pivoted(
                np.vstack([X1, E1]),
                np.concatenate([yt1, np.zeros(E1.shape[0], dtype=np.float64)]),
            )
        else:
            start1 = _pen_reg(X1, E1, yt1)
        start[jj[0]] = start1

        mu_init = self.linfo[0].linkinv(X1 @ start1)
        lres1 = np.log(np.abs(y - mu_init))
        if off2 is not None:
            lres1 = lres1 - off2

        X2 = np.asarray(X[:, jj[1]], dtype=np.float64)
        E2 = (
            np.zeros((0, X2.shape[1]), dtype=np.float64)
            if E_arr is None or E_arr.shape[1] == 0
            else np.asarray(E_arr[:, jj[1]], dtype=np.float64)
        )
        if use_unscaled and E2.shape[0] > 0:
            start2 = _qr_coef_pivoted(
                np.vstack([X2, E2]),
                np.concatenate([lres1, np.zeros(E2.shape[0], dtype=np.float64)]),
            )
        else:
            start2 = _pen_reg(X2, E2, lres1)
        start[jj[1]] = start2

        return start

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Standardized residuals (y - mu) / sigma = (y - mu) * tau.

        Mirrors mgcv ``gaulss$residuals``.
        """
        rtype = str(rtype).lower()
        if rtype not in {"deviance", "pearson", "response"}:
            raise ValueError(
                "gaulss residuals support only {'deviance', 'pearson', 'response'}."
            )
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        tau = np.asarray(fitted[:, 1], dtype=np.float64)
        rsd = y - mu
        if rtype == "response":
            return rsd
        return rsd * tau

    def null_deviance(
        self, y: np.ndarray, fitted: np.ndarray, prior_weights: np.ndarray
    ) -> float:
        """
        Mirror the gaulss ``postproc`` expression (mgcv/R/gamlss.r:910-918):
        ``sum(((y - mean(y)) * fitted[,2])^2)`` with ``fitted[,2] = tau``.
        Prior weights deliberately do not enter, as upstream.
        """
        del prior_weights
        y = np.asarray(y, dtype=np.float64)
        tau = np.asarray(fitted[:, 1], dtype=np.float64)
        return float(np.sum(((y - float(np.mean(y))) * tau) ** 2))

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
