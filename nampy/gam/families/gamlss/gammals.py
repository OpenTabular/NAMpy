from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted


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
        result: np.ndarray = np.where(
            x > 500.0,
            eta,
            self.b + np.log1p(np.exp(np.minimum(x, 500.0))),
        )
        return result

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
        result: np.ndarray = eta
        return result

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        """d linkinv / d eta = sigmoid(eta - b)."""
        eta = np.asarray(eta, dtype=np.float64)
        x = eta - self.b
        # numerically stable sigmoid
        pos = x >= 0.0
        ex = np.exp(-np.abs(x))
        result = np.where(pos, 1.0 / (1.0 + ex), ex / (1.0 + ex))
        output: np.ndarray = result
        return output

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        """d^2 eta / d mu^2.  Mirrors mgcv d2link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub = mu - self.b
        mub_v = np.exp(-mub * np.sign(mub))
        result: np.ndarray = -mub_v / (mub_v - 1.0) ** 2
        return result

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        """d^3 eta / d mu^3.  Mirrors mgcv d3link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub_raw = mu - self.b
        sm = -np.sign(mub_raw)
        mub_v = np.exp(mub_raw * sm)  # = exp(-|mu-b|)
        result: np.ndarray = sm * (mub_v + mub_v**2) / (mub_v - 1.0) ** 3
        return result

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        """d^4 eta / d mu^4.  Mirrors mgcv d4link for softplus-b."""
        mu = np.asarray(mu, dtype=np.float64)
        mub_raw = mu - self.b
        sm = -np.sign(mub_raw)
        mub_v = np.exp(mub_raw * sm)
        result: np.ndarray = (
            sm * (mub_v + 4.0 * mub_v**2 + mub_v**3) / (mub_v - 1.0) ** 4
        )
        return result


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
    parameter_names = ("mu", "sigma")

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

        linfo1: Any = _IdentityLinkInfo()
        linfo2: Any
        if link2_name == "log":
            linfo2 = _SoftplusBLinkInfo(b=b)
        else:
            linfo2 = _IdentityLinkInfo()

        self.linfo = [linfo1, linfo2]
        self.tri = trind_generator(2)
        self.link_names = (link1_name, link2_name)
        self.link_name = f"({link1_name}, {link2_name})"

    def distribution_parameters_from_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim != 2 or eta.shape[1] != 2:
            raise ValueError(f"gammals expected eta with shape (n, 2), got {eta.shape}.")
        mu = np.exp(eta[:, 0])
        log_dispersion = np.asarray(
            self.linfo[1].linkinv(eta[:, 1]), dtype=np.float64
        )
        sigma = np.exp(0.5 * log_dispersion)
        return np.column_stack([mu, sigma])

    def distribution_parameter_jacobian(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim != 2 or eta.shape[1] != 2:
            raise ValueError(f"gammals expected eta with shape (n, 2), got {eta.shape}.")
        mu = np.exp(eta[:, 0])
        log_dispersion = np.asarray(
            self.linfo[1].linkinv(eta[:, 1]), dtype=np.float64
        )
        sigma = np.exp(0.5 * log_dispersion)
        d_log_dispersion = np.asarray(
            self.linfo[1].mu_eta(eta[:, 1]), dtype=np.float64
        )
        return np.column_stack([mu, 0.5 * sigma * d_log_dispersion])

    def logpdf_from_parameters(
        self, y: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        params = np.asarray(parameters, dtype=np.float64)
        if params.shape != (y.size, 2):
            raise ValueError(
                f"gammals parameters must have shape ({y.size}, 2), got {params.shape}."
            )
        mu = params[:, 0]
        sigma = params[:, 1]
        if np.any(y <= 0.0):
            raise ValueError("gammals requires strictly positive response y > 0.")
        if (
            np.any(~np.isfinite(mu))
            or np.any(mu <= 0.0)
            or np.any(~np.isfinite(sigma))
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("gammals mu and sigma parameters must be finite and > 0.")
        dispersion = np.square(sigma)
        shape = 1.0 / dispersion
        scale = mu * dispersion
        return (
            (shape - 1.0) * np.log(y)
            - y / scale
            - gammaln(shape)
            - shape * np.log(scale)
        )

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

        eta_mat = self._eta_matrix_from_inputs(
            X,
            jj,
            coef,
            offset=offset,
            eta=kw.get("eta", None),
        )
        eta = np.asarray(eta_mat[:, 0], dtype=np.float64)
        etat = np.asarray(eta_mat[:, 1], dtype=np.float64)

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
        ll = float(np.sum(l0))

        if deriv == 0:
            return {"l": ll, "l0": l0}

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
        Initialize coefficients for gammals.

        Regresses X1 on log(y) for the log-mean predictor.
        Then regresses X2 on log|residuals| transformed through link2
        for the log-sigma predictor.

        Mirrors mgcv ``gammals$initialize`` regular matrix path.
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

        eps = np.max(y) * np.finfo(np.float64).eps ** 0.75
        use_unscaled = bool(E is not None and getattr(E, "use_unscaled", False))

        # --- Fit log-mean predictor on log(y) ---
        yt1 = np.log(y + eps)
        if off1 is not None:
            yt1 = yt1 - off1

        X1 = X[:, jj[0]]
        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            if use_unscaled:
                XE1 = np.vstack([X1, E1])
                y1e = np.concatenate([yt1, np.zeros(E1.shape[0])])
                start1 = _qr_coef_pivoted(XE1, y1e)
            else:
                start1 = _pen_reg(X1, E1, yt1)
        else:
            start1 = _qr_coef_pivoted(X1, yt1)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # --- Fit log-sigma predictor on transformed residuals ---
        lres1 = self.linfo[1].linkfun(
            np.log(np.abs(y - self.linfo[0].linkinv(X1 @ start1)))
        )
        if off2 is not None:
            lres1 = lres1 - off2

        X2 = X[:, jj[1]]
        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            if use_unscaled:
                XE2 = np.vstack([X2, E2])
                y2e = np.concatenate([lres1, np.zeros(E2.shape[0])])
                start2 = _qr_coef_pivoted(XE2, y2e)
            else:
                start2 = _pen_reg(X2, E2, lres1)
        else:
            start2 = _qr_coef_pivoted(X2, lres1)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        return start

    def residuals(
        self,
        y: np.ndarray,
        fitted: np.ndarray,
        rtype: str = "deviance",
        *,
        eta: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Residuals for gammals.  Mirrors mgcv ``gammals$residuals``."""
        del eta, kwargs
        rtype = str(rtype).lower()
        if rtype not in {"deviance", "pearson", "response"}:
            raise ValueError(
                "gammals residuals support only {'deviance', 'pearson', 'response'}."
            )
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        rho = np.asarray(fitted[:, 1], dtype=np.float64)  # log sigma
        if rtype == "deviance":
            rsd = 2.0 * ((y - mu) / mu - np.log(y / mu)) * np.exp(-rho)
            result: np.ndarray = np.sqrt(np.maximum(0.0, rsd)) * np.sign(y - mu)
            return result
        elif rtype == "pearson":
            result = (y - mu) / (np.exp(rho * 0.5) * mu)
            return result
        else:
            return y - mu

    def null_deviance(
        self, y: np.ndarray, fitted: np.ndarray, prior_weights: np.ndarray
    ) -> float:
        """
        Mirror the gammals ``postproc`` expression (mgcv/R/gamlss.r:2737-2742):
        ``2 * sum(((y - my)/my - log(y/my)) * exp(-fitted[,2]))`` with
        ``my = mean(y)`` and ``fitted[,2] = rho`` (log sigma). NAMpy already
        stores ``fitted[,1]`` exponentiated, matching the post-postproc state.
        """
        del prior_weights
        y = np.asarray(y, dtype=np.float64)
        rho = np.asarray(fitted[:, 1], dtype=np.float64)
        my = float(np.mean(y))
        return float(
            2.0 * np.sum(((y - my) / my - np.log(y / my)) * np.exp(-rho))
        )

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
