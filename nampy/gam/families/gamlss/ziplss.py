"""Two-predictor hurdle Poisson family, mgcv R/gamlss.r::ziplss."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import poisson

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .._zip_helpers import zipll
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted


class ZiplssFamily(GamlssFamily):
    """Log Poisson intensity and complementary-log-log presence probability."""

    name = "ziplss"
    nlp = n_linear_predictors = 2
    parameter_names = ("lambda", "presence")
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, link=("identity", "identity")):
        super().__init__()
        if not isinstance(link, (tuple, list)) or tuple(link) != ("identity", "identity"):
            raise ValueError("ziplss requires two identity links.")
        self.linfo = [_IdentityLinkInfo(), _IdentityLinkInfo()]
        self.link_names = tuple(link)
        self.link_name = "(identity, identity)"
        self.tri = trind_generator(2)

    def validate_y(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError("ziplss requires finite nonnegative counts.")
        if not np.allclose(y, np.round(y), rtol=np.sqrt(np.finfo(float).eps), atol=0):
            raise ValueError("Non-integer response variables are not allowed with ziplss.")
        if np.min(y) == 0 and np.max(y) == 1:
            raise ValueError("Using ziplss for binary data makes no sense.")
        return y

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0, d1b=0, d2b=0, fh=None, D=None, **kw):
        y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=offset, eta=kw.get("eta"))
        zl = zipll(y, eta[:, 0], eta[:, 1], deriv)
        ret = {"l": float(np.sum(zl["l"])), "l0": zl["l"]}
        if not deriv:
            return ret
        ones, zeros = np.ones((len(y), 2)), np.zeros((len(y), 2))
        de = gamlss_etamu(zl["l1"], zl["l2"], zl["l3"] if deriv > 1 else 0, zl["l4"] if deriv > 3 else 0, ones, zeros, zeros if deriv > 1 else 0, zeros if deriv > 3 else 0, self.tri["i2"], self.tri["i3"], self.tri["i4"], deriv - 1)
        out = gamlss_gH(X, jj, de["l1"], de["l2"], self.tri["i2"], l3=de["l3"], i3=self.tri["i3"], l4=de["l4"], i4=self.tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1, fh=fh, D=D, sandwich=bool(kw.get("sandwich", False)))
        if kw.get("ncv", False):
            out.update({key: de[key] for key in ("l1", "l2", "l3")})
        out.update(ret)
        return out

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
        self.validate_y(y)
        E = np.zeros((0, X.shape[1])) if E is None else E
        use_unscaled = bool(getattr(E, "use_unscaled", False))

        def regress(k, target, w=None):
            x, e = X[:, jj[k]], E[:, jj[k]]
            if w is not None:
                x = w[:, None] * x
            if use_unscaled:
                return _qr_coef_pivoted(np.vstack([x, e]), np.concatenate([target, np.zeros(E.shape[0])]))
            return _pen_reg(x, e, target)

        start = np.zeros(X.shape[1])
        start[jj[1]] = regress(1, (y != 0).astype(float))
        p = X[:, jj[1]] @ start[jj[1]]
        w = np.where((y == 0) & (p < .5), .1, 1.)
        target = np.log(np.abs(y) + (y == 0) * .2) * w
        start[jj[0]] = regress(0, target, w)
        return start

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.column_stack([np.exp(eta[:, 0]), 1 - np.exp(-np.exp(eta[:, 1]))])

    def distribution_parameter_jacobian(self, eta):
        eta = np.asarray(eta, dtype=float)
        et = np.exp(eta[:, 1])
        with np.errstate(invalid="ignore", over="ignore"):
            dp = np.exp(-et) * et
        dp[eta[:, 1] >= np.log(np.finfo(float).max) / 2] = 0
        return np.column_stack([np.exp(eta[:, 0]), dp])

    def point_prediction_from_parameters(self, parameters):
        params = np.asarray(parameters, dtype=float)
        lam, p = params[:, 0], params[:, 1]
        result = p.copy()
        ind = lam > np.sqrt(np.finfo(float).eps)
        result[ind] *= lam[ind] / (1 - np.exp(-lam[ind]))
        return result

    def logpdf_from_parameters(self, y, parameters):
        y = np.asarray(y, dtype=float).ravel()
        params = np.asarray(parameters, dtype=float)
        lam, p = params[:, 0], params[:, 1]
        if np.any(~np.isfinite(params)) or np.any(lam <= 0) or np.any((p < 0) | (p > 1)):
            raise ValueError("ziplss requires positive intensity and presence in [0,1].")
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.log1p(-p)
            ind = y > 0
            value[ind] = np.log(p[ind]) + y[ind] * np.log(lam[ind]) - lam[ind] - np.log(-np.expm1(-lam[ind])) - gammaln(y[ind] + 1)
        value[(y < 0) | (y != np.floor(y))] = -np.inf
        return value

    def predict_fitted(self, X, jj, coef, offset=None):
        # ziplss stores latent predictors as fitted.values; its optional
        # prediction hook instead returns the marginal expectation.
        return self._stacked_eta(X, jj, coef, offset)

    def _predict_response_from_eta(self, eta):
        return self.point_prediction_from_parameters(self.distribution_parameters_from_eta(eta))

    def predict(self, *, eta=None, X=None, jj=None, coef=None, offset=None, se=False, Vb=None):
        eta = self._stacked_eta(X, jj, coef, offset) if eta is None else np.asarray(eta, dtype=float)
        fit = self._predict_response_from_eta(eta)
        if not se or X is None:
            return fit
        if Vb is None:
            raise ValueError("Vb is required when se=True.")
        x1, x2 = X[:, jj[0]], X[:, jj[1]]
        vg = np.maximum(np.einsum("ij,jk,ik->i", x1, Vb[np.ix_(jj[0], jj[0])], x1), 0)
        # Preserve ziplss$predict's source second variance and nonnegative
        # cross covariance, including its repeated first-predictor block.
        ve = vg
        veg = np.maximum(np.einsum("ij,jk,ik->i", x1, Vb[np.ix_(jj[0], jj[1])], x2), 0)
        params = self.distribution_parameters_from_eta(eta)
        lam, p = params[:, 0], params[:, 1]
        conditional = np.ones(len(lam))
        ind = eta[:, 0] >= np.log(np.finfo(float).eps) / 2
        conditional[ind] = lam[ind] / (1 - np.exp(-lam[ind]))
        de = self.distribution_parameter_jacobian(eta)[:, 1] * conditional
        dg = ((lam + 1) * conditional - conditional**2) * p
        return fit, np.sqrt(dg**2 * vg + de**2 * ve + 2 * de * dg * veg)

    @staticmethod
    def saturated_loglik(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(y)
        g = y.copy()
        ind = (y > 1) & (y < 18)
        values = np.array([1.593624, 2.821439, 3.920690, 4.965114, 5.984901, 6.993576, 7.997309, 8.998888, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995, 15.999998, 16.999999])
        g[ind] = values[y[ind].astype(int) - 2]
        ind = y > 1
        out[ind] = zipll(y[ind], np.log(g[ind]), np.full(np.sum(ind), 1e10))["l"]
        return out

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, **kwargs):
        y, fitted = np.asarray(y, dtype=float), np.asarray(fitted, dtype=float)
        raw = y - self._predict_response_from_eta(fitted)
        if rtype == "response":
            return raw
        if rtype != "deviance":
            raise ValueError("ziplss residuals support only deviance and response.")
        return np.sqrt(np.maximum(0, 2 * (self.saturated_loglik(y) - zipll(y, fitted[:, 0], fitted[:, 1])["l"]))) * np.sign(raw)

    def null_deviance(self, y, fitted, prior_weights):
        y = np.asarray(y, dtype=float)
        tol = np.finfo(float).eps**.25
        positive = y[y > 0]

        def fp(p):
            l1p = np.log(1 - p) if p > np.sqrt(np.finfo(float).eps) else -p - p**2 / 2
            return -(l1p * np.sum(y == 0) + np.log(p) * len(positive))

        def flam(lam):
            return -np.sum(positive * np.log(lam) - np.log(np.exp(lam) - 1) - gammaln(positive + 1))

        pfit = minimize_scalar(fp, bounds=(1e-60, 1 - 1e-10), method="bounded", options={"xatol": tol})
        mean = float(np.mean(positive))
        lfit = minimize_scalar(flam, bounds=(mean / 2, mean * 2), method="bounded", options={"xatol": tol})
        return float(2 * (np.sum(self.saturated_loglik(y)) + pfit.fun + lfit.fun))

    def rd(self, mu, wt, scale, *, rng):
        mu = np.asarray(mu, dtype=float)
        lam = np.exp(mu[:, 0])
        p = 1 - np.exp(-np.exp(mu[:, 1]))
        present = p > rng.uniform(size=len(lam))
        out = np.zeros(len(lam))
        u = rng.uniform(np.exp(-lam[present]), 1)
        u = np.minimum(u, 1 - np.finfo(float).eps**.75)
        out[present] = poisson.ppf(u, lam[present])
        return out
