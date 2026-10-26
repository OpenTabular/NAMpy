"""Tweedie location/scale/shape family, mgcv R/gamlss.r::twlss."""

import numpy as np

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .._function_maps import LogLink, SqrtLink
from ..tweedie import _tw_dp_dth, _tw_p_from_theta, ldTweedie
from ._base import (
    GamlssFamily,
    _AdaptedLinkInfo,
    _IdentityLinkInfo,
    _pen_reg,
    _qr_coef_pivoted,
)


def _tw_null_fit(y, a=1.001, b=1.999):
    """Direct stabilized Newton port of R/gamlss.r::tw.null.fit."""
    y = np.asarray(y, dtype=float)
    th = np.zeros(3)

    def density(value):
        return ldTweedie(y, np.full(y.size, np.exp(value[0])),
                         rho=value[2], theta=value[1], a=a, b=b, all_derivs=True)

    lds = np.sum(density(th), axis=0)
    for _ in range(50):
        g = lds[[6, 3, 1]].copy()
        if not np.any(np.abs(g) > 1e-9 * abs(lds[0])):
            break
        g[0] *= np.exp(th[0])
        H = np.diag(lds[[7, 4, 2]])
        H[0, 1] = H[1, 0] = lds[8]
        H[0, 2] = H[2, 0] = lds[9]
        H[1, 2] = H[2, 1] = lds[5]
        H[:, 0] *= np.exp(th[0])
        H[0, 1:] *= np.exp(th[0])
        eigenvalues, vectors = np.linalg.eigh(H)
        eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
        tol = np.max(np.abs(eigenvalues)) * 1e-7
        eigenvalues[eigenvalues > -tol] = -tol
        step = vectors @ ((vectors.T @ g) / eigenvalues)
        ms = np.max(np.abs(step))
        if ms > 3:
            step = step * 3 / ms
        while True:
            th1 = th - step
            ld1 = density(th1)
            if np.sum(ld1[:, 0]) < lds[0]:
                step = step / 2
            else:
                th, lds = th1, np.sum(ld1, axis=0)
                break
    return np.array([np.exp(th[0]), float(_tw_p_from_theta(np.array([th[1]]), a, b)[0]), np.exp(th[2])])


class TwlssFamily(GamlssFamily):
    name = "twlss"
    nlp = n_linear_predictors = 3
    parameter_names = ("mu", "p", "phi")
    supports_gcv = supports_ubre = supports_pirls = False
    supports_analytic_outer_derivatives = False
    supports_analytic_outer_gradient = supports_analytic_outer_hessian = False
    max_derivative_order = 2
    available_derivs = 0

    def __init__(self, link=("log", "identity", "identity"), a=1.01, b=1.99):
        super().__init__()
        if not isinstance(link, (tuple, list)) or len(link) != 3:
            raise ValueError("twlss requires three links.")
        if link[0] not in {"log", "identity", "sqrt"} or tuple(link[1:]) != ("identity", "identity"):
            raise ValueError("twlss requires log/identity/sqrt mean and two identity links.")
        self.a, self.b = float(a), float(b)
        if not 1 < self.a < self.b < 2:
            raise ValueError("twlss requires 1 < a < b < 2.")
        mean_link = _IdentityLinkInfo() if link[0] == "identity" else _AdaptedLinkInfo(
            (LogLink if link[0] == "log" else SqrtLink)(eps=np.finfo(float).eps), link[0])
        self.linfo = [mean_link, _IdentityLinkInfo(), _IdentityLinkInfo()]
        self.tri = trind_generator(3)
        self.link_names, self.link_name = tuple(link), str(tuple(link))

    def validate_y(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("twlss requires finite nonnegative responses.")
        return y

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.ndim != 2 or eta.shape[1] != 3:
            raise ValueError("twlss eta must have shape (n, 3).")
        return np.column_stack((self.linfo[0].linkinv(eta[:, 0]),
                                _tw_p_from_theta(eta[:, 1], self.a, self.b),
                                np.exp(eta[:, 2])))

    def distribution_parameter_jacobian(self, eta):
        eta = np.asarray(eta, dtype=float)
        self.distribution_parameters_from_eta(eta)
        dp, _ = _tw_dp_dth(eta[:, 1], self.a, self.b)
        return np.column_stack((self.linfo[0].mu_eta(eta[:, 0]), dp, np.exp(eta[:, 2])))

    def logpdf_from_parameters(self, y, parameters):
        y, params = np.asarray(y, dtype=float).ravel(), np.asarray(parameters, dtype=float)
        if params.shape != (y.size, 3):
            raise ValueError("twlss parameters must have shape (n, 3).")
        if not np.isfinite(params).all() or np.any(params[:, [0, 2]] <= 0) or np.any((params[:, 1] < self.a) | (params[:, 1] > self.b)):
            raise ValueError("twlss requires positive mean/dispersion and bounded p.")
        return ldTweedie(y, params[:, 0], p=params[:, 1], phi=params[:, 2], a=self.a, b=self.b)[:, 0]

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0,
           d1b=0, d2b=0, fh=None, D=None, **kw):
        del weights, d1b, d2b, fh, D
        # Source available.derivs=0: its ll ignores higher derivative requests
        # and contracts analytic first/second likelihood derivatives only.
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=offset, eta=kw.get("eta"))
        mu = self.linfo[0].linkinv(eta[:, 0])
        ld = ldTweedie(y, mu, rho=eta[:, 2], theta=eta[:, 1], a=self.a, b=self.b, all_derivs=True)
        ret = {"l": float(np.sum(ld[:, 0])), "l0": ld[:, 0]}
        if not deriv:
            return ret
        l1 = ld[:, [6, 3, 1]]
        l2 = ld[:, [7, 8, 9, 4, 5, 2]]
        ig1 = np.column_stack([self.linfo[k].mu_eta(eta[:, k]) for k in range(3)])
        g2 = np.column_stack((self.linfo[0].d2link(mu), np.zeros((len(mu), 2))))
        de = gamlss_etamu(l1, l2, 0, 0, ig1, g2, 0, 0,
                          self.tri["i2"], self.tri["i3"], self.tri["i4"], 0)
        ret.update(gamlss_gH(X, jj, de["l1"], de["l2"], self.tri["i2"],
                             sandwich=kw.get("sandwich", False)))
        return ret

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        del offset, weights
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        E = np.zeros((0, X.shape[1])) if E is None else E
        use_unscaled = bool(getattr(E, "use_unscaled", False))
        start = np.zeros(X.shape[1])

        def regress(k, response):
            Xi, Ei = X[:, jj[k]], E[:, jj[k]]
            if use_unscaled:
                return _qr_coef_pivoted(np.vstack((Xi, Ei)), np.r_[response, np.zeros(E.shape[0])])
            return _pen_reg(Xi, Ei, response)

        yt1 = y if self.link_names[0] == "identity" else self.linfo[0].linkfun(np.abs(y) + np.max(y) * 1e-7)
        start[jj[0]] = regress(0, yt1)
        mu = self.linfo[0].linkinv(X[:, jj[0]] @ start[jj[0]])
        start[jj[2]] = regress(2, np.log(np.abs((y - mu) / mu**1.5)))
        return start

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, weights=None, **kwargs):
        del eta, kwargs
        fitted, y = np.asarray(fitted), np.asarray(y)
        mu = fitted[:, 0]
        p = _tw_p_from_theta(fitted[:, 1], self.a, self.b)
        phi = np.exp(fitted[:, 2])
        if rtype == "response":
            return y - mu
        if rtype == "pearson":
            return (y - mu) / np.sqrt(phi * mu**p)
        if rtype != "deviance":
            raise ValueError("twlss residual type must be deviance, pearson or response.")
        weights = np.ones(y.size) if weights is None else np.asarray(weights)
        y1 = y + (y == 0)
        theta = (y1**(1 - p) - mu**(1 - p)) / (1 - p)
        kappa = (y**(2 - p) - mu**(2 - p)) / (2 - p)
        return np.sign(y - mu) * np.sqrt(np.maximum(2 * (y * theta - kappa) * weights / phi, 0))

    def null_deviance(self, y, fitted, prior_weights):
        y, fitted = np.asarray(y), np.asarray(fitted)
        # twlss postproc deliberately calls tw.null.fit with its own defaults,
        # and divides the result by each fitted observation's dispersion.
        null_mu, null_p, _ = _tw_null_fit(y)
        y1 = y + (y == 0)
        theta = (y1**(1 - null_p) - null_mu**(1 - null_p)) / (1 - null_p)
        kappa = (y**(2 - null_p) - null_mu**(2 - null_p)) / (2 - null_p)
        return float(np.sum(np.maximum(2 * (y * theta - kappa) * prior_weights / np.exp(fitted[:, 2]), 0)))


def twlss(link=("log", "identity", "identity"), a=1.01, b=1.99):
    return TwlssFamily(link=link, a=a, b=b)
