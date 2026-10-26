"""Sinh-arcsinh location/scale/shape family, mgcv R/gamlss.r::shash."""

import numpy as np
from scipy.special import kv, log_ndtr, ndtr, ndtri

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted
from ._shash_derivatives import shash_fourth


class _LogEBLinkInfo:
    name = "logeb"

    def __init__(self, b):
        self.b = float(b)

    def linkfun(self, mu):
        return np.log(np.exp(np.asarray(mu)) - self.b)

    def linkinv(self, eta):
        return np.log(np.exp(np.asarray(eta)) + self.b)

    def mu_eta(self, eta):
        ee = np.exp(np.asarray(eta))
        return ee / (ee + self.b)

    def d2link(self, mu):
        em = np.exp(np.asarray(mu))
        fr = em / (em - self.b)
        return fr * (1 - fr)

    def d3link(self, mu):
        em = np.exp(np.asarray(mu))
        fr = em / (em - self.b)
        oo = fr * (1 - fr)
        return oo - 2 * oo * fr

    def d4link(self, mu):
        em = np.exp(np.asarray(mu))
        b = self.b
        return -b * em * (b**2 + 4 * b * em + em**2) / (em - b)**4


def _sqrt_x2pm(x, m):
    x = np.abs(np.asarray(x, dtype=float)).copy()
    small = x < 1e8
    x[small] = np.sqrt(x[small]**2 + m)
    return x


def _ax2m1_div_x2m2_sq(x, m1, m2, a=1):
    x = np.abs(np.asarray(x, dtype=float))
    negative = a * x**2 + m1 < 0
    out = np.zeros_like(x)
    out[negative] = (a * x[negative]**2 + m1) / (x[negative]**2 + m2)**2
    out[~negative] = ((_sqrt_x2pm(np.sqrt(a) * x[~negative], m1)
                       / _sqrt_x2pm(x[~negative], m2))
                      / _sqrt_x2pm(x[~negative], m2))**2
    return out


def _log1pexp(x):
    x = np.asarray(x, dtype=float).copy()
    # Source .bincode(..., right=TRUE) does not include the lower -Inf bound.
    first, second, third = (x > -np.inf) & (x <= -37), (x > -37) & (x <= 18), (x > 18) & (x <= 33.3)
    x[first] = np.exp(x[first])
    x[second] = np.log1p(np.exp(x[second]))
    x[third] = x[third] + np.exp(-x[third])
    return x


class ShashFamily(GamlssFamily):
    name = "shash"
    nlp = n_linear_predictors = 4
    parameter_names = ("mu", "sigma", "epsilon", "delta")
    supports_gcv = supports_ubre = supports_pirls = False
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = supports_analytic_outer_hessian = True
    max_derivative_order = 4
    available_derivs = 2

    def __init__(self, link=("identity", "logeb", "identity", "identity"),
                 b=1e-2, phi_pen=1e-3, phiPen=None):
        super().__init__()
        if not isinstance(link, (tuple, list)) or tuple(link) != ("identity", "logeb", "identity", "identity"):
            raise ValueError("shash requires identity/logeb/identity/identity links.")
        self.b = float(b)
        self.phi_pen = float(phi_pen if phiPen is None else phiPen)
        if not np.isfinite(self.b) or self.b <= 0:
            raise ValueError("shash b must be finite and positive.")
        if not np.isfinite(self.phi_pen) or self.phi_pen < 0:
            raise ValueError("shash phi_pen must be finite and nonnegative.")
        self.linfo = [_IdentityLinkInfo(), _LogEBLinkInfo(b),
                      _IdentityLinkInfo(), _IdentityLinkInfo()]
        self.link_names = tuple(link)
        self.link_name = str(self.link_names)
        self.tri = trind_generator(4)

    def validate_y(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("shash requires finite responses.")
        return y

    @staticmethod
    def _parameters(parameters):
        params = np.asarray(parameters, dtype=float)
        if params.ndim != 2 or params.shape[1] != 4:
            raise ValueError("shash parameters must have shape (n, 4).")
        if not np.isfinite(params).all() or np.any(params[:, [1, 3]] <= 0):
            raise ValueError("shash parameters must be finite with positive sigma and delta.")
        return params

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        if eta.ndim != 2 or eta.shape[1] != 4:
            raise ValueError("shash eta must have shape (n, 4).")
        return np.column_stack((eta[:, 0], np.exp(eta[:, 1]) + self.b,
                                eta[:, 2], np.exp(eta[:, 3])))

    def distribution_parameter_jacobian(self, eta):
        eta = np.asarray(eta, dtype=float)
        self.distribution_parameters_from_eta(eta)
        return np.column_stack((np.ones(eta.shape[0]), np.exp(eta[:, 1]),
                                np.ones(eta.shape[0]), np.exp(eta[:, 3])))

    def logpdf_from_parameters(self, y, parameters):
        mu, sigma, epsilon, delta = self._parameters(parameters).T
        y = np.asarray(y, dtype=float).ravel()
        z = (y - mu) / (sigma * delta)
        transformed = delta * np.arcsinh(z) - epsilon
        # The normalized density uses the literal source residual/Rd formula;
        # the fitting hook below separately preserves its .log1pexp anomaly.
        return (-np.log(sigma) - .5 * np.log(2 * np.pi)
                + np.log(np.cosh(transformed)) - .5 * np.log1p(z**2)
                - .5 * np.sinh(transformed)**2)

    def point_prediction_from_parameters(self, parameters):
        # The source quantile at probability 1/2 defines the conditional median.
        mu, sigma, epsilon, delta = self._parameters(parameters).T
        return mu + (delta * sigma) * np.sinh(epsilon / delta)

    def qf(self, p, mu, wt=None, scale=1):
        del wt, scale
        mu, tau, epsilon, phi = np.atleast_2d(np.asarray(mu, dtype=float)).T
        sigma, delta = np.exp(tau), np.exp(phi)
        return mu + (delta * sigma) * np.sinh(np.arcsinh(ndtri(p)) / delta + epsilon / delta)

    def cdf(self, q, mu, wt=None, scale=1, logp=False):
        del wt, scale
        mu, tau, epsilon, phi = np.atleast_2d(np.asarray(mu, dtype=float)).T
        sigma, delta = np.exp(tau), np.exp(phi)
        value = np.sinh((np.arcsinh((np.asarray(q) - mu) / (delta * sigma))
                         - epsilon / delta) * delta)
        return log_ndtr(value) if logp else ndtr(value)

    def rd(self, mu, wt=None, scale=1, *, rng=None):
        generator = np.random.default_rng() if rng is None else rng
        return self.qf(generator.uniform(size=np.atleast_2d(mu).shape[0]), mu, wt, scale)

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0,
           d1b=0, d2b=0, fh=None, D=None, **kw):
        del weights
        off = self._offset_list(offset)
        if any(value is not None and np.any(np.asarray(value) != 0) for value in off):
            raise ValueError("offset not still available for this family")
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=None, eta=kw.get("eta"))
        mu, tau, eps, phi = [self.linfo[k].linkinv(eta[:, k]) for k in range(4)]
        sig, delta = np.exp(tau), np.exp(phi)
        y = np.asarray(y, dtype=float)
        z = (y - mu) / (sig * delta)
        g = -(delta * np.arcsinh(z) - eps)
        with np.errstate(divide="ignore"):
            l0 = (-tau - .5 * np.log(2 * np.pi) + np.log(np.cosh(g))
                  - .5 * _log1pexp(2 * np.log(np.abs(z))) - .5 * np.sinh(g)**2
                  - self.phi_pen * phi**2)
        ret = {"l": float(np.sum(l0)), "l0": l0}
        if not deriv:
            return ret
        zsd = z * sig * delta
        sSp1, asinhZ = _sqrt_x2pm(z, 1), np.arcsinh(z)
        sech = lambda x: 1 / np.cosh(x)  # noqa: E731
        De = np.tanh(g) - .5 * np.sinh(2 * g)
        Dm = 1 / (delta * sig * sSp1) * (delta * De + z / sSp1)
        Dt = zsd * Dm - 1
        Dp = Dt + 1 - delta * asinhZ * De - 2 * self.phi_pen * phi
        L1 = np.column_stack((Dm, Dt, De, Dp))
        Dme = (sech(g)**2 - np.cosh(2 * g)) / (sig * sSp1)
        Dte = zsd * Dme
        Dmm = Dme / (sig * sSp1) + z * De / (sig**2 * delta * sSp1**3) + _ax2m1_div_x2m2_sq(z, -1, 1) / (delta * sig * delta * sig)
        Dmt = zsd * Dmm - Dm
        Dee = -2 * np.cosh(g)**2 + sech(g)**2 + 1
        Dtt = zsd * Dmt
        Dep = Dte - delta * asinhZ * Dee
        Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
        Dtp = zsd * Dmp
        Dpp = Dtp - delta * asinhZ * Dep + delta * (z / sSp1 - asinhZ) * De - 2 * self.phi_pen
        L2 = np.column_stack((Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee, Dep, Dpp))
        L3 = L4 = G3 = G4 = 0
        if deriv > 1:
            Deee = -2 * (np.sinh(2 * g) + sech(g)**2 * np.tanh(g))
            Dmee = Deee / (sig * sSp1)
            Dmme = Dmee / (sig * sSp1) + z * Dee / (sig * sig * delta * sSp1**3)
            Dmmm = 2 * z * Dme / (sig * sig * delta * sSp1**3) + Dmme / (sig * sSp1) + _ax2m1_div_x2m2_sq(z, -1, 1, 2) * De / (sig**3 * delta**2 * sSp1) + 2 * (z / sSp1) * _ax2m1_div_x2m2_sq(z, -3, 1) / ((sig * delta)**3 * sSp1)
            Dmmt = zsd * Dmmm - 2 * Dmm
            Dtee = zsd * Dmee
            Dmte = zsd * Dmme - Dme
            Dtte = zsd * Dmte
            Dmtt = zsd * Dmmt - Dmt
            Dttt = zsd * Dmtt
            Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
            Dtep = zsd * Dmep
            Deep = Dtee - delta * asinhZ * Deee
            Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
            Dmmp = Dmmt + 2 * Dme / (sig * sSp1) + z * De / (delta * sig * sig * sSp1**3) - delta * asinhZ * Dmme
            Dmtp = zsd * Dmmp - Dmp
            Dttp = zsd * Dmtp
            Dmpp = Dmtp + Dep / (sig * sSp1) + z**2 * De / (sig * sSp1**3) - delta * asinhZ * Dmep + delta * Dme * (z / sSp1 - asinhZ)
            Dtpp = zsd * Dmpp
            Dppp = Dtpp - delta * asinhZ * Depp + delta * (z / sSp1 - asinhZ) * (2 * Dep + De) + delta * (z / sSp1)**3 * De
            L3 = np.column_stack((Dmmm, Dmmt, Dmme, Dmmp, Dmtt, Dmte, Dmtp, Dmee, Dmep, Dmpp, Dttt, Dtte, Dttp, Dtee, Dtep, Dtpp, Deee, Deep, Depp, Dppp))
            G3 = np.column_stack([self.linfo[k].d3link(value) for k, value in enumerate((mu, tau, eps, phi))])
        if deriv > 3:
            L4 = shash_fourth(y, mu, tau, eps, phi)
            G4 = np.column_stack([self.linfo[k].d4link(value) for k, value in enumerate((mu, tau, eps, phi))])
        IG1 = np.column_stack([self.linfo[k].mu_eta(eta[:, k]) for k in range(4)])
        G2 = np.column_stack([self.linfo[k].d2link(value) for k, value in enumerate((mu, tau, eps, phi))])
        i2, i3, i4 = (self.tri[f"i{k}"] for k in (2, 3, 4))
        de = gamlss_etamu(L1, L2, L3, L4, IG1, G2, G3, G4, i2, i3, i4, deriv - 1)
        ret.update(gamlss_gH(X, jj, de["l1"], de["l2"], i2,
                             l3=de["l3"], i3=i3, l4=de["l4"], i4=i4,
                             d1b=d1b, d2b=d2b, deriv=deriv - 1, fh=fh, D=D,
                             sandwich=kw.get("sandwich", False)))
        if kw.get("ncv", False):
            ret.update({key: de[key] for key in ("l1", "l2", "l3", "l4")})
        return ret

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        del offset, weights
        y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
        E = np.zeros((0, X.shape[1])) if E is None else E
        start = np.zeros(X.shape[1])
        use_unscaled = bool(getattr(E, "use_unscaled", False))

        def regress(k, response):
            Xi, Ei = X[:, jj[k]], E[:, jj[k]]
            if use_unscaled:
                return _qr_coef_pivoted(np.vstack((Xi, Ei)), np.r_[response, np.zeros(E.shape[0])])
            return _pen_reg(Xi, Ei, response)

        start[jj[0]] = regress(0, y)
        residual = y - self.linfo[0].linkinv(X[:, jj[0]] @ start[jj[0]])
        start[jj[1]] = regress(1, np.log(np.abs(residual)))
        # Identity skewness/log-tailweight links initialize to Gaussian zero.
        return start

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, **kwargs):
        del eta, kwargs
        mu, tau, epsilon, phi = np.asarray(fitted, dtype=float).T
        sigma, delta = np.exp(tau), np.exp(phi)
        # Preserve the source residual centering, including its epsilon-independent
        # Bessel expression; it is not the density's conditional mean.
        rsd = (np.asarray(y) - mu - sigma * delta * np.exp(.25)
               * (kv((1 / delta + 1) / 2, .25) + kv((1 / delta - 1) / 2, .25))
               / np.sqrt(8 * np.pi))
        if rtype == "response":
            return rsd
        if rtype != "deviance":
            raise ValueError("shash residual type must be deviance or response.")
        params = np.column_stack((mu, sigma, epsilon, delta))
        return np.sqrt(np.maximum(-2 * self.logpdf_from_parameters(y, params), 0)) * np.sign(rsd)


def shash(link=("identity", "logeb", "identity", "identity"), b=.01,
          phi_pen=.001, phiPen=None):
    return ShashFamily(link=link, b=b, phi_pen=phi_pen, phiPen=phiPen)
