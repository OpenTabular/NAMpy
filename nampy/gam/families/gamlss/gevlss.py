"""GEV location-scale-shape family, mgcv 1.9-4 R/gamlss.r::gevlss."""

import numpy as np
from scipy.special import gamma

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .._function_maps import LINK_REGISTRY
from ._base import (
    GamlssFamily,
    _AdaptedLinkInfo,
    _IdentityLinkInfo,
    _pen_reg,
    _qr_coef_pivoted,
)
from ._gevlss_derivatives import gevlss_derivatives


class _ShiftedLogitLinkInfo:
    """Source logit confines GEV shape to (-1, .5)."""

    name = "logit"

    def __init__(self):
        self._logit = LINK_REGISTRY['logit'](eps=np.finfo(float).eps)

    def linkfun(self, mu):
        return self._logit((np.asarray(mu, float) + 1) / 1.5)

    def linkinv(self, eta):
        return 1.5 * self._logit.inverse(eta) - 1

    def mu_eta(self, eta):
        return self._logit.mu_eta(eta) * 1.5

    def d2link(self, mu):
        return self._logit.d2((np.asarray(mu, float) + 1) / 1.5) / 1.5**2

    def d3link(self, mu):
        return self._logit.d3((np.asarray(mu, float) + 1) / 1.5) / 1.5**3

    def d4link(self, mu):
        return self._logit.d4((np.asarray(mu, float) + 1) / 1.5) / 1.5**4


class GevlssFamily(GamlssFamily):
    """Three predictors for location, log scale and GEV shape."""

    name = "gevlss"
    nlp = n_linear_predictors = 3
    parameter_names = ('mu', 'sigma', 'xi')
    supports_gcv = False
    supports_ubre = False
    supports_pirls = False
    supports_closed_form_solve = False
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, link=('identity', 'identity', 'logit')):
        super().__init__()
        if len(link) != 3:
            raise ValueError('gevlss requires three links.')
        for name, allowed in zip(link, [('identity', 'log'), ('identity',), ('identity', 'logit')], strict=True):
            if name not in allowed:
                raise ValueError(f'{name} link not available for gevlss.')
        self.link_names = tuple(link)
        self.link_name = str(self.link_names)
        self.linfo = [
            _IdentityLinkInfo() if link[0] == 'identity' else _AdaptedLinkInfo(LINK_REGISTRY['log'](eps=self.eps), 'log'),
            _IdentityLinkInfo(),
            _ShiftedLogitLinkInfo() if link[2] == 'logit' else _IdentityLinkInfo(),
        ]
        self.tri = trind_generator(3)

    def validate_y(self, y):
        y = np.asarray(y, float).ravel()
        if np.any(~np.isfinite(y)):
            raise ValueError('gevlss response must be finite.')
        return y

    @staticmethod
    def _source_shape(xi):
        xi = np.asarray(xi, float).copy()
        xi[(xi >= 0) & (xi < 1e-7)] = 1e-7
        xi[(xi < 0) & (xi > -1e-7)] = -1e-7
        return xi

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, float)
        if eta.ndim != 2 or eta.shape[1] != 3:
            raise ValueError('gevlss expected eta shape (n, 3).')
        return np.column_stack([self.linfo[0].linkinv(eta[:, 0]), np.exp(eta[:, 1]), self.linfo[2].linkinv(eta[:, 2])])

    def distribution_parameter_jacobian(self, eta):
        eta = np.asarray(eta, float)
        return np.column_stack([self.linfo[0].mu_eta(eta[:, 0]), np.exp(eta[:, 1]), self.linfo[2].mu_eta(eta[:, 2])])

    def logpdf_from_parameters(self, y, parameters):
        y = self.validate_y(y)
        params = np.asarray(parameters, float)
        if params.shape != (len(y), 3):
            raise ValueError('gevlss requires (mu, sigma, xi) parameter columns.')
        if np.any(~np.isfinite(params)) or np.any(params[:, 1] <= 0):
            raise ValueError('gevlss parameters must be finite with sigma > 0.')
        xi = self._source_shape(params[:, 2])
        z = 1 + xi * (y - params[:, 0]) / params[:, 1]
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            value = -(1 + 1 / xi) * np.log(z) - z**(-1 / xi) - np.log(params[:, 1])
        return np.where(z <= 0, -np.inf, value)

    def point_prediction_from_parameters(self, parameters):
        mu, sigma, xi = np.asarray(parameters, float).T
        # Same conditional mean used by source residuals (including xi=0).
        with np.errstate(invalid='ignore', divide='ignore'):
            return mu + sigma * (gamma(1 - xi) - 1) / xi

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0,
           d1b=0, d2b=0, fh=None, D=None, **kw):
        del weights  # Source likelihood does not use wt.
        y = self.validate_y(y)
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=offset, eta=kw.get('eta'))
        mu = self.linfo[0].linkinv(eta[:, 0])
        rho = self.linfo[1].linkinv(eta[:, 1])
        xi = self._source_shape(self.linfo[2].linkinv(eta[:, 2]))
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            l0, l1, l2, l3, l4 = gevlss_derivatives(y, mu, rho, xi, deriv)
        result = {'l':float(np.sum(l0)), 'l0':l0}
        if not deriv:
            return result
        ig1 = np.column_stack([info.mu_eta(eta[:, k]) for k, info in enumerate(self.linfo)])
        parameters = [mu, rho, xi]
        g2 = np.column_stack([info.d2link(parameters[k]) for k, info in enumerate(self.linfo)])
        g3 = np.column_stack([info.d3link(parameters[k]) for k, info in enumerate(self.linfo)]) if deriv > 1 else 0
        g4 = np.column_stack([info.d4link(parameters[k]) for k, info in enumerate(self.linfo)]) if deriv > 3 else 0
        i2, i3, i4 = (self.tri[key] for key in ('i2', 'i3', 'i4'))
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4, i2, i3, i4, deriv-1)
        ret = gamlss_gH(X, jj, de['l1'], de['l2'], i2, l3=de['l3'], i3=i3, l4=de['l4'], i4=i4,
                        d1b=d1b, d2b=d2b, deriv=deriv-1, fh=fh, D=D, sandwich=bool(kw.get('sandwich', False)))
        if kw.get('ncv'):
            ret.update(de)
        ret.update(result)
        return ret

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y, X = np.asarray(y, float), np.asarray(X, float)
        E = np.zeros((0, X.shape[1])) if E is None else E
        start = np.zeros(X.shape[1])
        target = y if self.link_names[0] == 'identity' else self.linfo[0].linkfun(np.abs(y) + np.max(y)*1e-7)
        for k in (0, 1):
            Xi, Ei = X[:, jj[k]], np.asarray(E)[:, jj[k]]
            if getattr(E, 'use_unscaled', False):
                start[jj[k]] = _qr_coef_pivoted(np.vstack([Xi, Ei]), np.r_[target, np.zeros(len(Ei))])
            else:
                start[jj[k]] = _pen_reg(Xi, Ei, target)
            if k == 0:
                target = np.log(np.abs(y - self.linfo[0].linkinv(Xi @ start[jj[k]])))
        Xi = X[:, jj[2]]
        target = np.repeat(self.linfo[2].linkfun(np.array([1e-3])), len(y))

        def candidate(multiplier=1):
            value = start.copy()
            value[jj[2]] = _qr_coef_pivoted(Xi, target*multiplier)
            score = self.ll(y, X, jj, value, weights, offset)['l']
            return score, value

        best = candidate()
        step, multiplier, up = 0.2, 1.0, False
        trial = best
        while multiplier > -4.2 and multiplier < 4.2:
            trial = candidate(multiplier+step)
            if np.isfinite(trial[0]) and trial[0] > best[0]:
                up, best, multiplier = True, trial, multiplier + step
            elif up:
                break
            elif step > 0:
                step = -step
            else:
                break
        if not np.isfinite(trial[0]):
            trial = candidate(multiplier-step)
            if np.isfinite(trial[0]):
                best = trial
        return best[1]

    def residuals(self, y, fitted, rtype='deviance', *, eta=None, **kwargs):
        del eta, kwargs
        if rtype not in ('deviance', 'pearson', 'response'):
            raise ValueError('gevlss residuals support deviance, pearson and response.')
        mu, rho, original_xi = np.asarray(fitted, float).T
        y = np.asarray(y, float)
        fv = self.point_prediction_from_parameters(np.column_stack([mu, np.exp(rho), original_xi]))
        xi = self._source_shape(original_xi)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if rtype == 'response':
                return y-fv
            if rtype == 'pearson':
                sd = np.exp(rho)/xi * np.sqrt(np.maximum(0, gamma(1-2*xi)-gamma(1-xi)**2))
                return (y-fv)/sd
            base = 1 + (y-mu)*np.exp(-rho)*xi
            rsd = (xi+1)/xi*np.log(base) + base**(-1/xi) + (1+xi)*np.log(1+xi) - (1+xi)
            return np.sqrt(np.maximum(0, rsd)) * np.sign(y-fv)

    def null_deviance(self, y, fitted, prior_weights):
        return float('nan')  # Source explicitly declares null deviance undefined.

    def rd(self, mu, wt=None, scale=1.0, *, rng=None):
        del wt, scale
        rng = np.random.default_rng() if rng is None else rng
        mu, rho, xi = np.asarray(mu, float).T
        xi = xi.copy()
        xi[np.abs(xi) < 1e-8] = 1e-8
        z = rng.uniform(size=len(mu))
        return mu + ((-np.log(z))**(-xi)-1)*np.exp(rho)/xi


def gevlss(link=('identity', 'identity', 'logit')):
    return GevlssFamily(link=link)
