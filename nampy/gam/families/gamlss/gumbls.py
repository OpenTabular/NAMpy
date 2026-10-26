"""Gumbel location-scale family, mgcv R/gamlss.r::gumbls."""

from __future__ import annotations

import numpy as np

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted
from .gammals import _SoftplusBLinkInfo

_EULER = 0.5772156649015328606065121


class GumblsFamily(GamlssFamily):
    """Gumbel location and log scale with mgcv's bounded log-scale link."""

    name = "gumbls"
    nlp = n_linear_predictors = 2
    parameter_names = ("location", "scale")
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, link=("identity", "log"), b: float = -7.0):
        super().__init__()
        if not isinstance(link, (tuple, list)) or len(link) != 2:
            raise ValueError("gumbls requires two links.")
        if link[0] != "identity" or link[1] not in {"identity", "log"}:
            raise ValueError("gumbls supports identity location and identity/log scale links.")
        self.b = float(b)
        self.linfo = [_IdentityLinkInfo(), _SoftplusBLinkInfo(b) if link[1] == "log" else _IdentityLinkInfo()]
        self.link_names = tuple(link)
        self.link_name = f"({link[0]}, {link[1]})"
        self.tri = trind_generator(2)

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0, d1b=0, d2b=0, fh=None, D=None, **kw):
        y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=offset, eta=kw.get("eta"))
        mu = self.linfo[0].linkinv(eta[:, 0])
        beta = self.linfo[1].linkinv(eta[:, 1])
        eb = np.exp(-beta)
        z = (y - mu) * eb
        ez = np.exp(-z)
        l0 = -beta - z - ez
        ret = {"l": float(np.sum(l0)), "l0": l0}
        if not deriv:
            return ret
        lz, zm, zb = ez - 1, -eb, -z
        lzz, zmb, zbb = -ez, eb, z
        l1 = np.column_stack([lz * zm, lz * zb - 1])
        l2 = np.column_stack([lzz * zm**2, lzz * zm * zb + lz * zmb, lzz * zb**2 + lz * zbb])
        ig1 = np.column_stack([self.linfo[k].mu_eta(eta[:, k]) for k in range(2)])
        values = (mu, beta)
        g2 = np.column_stack([self.linfo[k].d2link(values[k]) for k in range(2)])
        l3 = l4 = g3 = g4 = 0
        if deriv > 1:
            lzzz, zbbb, zmbb = ez, -z, -eb
            l3 = np.column_stack([
                lzzz * zm**3,
                lzzz * zm**2 * zb + 2 * lzz * zm * zmb,
                lzzz * zb**2 * zm + 2 * lzz * zb * zmb + lzz * zbb * zm + lz * zmbb,
                lzzz * zb**3 + 3 * lzz * zb * zbb + lz * zbbb,
            ])
            g3 = np.column_stack([self.linfo[k].d3link(values[k]) for k in range(2)])
        if deriv > 3:
            lzzzz, zbbbb, zmbbb = -ez, z, eb
            l4 = np.column_stack([
                lzzzz * zm**4,
                lzzzz * zm**3 * zb + 3 * lzzz * zm**2 * zmb,
                lzzzz * zm**2 * zb**2 + 4 * lzzz * zm * zb * zmb + lzzz * zm**2 * zbb + 2 * lzz * zmb**2 + 2 * lzz * zm * zmbb,
                lzzzz * zb**3 * zm + 3 * lzzz * zb**2 * zmb + 3 * lzzz * zm * zb * zbb + 3 * lzz * zmb * zbb + 3 * lzz * zb * zmbb + lzz * zm * zbbb + lz * zmbbb,
                lzzzz * zb**4 + 6 * lzzz * zb**2 * zbb + 3 * lzz * zbb**2 + 4 * lzz * zb * zbbb + lz * zbbbb,
            ])
            g4 = np.column_stack([self.linfo[k].d4link(values[k]) for k in range(2)])
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4, self.tri["i2"], self.tri["i3"], self.tri["i4"], deriv - 1)
        out = gamlss_gH(X, jj, de["l1"], de["l2"], self.tri["i2"], l3=de["l3"], i3=self.tri["i3"], l4=de["l4"], i4=self.tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1, fh=fh, D=D, sandwich=bool(kw.get("sandwich", False)))
        if kw.get("ncv", False):
            out.update({key: de[key] for key in ("l1", "l2", "l3")})
        out.update(ret)
        return out

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
        E = np.zeros((0, X.shape[1])) if E is None else E
        use_unscaled = bool(getattr(E, "use_unscaled", False))
        off = self._offset_list(offset)

        def regress(k, target):
            x, e = X[:, jj[k]], E[:, jj[k]]
            if use_unscaled:
                return _qr_coef_pivoted(np.vstack([x, e]), np.concatenate([target, np.zeros(E.shape[0])]))
            return _pen_reg(x, e, target)

        start = np.zeros(X.shape[1])
        yt = y if off[0] is None else y - off[0]
        start[jj[0]] = regress(0, yt)
        lres = self.linfo[1].linkfun(np.log((y - self.linfo[0].linkinv(X[:, jj[0]] @ start[jj[0]]))**2) / 2 - 0.25)
        if off[1] is not None:
            lres -= off[1]
        start[jj[1]] = regress(1, lres)
        eta2 = X[:, jj[1]] @ start[jj[1]]
        if off[1] is not None:
            eta2 += off[1]
        yt = yt - 0.57721 * np.exp(self.linfo[1].linkinv(eta2))
        start[jj[0]] = regress(0, yt)
        return start

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.column_stack([eta[:, 0], np.exp(self.linfo[1].linkinv(eta[:, 1]))])

    def distribution_parameter_jacobian(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.column_stack([np.ones(eta.shape[0]), np.exp(self.linfo[1].linkinv(eta[:, 1])) * self.linfo[1].mu_eta(eta[:, 1])])

    def point_prediction_from_parameters(self, parameters):
        params = np.asarray(parameters, dtype=float)
        return params[:, 0] + _EULER * params[:, 1]

    def logpdf_from_parameters(self, y, parameters):
        params = np.asarray(parameters, dtype=float)
        if np.any(~np.isfinite(params)) or np.any(params[:, 1] <= 0):
            raise ValueError("gumbls requires finite location and positive scale.")
        z = (np.asarray(y, dtype=float).ravel() - params[:, 0]) / params[:, 1]
        return -np.log(params[:, 1]) - z - np.exp(-z)

    def predict_fitted(self, X, jj, coef, offset=None):
        fit = self._predict_response_from_eta(self._stacked_eta(X, jj, coef, offset))
        fit[:, 0] += _EULER * np.exp(fit[:, 1])
        return fit

    def predict(self, *, eta=None, X=None, jj=None, coef=None, offset=None, se=False, Vb=None):
        # Source gumbls$predict keeps location/log scale, while postproc adds
        # Euler*scale to fitted.values only. Its scale SE uses scalar ve[2].
        supplied_eta = eta is not None and X is None
        if supplied_eta:
            return self._predict_response_from_eta(eta)
        eta = self._stacked_eta(X, jj, coef, offset) if eta is None else np.asarray(eta, dtype=float)
        fit = self._predict_response_from_eta(eta)
        if not se:
            return fit
        if Vb is None:
            raise ValueError("Vb is required when se=True.")
        ve = np.column_stack([np.maximum(np.einsum("ij,jk,ik->i", X[:, cols], Vb[np.ix_(cols, cols)], X[:, cols]), 0) for cols in jj])
        return fit, np.column_stack([np.sqrt(ve[:, 0]), np.abs(self.linfo[1].mu_eta(eta[:, 1])) * np.sqrt(ve.ravel(order="F")[1])])

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, **kwargs):
        fitted, y = np.asarray(fitted, dtype=float), np.asarray(y, dtype=float)
        mean, beta = fitted[:, 0], np.exp(fitted[:, 1])
        mu = mean - beta * _EULER
        if rtype == "response":
            return y - mean
        if rtype == "pearson":
            return (y - mean) / (np.pi * beta / np.sqrt(6))
        if rtype != "deviance":
            raise ValueError("gumbls residuals support deviance, pearson and response.")
        z = (y - mu) / beta
        return np.sqrt(np.maximum(0, 2 * (z + np.exp(-z) - 1))) * np.sign(y - mu)

    def null_deviance(self, y, fitted, prior_weights):
        return float("nan")

    def rd(self, mu, wt, scale, *, rng):
        mu = np.asarray(mu, dtype=float)
        beta = np.exp(mu[:, 1])
        return mu[:, 0] - beta * (_EULER + np.log(-np.log(rng.uniform(size=mu.shape[0]))))
