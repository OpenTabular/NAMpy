"""Multinomial general family: mgcv 1.9-4 R/gamlss.r::multinom."""

from itertools import combinations_with_replacement

import numpy as np

from ...fit.solvers.gamlss_utils import gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted


class MultinomFamily(GamlssFamily):
    """K identity-linked log odds against reference category zero."""

    name = "multinom"
    supports_gcv = False
    supports_ubre = False
    supports_pirls = False
    supports_closed_form_solve = False
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, K=1):
        super().__init__()
        if not np.isscalar(K) or not np.isfinite(K) or K < 1 or K != round(K):
            raise ValueError("multinom K must be a positive integer (K+1 categories).")
        self.K = self.nlp = self.n_linear_predictors = int(K)
        self.linfo = [_IdentityLinkInfo() for _ in range(self.K)]
        self.tri = trind_generator(self.K)
        self.link_names = tuple("identity" for _ in range(self.K))
        self.link_name = str(self.link_names)
        self.parameter_names = tuple(f"eta{k}" for k in range(1, self.K + 1))
        self.distribution_parameter_names = tuple(f"p{k}" for k in range(self.K + 1))
        self.additive_predictor_indices = tuple(range(1, self.K + 1))

    def additive_link_from_eta(self, eta):
        eta = np.asarray(eta, float)
        return np.column_stack([np.zeros(len(eta)), eta])

    def validate_y(self, y):
        y = np.asarray(y, float).ravel()
        if not np.all(np.isfinite(y)):
            raise ValueError("multinom response must be finite.")
        rounded = np.rint(y)
        if np.any(rounded < 0) or np.any(rounded > self.K):
            raise ValueError("multinom response must round to categories 0 through K.")
        return y

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, float)
        if eta.ndim != 2 or eta.shape[1] != self.K:
            raise ValueError(f"multinom expected eta shape (n, {self.K}).")
        gamma = np.column_stack([np.ones(len(eta)), np.exp(eta)])
        return gamma / np.sum(gamma, axis=1)[:, None]

    def distribution_parameter_jacobian(self, eta):
        """Full probability Jacobian, including the reference category."""
        p = self.distribution_parameters_from_eta(eta)
        dp = -p[:, :, None] * p[:, None, 1:]
        for k in range(self.K):
            dp[:, k + 1, k] = p[:, k + 1] * (1 - p[:, k + 1])
        return dp

    def logpdf_from_parameters(self, y, parameters):
        y = np.rint(self.validate_y(y)).astype(int)
        p = np.asarray(parameters, float)
        if p.shape != (len(y), self.K + 1):
            raise ValueError(f"multinom probabilities require {self.K + 1} columns.")
        if np.any(~np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("multinom probabilities must be finite, nonnegative and sum to one.")
        with np.errstate(divide="ignore"):
            return np.log(p[np.arange(len(y)), y])

    def point_prediction_from_parameters(self, parameters):
        return np.argmax(np.asarray(parameters, float), axis=1)

    def _predict_response_from_eta(self, eta):
        return self.distribution_parameters_from_eta(eta)

    def predict_fitted(self, X, jj, coef, offset=None):
        # Source fitted.values retain identity-linked LPs, while predict
        # type="response" calls the family hook for K+1 probabilities.
        return self._stacked_eta(X, jj, coef, offset)

    def predict(self, *, eta=None, X=None, jj=None, coef=None, offset=None, se=False, Vb=None):
        if eta is not None and (not se or X is None):
            return self.distribution_parameters_from_eta(eta)
        if X is None or jj is None or coef is None:
            raise ValueError("Provide eta or X/jj/coef for multinom prediction.")
        eta = self._stacked_eta(X, jj, coef, offset) if eta is None else np.asarray(eta, float)
        p = self.distribution_parameters_from_eta(eta)
        if not se:
            return p
        if Vb is None:
            raise ValueError("Vb is required when se=True.")
        X, Vb = np.asarray(X, float), np.asarray(Vb, float)
        ve = np.zeros_like(eta)
        ce = {}
        for i, cols in enumerate(jj):
            Xi = X[:, cols]
            ve[:, i] = np.maximum(0, np.sum((Xi @ Vb[np.ix_(cols, cols)]) * Xi, axis=1))
            for j in range(i + 1, self.K):
                # Preserve source pmax(0, cross-covariance), including its
                # distinction from a conventional unrestricted delta method.
                ce[i, j] = np.maximum(0, np.sum((Xi @ Vb[np.ix_(cols, jj[j])]) * X[:, jj[j]], axis=1))
        dp = self.distribution_parameter_jacobian(eta)
        variance = np.sum(dp**2 * ve[:, None, :], axis=2)
        for (i, j), covariance in ce.items():
            variance += 2 * dp[:, :, i] * dp[:, :, j] * covariance[:, None]
        return p, np.sqrt(np.maximum(0, variance))

    def ll(self, y, X, jj, coef, weights, offset=None, deriv=0,
           d1b=0, d2b=0, fh=None, D=None, **kw):
        del weights  # Source multinom likelihood ignores wt.
        y = np.rint(self.validate_y(y)).astype(int)
        eta = self._eta_matrix_from_inputs(X, jj, coef, offset=offset, eta=kw.get("eta"))
        if eta.shape[1] != self.K:
            raise ValueError("number of linear predictors doesn't match multinom K")
        ee = np.exp(eta)
        beta = 1 + np.sum(ee, axis=1)
        alpha = np.log(beta)
        # Upstream constructs cbind(1, eta), so baseline observations add
        # one to ll. Preserve this source constant; public logpdf is normalized.
        l0 = np.column_stack([np.ones(len(y)), eta])[np.arange(len(y)), y] - alpha
        ret = {"l": float(np.sum(l0)), "l0": l0}
        if not deriv:
            return ret
        l1 = ee / beta[:, None]
        i2, i3, i4 = (self.tri[key] for key in ("i2", "i3", "i4"))
        b2 = beta**2
        l2 = np.empty((len(y), self.K * (self.K + 1) // 2))
        for index, (i, j) in enumerate(combinations_with_replacement(range(self.K), 2)):
            l2[:, index] = (-l1[:, i] + ee[:, i]**2 / b2 if i == j else ee[:, i] * ee[:, j] / b2)
        for i in range(self.K):
            l1[:, i] = (y == i + 1) - l1[:, i]
        l3 = l4 = 0
        if deriv > 1:
            b3 = b2 * beta
            l3 = np.empty((len(y), self.K * (self.K + 1) * (self.K + 2) // 6))
            for index, (i, j, k) in enumerate(combinations_with_replacement(range(self.K), 3)):
                product = ee[:, i] * ee[:, j] * ee[:, k]
                if i == j == k:
                    value = l2[:, i2[i, i]] + 2 * ee[:, i]**2 / b2 - 2 * ee[:, i]**3 / b3
                elif i != j and j != k and i != k:
                    value = -2 * product / b3
                else:
                    kk = k if i == j else j
                    value = l2[:, i2[i, kk]] - 2 * product / b3
                l3[:, index] = value
        if deriv > 3:
            b4 = b3 * beta
            l4 = np.empty((len(y), self.K * (self.K + 1) * (self.K + 2) * (self.K + 3) // 24))
            for index, indices in enumerate(combinations_with_replacement(range(self.K), 4)):
                i, j, k, ell = indices
                unique = list(dict.fromkeys(indices))
                n_unique = len(unique)
                product = ee[:, i] * ee[:, j] * ee[:, k] * ee[:, ell]
                if n_unique == 1:
                    value = l3[:, i3[i, i, i]] + 4 * ee[:, i]**2 / b2 - 10 * ee[:, i]**3 / b3 + 6 * ee[:, i]**4 / b4
                elif n_unique == 4:
                    value = 6 * product / b4
                elif n_unique == 3:
                    value = l3[:, i3[tuple(unique)]] + 6 * product / b4
                elif indices.count(unique[0]) == 2:
                    a, b = unique
                    value = l3[:, i3[a, b, b]] - 2 * ee[:, a]**2 * ee[:, b] / b3 + 6 * product / b4
                else:
                    a, b = unique if indices.count(unique[0]) != 1 else unique[::-1]
                    value = l3[:, i3[a, a, b]] - 4 * ee[:, a]**2 * ee[:, b] / b3 + 6 * product / b4
                l4[:, index] = value
        result = gamlss_gH(X, jj, l1, l2, i2, l3=l3, i3=i3, l4=l4, i4=i4,
                           d1b=d1b, d2b=d2b, deriv=deriv-1, fh=fh, D=D,
                           sandwich=bool(kw.get("sandwich", False)))
        if kw.get("ncv"):
            result.update(l1=l1, l2=l2, l3=l3, l4=l4)
        result.update(ret)
        if deriv > 3:
            # Source fourth-order loop uses `l` as its last category index,
            # overwriting the scalar likelihood with K before returning it.
            result["l"] = float(self.K)
        return result

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        del offset, weights  # Source initialization binarizes without offsets.
        y, X = np.asarray(y, float), np.asarray(X, float)
        E = np.zeros((0, X.shape[1])) if E is None else E
        start = np.zeros(X.shape[1])
        for k, cols in enumerate(jj):
            target = 6 * (y == k + 1) - 3
            Xi, Ei = X[:, cols], np.asarray(E)[:, cols]
            if getattr(E, "use_unscaled", False):
                start[cols] = _qr_coef_pivoted(np.vstack([Xi, Ei]), np.r_[target, np.zeros(len(Ei))])
            else:
                start[cols] = _pen_reg(Xi, Ei, target)
        return start

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, **kwargs):
        del kwargs
        if rtype != "deviance":
            raise ValueError("multinom supports only deviance residuals.")
        y = np.rint(self.validate_y(y)).astype(int)
        probabilities = self.distribution_parameters_from_eta(fitted if eta is None else eta)
        correct = np.argmax(probabilities, axis=1) == y
        return np.where(correct, 1, -1) * np.sqrt(-2 * np.log(np.maximum(np.finfo(float).eps, probabilities[np.arange(len(y)), y])))

    def null_deviance(self, y, fitted, prior_weights):
        del fitted, prior_weights
        y = np.rint(self.validate_y(y)).astype(int)
        counts = np.bincount(y, minlength=self.K + 1)
        return float(-2 * np.sum(np.log(counts[y] / len(y))))

    def rd(self, mu, wt=None, scale=1.0, *, rng=None):
        del wt, scale
        rng = np.random.default_rng() if rng is None else rng
        probabilities = self.distribution_parameters_from_eta(mu)
        return np.sum(np.cumsum(probabilities, axis=1) <= rng.uniform(size=(len(probabilities), 1)), axis=1)


def multinom(K=1):
    return MultinomFamily(K=K)
