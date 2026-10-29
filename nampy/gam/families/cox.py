"""Private Cox coefficient-likelihood adapter for the existing general solver.

No independent-observation density is supplied and no public ordinary-GAM
family alias is registered. Context binds event/stratum labels, not weights.
"""

import numpy as np

from ..linalg.cholesky import chol_solve_pivoted
from ..losses.cox import cox_higher_derivatives, cox_partial_likelihood
from .gamlss._base import GamlssFamily, _IdentityLinkInfo


class CoxPartialLikelihood(GamlssFamily):
    name = "cox_partial_likelihood"
    nlp = n_linear_predictors = 1
    parameter_names = ("log_relative_hazard",)
    linfo = [_IdentityLinkInfo()]
    supports_ml = supports_laml = False
    supports_reml = True
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = supports_analytic_outer_hessian = True
    max_derivative_order = 4
    supported_smoothing_optimizers = frozenset({"outer_newton", "bfgs"})
    known_scale = 1.0

    def __init__(self, event, strata):
        super().__init__()
        self.event = np.asarray(event, dtype=bool).copy()
        self.strata = np.asarray(strata).copy()

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        return np.zeros(X.shape[1])

    def distribution_parameters_from_eta(self, eta):
        # Internal fitted predictor bookkeeping only; no probability-law adapter.
        return np.asarray(eta)

    def predict_fitted(self, X, jj, coef, offset=None):
        return self._stacked_eta(X, jj, coef, offset)

    def _reported_deviance(self, y, fitted, *, loglik, weights=None):
        return -2 * loglik

    def ll(
        self,
        y,
        X,
        jj,
        coef,
        weights,
        offset=None,
        deriv=0,
        d1b=0,
        d2b=0,
        fh=None,
        D=None,
        **kwargs,
    ):
        if len(y) != len(self.event) or self.strata.shape != np.shape(y):
            raise ValueError(
                "Cox event/stratum context must align with coefficient design rows."
            )
        if not np.all(np.asarray(weights) == 1):
            raise NotImplementedError(
                "Cox case weights are unsupported; events belong to observation context."
            )
        if isinstance(offset, (list, tuple)):
            offset = offset[0]
        known = np.zeros(len(y)) if offset is None else np.asarray(offset)
        value, gradient, hessian = cox_partial_likelihood(
            coef, X, y, self.event, self.strata, known
        )
        result = {
            "l": value,
            "lb": gradient,
            "lbb": hessian,
            "d1H": None,
            "trHid2H": None,
        }
        if deriv < 2:
            return result
        third, fourth = cox_higher_derivatives(
            coef, X, y, self.event, self.strata, known, order=4 if deriv == 4 else 3
        )
        directions = np.asarray(d1b)
        dH = [
            np.einsum("abc,c->ab", third, directions[:, i])
            for i in range(directions.shape[1])
        ]
        result["d1H"] = (
            np.array([np.sum(fh * value) for value in dH]) if deriv == 2 else dH
        )
        if deriv == 4:
            p = len(coef)
            piv = np.asarray(getattr(fh, "pivot", np.arange(p)), dtype=int)
            ipiv = np.argsort(piv)
            inverse = np.asarray(D)[:, None] * chol_solve_pivoted(
                fh, np.diag(D), piv=piv, ipiv=ipiv
            )
            traces = []
            k = 0
            for i in range(directions.shape[1]):
                for j in range(i, directions.shape[1]):
                    second = np.einsum(
                        "abcd,c,d->ab", fourth, directions[:, i], directions[:, j]
                    ) + np.einsum("abc,c->ab", third, np.asarray(d2b)[:, k])
                    traces.append(np.sum(inverse * second))
                    k += 1
            result["trHid2H"] = np.array(traces)
        return result
