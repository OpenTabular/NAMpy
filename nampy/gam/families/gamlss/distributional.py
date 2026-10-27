"""Additional joint distributional families with explicit natural parameters."""

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_distribution
from scipy.stats import t

from ...fit.solvers.gamlss_utils import gamlss_gH, trind_generator
from .._function_maps import LINK_REGISTRY
from ._base import GamlssFamily, _AdaptedLinkInfo, _IdentityLinkInfo, _qr_coef_pivoted
from ._taylor import Taylor


class _DistributionalFamily(GamlssFamily):
    reference_group = "distributional_extensions"
    nlp = n_linear_predictors = 2
    supports_analytic_outer_derivatives = True
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self):
        super().__init__()
        self.tri = trind_generator(2)
        self.linfo = [
            _IdentityLinkInfo()
            if name == "identity"
            else _AdaptedLinkInfo(LINK_REGISTRY[name](eps=self.eps), name)
            for name in self.link_names
        ]
        self.link_name = str(self.link_names)

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
        **kw,
    ):
        y = self.validate_y(y)
        eta = self._eta_matrix_from_inputs(
            X, jj, coef, offset=offset, eta=kw.get("eta")
        )
        order = 4 if deriv > 3 else 3 if deriv > 1 else 2 if deriv else 0
        variables = [Taylor(eta[:, k], 2, order, k) for k in range(2)]
        value = self._log_density(y, *variables)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        value = value * w
        result = {"l": float(np.sum(value.value)), "l0": value.value}
        if not deriv:
            return result
        l1, l2 = value.packed_derivatives(1), value.packed_derivatives(2)
        l3 = value.packed_derivatives(3) if order >= 3 else 0
        l4 = value.packed_derivatives(4) if order >= 4 else 0
        result.update(
            gamlss_gH(
                X,
                jj,
                l1,
                l2,
                self.tri["i2"],
                l3=l3,
                i3=self.tri["i3"],
                l4=l4,
                i4=self.tri["i4"],
                d1b=d1b,
                d2b=d2b,
                deriv=deriv - 1,
                fh=fh,
                D=D,
                sandwich=bool(kw.get("sandwich", False)),
            )
        )
        if kw.get("ncv"):
            result.update(l1=l1, l2=l2, l3=l3)
        return result

    def initialize(self, y, X, jj, offset=None, weights=None, E=None):
        y, X = self.validate_y(y), np.asarray(X, dtype=float)
        targets = self._initial_predictors(y)
        offsets = self._offset_list(offset)
        start = np.zeros(X.shape[1])
        for k, target in enumerate(targets):
            target = np.broadcast_to(target, y.shape).copy()
            if offsets[k] is not None:
                target -= offsets[k]
            start[jj[k]] = _qr_coef_pivoted(X[:, jj[k]], target)
        return start

    def predict_fitted(self, X, jj, coef, offset=None):
        return self.distribution_parameters_from_eta(
            self._stacked_eta(X, jj, coef, offset)
        )

    def predict_response(self, X, jj, coef, offset=None, se=False, Vb=None, eta=None):
        eta = self._stacked_eta(X, jj, coef, offset) if eta is None else np.asarray(eta)
        parameters = self.distribution_parameters_from_eta(eta)
        if not se:
            return parameters
        if Vb is None:
            raise ValueError("Vb is required for standard errors.")
        se_eta = np.column_stack(
            [
                np.sqrt(
                    np.maximum(
                        np.einsum(
                            "ij,jk,ik->i",
                            X[:, cols],
                            Vb[np.ix_(cols, cols)],
                            X[:, cols],
                        ),
                        0,
                    )
                )
                for cols in jj
            ]
        )
        return parameters, np.abs(self.distribution_parameter_jacobian(eta)) * se_eta

    def null_deviance(self, y, fitted, prior_weights):
        # A family-specific null optimization is needed; avoid silently using
        # a Gaussian null or a saturated log-density convention.
        return float("nan")

    def residuals(self, y, fitted, rtype="deviance", *, eta=None, **kw):
        y = self.validate_y(y)
        params = np.asarray(fitted, dtype=float)
        mu = self.point_prediction_from_parameters(params)
        if rtype == "response":
            return y - mu
        if rtype == "pearson":
            return (y - mu) / np.sqrt(self.variance_from_parameters(params))
        if rtype == "deviance":
            # Quantile residuals for continuous distributions are not deviance
            # residuals. Do not label them as such or invent a saturation rule.
            raise NotImplementedError(
                f"{self.name} provides response and Pearson residuals; deviance residuals are unavailable."
            )
        raise ValueError("Supported residual types are response and pearson.")

    def _check_parameters(self, parameters):
        params = np.asarray(parameters, dtype=float)
        if (
            params.ndim != 2
            or params.shape[1] != 2
            or not np.all(np.isfinite(params))
            or np.any(params[:, 1] <= 0)
        ):
            raise ValueError(
                "Expected finite (n,2) parameters with positive second column."
            )
        return params


class StudentLocationScale(_DistributionalFamily):
    """Student-t with varying location and scale and fixed degrees of freedom.

    ``df`` is a finite float >2, default 5, ensuring finite response variance.
    Natural parameter columns are (mu,sigma), sigma>0. Predictors are
    (mu,log(sigma)); sigma is a t scale, so Var(Y)=sigma**2*df/(df-2).
    This matches R gamlss.dist::TF(mu,sigma,nu=df), with fixed nu; it does
    not fit a third degrees-of-freedom predictor. Identity/log links are fixed.
    """

    name = "student_lss"
    parameter_names = ("mu", "sigma")
    link_names = ("identity", "log")

    def __init__(self, df=5.0):
        if not np.isfinite(df) or df <= 2:
            raise ValueError("df must be finite and greater than two.")
        self.df = float(df)
        super().__init__()

    def _log_density(self, y, mu, log_sigma):
        from scipy.special import gammaln

        z = (y - mu) * (-log_sigma).exp()
        return (
            gammaln((self.df + 1) / 2)
            - gammaln(self.df / 2)
            - 0.5 * np.log(self.df * np.pi)
            - log_sigma
            - 0.5 * (self.df + 1) * (1 + z * z / self.df).log()
        )

    def _initial_predictors(self, y):
        return y, np.log(max(np.std(y) * np.sqrt((self.df - 2) / self.df), 1e-3))

    def validate_y(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("student_lss requires a finite response vector.")
        return y

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.column_stack([eta[:, 0], np.exp(eta[:, 1])])

    def distribution_parameter_jacobian(self, eta):
        return np.column_stack([np.ones(len(eta)), np.exp(np.asarray(eta)[:, 1])])

    def logpdf_from_parameters(self, y, parameters):
        mu, sigma = self._check_parameters(parameters).T
        return t.logpdf((self.validate_y(y) - mu) / sigma, self.df) - np.log(sigma)

    def variance_from_parameters(self, parameters):
        return self._check_parameters(parameters)[:, 1] ** 2 * self.df / (self.df - 2)

    def rd(self, mu, wt, scale, *, rng):
        params = self._check_parameters(mu)
        return params[:, 0] + params[:, 1] * rng.standard_t(self.df, len(params))


class BetaLocationPrecision(_DistributionalFamily):
    """Beta regression with varying mean mu and precision phi.

    No constructor options. Natural columns (mu,phi) satisfy 0<mu<1, phi>0;
    predictors are (logit(mu),log(phi)). Shapes are a=mu*phi, b=(1-mu)*phi;
    Var(Y)=mu*(1-mu)/(1+phi). Relative to gamlss.dist::BE, its sigma equals
    1/sqrt(1+phi). Endpoints 0 and 1 are rejected, never silently clipped.
    """

    name = "beta_lss"
    parameter_names = ("mu", "phi")
    link_names = ("logit", "log")

    def _log_density(self, y, eta_mu, log_phi):
        mu, phi = 1 / (1 + (-eta_mu).exp()), log_phi.exp()
        a, b = mu * phi, (1 - mu) * phi
        return (
            phi.lgamma()
            - a.lgamma()
            - b.lgamma()
            + (a - 1) * np.log(y)
            + (b - 1) * np.log1p(-y)
        )

    def validate_y(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any((y <= 0) | (y >= 1)):
            raise ValueError(
                "beta_lss requires finite responses strictly inside (0,1)."
            )
        return y

    def _initial_predictors(self, y):
        mean = np.mean(y)
        phi = max(mean * (1 - mean) / max(np.var(y), 1e-6) - 1, 1)
        return np.log(y) - np.log1p(-y), np.log(phi)

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.column_stack([expit(eta[:, 0]), np.exp(eta[:, 1])])

    def distribution_parameter_jacobian(self, eta):
        params = self.distribution_parameters_from_eta(eta)
        return np.column_stack([params[:, 0] * (1 - params[:, 0]), params[:, 1]])

    def _check_parameters(self, parameters):
        params = super()._check_parameters(parameters)
        if np.any((params[:, 0] <= 0) | (params[:, 0] >= 1)):
            raise ValueError("beta_lss mean must be strictly inside (0,1).")
        return params

    def logpdf_from_parameters(self, y, parameters):
        mu, phi = self._check_parameters(parameters).T
        return beta_distribution.logpdf(self.validate_y(y), mu * phi, (1 - mu) * phi)

    def variance_from_parameters(self, parameters):
        mu, phi = self._check_parameters(parameters).T
        return mu * (1 - mu) / (1 + phi)

    def rd(self, mu, wt, scale, *, rng):
        mu, phi = self._check_parameters(mu).T
        return rng.beta(mu * phi, (1 - mu) * phi)


class NegativeBinomialLocationDispersion(_DistributionalFamily):
    """NB2 count regression with varying mean mu and dispersion alpha.

    No constructor options. Natural columns (mu,alpha) are positive; both
    predictors use log links. Var(Y)=mu+alpha*mu**2, size=1/alpha.
    This matches gamlss.dist::NBI(mu,sigma=alpha). Zero counts are supported;
    negative and noninteger responses are rejected.
    """

    name = "nb_lss"
    parameter_names = ("mu", "alpha")
    link_names = ("log", "log")

    def _log_density(self, y, log_mu, log_alpha):
        small = log_alpha.value < np.log(0.01)
        if np.any(small):
            stable = self._poisson_limit_density(
                y[small], log_mu.subset(small), log_alpha.subset(small)
            )
            regular = self._regular_density(
                y[~small], log_mu.subset(~small), log_alpha.subset(~small)
            )
            return Taylor.merge(small, stable, regular)
        return self._regular_density(y, log_mu, log_alpha)

    @staticmethod
    def _poisson_limit_density(y, log_mu, log_alpha):
        """Stable Stirling difference; size>100, including all derivatives.

        Subtract y*log(size) symbolically rather than subtracting two enormous
        log-gamma values. Five Stirling corrections leave an O(size**-11)
        remainder. log1p divided differences preserve the Poisson limit.
        """
        alpha, mu = log_alpha.exp(), log_mu.exp()
        t = alpha * y
        log_t = t.log1p()
        rising = y * t.log1p_over_x(minus_one=True) + (y - 0.5) * log_t
        for power, coefficient in [
            (1, 1 / 12),
            (3, -1 / 360),
            (5, 1 / 1260),
            (7, -1 / 1680),
            (9, 1 / 1188),
        ]:
            rising += coefficient * alpha**power * (-power * log_t).expm1()
        z = alpha * mu
        return (
            rising
            - Taylor(y + 1, 2, log_mu.order).lgamma()
            + y * log_mu
            - mu * z.log1p_over_x()
            - y * z.log1p()
        )

    @staticmethod
    def _regular_density(y, log_mu, log_alpha):
        mu, size = log_mu.exp(), (-log_alpha).exp()
        total = size + mu
        return (
            (size + y).lgamma()
            - size.lgamma()
            + size * (size.log() - total.log())
            + y * (log_mu - total.log())
            - Taylor(y + 1, 2, log_mu.order).lgamma()
        )

    def validate_y(self, y):
        y = np.asarray(y, dtype=float)
        if (
            y.ndim != 1
            or not np.all(np.isfinite(y))
            or np.any(y < 0)
            or np.any(y != np.floor(y))
        ):
            raise ValueError("nb_lss requires finite nonnegative integer counts.")
        return y

    def _initial_predictors(self, y):
        mean = max(np.mean(y), 1e-3)
        alpha = max((np.var(y) - mean) / mean**2, 0.1)
        return np.log(y + 0.1), np.log(alpha)

    def distribution_parameters_from_eta(self, eta):
        return np.exp(np.asarray(eta, dtype=float))

    distribution_parameter_jacobian = distribution_parameters_from_eta

    def _check_parameters(self, parameters):
        params = super()._check_parameters(parameters)
        if np.any(params[:, 0] <= 0):
            raise ValueError("nb_lss mean must be positive.")
        return params

    def logpdf_from_parameters(self, y, parameters):
        mu, alpha = self._check_parameters(parameters).T
        return self._log_density(
            self.validate_y(y), Taylor(np.log(mu), 2, 0), Taylor(np.log(alpha), 2, 0)
        ).value

    def variance_from_parameters(self, parameters):
        mu, alpha = self._check_parameters(parameters).T
        return mu + alpha * mu**2

    def rd(self, mu, wt, scale, *, rng):
        mu, alpha = self._check_parameters(mu).T
        return rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
