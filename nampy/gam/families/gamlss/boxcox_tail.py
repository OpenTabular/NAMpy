"""Normalized BCT/BCPE likelihoods in the existing joint fitting engine.

Density and initialization: gamlss.dist 6.1-12, R/BCT.R::dBCT and
R/BCPE.R::dBCPE. Observed derivatives differentiate the normalized density,
not GAMLSS's working curvature. CDF parameter derivatives are integrals of
analytic density derivatives, with a checked adaptive quadrature error.
"""

import numpy as np
from scipy.integrate import quad_vec

from ....distributions import BCPELaw, BCTLaw, DistributionPrediction
from ._taylor import Taylor, indices
from .positive import _PositiveDistributionalFamily


def _latent_log_density(log_abs, log_tau, kind):
    tau = log_tau.exp()
    if kind == "bct":
        return (
            ((tau + 1) / 2).lgamma()
            - (tau / 2).lgamma()
            - (log_tau + np.log(np.pi)) / 2
            - (tau + 1) / 2 * (2 * log_abs - log_tau).softplus()
        )
    inverse = (-log_tau).exp()
    log_alpha = (inverse.lgamma() - (3 * inverse).lgamma()) / 2
    return (
        log_tau
        - np.log(2)
        - log_alpha
        - inverse.lgamma()
        - (tau * (log_abs - log_alpha)).exp()
    )


def _log_normalizer(log_sigma, nu, log_tau, kind):
    """log F(1/(sigma*abs(nu))); Taylor coefficients by differentiation.

    Set z=a*exp(-s), so F(a)=1/2+integral_0^infinity z*f(z) ds.
    The integrand remains smooth at the origin even for PE shape below one.
    The scalar value is independently checked against the shared probability
    law; numerical failure is explicit, never an omitted normalizer.
    """
    law = BCTLaw() if kind == "bct" else BCPELaw()
    tau = np.exp(log_tau.value)
    zero = nu.value == 0
    if kind == "bct" and np.any(zero & (tau <= nu.order)):
        raise NotImplementedError(
            "BCT derivatives at nu=0 require tau greater than derivative order."
        )
    result = nu.constant(np.zeros_like(nu.value))
    if np.all(zero):
        return result
    ls, nn, lt = (value.subset(~zero) for value in (log_sigma, nu, log_tau))
    sign = np.sign(nn.value)
    log_a = -ls - (nn * sign).log()
    expected = law.latent.logcdf(
        np.ones_like(log_a.value), log_a.value, np.exp(lt.value)
    )
    if not nu.order:
        active = nn.constant(expected)
    else:
        keys = indices(nu.dimension, nu.order)

        def integrand(s):
            log_z = log_a - s
            value = (_latent_log_density(log_z, lt, kind) + log_z).exp()
            return np.stack(
                [
                    np.broadcast_to(value.coefficients.get(key, 0), nn.value.shape)
                    for key in keys
                ]
            )

        integral, error, info = quad_vec(
            integrand,
            0,
            np.inf,
            epsabs=2e-11,
            epsrel=2e-11,
            norm="max",
            limit=300,
            full_output=True,
        )
        if not info.success or not np.isfinite(integral).all() or error > 2e-9:
            raise NotImplementedError(
                "Box-Cox likelihood normalization derivatives did not converge."
            )
        active = nn.constant(0)
        active.coefficients = dict(zip(keys, integral, strict=True))
        active = (active + 0.5).log()
        if not np.allclose(active.value, expected, rtol=0, atol=2e-10):
            raise NotImplementedError("Box-Cox normalization integral lost accuracy.")
    return Taylor.merge(~zero, active, result.subset(zero))


class _BoxCoxTailLikelihood(_PositiveDistributionalFamily):
    kind: str
    law: type[BCTLaw] | type[BCPELaw]
    initial_nu: float
    initial_tau: float
    parameter_names = ("mu", "sigma", "nu", "tau")
    link_names = ("log", "log", "identity", "log")
    parameter_bounds = {"mu": (0, None), "sigma": (0, None), "tau": (0, None)}
    # Higher derivatives need separate outer-smoothing validation; fixed
    # smoothing uses the exact joint gradient and observed Hessian.
    max_derivative_order = 2

    def _log_density(self, y, log_mu, log_sigma, nu, log_tau):
        ratio = np.log(y) - log_mu
        z = ratio * (-log_sigma).exp() * (nu * ratio).expm1_over_x()
        if self.kind == "bct":
            tau = log_tau.exp()
            latent = (
                ((tau + 1) / 2).lgamma()
                - (tau / 2).lgamma()
                - (log_tau + np.log(np.pi)) / 2
                - (tau + 1) / 2 * (z * z * (-log_tau).exp()).log1p()
            )
        else:
            if np.any(z.value == 0):
                raise NotImplementedError(
                    "BCPE observed derivatives at y=mu are unsupported for varying shape."
                )
            latent = _latent_log_density(
                (z * np.sign(z.value)).log(), log_tau, self.kind
            )
        return (
            nu * ratio
            - log_sigma
            - np.log(y)
            + latent
            - _log_normalizer(log_sigma, nu, log_tau, self.kind)
        )

    def _initial_predictors(self, y):
        return (
            np.log((y + np.mean(y)) / 2),
            np.log(0.1),
            self.initial_nu,
            np.log(self.initial_tau),
        )

    def _distribution(self, parameters):
        return DistributionPrediction(
            self.law(),
            dict(
                zip(
                    self.parameter_names,
                    self._check_parameters(parameters).T,
                    strict=True,
                )
            ),
        )

    def logpdf_from_parameters(self, y, parameters):
        return self._distribution(parameters).log_prob(self.validate_y(y))

    def point_prediction_from_parameters(self, parameters):
        return self._distribution(parameters).mean()

    def variance_from_parameters(self, parameters):
        return self._distribution(parameters).variance()

    def rd(self, mu, wt, scale, *, rng):
        return self._distribution(mu).sample(1, random_state=rng)[0]


class BCTLikelihood(_BoxCoxTailLikelihood):
    """Four-predictor BCT fitting with fixed smoothing, log/log/identity/log links.

    Uses the exact Student-t density, including above the upstream density-only
    Gaussian-switch threshold. Automatic smoothing is intentionally gated until
    third/fourth-order normalization derivatives have independent validation.
    """

    name, kind, law = "bct_lss", "bct", BCTLaw
    initial_nu, initial_tau = 0.5, 10.0


class BCPELikelihood(_BoxCoxTailLikelihood):
    """Four-predictor variance-one BCPE fitting with fixed smoothing.

    Links are log/log/identity/log. At y=mu, arbitrary varying PE shape need
    not have an observed Hessian; this nondifferentiable surface is explicit.
    Automatic smoothing requires independently validated higher derivatives.
    """

    name, kind, law = "bcpe_lss", "bcpe", BCPELaw
    initial_nu, initial_tau = 1.0, 2.0
