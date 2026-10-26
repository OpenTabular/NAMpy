"""stats quasi families and mgcv's extended quasi-likelihood scale contract."""

from numbers import Real

import numpy as np

from ._function_maps import LINK_REGISTRY, PowerLink
from .family_base import GLMFamily, JointOuterStrategy, _BinomialBase

_VARIANCES = {
    "constant": "constant",
    "mu": "poisson",
    "mu(1-mu)": "binomial",
    "mu^2": "gamma",
    "mu^3": "inverse_gaussian",
}


class _QuasiBase(GLMFamily):
    is_quasi = True
    supports_pirls = True
    supports_gcv = True
    supports_ubre = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.GAMMA_SCALE
    known_scale = None
    max_derivative_order = 4

    def __init__(self, link, variance, *, power=None, **kwargs):
        if isinstance(variance, str):
            variance = variance.replace(" ", "")
        if not isinstance(variance, str) or variance not in _VARIANCES:
            raise ValueError(
                "quasi variance must be constant, mu, mu(1-mu), mu^2 or mu^3"
            )
        self.varfun = variance
        self._variance_key = _VARIANCES[variance]
        self._power = None
        if power is not None:
            if link not in {"identity", "power"}:
                raise ValueError("power cannot be combined with a named link")
            if isinstance(power, bool) or not isinstance(power, Real):
                raise ValueError("power must be a finite numeric scalar")
            power = float(power)
            if not np.isfinite(power):
                raise ValueError("power must be a finite numeric scalar")
            link = "log" if power <= 0 else "identity"
            if power > 0 and power != 1:
                self._power = power
        if not isinstance(link, str) or link not in LINK_REGISTRY:
            raise ValueError(f"Unsupported quasi link {link!r}")
        self._link_key = self.link_name = link
        super().__init__(**kwargs)
        if self._power is not None:
            self.link = PowerLink(eps=self.eps, power=self._power)
            self.link_name = f"mu^{round(self._power, 3):g}"
        self.canonical_link = (
            self.name == "quasipoisson" and self.link_name == "log"
        ) or (self.name == "quasibinomial" and self.link_name == "logit")

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        if self.link_name in {"sqrt", "1/mu^2"} or self._power is not None:
            return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))
        if self.link_name == "inverse":
            return bool(np.all(np.isfinite(eta)) and np.all(eta != 0))
        return True

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=float)
        if self.varfun == "constant":
            return True
        valid = np.all(mu > 0)
        if self.varfun == "mu(1-mu)":
            valid = valid and np.all(mu < 1)
        if self.name != "quasi":
            valid = valid and np.all(np.isfinite(mu))
        return bool(valid)

    def initialize_mu(self, y, weights=None):
        del weights
        y = np.asarray(y, dtype=float)
        if self.varfun == "constant":
            return y.copy()
        if self.varfun == "mu(1-mu)":
            return np.maximum(0.001, np.minimum(0.999, y))
        return y + 0.1 * (y == 0)

    def deviance_obs(self, y, mu, weights=None):
        y, mu = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        )
        w = self._check_weights(y, weights)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.varfun == "constant":
                return w * (y - mu) ** 2
            if self.varfun == "mu":
                return 2 * w * (y * np.log(np.where(y == 0, 1, y / mu)) - (y - mu))
            if self.varfun == "mu(1-mu)":
                a = np.zeros_like(y)
                b = np.zeros_like(y)
                pos = y != 0
                a[pos] = y[pos] * np.log(y[pos] / mu[pos])
                pos = y != 1
                b[pos] = (1 - y[pos]) * np.log((1 - y[pos]) / (1 - mu[pos]))
                return 2 * w * (a + b)
            if self.varfun == "mu^2":
                return np.maximum(
                    -2 * w * (np.log(np.where(y == 0, 1, y) / mu) - (y - mu) / mu), 0
                )
            return w * (y - mu) ** 2 / (y * mu**2)

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def deviance_derivatives_mu(self, y, mu, order=1):
        # gam.fit3's ordinary-family derivative identities use V and its
        # derivatives, also for quasi families without a probability density.
        mu, y = np.asarray(mu, dtype=float), np.asarray(y, dtype=float)
        v = self.variance(mu)
        r = mu - y
        if order == 1:
            return 2 * r / v
        v1 = self.dvar(mu)
        if order == 2:
            return 2 / v - 2 * r * v1 / v**2
        v2 = self.d2var(mu)
        if order == 3:
            return -4 * v1 / v**2 - 2 * r * v2 / v**2 + 4 * r * v1**2 / v**3
        if order == 4:
            return (
                -6 * v2 / v**2
                + 12 * v1**2 / v**3
                - 2 * r * self.d3var(mu) / v**2
                + 12 * r * v1 * v2 / v**3
                - 12 * r * v1**3 / v**4
            )
        raise ValueError("deviance derivative order must be 1 through 4")

    def working_weight_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        a, b, c = (self.inverse_link_derivatives(eta, k) for k in (1, 2, 3))
        if y is None:
            v = self.variance(mu)
            return 2 * a * b / v - a**3 * self.dvar(mu) / v**2
        d1, d2, d3 = (self.deviance_derivatives_mu(y, mu, k) for k in (1, 2, 3))
        return (d3 * a**3 + 3 * d2 * a * b + d1 * c) / 2

    def working_weight_second_derivative_eta(self, eta, y=None):
        mu = self.inverse_link(eta)
        a, b, c, d = (self.inverse_link_derivatives(eta, k) for k in (1, 2, 3, 4))
        if y is None:
            v, v1, v2 = self.variance(mu), self.dvar(mu), self.d2var(mu)
            return (
                2 * (b * b + a * c) / v
                - 5 * a * a * b * v1 / v**2
                - a**4 * v2 / v**2
                + 2 * a**4 * v1 * v1 / v**3
            )
        d1, d2, d3, d4 = (self.deviance_derivatives_mu(y, mu, k) for k in (1, 2, 3, 4))
        return (
            d4 * a**4 + 6 * d3 * a * a * b + 3 * d2 * b * b + 4 * d2 * a * c + d1 * d
        ) / 2

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
        w = self._check_weights(y, weights)
        return float(
            np.sum(w * (y - mu) ** 2 / self.variance(mu))
            / (len(y) - (0 if edf is None else edf))
        )

    def aic(self, y, mu, *, edf=0.0, scale=1.0, weights=None, n=None, dev=None):
        return float("nan")

    def loglik_obs(self, y, mu, scale=1.0, n=None):
        return np.full(np.broadcast_shapes(np.shape(y), np.shape(mu)), np.nan)

    def ls(self, y, w, n=None, scale=1.0):
        """mgcv EQL normalization for selection, not a sampling likelihood."""
        w = self._check_weights(y, w)
        nobs = np.sum(w > 0)
        return np.array(
            [
                -nobs * np.log(scale) / 2 + np.sum(np.log(w[w > 0])) / 2,
                -nobs / (2 * scale),
                nobs / (2 * scale * scale),
            ]
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        return float(self.ls(y, weights, n, scale)[0])

    def quantile_residual_bounds(self, y, mu, *, weights=None, scale=1.0):
        raise NotImplementedError(
            "Quasi families do not define a response distribution for quantile residuals"
        )


class QuasiPoissonFamily(_QuasiBase):
    name = "quasipoisson"

    def __init__(self, link="log", **kwargs):
        super().__init__(link, "mu", **kwargs)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0):
            raise ValueError("negative values not allowed for the quasiPoisson family")
        return y

    def initialize_mu(self, y, weights=None):
        return np.asarray(y, dtype=float) + 0.1


class QuasiBinomialFamily(_QuasiBase, _BinomialBase):
    name = "quasibinomial"

    def __init__(self, link="logit", **kwargs):
        super().__init__(link, "mu(1-mu)", **kwargs)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("quasibinomial requires targets in [0,1]")
        return y

    def initialize_mu(self, y, weights=None):
        y = np.asarray(y, dtype=float)
        trials = getattr(self, "_trial_counts", None)
        w = (
            self._check_weights(y, weights)
            if trials is None
            else np.asarray(trials, dtype=float)
        )
        return (w * y + 0.5) / (w + 1)


class QuasiFamily(_QuasiBase):
    name = "quasi"

    def __init__(self, link="identity", variance="constant", *, power=None, **kwargs):
        super().__init__(link, variance, power=power, **kwargs)
