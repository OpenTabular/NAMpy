"""Censored Gaussian extended family, mgcv/R/efam.r::cnorm.

Censor bounds are explicit response data, never family-global fitted state.
The constructor and AIC follow source behavior, including its documented
source/manual discrepancies; this is not an ordinary Gaussian shortcut.
"""

import numpy as np

from ..response import normalize_censored_response
from ._cnorm_helpers import ddnorm, dpnorm, log_dnorm, log_pnorm
from ._function_maps import LINK_REGISTRY
from .family_base import ExtendedFamily, JointOuterStrategy


def _unpack(y):
    response = normalize_censored_response(y)
    return response.values, response.bounds


class CensoredNormalFamily(ExtendedFamily):
    """Normal location with fixed or estimated SD and optional censoring."""

    name = "cnorm"
    canonical_link = False
    supports_pirls = True
    supports_gcv = False
    supports_ubre = False
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    known_scale = 1.0
    joint_outer_strategy = JointOuterStrategy.CNORM_THETA
    max_derivative_order = 4

    def __init__(self, theta=None, link="identity"):
        super().__init__()
        if link not in {"identity", "log", "sqrt"}:
            raise ValueError("cnorm link must be 'identity', 'log', or 'sqrt'.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)
        self.known_scale = 1.0
        self.n_theta = 1
        self.ini_theta = np.zeros(1)
        if theta is not None:
            value = np.asarray(theta, float).reshape(-1)
            if value.size != 1 or not np.all(np.isfinite(value)):
                raise ValueError("cnorm theta must contain one finite value.")
            # Source stores the negative input itself, not log(-theta).
            self.ini_theta = np.log(value) if value[0] > 0 else value.copy()
            if value[0] > 0:
                self.n_theta = 0
        self._theta_working = self.ini_theta.copy()

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        return np.exp(self._theta_working) if trans else self._theta_working.copy()

    def putTheta(self, theta):
        value = np.asarray(theta, float).reshape(-1)
        if value.size != 1 or not np.all(np.isfinite(value)):
            raise ValueError("cnorm working theta must contain one finite value.")
        self._theta_working = value.copy()

    def validate_y(self, y):
        return normalize_censored_response(y)

    def initialize_mu(self, y, weights=None):
        del weights
        values = _unpack(y)[0]
        return (
            values.copy()
            if self.link_name == "identity"
            else np.maximum(values, np.min(values > 0))
        )

    def initialize_linear_predictors(self, y):
        return [self.link(self.initialize_mu(y))]

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def valid_mu(self, mu):
        mu = np.asarray(mu)
        return bool(
            np.all(np.isfinite(mu)) if self.link_name == "identity" else np.all(mu > 0)
        )

    def valid_eta(self, eta):
        eta = np.asarray(eta)
        return (
            bool(np.all(np.isfinite(eta)) and np.all(eta >= 0))
            if self.link_name == "sqrt"
            else True
        )

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        return 1.0

    def _state(self, y, weights=None, theta=None):
        values, bounds = _unpack(y)
        w = (
            np.ones_like(values)
            if weights is None
            else np.broadcast_to(np.asarray(weights, float), values.shape)
        )
        theta = (
            self.getTheta() if theta is None else np.asarray(theta, float).reshape(-1)
        )
        if theta.size != 1:
            raise ValueError("cnorm working theta must have length one.")
        with np.errstate(divide="ignore", invalid="ignore"):
            th = theta[0] - np.log(w) / 2
        return values, bounds, th

    def deviance_obs(self, y, mu, weights=None, theta=None):
        values, bounds, th = self._state(y, weights, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth = np.exp(-th)
        result = np.zeros_like(values)
        exact = bounds == values
        result[exact] = (values[exact] - mu[exact]) ** 2 * np.exp(-2 * th[exact])
        interval = np.isfinite(bounds) & ~exact
        lo, hi = (
            np.minimum(values[interval], bounds[interval]),
            np.maximum(values[interval], bounds[interval]),
        )
        zz = (hi - lo) * eth[interval] / 2
        result[interval] = 2 * dpnorm(-zz, zz) - 2 * dpnorm(
            (lo - mu[interval]) * eth[interval], (hi - mu[interval]) * eth[interval]
        )
        left, right = np.isneginf(bounds), np.isposinf(bounds)
        result[left] = -2 * log_pnorm((values[left] - mu[left]) * eth[left])
        result[right] = -2 * log_pnorm(-(values[right] - mu[right]) * eth[right])
        return result

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def aic(
        self, y, mu, theta=None, wt=None, dev=None, *, edf=0.0, scale=1.0, weights=None
    ):
        del dev, edf, scale
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        values, bounds, th = self._state(y, wt if weights is None else weights, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth = np.exp(-th)
        result = np.zeros_like(values)
        exact = bounds == values
        # Source deliberately has no log SD Jacobian here.
        result[exact] = -2 * log_dnorm((values[exact] - mu[exact]) * eth[exact])
        interval = np.isfinite(bounds) & ~exact
        lo, hi = (
            np.minimum(values[interval], bounds[interval]),
            np.maximum(values[interval], bounds[interval]),
        )
        result[interval] = -2 * dpnorm(
            (lo - mu[interval]) * eth[interval], (hi - mu[interval]) * eth[interval]
        )
        # Preserve source <=0 override, including nonpositive exact bounds.
        left, right = bounds <= 0, np.isposinf(bounds)
        result[left] = -2 * log_pnorm((values[left] - mu[left]) * eth[left])
        result[right] = -2 * log_pnorm(-(values[right] - mu[right]) * eth[right])
        return float(np.sum(result))

    def weighted_loglik(self, y, mu, weights=None, scale=1.0):
        return -self.aic(y, mu, weights=weights, scale=scale) / 2

    def loglik(self, y, mu, scale=1.0):
        return self.weighted_loglik(y, mu, scale=scale)

    def ls(self, y, w, theta=None, scale=1.0):
        del scale
        values, bounds, th = self._state(y, w, theta)
        exact = bounds == values
        value = np.sum(-th[exact] - np.log(2 * np.pi) / 2)
        interval = np.isfinite(bounds) & ~exact
        lo, hi = (
            np.minimum(values[interval], bounds[interval]),
            np.maximum(values[interval], bounds[interval]),
        )
        zz = (hi - lo) * np.exp(-th[interval]) / 2
        value += np.sum(dpnorm(-zz, zz))
        return {
            "ls": float(value),
            "lsth1": np.zeros(1),
            "LSTH1": np.zeros((len(values), 1)),
            "lsth2": np.zeros((1, 1)),
        }

    def Dd(self, y, mu, theta, wt, level=0):
        values, bounds, th = self._state(y, wt, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth = np.exp(-th)
        e2th = eth * eth
        names = ["Dmu", "Dmu2"]
        if level > 0:
            names += ["Dth", "Dmuth", "Dmu2th", "Dmu3"]
        if level > 1:
            names += ["Dth2", "Dmuth2", "Dmu2th2", "Dmu4", "Dmu3th"]
        out = {key: np.zeros_like(values) for key in names}
        exact = bounds == values
        z = (values[exact] - mu[exact]) * eth[exact]
        dm = -2 * z * eth[exact]
        out["Dmu"][exact] = dm
        out["Dmu2"][exact] = 2 * e2th[exact]
        if level > 0:
            out["Dth"][exact] = -2 * (z**2 - 1)
            out["Dmuth"][exact] = -2 * dm
            out["Dmu2th"][exact] = -4 * e2th[exact]
        if level > 1:
            out["Dth2"][exact] = 4 * z**2
            out["Dmuth2"][exact] = 4 * dm
            out["Dmu2th2"][exact] = 8 * e2th[exact]
        # All three source censor blocks share the displayed derivative
        # recurrences. Only the signed endpoint-density differences differ.
        for case, mask in (
            ("interval", np.isfinite(bounds) & ~exact),
            ("left", np.isneginf(bounds)),
            ("right", np.isposinf(bounds)),
        ):
            if not np.any(mask):
                continue
            t, e, e2 = th[mask], eth[mask], e2th[mask]
            if case == "interval":
                z0 = (np.minimum(values[mask], bounds[mask]) - mu[mask]) * e
                z1 = (np.maximum(values[mask], bounds[mask]) - mu[mask]) * e
                lp = dpnorm(z0, z1)
            else:
                z0 = z1 = (values[mask] - mu[mask]) * e
                lp = log_pnorm(z1 if case == "left" else -z0)

            def difference(
                a0=None,
                a1=None,
                power=None,
                shift=0,
                *,
                shift_first=False,
                z0=z0,
                z1=z1,
                lp=lp,
                case=case,
            ):
                if power is not None:
                    a0, a1 = z0.copy(), z1.copy()
                    for _ in range(power - 1):
                        a0, a1 = a0 * z0, a1 * z1
                with np.errstate(divide="ignore", invalid="ignore"):
                    if case == "interval":
                        if a0 is None:
                            ld, sign = ddnorm(z0, z1)
                        else:
                            ld, sign = ddnorm(
                                z0,
                                z1,
                                np.log(np.abs(a0)),
                                np.log(np.abs(a1)),
                                np.ones_like(a0) if power in {2, 4} else np.sign(a0),
                                np.ones_like(a1) if power in {2, 4} else np.sign(a1),
                            )
                    else:
                        a = a1 if case == "left" else a0
                        ld = log_dnorm(z1)
                        sign = np.ones_like(ld)
                        if a is not None:
                            ld = ld + np.log(np.abs(a))
                            sign = np.sign(a)
                        if case == "right":
                            sign = -sign
                    return sign * np.exp(
                        -shift + ld - lp if shift_first else ld - lp - shift
                    )

            dm = 2 * difference(shift=t, shift_first=True)
            dt = 2 * difference(power=1)
            dm2 = dm**2 / 2 + e2 * dt
            out["Dmu"][mask], out["Dmu2"][mask] = dm, dm2
            if level > 0:
                dm3 = dm * (3 * dm2 / 2 - dm**2 / 4 - e2) + 2 * difference(
                    power=2, shift=3 * t
                )
                dmt = dm * dt / 2 - dm + 2 * difference(power=2, shift=t)
                dtt = dt**2 / 2 - dt + 2 * difference(power=3)
                dm2t = dm * dmt + e2 * (dtt - 2 * dt)
                for key, value in (
                    ("Dth", dt),
                    ("Dmuth", dmt),
                    ("Dmu2th", dm2t),
                    ("Dmu3", dm3),
                ):
                    out[key][mask] = value
            if level > 1:
                a0 = 2 * (z0 * z0 * z0) * e + dm * z0**2 - 4 * z0 * e
                a1 = 2 * (z1 * z1 * z1) * e + dm * z1**2 - 4 * z1 * e
                dm4 = (
                    dm2 * (3 * dm2 / 2 - dm**2 / 4 - e2)
                    + dm * (3 * dm3 - dm * dm2) / 2
                    + difference(a0, a1, shift=3 * t)
                )
                dm3t = (
                    dmt * (3 * dm2 / 2 - dm**2 / 4)
                    + dm * (3 * dm2t - dm * dmt) / 2
                    + e2 * (2 * dm - dmt)
                    + (dt - 10) * difference(power=2, shift=3 * t)
                    + 2 * difference(power=4, shift=3 * t)
                )
                dmtt = (
                    (dmt * dt + dm * dtt) / 2
                    - dmt
                    + (dt - 6) * difference(power=2, shift=t)
                    + 2 * difference(power=4, shift=t)
                )
                dttt = (
                    dtt * (dt - 1)
                    + dt * difference(power=3)
                    + 2
                    * difference(
                        (z0 * z0 * z0) * (z0**2 - 3), (z1 * z1 * z1) * (z1**2 - 3)
                    )
                )
                dm2tt = dmt**2 + dm * dmtt + e2 * (dttt - 4 * dtt + 4 * dt)
                for key, value in (
                    ("Dmu4", dm4),
                    ("Dmu3th", dm3t),
                    ("Dth2", dtt),
                    ("Dmuth2", dmtt),
                    ("Dmu2th2", dm2tt),
                ):
                    out[key][mask] = value
        out["EDmu2"] = out["Dmu2"].copy()
        if level > 0:
            out["EDmu2th"] = out["Dmu2th"].copy()
        return out
