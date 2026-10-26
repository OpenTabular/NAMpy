"""Censored logistic location family, mgcv 1.9-4 R/efam.r::clog."""

import warnings

import numpy as np

from ._censored_location import CensoredLocationFamily
from ._function_maps import LINK_REGISTRY
from .family_base import JointOuterStrategy


def _log1pexp(x):
    """Preserve the four source branches and their thresholds."""
    x = np.asarray(x, float)
    result = x.copy()
    mask = x <= 37
    result[mask] = np.exp(x[mask])
    mask = (-37 < x) & (x <= 18)
    result[mask] = np.log1p(np.exp(x[mask]))
    mask = (18 < x) & (x <= 33.3)
    result[mask] = x[mask] + np.exp(-x[mask])
    return result


def _log1mexp(a):
    a = np.asarray(a, float)
    result = np.zeros_like(a)
    mask = (0 < a) & (a <= np.log(2))
    result[mask] = np.log(-np.expm1(-a[mask]))
    mask = a > np.log(2)
    result[mask] = np.log1p(-np.exp(-a[mask]))
    return result


class CensoredLogisticFamily(CensoredLocationFamily):
    """One location predictor and fixed or estimated logistic scale."""

    name = "clog"
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
    joint_outer_strategy = JointOuterStrategy.CLOG_THETA
    max_derivative_order = 4

    def __init__(self, theta=None, link="identity"):
        super().__init__()
        if link not in {"identity", "log", "sqrt"}:
            raise ValueError("clog link must be 'identity', 'log', or 'sqrt'.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)
        self.known_scale = 1.0
        self.n_theta = 1
        self.ini_theta = np.zeros(1)
        if theta is not None:
            value = np.asarray(theta, float).reshape(-1)
            if value.size != 1 or not np.all(np.isfinite(value)):
                raise ValueError("clog theta must contain one finite value.")
            if value[0] != 0:
                self.ini_theta = np.log(np.abs(value))
            if value[0] > 0:
                self.n_theta = 0
        self._theta_working = self.ini_theta.copy()

    def _parts(self, y, mu, weights=None, theta=None):
        values, bounds, th = self._state(y, weights, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        exact = bounds == values
        with np.errstate(invalid="ignore", over="ignore"):
            interval = np.isfinite(bounds * values) & ~exact
        return values, bounds, th, mu, exact, interval

    def _precision_scale(self, y, weights=None, theta=None):
        # Unlike cnorm's exp(theta-log(w)/2), clog explicitly computes
        # exp(theta)/sqrt(w) in every source block. Preserve that ordering.
        values, _, _ = self._state(y, weights, theta)
        w = np.ones_like(values) if weights is None else np.broadcast_to(np.asarray(weights, float), values.shape)
        working = self.getTheta() if theta is None else np.asarray(theta, float).reshape(-1)
        return np.exp(working[0]) / np.sqrt(w)

    def deviance_obs(self, y, mu, weights=None, theta=None):
        v, b, th, mu, exact, interval = self._parts(y, mu, weights, theta)
        result = np.zeros_like(v)
        s = self._precision_scale(y, weights, theta)
        z = (v[exact] - mu[exact]) / s[exact]
        result[exact] = 2 * (-2 * np.log(2) + z + 2 * _log1pexp(-z))
        lo, hi = np.minimum(v[interval], b[interval]), np.maximum(v[interval], b[interval])
        half_width = (hi - lo) / (2 * s[interval])
        lm = half_width + _log1mexp(half_width) - _log1pexp(half_width)
        ll = _log1mexp((hi - lo) / s[interval]) - _log1pexp(
            (lo - mu[interval]) / s[interval]
        ) - _log1pexp(-(hi - mu[interval]) / s[interval])
        result[interval] = 2 * (lm - ll)
        left, right = np.isneginf(b), np.isposinf(b)
        result[left] = 2 * _log1pexp(-(v[left] - mu[left]) / s[left])
        z = (v[right] - mu[right]) / s[right]
        result[right] = 2 * (z + _log1pexp(-z))
        return result

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def aic(self, y, mu, theta=None, wt=None, dev=None, *, edf=0.0, scale=1.0, weights=None):
        del dev, edf, scale
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        v, b, th, _, exact, interval = self._parts(y, mu, wt if weights is None else weights, theta)
        result = np.zeros_like(v)
        s = self._precision_scale(y, wt if weights is None else weights, theta)
        # Source aic is independent of mu and does not add dev internally.
        lm = -np.log1p(s[exact] - 1) - 2 * np.log(2)
        result[exact] = -2 * lm + 2 * th[exact]
        lo, hi = np.minimum(v[interval], b[interval]), np.maximum(v[interval], b[interval])
        z = (hi - lo) / (2 * s[interval])
        lm = z + _log1mexp(z) - _log1pexp(z)
        result[interval] = -2 * lm + 2 * th[interval]
        one_sided = np.isinf(b)
        result[one_sided] = 2 * th[one_sided]
        return float(np.sum(result))

    def weighted_loglik(self, y, mu, weights=None, scale=1.0):
        return -self.aic(y, mu, weights=weights, scale=scale) / 2

    def loglik(self, y, mu, scale=1.0):
        return self.weighted_loglik(y, mu, scale=scale)

    def update_theta_efs(self, y, mu, weights, scale=1.0):
        """Scalar fixed-dispersion branch of efam.r::estimate.theta.

        Called inside gam.fit4's EFS PIRLS loop, before its fresh derivative
        and convergence checks. No outer-optimizer approximation is involved.
        """
        theta = self.getTheta()

        def state(value, derivatives=True):
            dev = np.sum(self.deviance_obs(y, mu, weights, theta=value)) / scale
            saturated = self.ls(y, weights, theta=value, scale=scale)
            objective = dev / 2 - saturated["ls"]
            if not derivatives:
                return objective, None, None
            dd = self.Dd(y, mu, value, weights, level=2)
            gradient = np.sum(dd["Dth"]) / (2 * scale) - saturated["lsth1"][0]
            hessian = np.sum(dd["Dth2"]) / (2 * scale) - saturated["lsth2"][0, 0]
            return objective, gradient, hessian

        value, gradient, hessian = state(theta)
        if abs(gradient) > 1e-7 * (abs(value) + 1):
            for _ in range(100):
                # For one parameter the source symmetric eigendecomposition
                # is scalar; its negative eigenvalue is replaced by abs(H).
                step = np.array([-gradient / abs(hessian)])
                length = np.max(np.abs(step))
                if length > 4:
                    step = step * 4 / length
                candidate = state(theta + step)
                halvings = 0
                failed = False
                while candidate[0] - value > np.finfo(float).eps**0.75 * abs(value):
                    step = step / 2
                    halvings += 1
                    if not np.any(theta != theta + step) or halvings > 25:
                        failed = True
                        break
                    candidate = state(theta + step, derivatives=False)
                if failed:
                    warnings.warn("step failure in theta estimation", RuntimeWarning, stacklevel=2)
                    break
                theta = theta + step
                value, gradient, hessian = state(theta) if halvings else candidate
                if abs(gradient) <= 1e-7 * (abs(value) + 1):
                    break
        self.putTheta(theta)

    def ls(self, y, w, theta=None, scale=1.0):
        del scale
        v, b, th = self._state(y, w, theta)
        exact = b == v
        with np.errstate(invalid="ignore", over="ignore"):
            interval = np.isfinite(b * v) & ~exact
        s = self._precision_scale(y, w, theta)
        loglik, l1, l2 = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
        loglik[exact] = -np.log1p(s[exact] - 1) - 2 * np.log(2)
        l1[exact] = -1
        width = np.abs(v[interval] - b[interval])
        z = width / (2 * s[interval])
        loglik[interval] = z + _log1mexp(z) - _log1pexp(z)
        denominator = np.expm1(-z) - np.expm1(z)
        lmth = width / s[interval] / denominator
        l1[interval] = lmth
        l2[interval] = -lmth - 0.5 * width**2 / s[interval]**2 * (1 / denominator)**2 * (
            2 + np.expm1(-z) + np.expm1(z)
        )
        return {"ls": float(np.sum(loglik)), "lsth1": np.array([np.sum(l1)]),
                "LSTH1": l1[:, None], "lsth2": np.array([[np.sum(l2)]])}

    def Dd(self, y, mu, theta, wt, level=0):
        v, b, th, mu, exact, interval = self._parts(y, mu, wt, theta)
        names = ["Dmu", "Dmu2"]
        if level > 0:
            names += ["Dth", "Dmuth", "Dmu3", "Dmu2th"]
        if level > 1:
            names += ["Dmu4", "Dth2", "Dmuth2", "Dmu2th2", "Dmu3th"]
        out = {name: np.zeros_like(v) for name in names}
        precision_scale = self._precision_scale(y, wt, theta)
        for kind, mask in (("exact", exact), ("interval", interval),
                           ("left", np.isneginf(b)), ("right", np.isposinf(b))):
            if not np.any(mask):
                continue
            s = precision_scale[mask]
            yi, mui = v[mask], mu[mask]
            z = yi - mui
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if kind == "exact":
                    a = 1 / (2 + np.expm1(z / s))
                    dm = 2 / s * (2 * a - 1)
                    dm2 = 4 / s**2 * a * (1 - a)
                    if level > 0:
                        dmt = -dm + z * dm2
                        dt = z * dm
                        dm3 = -0.5 * dm * dm2
                        dm2t = -2 * dm2 - 0.5 * z * dm * dm2
                    if level > 1:
                        dt2 = z * dmt
                        dmt2 = -dmt + z * dm2t
                        dm4 = -0.5 * (dm2**2 + dm * dm3)
                        dm3t = -0.5 * (dm2 * dmt + dm * dm2t)
                        dm2t2 = -2 * dm2t - 0.5 * z * (dm * dm2t + dm2 * dmt)
                elif kind == "interval":
                    lo, hi = np.minimum(yi, b[mask]), np.maximum(yi, b[mask])
                    zr, zl, width = hi - mui, lo - mui, hi - lo
                    a, beta = 1 / (2 + np.expm1(-zr / s)), 1 / (2 + np.expm1(-zl / s))
                    aa, bb = a - a**2, beta - beta**2
                    ac, bc = a**2 - a**3, beta**2 - beta**3
                    dm = 2 / s * (1 - a - beta)
                    dm2 = 2 / s**2 * (aa + bb)
                    if level > 0:
                        dmt = -dm + 2 / s**2 * (zr * aa + zl * bb)
                        denom = np.expm1(-width / (2 * s)) - np.expm1(width / (2 * s))
                        lmth = width / s / denom
                        lth = 1 / s * (-width / np.expm1(width / s)
                            + zl / (np.expm1(-zl / s) + 2) - zr / (np.expm1(zr / s) + 2))
                        dt = 2 * (lmth - lth)
                        # Preserve the displayed source powers, even where a
                        # derivative rederivation would produce another form.
                        dm3 = 4 / s**2 * (ac + bc) - dm2
                        dm2t = -2 * dm2 - (dmt + dm) / s + 4 / s**3 * (zr * ac + zl * bc)
                    if level > 1:
                        lmth2 = -lmth - 0.5 * width**2 / s**2 / denom**2 * (
                            2 + np.expm1(-width / (2 * s)) + np.expm1(width / (2 * s)))
                        lth2 = -lth - 1 / s**2 * (
                            width**2 / np.expm1(width / s) / (-np.expm1(-width / s))
                            + zl**2 / (np.expm1(-zl / s) + 2) / (np.expm1(zl / s) + 2)
                            + zr**2 / (np.expm1(zr / s) + 2) / (np.expm1(-zr / s) + 2))
                        dt2 = 2 * (lmth2 - lth2)
                        dmt2 = -3 * dmt - 2 * dm - 2 / s**3 * (
                            zr**2 * (1 - 2 * a) * aa + zl**2 * (1 - 2 * beta) * bb)
                        dm4 = -4 / s**3 * ((2 * a - 3 * a**2) * aa + (2 * beta - 3 * beta**2) * bb) - dm3
                        dm3t = -2 * dm3 + 2 / s**3 * (zr * aa * (1 - 6 * a + 6 * a**2)
                            + zl * bb * (1 - 6 * beta + 6 * beta**2))
                        dm2t2 = -2 * dm2t + (dm - dmt2) / s - 12 / s**3 * (zr * ac + zl * bc) - 4 / s**4 * (
                            zr**2 * (2 * a - 3 * a**2) * aa + zl**2 * (2 * beta - 3 * beta**2) * bb)
                else:
                    a = 1 / (2 + np.expm1(-z / s))
                    dm = 2 / s * (1 - a) if kind == "left" else -2 / s * a
                    dm2 = 2 / s**2 * (a - a**2)
                    if level > 0:
                        dmt = -dm + z * dm2
                        dt = z * dm
                        dm3 = (2 * a - 1) / s * dm2
                        dm2t = -2 * dm2 + z / s * (2 * a - 1) * dm2
                    if level > 1:
                        dt2 = z * dmt
                        dmt2 = -dmt + z * dm2t
                        dm4 = -dm2**2 + (2 * a - 1) / s * dm3
                        dm3t = -z * dm2**2 + (2 * a - 1) / s * (dm2t - dm2)
                        dm2t2 = -2 * dm2t - z**2 * dm2**2 + z / s * (2 * a - 1) * (dm2t - dm2)
            out["Dmu"][mask], out["Dmu2"][mask] = dm, dm2
            if level > 0:
                for name, value in (("Dth", dt), ("Dmuth", dmt), ("Dmu3", dm3), ("Dmu2th", dm2t)):
                    out[name][mask] = value
            if level > 1:
                for name, value in (("Dth2", dt2), ("Dmuth2", dmt2), ("Dmu4", dm4),
                                    ("Dmu3th", dm3t), ("Dmu2th2", dm2t2)):
                    out[name][mask] = value
        out["EDmu2"] = np.maximum(out["Dmu2"], 0)
        if level > 0:
            out["EDmu2th"] = out["Dmu2th"].copy()
        return out
