"""Censored Box-Cox Gaussian extended family, mgcv R/efam.r::bcg."""

import numpy as np

from ._censored_location import CensoredLocationFamily, unpack_censored
from ._cnorm_helpers import dpnorm, log_dnorm, log_pnorm, logexm1, logexp1
from ._function_maps import LINK_REGISTRY
from ._scalar_math import exp, log
from .family_base import JointOuterStrategy


def box_cox(y, lam, deriv=0):
    """Source bcg nested bc: transform and first two lambda derivatives."""
    y = np.asarray(y, float)
    with np.errstate(all="ignore"):
        ly = log(y)
        yl = y**lam
        ylly = ly * yl
        if abs(lam) < 1e-7:
            yl2 = ylly * ly
            z = ylly - yl2 * lam / 2 + yl2 * ly * lam**2 / 6
        else:
            z = (yl - 1) / lam
        z = np.asarray(z).copy()
        z[(y == np.inf) & (lam >= 0)] = np.inf
        z[(y == 0) & (lam <= 0)] = -np.inf
        z1 = z2 = np.full_like(z, np.nan)
        if deriv:
            if abs(lam) < 1e-4:
                c0, c1 = ylly * ly, ylly * ly * ly
                z1 = c0 / 2 - c1 * lam / 6
                if deriv > 1:
                    z2 = c1 / 3 - c1 * ly * lam / 12
            else:
                z1 = ylly / lam - z / lam
                if deriv > 1:
                    z2 = ylly * (ly - 2 / lam) / lam + 2 * z / lam**2
            limit = ((y == np.inf) & (lam < 0)) | ((y == 0) & (lam > 0))
            z1[limit] = 1 / lam**2
            if deriv > 1:
                z2[limit] = -2 * z1[limit] / lam
            z1[((y == np.inf) & (lam >= 0)) | ((y == 0) & (lam <= 0))] = np.inf
            if deriv > 1:
                z2[(y == np.inf) & (lam >= 0)] = np.inf
                z2[(y == 0) & (lam <= 0)] = -np.inf
    return z, z1, z2


def _ddnorm(x0, x1, a0=0, a1=0, s0=1, s1=1):
    """bcg's signed normal difference (without cnorm's zero-sign branches)."""
    x0, x1, a0, a1, s0, s1 = np.broadcast_arrays(*[np.asarray(x, float) for x in (x0, x1, a0, a1, s0, s1)])
    p0, p1 = log_dnorm(x0) + a0, log_dnorm(x1) + a1
    sign = np.where(((s1 < 0) & (s0 > 0)) | ((s1 > 0) & (s0 > 0) & (p1 < p0)) | ((s1 < 0) & (s0 < 0) & (p1 > p0)), -1., 1.)
    p0, p1 = np.minimum(p0, p1), np.maximum(p0, p1)
    same = s0 * s1 > 0
    out = p0.copy()
    out[same] += logexm1(p1[same] - p0[same])
    out[~same] += logexp1(p1[~same] - p0[~same])
    return out, sign


class CensoredBoxCoxGaussianFamily(CensoredLocationFamily):
    """Box-Cox lambda and log SD ancillary parameters with explicit censoring."""

    name = "bcg"
    canonical_link = False
    supports_pirls = True
    supports_gcv = supports_ubre = False
    supports_ml = supports_reml = supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    known_scale = 1.
    max_derivative_order = 4
    joint_outer_strategy = JointOuterStrategy.BCG_THETA

    def update_theta_efs(self, y, mu, weights, scale=1.0):
        from ._estimate_theta import update_theta_efs

        update_theta_efs(self, y, mu, weights, scale=scale)

    def __init__(self, theta=None, link="identity"):
        super().__init__()
        if link not in {"identity", "log", "sqrt"}:
            raise ValueError("bcg link must be identity, log or sqrt.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)
        self.n_theta = 2
        self.ini_theta = np.array([1., 0.])
        if theta is not None:
            value = np.asarray(theta, float).reshape(-1)
            if value.size != 2 or np.any(~np.isfinite(value)):
                raise ValueError("bcg theta must have length two and be finite.")
            if value[1] <= 0:
                # Source copies iniTheta before converting theta[2].
                self.ini_theta = value.copy()
            else:
                # Preserve c(theta[1], log(theta)), including its third entry.
                self.ini_theta = np.r_[value[0], log(value)]
                self.n_theta = 0
        self._theta_working = self.ini_theta.copy()

    def getTheta(self, trans=False):
        value = self._theta_working.copy()
        if trans:
            value[1] = exp(value[1])
        return value

    def putTheta(self, theta):
        value = np.asarray(theta, float).reshape(-1)
        if value.size not in {2, 3} or np.any(~np.isfinite(value)):
            raise ValueError("bcg working theta must contain two or three finite values.")
        self._theta_working = value.copy()

    def validate_y(self, y):
        result = super().validate_y(y)
        if np.any(result.values < 0):
            raise ValueError("response must be non-negative")
        return result

    def _bcg_state(self, y, weights=None, theta=None):
        values, bounds = unpack_censored(y)
        w = np.ones_like(values) if weights is None else np.broadcast_to(np.asarray(weights, float), values.shape)
        theta = self.getTheta() if theta is None else np.asarray(theta, float).reshape(-1)
        if theta.size not in {2, 3}:
            raise ValueError("bcg working theta must have length two or three.")
        return values, bounds, theta[1] - log(w) / 2, theta[0]

    def deviance_obs(self, y, mu, weights=None, theta=None):
        values, bounds, th, lam = self._bcg_state(y, weights, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth = exp(-th)
        out = np.zeros_like(values)
        exact = bounds == values
        out[exact] = (box_cox(values[exact], lam)[0] - mu[exact])**2 * exp(-2 * th[exact])
        interval = np.isfinite(bounds) & (bounds > 0) & ~exact
        lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
        z0, z1 = box_cox(lo, lam)[0], box_cox(hi, lam)[0]
        zz = (z1 - z0) * eth[interval] / 2
        out[interval] = 2 * dpnorm(-zz, zz) - 2 * dpnorm((z0 - mu[interval]) * eth[interval], (z1 - mu[interval]) * eth[interval])
        left, right = bounds <= 0, np.isposinf(bounds)
        out[left] = -2 * log_pnorm((box_cox(values[left], lam)[0] - mu[left]) * eth[left])
        out[right] = -2 * log_pnorm(-(box_cox(values[right], lam)[0] - mu[right]) * eth[right])
        return out

    def deviance_residual_sign(self, y, mu):
        values, _ = unpack_censored(y)
        return np.sign(box_cox(values, self.getTheta()[0])[0] - np.asarray(mu, float))

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def aic(self, y, mu, theta=None, wt=None, dev=None, *, edf=0., scale=1., weights=None):
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        values, bounds, th, lam = self._bcg_state(y, wt if weights is None else weights, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth = exp(-th)
        out = np.zeros_like(values)
        exact = bounds == values
        out[exact] = -2 * log_dnorm((box_cox(values[exact], lam)[0] - mu[exact]) * eth[exact])
        interval = np.isfinite(bounds) & (bounds > 0) & ~exact
        lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
        out[interval] = -2 * dpnorm((box_cox(lo, lam)[0] - mu[interval]) * eth[interval], (box_cox(hi, lam)[0] - mu[interval]) * eth[interval])
        left, right = bounds <= 0, np.isposinf(bounds)
        out[left] = -2 * log_pnorm((box_cox(values[left], lam)[0] - mu[left]) * eth[left])
        out[right] = -2 * log_pnorm(-(box_cox(values[right], lam)[0] - mu[right]) * eth[right])
        return float(np.sum(out))

    def weighted_loglik(self, y, mu, weights=None, scale=1.):
        return -self.aic(y, mu, weights=weights) / 2

    def loglik(self, y, mu, scale=1.):
        return self.weighted_loglik(y, mu, scale=scale)

    def ls(self, y, w, theta=None, scale=1.):
        values, bounds, th, lam = self._bcg_state(y, w, theta)
        exact = bounds == values
        value = np.sum((lam - 1) * log(values[exact]) - th[exact] - log(2 * np.pi) / 2)
        interval = np.isfinite(bounds) & (bounds > 0) & ~exact
        lo, hi = np.minimum(values[interval], bounds[interval]), np.maximum(values[interval], bounds[interval])
        zz = (box_cox(hi, lam)[0] - box_cox(lo, lam)[0]) * exp(-th[interval]) / 2
        value += np.sum(dpnorm(-zz, zz))
        return {"ls": float(value), "lsth1": np.zeros(2), "LSTH1": np.zeros((len(values), 2)), "lsth2": np.zeros((2, 2))}

    def Dd(self, y, mu, theta, wt, level=0):
        values, bounds, th, lam = self._bcg_state(y, wt, theta)
        mu = np.broadcast_to(np.asarray(mu, float), values.shape)
        eth, e2th = exp(-th), exp(-th)**2
        out = {"Dmu": np.zeros_like(values), "Dmu2": np.zeros_like(values)}
        if level > 0:
            out.update({key: np.zeros((len(values), 2)) for key in ("Dth", "Dmuth", "Dmu2th")})
            out["Dmu3"] = np.zeros_like(values)
        if level > 1:
            out.update({key: np.zeros((len(values), 3)) for key in ("Dth2", "Dmuth2", "Dmu2th2")})
            out["Dmu3th"] = np.zeros((len(values), 2))
            out["Dmu4"] = np.zeros_like(values)
        exact = bounds == values
        b, bl, bll = box_cox(values[exact], lam, level)
        e, e2 = eth[exact], e2th[exact]
        z = (b - mu[exact]) * e
        dm = -2 * z * e
        out["Dmu"][exact], out["Dmu2"][exact] = dm, 2 * e2
        if level > 0:
            out["Dth"][exact] = np.column_stack([-2 * (log(values[exact]) - e * z * bl), -2 * (z**2 - 1)])
            out["Dmuth"][exact] = np.column_stack([-2 * e2 * bl, -2 * dm])
            out["Dmu2th"][exact, 1] = -4 * e2
        if level > 1:
            out["Dth2"][exact] = np.column_stack([2 * (e2 * bl**2 + e * z * bll), -4 * e * z * bl, 4 * z**2])
            out["Dmuth2"][exact] = np.column_stack([-2 * e2 * bll, 4 * e2 * bl, 4 * dm])
            out["Dmu2th2"][exact, 2] = 8 * e2
        for case, mask in (("interval", np.isfinite(bounds) & (bounds > 0) & ~exact), ("left", bounds <= 0), ("right", np.isposinf(bounds))):
            if not np.any(mask):
                continue
            t, e, e2 = th[mask], eth[mask], e2th[mask]
            if case == "interval":
                b0, bl0, bll0 = box_cox(np.minimum(values[mask], bounds[mask]), lam, level)
                b1, bl1, bll1 = box_cox(np.maximum(values[mask], bounds[mask]), lam, level)
                z0, z1 = (b0 - mu[mask]) * e, (b1 - mu[mask]) * e
                lp = dpnorm(z0, z1)
            else:
                b, bl, bll = box_cox(values[mask], lam, level)
                z0 = z1 = (b - mu[mask]) * e
                bl0 = bl1 = bl
                bll0 = bll1 = bll
                lp = log_pnorm(z1 if case == "left" else -z0)

            def diff(a0=1., a1=1., shift=0, *, case=case, z0=z0, z1=z1, lp=lp):
                a0, a1 = np.broadcast_arrays(np.asarray(a0, float), np.asarray(a1, float), z0)[:2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    if case == "interval":
                        ld, sign = _ddnorm(z0, z1, log(np.abs(a0)), log(np.abs(a1)), np.sign(a0), np.sign(a1))
                    else:
                        a = a1 if case == "left" else a0
                        ld, sign = log_dnorm(z1) + log(np.abs(a)), np.sign(a) * (1 if case == "left" else -1)
                    return sign * exp(ld - lp - shift)

            dm = 2 * diff(shift=t)
            dt = 2 * diff(z0, z1)
            dm2 = dm**2 / 2 + e2 * dt
            out["Dmu"][mask], out["Dmu2"][mask] = dm, dm2
            if level > 0:
                dm3 = dm * (3 * dm2 / 2 - dm**2 / 4 - e2) + 2 * diff(z0**2, z1**2, 3 * t)
                dl = -2 * diff(bl0, bl1, t)
                dml = dl * dm / 2 - 2 * diff(bl0 * z0, bl1 * z1, 2 * t)
                dmt = dm * dt / 2 - dm + 2 * diff(z0**2, z1**2, t)
                dtt = dt**2 / 2 - dt + 2 * diff(z0**3, z1**3)
                dlt = dl * dt / 2 - dl - 2 * diff(bl0 * z0**2, bl1 * z1**2, t)
                dm2l, dm2t = dm * dml + e2 * dlt, dm * dmt + e2 * (dtt - 2 * dt)
                out["Dmu3"][mask] = dm3
                out["Dth"][mask] = np.column_stack([dl, dt])
                out["Dmuth"][mask] = np.column_stack([dml, dmt])
                out["Dmu2th"][mask] = np.column_stack([dm2l, dm2t])
            if level > 1:
                a0, a1 = 2 * z0**3 * e + dm * z0**2 - 4 * z0 * e, 2 * z1**3 * e + dm * z1**2 - 4 * z1 * e
                dm4 = dm2 * (3 * dm2 / 2 - dm**2 / 4 - e2) + dm * (3 * dm3 - dm * dm2) / 2 + diff(a0, a1, 3 * t)
                dm3l = dml * (3 * dm2 / 2 - dm**2 / 4) + dm * (3 * dm2l - dm * dml) / 2 - e2 * dml + dl * diff(z0**2, z1**2, 3 * t) + 2 * diff(bl0 * z0 * (2 - z0**2), bl1 * z1 * (2 - z1**2), 4 * t)
                dm3t = dmt * (3 * dm2 / 2 - dm**2 / 4) + dm * (3 * dm2t - dm * dmt) / 2 + e2 * (2 * dm - dmt) + (dt - 10) * diff(z0**2, z1**2, 3 * t) + 2 * diff(z0**4, z1**4, 3 * t)
                dll = dl**2 / 2 + 2 * diff(bl0**2 * z0 * e - bll0, bl1**2 * z1 * e - bll1, t)
                dmll = (dl * dml + dm * dll) / 2 - dl * diff(bl0 * z0, bl1 * z1, 2 * t) + 2 * diff(bl0**2 * (z0**2 - 1) * e - bll0 * z0, bl1**2 * (z1**2 - 1) * e - bll1 * z1, 2 * t)
                dmlt = (dl * dmt + dm * dlt) / 2 + (6 - dt) * diff(bl0 * z0, bl1 * z1, 2 * t) - 2 * diff(bl0 * z0**3, bl1 * z1**3, 2 * t)
                dmtt = (dmt * dt + dm * dtt) / 2 - dmt + (dt - 6) * diff(z0**2, z1**2, t) + 2 * diff(z0**4, z1**4, t)
                dllt = (dl * dlt + dt * dll) / 2 - dll - dl * diff(bl0 * z0**2, bl1 * z1**2, t) + 2 * diff(z0 * (bl0**2 * (z0**2 - 2) * e - bll0 * z0), z1 * (bl1**2 * (z1**2 - 2) * e - bll1 * z1), t)
                dltt = (dl * dtt + dt * dlt) / 2 - dlt + (6 - dt) * diff(bl0 * z0**2, bl1 * z1**2, t) - 2 * diff(bl0 * z0**4, bl1 * z1**4, t)
                dttt = dtt * (dt - 1) + dt * diff(z0**3, z1**3) + 2 * diff(z0**3 * (z0**2 - 3), z1**3 * (z1**2 - 3))
                out["Dmu4"][mask] = dm4
                out["Dmu3th"][mask] = np.column_stack([dm3l, dm3t])
                out["Dth2"][mask] = np.column_stack([dll, dlt, dtt])
                out["Dmuth2"][mask] = np.column_stack([dmll, dmlt, dmtt])
                out["Dmu2th2"][mask] = np.column_stack([dml**2 + dm * dmll + e2 * dllt, dml * dmt + dm * dmlt + e2 * (dltt - 2 * dlt), dmt**2 + dm * dmtt + e2 * (dttt - 4 * dtt + 4 * dt)])
        out["EDmu2"] = out["Dmu2"].copy()
        if level > 0:
            out["EDmu2th"] = out["Dmu2th"].copy()
        return out
