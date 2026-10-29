"""Single-predictor hurdle Poisson family from mgcv/R/efam.r::ziP.

References
----------
- Lambert, D. (1992). Zero-Inflated Poisson Regression, With an Application to Defects
  in Manufacturing.
  https://doi.org/10.1080/00401706.1992.10485228
- Wood, S. N., Pya, N., and Säfken, B. (2016). Smoothing Parameter and Model Selection
  for General Smooth Models.
  https://doi.org/10.1080/01621459.2016.1180986
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from ._function_maps import LINK_REGISTRY
from ._zip_helpers import zipll
from .family_base import ExtendedFamily, JointOuterStrategy


class ZeroInflatedPoissonFamily(ExtendedFamily):
    """Model log Poisson intensity with a linked probability of presence.

    ``theta`` is the *working* intercept/log-slope pair. Any supplied pair
    fixes both parameters; omitted theta estimates both, starting at zero.
    Solver fitted values are log intensity, not expected response counts.
    """

    pirls_curvature_mode = "observed"
    name = "zip"
    canonical_link = False
    supports_pirls = True
    supports_ml = True
    supports_reml = True
    supports_laml = True
    supports_exact_pirls_first_derivatives = True
    supports_exact_pirls_second_derivatives = True
    joint_outer_strategy = JointOuterStrategy.ZIP_THETA
    known_scale = 1.0
    max_derivative_order = 4
    no_r_sq = True

    def __init__(self, theta=None, link="identity", b=0.0):
        super().__init__()
        if link != "identity":
            raise ValueError("ziP supports only the identity link.")
        self.link_name = link
        self.link = LINK_REGISTRY[link](eps=self.eps)
        if not np.isscalar(b) or not np.isfinite(b):
            raise ValueError("ziP b must be a finite scalar.")
        self.b = max(float(b), 0.0)
        self.n_theta = 2 if theta is None else 0
        self.ini_theta = np.zeros(2) if theta is None else self._check_theta(theta)
        self._theta_working = self.ini_theta.copy()
        # Unlike SCAT, ziP has no data-dependent preinitialize expression.
        self._initialized_from_data = True

    @staticmethod
    def _check_theta(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,) or not np.isfinite(theta).all():
            raise ValueError("ziP theta must contain two finite working values.")
        return theta.copy()

    @property
    def estimate_theta(self):
        return self.n_theta > 0

    def getTheta(self, trans=False):
        theta = self._theta_working.copy()
        if trans:
            theta[1] = self.b + np.exp(theta[1])
        return theta

    def putTheta(self, theta):
        self._theta_working = self._check_theta(theta)

    def _presence_predictor(self, mu, theta=None):
        theta = self.getTheta() if theta is None else np.asarray(theta, dtype=float)
        return theta[0] + (self.b + np.exp(theta[1])) * np.asarray(mu, dtype=float)

    def validate_y(self, y):
        y = super().validate_y(y)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the zero inflated Poisson family"
            )
        # R all.equal.numeric uses mean absolute difference, scaled by the
        # mean absolute target unless that target is close to zero.
        rounded = np.round(y)
        difference = np.abs(y - rounded)
        changed = difference > 0
        if np.any(changed):
            error = float(np.mean(difference[changed]))
            scale = float(np.mean(np.abs(y[changed])))
            if scale > np.sqrt(np.finfo(float).eps):
                error /= scale
            if error > np.sqrt(np.finfo(float).eps):
                raise ValueError(
                    "Non-integer response variables are not allowed with ziP"
                )
        if y.size and np.min(y) == 0 and np.max(y) == 1:
            raise ValueError("Using ziP for binary data makes no sense")
        return y

    def initialize_mu(self, y, weights=None):
        del weights
        y = self.validate_y(y)
        return np.log(y + (y == 0) / 5)

    def initialize_linear_predictors(self, y):
        return [self.initialize_mu(y)]

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        return self.link.mu_eta(eta)

    def valid_mu(self, mu):
        return bool(np.isfinite(mu).all())

    def valid_eta(self, eta):
        return True

    @staticmethod
    def _weights(y, weights=None):
        return (
            np.ones(np.shape(y))
            if weights is None
            else np.broadcast_to(np.asarray(weights, dtype=float), np.shape(y))
        )

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        return zipll(y, mu, self._presence_predictor(mu))["l"]

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale)))

    def deviance_obs(self, y, mu, weights=None, theta=None):
        # Upstream dev.resids intentionally ignores wt; Dd and aic do not.
        del weights
        return -2 * zipll(y, mu, self._presence_predictor(mu, theta))["l"]

    def deviance(self, y, mu, weights=None):
        return float(np.sum(self.deviance_obs(y, mu, weights)))

    def aic(
        self, y, mu, theta=None, wt=None, dev=None, *, edf=0.0, scale=1.0, weights=None
    ):
        del dev, edf, scale
        if wt is not None and weights is not None:
            raise TypeError("pass only one of wt or weights")
        w = self._weights(y, wt if weights is None else weights)
        return float(np.sum(w * self.deviance_obs(y, mu, theta=theta)))

    def ls(self, y, w, theta=None, scale=1.0):
        return {
            "ls": 0.0,
            "lsth1": np.zeros(2),
            "LSTH1": np.zeros((np.size(y), 2)),
            "lsth2": np.zeros((2, 2)),
        }

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        return 1.0

    def Dd(self, y, mu, theta=None, wt=None, level=0):
        y, mu = np.broadcast_arrays(
            np.atleast_1d(np.asarray(y, dtype=float)),
            np.atleast_1d(np.asarray(mu, dtype=float)),
        )
        theta = self.getTheta() if theta is None else np.asarray(theta, dtype=float)
        w = self._weights(y, wt)
        e = np.exp(theta[1])
        s = self.b + e
        z = zipll(y, mu, theta[0] + s * mu, 1 if level == 0 else 2 if level == 1 else 4)
        # zipll has identically zero gamma/presence mixed derivatives, and
        # lind has zero second and higher mu derivatives. These are the
        # surviving terms in efam.r::ziP$Dd, retaining packed theta order.
        le1 = z["l1"][:, 1]
        le2 = z["l2"][:, 2]
        out = {
            "Dmu": -2 * w * (z["l1"][:, 0] + le1 * s),
            "Dmu2": -2 * w * (z["l2"][:, 0] + le2 * s**2),
            "EDmu2": -2 * w * (z["El2"][:, 0] + z["El2"][:, 2] * s**2),
        }
        if level > 0:
            a = np.column_stack((np.ones(y.size), e * mu))
            c = np.column_stack((np.zeros(y.size), np.full(y.size, e)))
            le3 = z["l3"][:, 3]
            wc = -2 * w[:, None]
            out["Dth"] = wc * le1[:, None] * a
            out["Dmuth"] = wc * (le2[:, None] * s * a + le1[:, None] * c)
            out["Dmu2th"] = wc * (le3[:, None] * s**2 * a + le2[:, None] * 2 * s * c)
            out["Dmu3"] = -2 * w * (z["l3"][:, 0] + le3 * s**3)
        if level > 1:
            aa = np.column_stack((a[:, 0] ** 2, a[:, 0] * a[:, 1], a[:, 1] ** 2))
            a2 = np.column_stack((np.zeros(y.size), np.zeros(y.size), e * mu))
            c2 = np.column_stack(
                (np.zeros(y.size), np.zeros(y.size), np.full(y.size, e))
            )
            ac = np.column_stack(
                (
                    2 * a[:, 0] * c[:, 0],
                    a[:, 0] * c[:, 1] + a[:, 1] * c[:, 0],
                    2 * a[:, 1] * c[:, 1],
                )
            )
            cc = np.column_stack(
                (2 * c[:, 0] ** 2, 2 * c[:, 0] * c[:, 1], 2 * c[:, 1] ** 2)
            )
            le4 = z["l4"][:, 4]
            out["Dth2"] = wc * (le2[:, None] * aa + le1[:, None] * a2)
            out["Dmuth2"] = wc * (
                le3[:, None] * s * aa + le2[:, None] * (s * a2 + ac) + le1[:, None] * c2
            )
            out["Dmu2th2"] = wc * (
                le4[:, None] * s**2 * aa
                + le3[:, None] * (s**2 * a2 + 2 * s * ac)
                + le2[:, None] * (cc + 2 * s * c2)
            )
            out["Dmu3th"] = wc * (le4[:, None] * s**3 * a + 3 * le3[:, None] * s**2 * c)
            out["Dmu4"] = -2 * w * (z["l4"][:, 0] + le4 * s**4)
        return out

    def deviance_derivatives_mu(self, y, mu, order=1):
        return self.Dd(y, mu, level=0 if order < 3 else order - 2)[
            "Dmu" if order == 1 else f"Dmu{order}"
        ]

    def score_eta(self, y, eta):
        return -0.5 * self.Dd(y, eta)["Dmu"]

    def hessian_eta(self, y, eta):
        return -0.5 * self.Dd(y, eta)["Dmu2"]

    def _response_parts(self, eta):
        gamma = np.atleast_1d(np.asarray(eta, dtype=float))
        with np.errstate(all="ignore"):
            et = np.exp(self._presence_predictor(gamma))
            p = 1 - np.exp(-et)
            lam = np.exp(gamma)
            mean = p.copy()
            small = gamma < np.log(np.finfo(float).eps) / 2
            mean[~small] = lam[~small] / (1 - np.exp(-lam[~small]))
            mean[small] = 1
        return p, mean, et, lam

    def response_from_eta(self, eta):
        p, mean, _, _ = self._response_parts(eta)
        return p * mean

    def response_se_from_eta(self, eta, se_eta):
        p, mean, et, lam = self._response_parts(eta)
        with np.errstate(all="ignore"):
            dp = np.exp(-et) * et * (self.b + np.exp(self.getTheta()[1]))
            dmean = (lam + 1) * mean - mean**2
            return np.abs(dp * mean + dmean * p) * np.asarray(se_eta)

    def saturated_deviance_obs(self, y, weights=None):
        y = np.asarray(y, dtype=float).ravel()
        positive = y > 0
        result = np.zeros(y.size)
        if not np.any(positive):
            return result
        w = self._weights(y, weights)[positive]
        yp = y[positive]
        mu = np.log(yp)
        theta = self.getTheta()
        r = self.Dd(yp, mu, theta, w)
        value = self.deviance_obs(yp, mu, w, theta)
        threshold = np.max(np.abs(value)) * 1e-7
        unconverged = np.abs(r["Dmu"]) > threshold
        for _ in range(100):
            with np.errstate(all="ignore"):
                step = -r["Dmu"] / r["Dmu2"]
            step[~unconverged] = 0
            mu1 = mu + step
            v1 = self.deviance_obs(yp, mu1, w, theta)
            bad = (v1 > value) & unconverged
            for _half in range(50):
                if not np.any(bad):
                    break
                step[bad] /= 2
                mu1 = mu + step
                v1 = self.deviance_obs(yp, mu1, w, theta)
                bad = (v1 > value) & unconverged
            mu, value = mu1, v1
            r = self.Dd(yp, mu, theta, w)
            unconverged = np.abs(r["Dmu"]) > threshold
            if not np.any(unconverged):
                break
        result[positive] = value
        return result

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        return float(-0.5 * np.sum(self.saturated_deviance_obs(y, weights)))

    def postprocess_deviance(self, y, mu, weights=None):
        return float(
            np.sum(
                self.deviance_obs(y, mu, weights)
                - self.saturated_deviance_obs(y, weights)
            )
        )

    def null_deviance(self, y, mu, weights=None, offset=None, intercept=True):
        del mu, offset, intercept
        y = np.asarray(y, dtype=float)
        mean = np.mean(y)
        if mean == 0:
            raise ValueError(
                "ziP null-deviance optimization requires a nonzero response."
            )
        out = minimize_scalar(
            lambda g: self.deviance(y, np.full(y.shape, g), weights),
            bounds=(mean / 5, mean * 3),
            method="bounded",
            options={"xatol": np.finfo(float).eps ** 0.25},
        )
        return float(out.fun - np.sum(self.saturated_deviance_obs(y, weights)))

    def residuals(self, model, rtype="deviance"):
        from ..model_state import _fit_state, _fitted_eta

        y = np.asarray(model.y_, dtype=float)
        eta = np.asarray(_fitted_eta(model), dtype=float)
        weights = self._weights(y, getattr(model, "prior_weights_", None))
        if rtype == "response":
            return y - np.asarray(model.predict(type="response"), dtype=float)
        if rtype == "deviance":
            value = self.deviance_obs(y, eta, weights) - self.saturated_deviance_obs(
                y, weights
            )
            return np.sqrt(np.maximum(value, 0)) * np.sign(
                y - np.asarray(model.predict(type="response"), dtype=float)
            )
        if rtype == "working":
            state = _fit_state(model)
            if state is None or state.working_response is None:
                raise RuntimeError(
                    "ziP working residuals require stored working response."
                )
            result = np.asarray(state.working_response, dtype=float) - eta
            offset = getattr(model, "offset_train_", None)
            if offset is not None:
                result = result + np.asarray(offset)
            result[~np.isfinite(result)] = np.nan
            return result
        raise NotImplementedError(
            "ziP residuals support deviance, response, and working only."
        )

    def rd(self, mu, wt=None, scale=1.0, *, rng=None):
        del wt, scale
        rng = np.random.default_rng() if rng is None else rng
        gamma = np.atleast_1d(np.asarray(mu, dtype=float))
        lam = np.exp(gamma)
        finite = lam[np.isfinite(lam)]
        maximum = max(np.max(finite) if finite.size else 0, np.finfo(float).eps ** 0.2)
        lam = np.where(np.isfinite(lam), lam, maximum)
        p = 1 - np.exp(-np.exp(self._presence_predictor(gamma)))
        present = p > rng.uniform(size=gamma.size)
        y = np.zeros(gamma.shape)
        lami = lam[present]
        p0 = poisson.pmf(0, lami)
        near = 1 - np.finfo(float).eps * 10
        tiny = p0 > near
        values = p0.copy()
        values[tiny] = 1
        values[~tiny] = poisson.ppf(rng.uniform(p0[~tiny], near), lami[~tiny])
        y[present] = values
        return y
