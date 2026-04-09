"""
Family base classes for GAM smooth models.

Three family tiers are defined here:

:class:`BaseFamily`
    Abstract root class.  Declares capability flags (``supports_gcv``,
    ``supports_reml``, …) and response validation.

:class:`GLMFamily`
    Standard single-linear-predictor GLM family.  Provides link / inverse-link /
    variance / deviance / log-likelihood interface.  Fit via exact Gaussian solver
    (when ``supports_closed_form_solve = True``) or penalized IRLS.

:class:`ExtendedFamily`
    Non-standard single-predictor likelihoods requiring a bespoke solver.
    Not yet implemented in the fitting backends.

:class:`GeneralFamily`
    Multi-linear-predictor families (e.g. GAMLSS-style location-scale models).
    Not yet implemented.
"""

import abc

import numpy as np
from scipy.special import gammaln

_EPS = 1e-9
_CAPABILITY_FLAGS = (
    "supports_closed_form_solve",
    "supports_pirls",
    "supports_gcv",
    "supports_ubre",
    "supports_ml",
    "supports_reml",
    "supports_laml",
    "supports_ncv",
    "supports_qncv",
    "supports_exact_pirls_first_derivatives",
    "supports_exact_pirls_second_derivatives",
)


class _FamilyMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        type.__setattr__(cls, "_capability_flags_locked", True)
        return cls

    def __setattr__(cls, name, value):
        if (
            getattr(cls, "_capability_flags_locked", False)
            and name in _CAPABILITY_FLAGS
        ):
            raise AttributeError(f"{name} is read-only on family classes.")
        super().__setattr__(name, value)


class BaseFamily(metaclass=_FamilyMeta):
    """
    Abstract root class for all GAM families.

    Declares capability flags used by the fitting orchestrator and smoothness
    selection backends to choose the right solver and scoring method.
    """

    name = "base"
    link_name = "identity"
    family_class = "base"

    supports_closed_form_solve = False
    supports_pirls = False

    supports_gcv = False
    supports_ubre = False
    supports_ml = False
    supports_reml = False
    supports_laml = False
    supports_ncv = False
    supports_qncv = False
    supports_exact_pirls_first_derivatives = False
    supports_exact_pirls_second_derivatives = False

    n_linear_predictors = 1
    known_scale = None  # None -> unknown; numeric -> fixed/known scale
    max_derivative_order = 0

    def __init__(self, eps: float = _EPS):
        self.eps = float(eps)

    def __setattr__(self, name, value):
        if name in _CAPABILITY_FLAGS and hasattr(self, name):
            raise AttributeError(f"{name} is read-only on family instances.")
        super().__setattr__(name, value)

    def validate_y(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or Inf")
        return y

    def validate_predictor_count(self, n_predictors: int):
        if int(n_predictors) != int(self.n_linear_predictors):
            raise ValueError(
                f"Family {self.name!r} expects {self.n_linear_predictors} linear predictor(s), "
                f"got {n_predictors}."
            )

    def supports_method(self, method: str) -> bool:
        method = str(method).lower()
        attr_map = {
            "fixed": None,
            "gcv": "supports_gcv",
            "ubre": "supports_ubre",
            "aic": "supports_ubre",
            "ubreaic": "supports_ubre",
            "ml": "supports_ml",
            "reml": "supports_reml",
            "laml": "supports_laml",
            "ncv": "supports_ncv",
            "qncv": "supports_qncv",
        }
        if method not in attr_map:
            raise ValueError(
                "method must be one of "
                "{'fixed', 'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml', 'ncv', 'qncv'}"
            )
        attr = attr_map[method]
        if attr is None:
            return True
        return bool(getattr(self, attr, False))

    # ------------------------------------------------------------------
    # Future derivative contracts
    # ------------------------------------------------------------------
    def inverse_link_derivatives(self, eta, order=1):
        """
        Derivatives of inverse link mu(eta) w.r.t. eta.

        Phase 1 only guarantees order=1 for GLM-style families via mu_eta().
        Higher orders are intentionally deferred to later phases where they will
        be hardcoded and tested carefully.
        """
        if int(order) != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet implement inverse-link "
                f"derivatives of order {order}."
            )
        return self.mu_eta(eta)

    def deviance_derivatives_mu(self, y, mu, order=1):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement deviance derivatives."
        )

    def working_weight_derivative_eta(self, eta, y=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement working-weight derivatives."
        )

    def working_weight_second_derivative_eta(self, eta, y=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement second working-weight derivatives."
        )

    def initialize_linear_predictors(self, y):
        """
        Default single-predictor initialization for GLM-style families.
        """
        mu0 = self.initialize_mu(y)
        return [self.link(mu0)]


class GLMFamily(BaseFamily):
    """
    One-linear-predictor family contract for exact Gaussian / PIRLS GAM fitting.
    """

    family_class = "glm"
    n_linear_predictors = 1
    canonical_link = False

    _link_key: str | None = None
    _variance_key: str | None = None

    def __init__(self, eps: float = _EPS):
        super().__init__(eps=eps)
        from ._function_maps import LINK_REGISTRY, VARIANCE_REGISTRY

        if self._link_key is not None:
            self.link = LINK_REGISTRY[self._link_key](eps=self.eps)
        if self._variance_key is not None:
            self.variance = VARIANCE_REGISTRY[self._variance_key](eps=self.eps)

    def initialize_mu(self, y):
        raise NotImplementedError

    def link(self, mu):
        raise NotImplementedError(
            f"{self.__class__.__name__} must assign self.link to a LinkFunction."
        )

    def inverse_link(self, eta):
        return self.link.inverse(eta)

    def mu_eta(self, eta):
        """d mu / d eta"""
        return self.link.mu_eta(eta)

    def variance(self, mu):
        raise NotImplementedError(
            f"{self.__class__.__name__} must assign self.variance to a VarianceFunction."
        )

    def _check_weights(self, y, weights=None):
        y = np.asarray(y, dtype=np.float64)
        if weights is None:
            return np.ones_like(y, dtype=np.float64)
        return np.asarray(weights, dtype=np.float64)

    def dvar(self, mu):
        return self.variance.d1(mu)

    def d2var(self, mu):
        return self.variance.d2(mu)

    def d3var(self, mu):
        return self.variance.d3(mu)

    def d2link(self, mu):
        return self.link.d2(mu)

    def d3link(self, mu):
        return self.link.d3(mu)

    def d4link(self, mu):
        return self.link.d4(mu)

    @abc.abstractmethod
    def deviance(self, y, mu, weights=None):
        raise NotImplementedError

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        if self.known_scale is not None:
            return float(self.known_scale)
        return 1.0

    def loglik_obs(self, y, mu, scale=1.0):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement per-observation log-likelihoods."
        )

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def aic(self, y, mu, *, edf=0.0, scale=1.0, weights=None):
        loglik_obs = np.asarray(self.loglik_obs(y, mu, scale=scale), dtype=np.float64)
        obs_weights = self._check_weights(loglik_obs, weights)
        return float(-2.0 * np.sum(obs_weights * loglik_obs) + 2.0 * float(edf))

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement the saturated log-likelihood "
            "term required for Laplace ML/REML criteria."
        )


class _BinomialBase(GLMFamily):
    _variance_key = "binomial"

    def deviance(self, y, mu, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        weights = self._check_weights(y, weights)
        term1 = np.zeros_like(y, dtype=np.float64)
        mask1 = y > 0
        term1[mask1] = y[mask1] * np.log(y[mask1] / mu[mask1])
        term2 = np.zeros_like(y, dtype=np.float64)
        mask2 = y < 1
        term2[mask2] = (1.0 - y[mask2]) * np.log((1.0 - y[mask2]) / (1.0 - mu[mask2]))
        return float(2.0 * np.sum(weights * (term1 + term2)))

    def loglik_obs(self, y, mu, scale=1.0):
        del scale
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, 1.0 - self.eps)
        return y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del scale, n
        y = np.asarray(y, dtype=np.float64)
        weights = self._check_weights(y, weights)
        term = np.zeros_like(y, dtype=np.float64)
        mask1 = y > 0.0
        term[mask1] += y[mask1] * np.log(y[mask1])
        mask2 = y < 1.0
        term[mask2] += (1.0 - y[mask2]) * np.log(1.0 - y[mask2])
        return float(np.sum(weights * term))


class _GammaBase(GLMFamily):
    _variance_key = "gamma"

    def deviance(self, y, mu, weights=None):
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        weights = self._check_weights(y, weights)
        return float(2.0 * np.sum(weights * ((y - mu) / mu - np.log(y / mu))))

    def estimate_dispersion(self, y, mu, edf=None, weights=None):
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        w = self._check_weights(y, weights)
        pearson = float(np.sum(w * (y - mu) ** 2 / self.variance(mu)))
        w_sum = float(np.sum(w))
        if edf is None:
            return pearson / max(w_sum, 1.0)
        return pearson / max(w_sum - float(edf), 1.0)

    def loglik_obs(self, y, mu, scale=1.0):
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        mu = np.clip(np.asarray(mu, dtype=np.float64), self.eps, None)
        scale = float(max(scale, self.eps))
        shape = 1.0 / scale
        return (
            (shape - 1.0) * np.log(y)
            - y * shape / mu
            - gammaln(shape)
            - shape * np.log(mu / shape)
        )

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        del n
        y = np.clip(np.asarray(y, dtype=np.float64), self.eps, None)
        weights = self._check_weights(y, weights)
        scale = float(max(scale, self.eps))
        shape = 1.0 / scale
        sat = -np.log(y) - shape - gammaln(shape) + shape * np.log(shape)
        return float(np.sum(weights * sat))


class ExtendedFamily(BaseFamily):
    """
    Contract for extended exponential-family models:
    - one linear predictor
    - richer likelihood structure than ordinary GLM families
    - will later supply higher-order derivatives for outer REML/LAML machinery
    """

    family_class = "extended"
    n_linear_predictors = 1

    def initialize_linear_predictors(self, y):
        raise NotImplementedError

    def loglik_obs(self, y, eta):
        raise NotImplementedError

    def score_eta(self, y, eta):
        raise NotImplementedError

    def hessian_eta(self, y, eta):
        raise NotImplementedError


class GeneralFamily(BaseFamily):
    """
    Contract for general (non-exponential) family models:
    - potentially multiple linear predictors
    - classical LSS / distributional models belong here, not in the neural path
    """

    family_class = "general"

    def initialize_linear_predictors(self, y):
        raise NotImplementedError

    def loglik_obs(self, y, eta_list):
        raise NotImplementedError

    def score_eta(self, y, eta_list):
        raise NotImplementedError

    def hessian_eta(self, y, eta_list):
        raise NotImplementedError
