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

_EPS = 1e-9


class BaseFamily(abc.ABC):
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

    @abc.abstractmethod
    def initialize_mu(self, y):
        raise NotImplementedError

    @abc.abstractmethod
    def link(self, mu):
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_link(self, eta):
        raise NotImplementedError

    @abc.abstractmethod
    def mu_eta(self, eta):
        """d mu / d eta"""
        raise NotImplementedError

    @abc.abstractmethod
    def variance(self, mu):
        raise NotImplementedError

    def dvar(self, mu):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement dvar(mu)."
        )

    def d2link(self, mu):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement d2link(mu)."
        )

    @abc.abstractmethod
    def deviance(self, y, mu):
        raise NotImplementedError

    def estimate_dispersion(self, y, mu, edf=None):
        if self.known_scale is not None:
            return float(self.known_scale)
        return 1.0

    def loglik_obs(self, y, mu, scale=1.0):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement per-observation log-likelihoods."
        )

    def loglik(self, y, mu, scale=1.0):
        return float(np.sum(self.loglik_obs(y, mu, scale=scale)))

    def saturated_loglik(self, y, weights=None, n=None, scale=1.0):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement the saturated log-likelihood "
            "term required for Laplace ML/REML criteria."
        )


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
