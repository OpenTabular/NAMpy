from __future__ import annotations

from typing import Any

import numpy as np

from ..family_base import GeneralFamily


class _IdentityLinkInfo:
    """Identity link: mu = eta.  Mirrors mgcv make.link("identity") + fix.family.link."""

    name = "identity"

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return np.asarray(mu, dtype=np.float64)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=np.float64)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(eta).shape, dtype=np.float64)

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(mu).shape, dtype=np.float64)


class GamlssFamily(GeneralFamily):
    """
    Base class for multi-predictor GAMLSS families.

    Concrete subclasses must set:
      ``nlp``       — number of linear predictors
      ``linfo``     — list of link-info objects (one per predictor)
      ``tri``       — dict from ``trind_generator(nlp)``
      ``name``      — family name string

    And implement:
      ``ll(y, X, jj, coef, weights, offset, deriv, **kw)``
      ``initialize(y, X, offset, weights)``
    """

    family_class = "general"
    nlp: int = 1
    linfo: list = []
    tri: dict = {}

    supports_laml = True
    supports_ml = True
    supports_reml = True
    supports_analytic_outer_derivatives = False
    supports_analytic_outer_gradient = False
    supports_analytic_outer_hessian = False

    n_linear_predictors: int = 1

    def validate_y(self, y):
        return np.asarray(y, dtype=np.float64).ravel()

    def ll(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        weights: np.ndarray,
        offset: Any = None,
        deriv: int = 0,
        d1b: Any = 0,
        d2b: Any = 0,
        fh: Any = None,
        D: Any = None,
        **kw,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        offset: Any = None,
        weights: Any = None,
        E: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def _stacked_eta(
        self,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        offset: Any = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        eta_cols = []
        off_list: list[Any] | None = None
        if offset is not None:
            off_list = list(offset) if isinstance(offset, (list, tuple)) else [offset]
        for k, cols in enumerate(jj):
            eta_k = X[:, cols] @ coef[cols]
            if off_list is not None and k < len(off_list) and off_list[k] is not None:
                eta_k = eta_k + np.asarray(off_list[k], dtype=np.float64)
            eta_cols.append(np.asarray(eta_k, dtype=np.float64))
        return (
            np.column_stack(eta_cols)
            if eta_cols
            else np.empty((X.shape[0], 0), dtype=np.float64)
        )

    def _predict_response_from_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.ndim == 1:
            eta = eta[:, None]
        cols = []
        for k in range(eta.shape[1]):
            cols.append(np.asarray(self.linfo[k].linkinv(eta[:, k]), dtype=np.float64))
        return np.column_stack(cols) if cols else np.empty((eta.shape[0], 0))

    def predict(
        self,
        *,
        eta: np.ndarray | None = None,
        X: np.ndarray | None = None,
        jj: list[np.ndarray] | None = None,
        coef: np.ndarray | None = None,
        offset: Any = None,
        se: bool = False,
        Vb: np.ndarray | None = None,
    ) -> np.ndarray:
        if eta is None:
            if X is None or jj is None or coef is None:
                raise ValueError("Provide either eta or X/jj/coef for prediction.")
            eta = self._stacked_eta(X, jj, coef, offset=offset)
        eta = np.asarray(eta, dtype=np.float64)
        fit = np.asarray(self._predict_response_from_eta(eta), dtype=np.float64)
        if not se:
            return fit

        if Vb is None:
            raise ValueError("Vb is required when se=True.")
        if X is None or jj is None:
            raise ValueError("X and jj are required when se=True.")

        X = np.asarray(X, dtype=np.float64)
        Vb = np.asarray(Vb, dtype=np.float64)
        ve = np.zeros_like(eta, dtype=np.float64)
        for k, cols in enumerate(jj):
            Xi = X[:, cols]
            Vk = Vb[np.ix_(cols, cols)]
            ve[:, k] = np.maximum(
                np.einsum("ij,jk,ik->i", Xi, Vk, Xi),
                0.0,
            )

        se_fit = np.zeros_like(fit, dtype=np.float64)
        for k in range(min(fit.shape[1], eta.shape[1], len(self.linfo))):
            se_fit[:, k] = np.abs(
                np.asarray(self.linfo[k].mu_eta(eta[:, k]), dtype=np.float64)
            ) * np.sqrt(ve[:, k])
        return fit, se_fit

    def predict_fitted(
        self,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        offset: Any = None,
    ) -> np.ndarray:
        return self.predict(X=X, jj=jj, coef=coef, offset=offset)

    def sandwich(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        weights: np.ndarray | None,
        *,
        offset: Any = None,
    ) -> np.ndarray:
        ll = self.ll(
            np.asarray(y, dtype=np.float64),
            np.asarray(X, dtype=np.float64),
            jj,
            np.asarray(coef, dtype=np.float64),
            (
                np.ones(len(np.asarray(y, dtype=np.float64).ravel()), dtype=np.float64)
                if weights is None
                else np.asarray(weights, dtype=np.float64)
            ),
            offset=offset,
            deriv=1,
            sandwich=True,
        )
        return np.asarray(ll["lbb"], dtype=np.float64)


class _AdaptedLinkInfo:
    """Wraps a _function_maps.LinkFunction into the linfo interface."""

    def __init__(self, lobj: Any, name: str):
        self._lobj = lobj
        self.name = name

    def linkfun(self, mu):
        return self._lobj(mu)

    def linkinv(self, eta):
        return self._lobj.inverse(eta)

    def mu_eta(self, eta):
        return self._lobj.mu_eta(eta)

    def d2link(self, mu):
        return self._lobj.d2(mu)

    def d3link(self, mu):
        return self._lobj.d3(mu)

    def d4link(self, mu):
        return self._lobj.d4(mu)
