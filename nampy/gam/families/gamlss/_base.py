from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import get_lapack_funcs

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


def _r_matrix_norm_one(M: np.ndarray) -> float:
    """Mirror R matrix ``norm()`` default one-norm."""
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return 0.0
    if M.ndim == 1:
        return float(np.sum(np.abs(M)))
    return float(np.max(np.sum(np.abs(M), axis=0)))


def _rrank_upper_triangular(R: np.ndarray, tol: float | None = None) -> int:
    """Mirror ``mgcv::Rrank`` for an upper-triangular factor."""
    R = np.asarray(R, dtype=np.float64)
    m = int(R.shape[0])
    rank = min(m, int(R.shape[1]))
    if tol is None:
        tol = float(np.finfo(np.float64).eps ** 0.9)
    trcon = get_lapack_funcs("trcon", (np.asfortranarray(R),))
    while rank > 0:
        block = np.asfortranarray(R[:rank, :rank], dtype=np.float64)
        rcond, info = trcon(block, norm="1", uplo="U", diag="N")
        if info != 0 or not np.isfinite(rcond):
            rcond = 0.0
        if float(rcond) > float(tol):
            break
        rank -= 1
    return int(rank)


def _qr_coef_pivoted(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mirror ``qr.coef(qr(X), y)`` with pivoted QR."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ncol = int(X.shape[1])
    coef = np.zeros(ncol, dtype=np.float64)
    if ncol == 0:
        return coef

    Q, R, piv = scipy_qr(
        X,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    rank = _rrank_upper_triangular(R)
    if rank > 0:
        qty = np.asarray(Q.T @ y, dtype=np.float64)
        sol = solve_triangular(
            R[:rank, :rank],
            qty[:rank],
            lower=False,
            check_finite=False,
        )
        coef[np.asarray(piv[:rank], dtype=int)] = sol
    coef[~np.isfinite(coef)] = 0.0
    return coef


def _pen_reg(X: np.ndarray, E: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Port of ``mgcv/R/gamlss.r::pen.reg``."""
    X = np.asarray(X, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if float(np.sum(np.abs(E))) == 0.0:
        return _qr_coef_pivoted(X, y)

    Qx, R_piv, piv = scipy_qr(
        X,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    r = int(R_piv.shape[1])
    rr = _rrank_upper_triangular(R_piv)

    R = np.zeros_like(R_piv)
    R[:, np.asarray(piv, dtype=int)] = R_piv
    Qy = np.asarray(Qx.T @ y, dtype=np.float64)[:r]

    norm_R = _r_matrix_norm_one(R)
    norm_E = _r_matrix_norm_one(E)
    if not np.isfinite(norm_R) or norm_R <= 0.0:
        return np.zeros(r, dtype=np.float64)
    if not np.isfinite(norm_E) or norm_E <= 0.0:
        return _qr_coef_pivoted(X, y)

    k = 0.01 * norm_R / norm_E

    def _qrr_stats(k_scale: float):
        A = np.vstack([R, E * float(k_scale)])
        Q, Rq, pivq = scipy_qr(
            A,
            mode="economic",
            pivoting=True,
            check_finite=False,
        )
        edf = float(np.sum(np.asarray(Q[:r, :], dtype=np.float64) ** 2))
        rank_q = _rrank_upper_triangular(Rq)
        return A, Q, Rq, pivq, edf, rank_q

    A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)
    re = min(int(np.sum(np.sum(np.abs(E), axis=0) != 0.0)), int(E.shape[0])) - rank_q + rr

    while edf > rr - 0.1 * re:
        k *= 10.0
        A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)

    while edf < 0.85 * rr:
        k /= 5.0
        A, Qq, Rq, pivq, edf, rank_q = _qrr_stats(k)

    coef = _qr_coef_pivoted(
        A,
        np.concatenate([Qy, np.zeros(E.shape[0], dtype=np.float64)]),
    )
    coef[~np.isfinite(coef)] = 0.0
    return coef


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
    supports_ncv = True
    supports_qncv = True
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

    def _offset_list(self, offset: Any = None) -> list[Any]:
        if offset is None:
            return [None] * int(self.nlp)
        if isinstance(offset, (list, tuple)):
            out = list(offset)
        else:
            out = [offset]
        if len(out) < int(self.nlp):
            out = out + [None] * (int(self.nlp) - len(out))
        return out

    def _eta_matrix_from_inputs(
        self,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        *,
        offset: Any = None,
        eta: np.ndarray | None = None,
    ) -> np.ndarray:
        if eta is not None:
            eta_arr = np.asarray(eta, dtype=np.float64)
            if eta_arr.ndim == 1:
                eta_arr = eta_arr[:, None]
            if eta_arr.shape[1] != int(self.nlp):
                raise ValueError(
                    f"{self.name!r} expected eta with {int(self.nlp)} columns, "
                    f"got {eta_arr.shape}."
                )
            return eta_arr
        return self._stacked_eta(X, jj, coef, offset=offset)

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
