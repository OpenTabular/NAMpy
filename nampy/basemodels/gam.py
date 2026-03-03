"""Gaussian additive model core (penalised least squares with cubic splines).

Phase A: statistically correct intercept handling, parameter-space EDF,
         proper covariance matrices, honest summary.
Phase B: GCV / exact ML / exact REML smoothing selection, prediction SEs,
         lpmatrix, term-drop tests, Kass–Steffey covariance via delta
         method, concurvity diagnostics.

ML and REML are implemented via a mixed-model reparameterization: each
smooth's basis is split into a null-space (unpenalized → fixed effects)
and a penalized space (whitened so penalty becomes λ_j I → random
effects).  The exact profiled criteria then follow from the resulting
block-structured normal equations, using Woodbury / matrix-determinant-
lemma identities to stay in the (small) coefficient space rather than
forming the n × n covariance.

ML/REML smoothing parameters are optimised in the reparameterised system;
coefficient fitting uses the equivalent original-basis penalised LS solve.
"""

import warnings

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.linalg import qr as scipy_qr, solve_triangular
from scipy.optimize import minimize
from scipy.stats import f as f_dist
from scipy.stats import norm

from ..splines.cubic import CubicSplines

_SP_LOG_BOUNDS = (-20.0, 20.0)


# ======================================================================
# Reparameterization helper (module-level, reusable)
# ======================================================================

def _reparameterize_smooth(B, P, tol=1e-10):
    """Split a smooth basis into null-space and whitened penalized-space.

    Parameters
    ----------
    B : ndarray, shape (n, d)
        Basis matrix (already centered / identifiability-constrained).
    P : ndarray, shape (d, d)
        Penalty matrix (symmetric PSD, typically rank-deficient).
    tol : float
        *Relative* eigenvalue threshold: eigenvalues <= tol * max(|evals|)
        are treated as null space.

    Returns
    -------
    B0 : ndarray, shape (n, n_null)
        Null-space basis columns (unpenalized → fixed effects).
    Zr : ndarray, shape (n, n_pen)
        Whitened penalized-space columns (penalty = λ I).
    meta : dict
        Reparameterization metadata for coefficient reconstruction:
        U0, U1, d_pos, n_null, n_pen.
    """
    P_sym = 0.5 * (P + P.T)
    evals, U = np.linalg.eigh(P_sym)

    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]

    tol_eff = tol * max(1.0, np.max(np.abs(evals)))
    null_mask = evals <= tol_eff
    pos_mask = ~null_mask

    U0 = U[:, null_mask]
    U1 = U[:, pos_mask]
    d_pos = evals[pos_mask]

    B0 = B @ U0
    B1 = B @ U1

    Zr = B1 / np.sqrt(d_pos)[np.newaxis, :] if d_pos.size else B1

    return B0, Zr, {
        "U0": U0,
        "U1": U1,
        "d_pos": d_pos,
        "n_null": int(null_mask.sum()),
        "n_pen": int(pos_mask.sum()),
    }


class GAM:
    """Gaussian additive model with penalised cubic regression splines.

    Fits:  y = alpha + Z @ beta + eps,   eps ~ N(0, sigma^2 I)

    where Z is a column-stack of sum-to-zero-constrained cubic spline
    bases (one per feature) and alpha = mean(y).  Smoothing parameters
    are selected by minimising GCV, exact ML, or exact REML.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training features (numerical only).
    k : int, default=10
        Number of basis functions (knots) per feature.  Must be >= 3.
    s : array-like, float, or None, default=None
        Initial smoothing parameters (one per feature).  ``None`` → 1.0
        per feature.  A scalar is broadcast.
    feature_names : list of str or None
        Display names; auto-generated if ``None``.
    """

    def __init__(self, X, k=10, s=None, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN / Inf")
        if k < 3:
            raise ValueError("k must be >= 3")

        self.X = X
        self.k_ = int(k)
        self.n_samples_, self.n_features_ = X.shape
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_)]
        )

        # ----- Original (constrained) basis/penalty system -----
        self.splines = [CubicSplines(X[:, i], k) for i in range(self.n_features_)]
        self.Z = np.column_stack([sp.basis for sp in self.splines])
        if self.Z.shape[0] != self.n_samples_:
            raise ValueError("Design matrix row count mismatch")

        self.penalties = [sp.penalty for sp in self.splines]

        self.slices = []
        start = 0
        for sp in self.splines:
            nb = sp.basis.shape[1]
            self.slices.append(slice(start, start + nb))
            start += nb
        self.n_coef_ = start

        self.ZTZ = self.Z.T @ self.Z

        # ----- Penalty eigendecomps (for outer-Newton / LAML) -----
        self._penalty_ranks = np.empty(self.n_features_, dtype=np.int64)
        self._penalty_logdet_plus_fixed = np.empty(
            self.n_features_, dtype=np.float64
        )
        for j in range(self.n_features_):
            Sj = self.penalties[j]
            evals_j = np.linalg.eigvalsh(0.5 * (Sj + Sj.T))
            tol_j = 1e-10 * max(1.0, np.max(np.abs(evals_j)))
            pos = evals_j[evals_j > tol_j]
            self._penalty_ranks[j] = len(pos)
            self._penalty_logdet_plus_fixed[j] = float(
                np.sum(np.log(pos)) if len(pos) > 0 else 0.0
            )

        self.smoothing_params = self._validate_smoothing_params(s)

        # ----- Reparameterized system for ML/REML -----
        self._build_reparameterized_system()

        # ----- Fitted state -----
        self.intercept_ = None
        self.coef_ = None
        self.beta = None
        self.scale_ = None
        self.edf_ = None
        self.trace_S_ = None
        self.rss_ = None
        self.Vp_ = None
        self.Vf_ = None
        self.Vp_kass_steffey_ = None
        self.Vp_wood_ = None
        self._y_train = None
        self._optim_method = None

    # ------------------------------------------------------------------
    # Reparameterized representation
    # ------------------------------------------------------------------

    def _build_reparameterized_system(self):
        """Build the mixed-model matrices (X_fix, Z_rand) once at init."""
        fix_blocks = [np.ones((self.n_samples_, 1))]
        rand_blocks = []
        self._reparam_meta = []
        self.rand_dims_per_term_ = []

        for i in range(self.n_features_):
            B = self.Z[:, self.slices[i]]
            P = self.penalties[i]
            B0, Zr, meta = _reparameterize_smooth(B, P)
            fix_blocks.append(B0)
            if Zr.shape[1] > 0:
                rand_blocks.append(Zr)
            self._reparam_meta.append(meta)
            self.rand_dims_per_term_.append(meta["n_pen"])

        X_fix_raw = np.column_stack(fix_blocks)

        _Q, R, piv = scipy_qr(X_fix_raw, pivoting=True)
        diag_R = np.abs(np.diag(R[:min(X_fix_raw.shape), :]))
        rank_tol = (
            max(X_fix_raw.shape) * np.finfo(float).eps * diag_R[0]
            if diag_R[0] > 0 else 1e-12
        )
        rank = int(np.sum(diag_R > rank_tol))
        keep_cols = np.sort(piv[:rank])
        self.X_fix_ = X_fix_raw[:, keep_cols]
        self.rank_X_fix_ = rank
        self._fix_pivot_keep = keep_cols

        if rand_blocks:
            self.Z_rand_ = np.column_stack(rand_blocks)
        else:
            self.Z_rand_ = np.empty((self.n_samples_, 0), dtype=np.float64)
        self.n_rand_ = self.Z_rand_.shape[1]

        self.ZtZ_rand_ = self.Z_rand_.T @ self.Z_rand_

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_smoothing_params(self, s):
        if s is None:
            return np.ones(self.n_features_, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(self.n_features_, s.item())
        if s.shape != (self.n_features_,):
            raise ValueError(
                f"smoothing_params shape must be ({self.n_features_},), got {s.shape}"
            )
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("smoothing_params must be finite and > 0")
        return s.copy()

    @staticmethod
    def _validate_y(y, n_expected):
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != n_expected:
            raise ValueError(
                f"y length {y.shape[0]} != n_samples {n_expected}"
            )
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN / Inf")
        return y

    # ------------------------------------------------------------------
    # Core linear algebra (original parameterization – used by all paths
    # for coefficient fitting after smoothing params are chosen)
    # ------------------------------------------------------------------

    def _assemble_penalty_block(self, smoothing_params):
        blocks = [
            smoothing_params[i] * self.penalties[i]
            for i in range(self.n_features_)
        ]
        return block_diag(*blocks)

    def _solve_given_smoothing(self, y, smoothing_params, store=False):
        """Penalised Gaussian LS for fixed smoothing parameters.

        Works in the original (constrained) parameterization.  Uses
        Cholesky factorisation of A = Z'Z + S_lambda (SPD).
        """
        y = self._validate_y(y, self.n_samples_)

        intercept = float(np.mean(y))
        y_centered = y - intercept

        P = self._assemble_penalty_block(smoothing_params)
        A = self.ZTZ + P
        ZTy = self.Z.T @ y_centered

        try:
            cA, loA = cho_factor(A, check_finite=False)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "Penalised normal equations not SPD; check penalty / data"
            )

        beta = cho_solve((cA, loA), ZTy, check_finite=False)
        fitted = intercept + self.Z @ beta
        resid = y - fitted
        rss = float(resid @ resid)

        # tr(H) = 1 (intercept) + tr(A^{-1} Z'Z)
        AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        trace_smooth = float(np.trace(AinvZTZ))
        trace_S = 1.0 + trace_smooth
        edf = trace_S

        out = {
            "intercept": intercept,
            "beta": beta,
            "fitted": fitted,
            "resid": resid,
            "rss": rss,
            "trace_S": trace_S,
            "edf": edf,
            "A": A,
            "cA": (cA, loA),
            "P": P,
            "y_centered": y_centered,
        }

        if store:
            self.smoothing_params = np.asarray(smoothing_params, dtype=np.float64).copy()
            self.intercept_ = intercept
            self.coef_ = beta
            self.beta = [beta[sl] for sl in self.slices]
            self.rss_ = rss
            self.trace_S_ = trace_S
            self.edf_ = edf

            denom = max(self.n_samples_ - edf, 1.0)
            self.scale_ = rss / denom

            A_inv = cho_solve((cA, loA), np.eye(A.shape[0]), check_finite=False)
            self.Vp_ = self.scale_ * A_inv
            self.Vf_ = self.scale_ * (A_inv @ self.ZTZ @ A_inv.T)

        return out

    # ------------------------------------------------------------------
    # Smoothing criteria
    # ------------------------------------------------------------------

    def gcv_score(self, y, log_smoothing_params):
        """GCV score using parameter-space trace (no n×n hat matrix)."""
        sp = np.exp(np.asarray(log_smoothing_params, dtype=np.float64))
        sol = self._solve_given_smoothing(y, sp, store=False)
        n = self.n_samples_
        den = 1.0 - sol["trace_S"] / n
        if den <= 1e-12 or not np.isfinite(den):
            return np.inf
        return (sol["rss"] / n) / (den ** 2)

    def _criterion_gcv(self, y, log_sp):
        """GCV criterion (original parameterization)."""
        sp = np.exp(np.asarray(log_sp, dtype=np.float64))
        sol = self._solve_given_smoothing(y, sp, store=False)
        n = self.n_samples_
        den = 1.0 - sol["trace_S"] / n
        if den <= 1e-12:
            return np.inf
        return (sol["rss"] / n) / (den ** 2)

    def _criterion_ml_reml_exact(self, y, log_sp, method):
        """Exact Gaussian ML or REML via the mixed-model reparameterization.

        ML:   J = n     * log(RSS_V / n)     + log|V_tilde|
        REML: J = (n-p) * log(RSS_V / (n-p)) + log|V_tilde| + log|X'K X|

        where K = V_tilde^{-1}, computed via Woodbury in coefficient space.
        """
        y = self._validate_y(y, self.n_samples_)
        sp = np.exp(np.asarray(log_sp, dtype=np.float64))

        Xf = self.X_fix_
        Zr = self.Z_rand_
        n = Xf.shape[0]
        p = self.rank_X_fix_
        q = self.n_rand_

        if q == 0:
            # No penalized columns → ordinary LS (degenerate case)
            XtX = Xf.T @ Xf
            try:
                cXtX, lo = cho_factor(XtX, check_finite=False)
            except np.linalg.LinAlgError:
                return np.inf
            b_hat = cho_solve((cXtX, lo), Xf.T @ y, check_finite=False)
            resid = y - Xf @ b_hat
            rss_v = max(float(resid @ resid), 1e-14)

            if method == "ML":
                return n * np.log(rss_v / n)

            # REML: need the extra log|X'X| term (K=I when q=0)
            if n <= p:
                return np.inf
            logdet_XtX = 2.0 * float(np.sum(np.log(np.diag(cXtX))))
            return (n - p) * np.log(rss_v / (n - p)) + logdet_XtX

        # Build Λ = blockdiag(λ_j I_{r_j})
        lam_vec = np.concatenate([
            np.full(rj, sp[j], dtype=np.float64)
            for j, rj in enumerate(self.rand_dims_per_term_)
            if rj > 0
        ])

        # M = Z_r' Z_r + Λ
        M = self.ZtZ_rand_ + np.diag(lam_vec)

        try:
            cM, loM = cho_factor(M, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        # V_tilde^{-1} y = y - Z_r M^{-1} Z_r' y   (Woodbury)
        ZTy = Zr.T @ y
        Minv_ZTy = cho_solve((cM, loM), ZTy, check_finite=False)
        Ky = y - Zr @ Minv_ZTy

        # V_tilde^{-1} X
        ZTX = Zr.T @ Xf
        Minv_ZTX = cho_solve((cM, loM), ZTX, check_finite=False)
        KX = Xf - Zr @ Minv_ZTX

        XtKX = Xf.T @ KX

        try:
            cXKX, loXKX = cho_factor(XtKX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        XtKy = Xf.T @ Ky
        b_hat = cho_solve((cXKX, loXKX), XtKy, check_finite=False)

        rss_v = max(float(y @ Ky - XtKy @ b_hat), 1e-14)

        # log|V_tilde| = log|M| - log|Λ|   (matrix determinant lemma)
        logdet_M = 2.0 * float(np.sum(np.log(np.diag(cM))))
        logdet_Lam = float(np.sum(np.log(lam_vec)))
        logdet_Vtilde = logdet_M - logdet_Lam

        if method == "ML":
            return n * np.log(rss_v / n) + logdet_Vtilde

        # REML
        if n <= p:
            return np.inf
        logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
        return (n - p) * np.log(rss_v / (n - p)) + logdet_Vtilde + logdet_XtKX

    def _criterion(self, y, log_smoothing_params, method="GCV"):
        m = method.upper()
        if m == "GCV":
            return self._criterion_gcv(y, log_smoothing_params)
        if m in {"ML", "REML"}:
            return self._criterion_ml_reml_exact(y, log_smoothing_params, m)
        if m == "LAML":
            return self._criterion_ml_reml_exact(y, log_smoothing_params, "REML")
        raise ValueError("method must be 'GCV', 'ML', 'REML', or 'LAML'")

    # ==================================================================
    # Outer-Newton machinery  (Wood Section 3.1)
    # ==================================================================

    # ---- A. Stable penalty log-determinant (Section 3.1.1) -----------

    def _penalty_logdet_plus_and_derivs(self, rho):
        """Stable penalty log-determinant and its first/second derivatives.

        For single-parameter-per-term blocks we have

            log|S_lambda|_+ = sum_j (rank_j * rho_j + c_j)

        where ``rank_j`` is the rank of ``S_j`` and ``c_j = log|S_j|_+`` was
        precomputed at initialisation.

        Returns ``(val, grad, hess)`` where ``hess`` is a diagonal matrix
        (cross-derivatives are zero for single-parameter blocks).
        """
        rho = np.asarray(rho, dtype=np.float64)
        m = self.n_features_
        ranks = self._penalty_ranks.astype(np.float64)
        fixed = self._penalty_logdet_plus_fixed

        val = float(np.dot(ranks, rho) + np.sum(fixed))
        grad = ranks.copy()
        hess = np.zeros((m, m), dtype=np.float64)
        return val, grad, hess

    # ---- B. Coefficient solve + implicit derivatives (Section 3.1.3) -

    def _solve_given_rho_with_derivs(self, y_centered, rho, need_second=True):
        """Penalised least-squares solve with analytic implicit derivatives.

        Solves for the spline coefficients beta given log smoothing parameters
        ``rho = log(lambda)`` and returns analytic first- and (optionally)
        second-order derivatives of beta with respect to ``rho``.

        Parameters
        ----------
        y_centered : ndarray, shape (n,)
            Response centred by ``mean(y)`` (intercept already removed).
        rho : ndarray, shape (m,)
            Log smoothing parameters.
        need_second : bool
            Whether to compute second derivatives.

        Returns
        -------
        dict with keys ``beta``, ``A``, ``cA``, ``A_inv``, ``rss``,
        ``trace_S``, ``P``, ``D`` (list of D_k), ``dbeta`` (shape p×m),
        ``d2beta`` (shape p×m×m or None), ``logdet_A``.
        """
        sp = np.exp(rho)
        m = self.n_features_
        p = self.n_coef_

        P = self._assemble_penalty_block(sp)
        A = self.ZTZ + P
        ZTy = self.Z.T @ y_centered

        cA, loA = cho_factor(A, check_finite=False)
        beta = cho_solve((cA, loA), ZTy, check_finite=False)

        fitted = self.Z @ beta
        rss = float(np.sum((y_centered - fitted) ** 2))

        AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        trace_S = 1.0 + float(np.trace(AinvZTZ))

        A_inv = cho_solve((cA, loA), np.eye(p), check_finite=False)
        logdet_A = 2.0 * float(np.sum(np.log(np.diag(cA))))

        # D_k = dA/d(rho_k) = lambda_k S_k  (since d(lambda_k S_k)/d(rho_k) = lambda_k S_k)
        D = []
        for j in range(m):
            Dj = np.zeros((p, p), dtype=np.float64)
            sl = self.slices[j]
            Dj[sl, sl] = sp[j] * self.penalties[j]
            D.append(Dj)

        # First derivatives:  d(beta)/d(rho_k) = -A^{-1} D_k beta
        dbeta = np.zeros((p, m), dtype=np.float64)
        AinvD = []
        for k in range(m):
            AiDk = A_inv @ D[k]
            AinvD.append(AiDk)
            dbeta[:, k] = -AiDk @ beta

        d2beta = None
        if need_second:
            # d2(beta)/d(rho_k)d(rho_l) =
            #   A^{-1} D_l A^{-1} D_k beta
            #   + A^{-1} D_k A^{-1} D_l beta
            #   - delta_{kl} A^{-1} D_k beta
            d2beta = np.zeros((p, m, m), dtype=np.float64)
            for k in range(m):
                Dk_beta = D[k] @ beta
                AinvDk_beta = A_inv @ Dk_beta
                for l in range(k, m):
                    Dl_beta = D[l] @ beta
                    AinvDl_beta = A_inv @ Dl_beta
                    v = (
                        A_inv @ (D[l] @ AinvDk_beta)
                        + A_inv @ (D[k] @ AinvDl_beta)
                    )
                    if k == l:
                        v -= AinvDk_beta
                    d2beta[:, k, l] = v
                    if l != k:
                        d2beta[:, l, k] = v

        return {
            "beta": beta,
            "A": A,
            "cA": (cA, loA),
            "A_inv": A_inv,
            "rss": rss,
            "trace_S": trace_S,
            "P": P,
            "D": D,
            "AinvD": AinvD,
            "dbeta": dbeta,
            "d2beta": d2beta,
            "logdet_A": logdet_A,
        }

    # ---- C. LAML/REML objective with analytic gradient & Hessian -----

    def _laml_objective_gradient_hessian(self, y_centered, rho):
        """Negative REML / LAML with analytic gradient and Hessian.

        Implements the Gaussian penalised least-squares REML / LAML objective
        in terms of ``rho = log(lambda)`` and returns its value, gradient, and
        Hessian with respect to ``rho``.  The implementation follows Wood
        (2016) where, for the Gaussian case, the Laplace-approximate marginal
        likelihood coincides with REML up to additive constants that do not
        affect optimisation.

        Returns ``(val, grad, hess)`` where ``val`` is the objective to
        minimise.
        """
        n = self.n_samples_
        m = self.n_features_
        p = self.n_coef_
        sp = np.exp(rho)

        sol = self._solve_given_rho_with_derivs(y_centered, rho, need_second=True)
        beta = sol["beta"]
        A_inv = sol["A_inv"]
        D = sol["D"]
        AinvD = sol["AinvD"]
        dbeta = sol["dbeta"]
        d2beta = sol["d2beta"]
        rss = sol["rss"]
        logdet_A = sol["logdet_A"]

        # Penalty null-space dim = sum of (k_j - rank_j)
        Mp = int(np.sum(
            np.array([sl.stop - sl.start for sl in self.slices])
            - self._penalty_ranks
        ))
        n_reml = n - Mp
        if n_reml <= 0:
            return np.inf, np.zeros(m), np.eye(m) * 1e8

        # Penalty quadratic  b^T S_lambda b
        P = sol["P"]
        bPb = float(beta @ P @ beta)

        sigma2 = max((rss + bPb) / n_reml, 1e-15)

        # log|S_lambda|+ and its derivatives
        ldet_S, dldet_S, d2ldet_S = self._penalty_logdet_plus_and_derivs(rho)

        # Objective (to minimise):
        # V = n_reml * log(sigma2) + logdet_A - ldet_S
        # (dropped constant n_reml * log(2*pi) + n_reml since irrelevant for optim)
        val = n_reml * np.log(sigma2) + logdet_A - ldet_S

        # ---------- gradient --------------------------------------------------
        grad = np.zeros(m, dtype=np.float64)
        # Pre-compute tr(A^{-1} D_k) for each k
        trAiD = np.array([float(np.trace(AinvD[k])) for k in range(m)])

        for k in range(m):
            # d(b^T P b)/d(rho_k) = 2 beta^T P dbeta_k + beta^T D_k beta
            dbPb_k = 2.0 * beta @ P @ dbeta[:, k] + beta @ D[k] @ beta
            # d(rss)/d(rho_k) = -2 y_c^T Z dbeta_k  (since rss = ||y_c - Z beta||^2)
            drss_k = -2.0 * float((y_centered - self.Z @ beta) @ (self.Z @ dbeta[:, k]))
            dsigma2_k = (drss_k + dbPb_k) / n_reml

            grad[k] = (
                n_reml * dsigma2_k / sigma2
                + trAiD[k]
                - dldet_S[k]
            )

        # ---------- Hessian ---------------------------------------------------
        hess = np.zeros((m, m), dtype=np.float64)
        resid = y_centered - self.Z @ beta

        for k in range(m):
            dbPb_k = 2.0 * beta @ P @ dbeta[:, k] + beta @ D[k] @ beta
            drss_k = -2.0 * float(resid @ (self.Z @ dbeta[:, k]))
            dsigma2_k = (drss_k + dbPb_k) / n_reml

            for l in range(k, m):
                dbPb_l = 2.0 * beta @ P @ dbeta[:, l] + beta @ D[l] @ beta
                drss_l = -2.0 * float(resid @ (self.Z @ dbeta[:, l]))
                dsigma2_l = (drss_l + dbPb_l) / n_reml

                # d2(b^T P b)/d(rho_k)d(rho_l)
                d2bPb_kl = (
                    2.0 * (dbeta[:, k] @ P @ dbeta[:, l])
                    + 2.0 * (beta @ P @ d2beta[:, k, l])
                    + 2.0 * (dbeta[:, l] @ D[k] @ beta)
                    + 2.0 * (beta @ D[l] @ dbeta[:, k])
                    + float(k == l) * (beta @ D[k] @ beta)
                )
                # d2(rss)/d(rho_k)d(rho_l)
                Zdb_k = self.Z @ dbeta[:, k]
                Zdb_l = self.Z @ dbeta[:, l]
                d2rss_kl = (
                    2.0 * float(Zdb_k @ Zdb_l)
                    - 2.0 * float(resid @ (self.Z @ d2beta[:, k, l]))
                )
                d2sigma2_kl = (d2rss_kl + d2bPb_kl) / n_reml

                # d2(logdet_A)/d(rho_k)d(rho_l)
                # = -tr(A^{-1} D_k A^{-1} D_l) + delta_{kl} tr(A^{-1} D_k)
                d2logdetA_kl = -float(np.trace(AinvD[k] @ AinvD[l]))
                if k == l:
                    d2logdetA_kl += trAiD[k]

                hess[k, l] = (
                    n_reml * (d2sigma2_kl / sigma2 - dsigma2_k * dsigma2_l / sigma2**2)
                    + d2logdetA_kl
                    - d2ldet_S[k, l]
                )
                if l != k:
                    hess[l, k] = hess[k, l]

        return float(val), grad, hess

    # ---- D. Outer Newton driver (Section 3.1 / 3.2) -----------------

    def _optimize_smoothing_outer_newton(
        self,
        y,
        method="REML",
        initial_rho=None,
        max_iter=50,
        tol=1e-6,
        max_half_steps=10,
        working_inf_pos_threshold=15.0,
        working_inf_grad_tol=1e-3,
        working_inf_hess_tol=1e-4,
    ):
        r"""Wood-style outer Newton optimiser for smoothing parameters.

        Iteratively applies Newton updates in ``rho = log(lambda)`` using
        analytic gradient and Hessian of the REML / LAML criterion.  Includes
        step-halving and faithful Wood-style "working infinity" detection.

        Parameters
        ----------
        y : ndarray, shape (n,)
            Validated response.
        method : {'REML', 'LAML'}
            Criterion.  ``'LAML'`` is a Laplace-approximate marginal
            likelihood alias; for Gaussian it coincides with ``'REML'``.
        initial_rho : ndarray or None
            Starting values for ``rho = log(lambda)``.
            ``None`` → ``log(self.smoothing_params)``.
        max_iter : int
            Maximum outer Newton iterations.
        tol : float
            Convergence tolerance on the active-set gradient norm.
        max_half_steps : int
            Maximum step halvings per iteration.
        working_inf_pos_threshold : float, default=15.0
            A coordinate ``rho_k`` is a candidate for working infinity only
            when it is large and positive (that is, ``rho_k`` above this
            threshold, meaning ``lambda_k`` tends to infinity and the
            corresponding variance component tends to zero).  Large negative
            ``rho_k`` (weak penalty) is not treated as working infinity.
        working_inf_grad_tol : float, default=1e-3
            A candidate coordinate is frozen when its gradient component
            ``|grad[k]| < working_inf_grad_tol`` (near-stationary in that
            direction).
        working_inf_hess_tol : float, default=1e-4
            A candidate coordinate is frozen when its diagonal Hessian entry
            ``hess[k, k] < working_inf_hess_tol`` (flat or indefinite curvature
            at large positive ``rho_k``).

        Returns
        -------
        dict with keys ``rho``, ``sp``, ``converged``, ``n_iter``,
        ``history``, ``frozen`` (mask of frozen coordinates).

        Notes
        -----
        "Working infinity" in the Wood sense refers specifically to
        ``lambda_k`` tending to infinity (large positive ``rho_k``), where
        the corresponding variance component or smooth is effectively zero.
        The gradient tends to zero and the Hessian diagonal becomes flat or
        indefinite at such coordinates.  The three-condition test used here
        (large positive ``rho_k`` and small gradient and small Hessian
        diagonal) is substantially more faithful to that criterion than a
        simple absolute-value threshold on ``rho_k``.
        """
        m = self.n_features_
        y = self._validate_y(y, self.n_samples_)
        intercept = float(np.mean(y))
        y_c = y - intercept

        if initial_rho is not None:
            rho = np.asarray(initial_rho, dtype=np.float64).copy()
        else:
            rho = np.log(self.smoothing_params).copy()

        active = np.ones(m, dtype=bool)
        history = []
        n_iter = 0

        for it in range(max_iter):
            n_iter = it + 1
            val, grad, hess = self._laml_objective_gradient_hessian(y_c, rho)
            active_grad_norm = float(np.linalg.norm(grad[active]))
            history.append({
                "iter": it,
                "objective": val,
                "grad_norm": active_grad_norm,
                "n_active": int(active.sum()),
            })

            if active_grad_norm < tol:
                break

            # Solve Newton system on active coordinates
            idx_a = np.where(active)[0]
            if len(idx_a) == 0:
                break

            g_a = grad[idx_a]
            H_a = hess[np.ix_(idx_a, idx_a)]

            # Stabilise: ensure H_a is PD by adding a small ridge if needed
            evals_H = np.linalg.eigvalsh(H_a)
            if evals_H.min() < 1e-8:
                H_a = H_a + (abs(evals_H.min()) + 1e-6) * np.eye(len(idx_a))

            try:
                delta_a = np.linalg.solve(H_a, g_a)
            except np.linalg.LinAlgError:
                delta_a = np.linalg.lstsq(H_a, g_a, rcond=None)[0]

            delta = np.zeros(m, dtype=np.float64)
            delta[idx_a] = delta_a

            # Step-halving line search
            step = 1.0
            rho_new = rho - step * delta
            val_new, _, _ = self._laml_objective_gradient_hessian(y_c, rho_new)

            for _ in range(max_half_steps):
                if np.isfinite(val_new) and val_new < val:
                    break
                step *= 0.5
                rho_new = rho - step * delta
                val_new, _, _ = self._laml_objective_gradient_hessian(y_c, rho_new)
            else:
                if not (np.isfinite(val_new) and val_new < val):
                    warnings.warn(
                        f"Outer Newton: step-halving failed at iteration {it}"
                    )
                    break

            rho = rho_new

            # ------------------------------------------------------------------
            # Wood-style "working infinity" detection
            #
            # A coordinate qualifies if ALL THREE conditions hold:
            #   1. rho_k is large and POSITIVE  (lambda_k -> inf; smooth -> 0)
            #      Large negative rho means near-zero penalty, which is not
            #      the same and must NOT be frozen.
            #   2. |grad[k]| is near zero  (stationary in this direction)
            #   3. hess[k,k] is near zero or non-positive
            #      (flat/indefinite curvature at the boundary)
            # ------------------------------------------------------------------
            for k in range(m):
                if (
                    rho[k] > working_inf_pos_threshold
                    and abs(grad[k]) < working_inf_grad_tol
                    and hess[k, k] < working_inf_hess_tol
                ):
                    active[k] = False

        sp_final = np.exp(rho)
        converged = float(np.linalg.norm(grad[active])) < tol if np.any(active) else True

        return {
            "rho": rho,
            "sp": sp_final,
            "converged": converged,
            "n_iter": n_iter,
            "history": history,
            "frozen": ~active,
        }

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_without_optimization(self, y):
        """Fit with current smoothing parameters (no optimisation)."""
        self._solve_given_smoothing(y, self.smoothing_params, store=True)
        return self

    def optimize_smoothing_params(
        self, y, initial_smoothing_params=None, method="GCV", optimizer="lbfgsb"
    ):
        """Optimise smoothing parameters.

        Parameters
        ----------
        y : ndarray
            Response (already validated by caller).
        initial_smoothing_params : array-like or None
            Starting values.  ``None`` → current ``self.smoothing_params``.
        method : {'GCV', 'ML', 'REML', 'LAML'}
            Smoothing-selection criterion.  ``'LAML'`` is a Laplace-
            approximate marginal likelihood; for Gaussian it coincides
            with ``'REML'`` up to constants.
        optimizer : {'lbfgsb', 'outer_newton'}
            ``'lbfgsb'`` — L-BFGS-B on the criterion (existing path).
            ``'outer_newton'`` — Wood-style outer Newton with analytic
            gradient / Hessian (requires ``method`` in ``{'REML', 'LAML'}``).

        Returns
        -------
        self
        """
        method = method.upper()
        optimizer = optimizer.lower()

        valid_methods = {"GCV", "ML", "REML", "LAML"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        if initial_smoothing_params is None:
            x0 = np.log(self.smoothing_params)
        else:
            x0 = np.log(self._validate_smoothing_params(initial_smoothing_params))

        if optimizer == "lbfgsb":
            crit_method = "REML" if method == "LAML" else method
            bounds = [_SP_LOG_BOUNDS] * self.n_features_
            result = minimize(
                lambda log_s: self._criterion(y, log_s, method=crit_method),
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if not result.success:
                warnings.warn(
                    f"Smoothing optimisation did not converge: {result.message}"
                )
            self.smoothing_params = np.exp(result.x)
            self._optim_result = result

        elif optimizer == "outer_newton":
            if method not in {"REML", "LAML"}:
                raise ValueError(
                    "outer_newton optimizer requires method='REML' or 'LAML'"
                )
            result = self._optimize_smoothing_outer_newton(
                y, method=method, initial_rho=x0,
            )
            if not result["converged"]:
                warnings.warn(
                    f"Outer Newton did not converge after {result['n_iter']} iterations"
                )
            self.smoothing_params = result["sp"]
            self._optim_result = result

        else:
            raise ValueError(f"optimizer must be 'lbfgsb' or 'outer_newton'")

        self._optim_method = method
        return self

    def fit(self, y, optimize=True, method="GCV", optimizer="lbfgsb"):
        """Fit the model.

        Parameters
        ----------
        y : array-like
            Response.
        optimize : bool
            If ``True``, optimise smoothing parameters before fitting.
        method : {'GCV', 'ML', 'REML', 'LAML'}
            Smoothing-selection criterion.
        optimizer : {'lbfgsb', 'outer_newton'}
            Which optimizer to use for smoothing selection.

        Returns
        -------
        self
        """
        y = self._validate_y(y, self.n_samples_)
        if optimize:
            self.optimize_smoothing_params(y, method=method, optimizer=optimizer)
        self.fit_without_optimization(y)
        self._y_train = y.copy()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _build_new_design_matrix(self, X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_new.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_new.shape[1]}"
            )
        blocks = []
        for i, spline in enumerate(self.splines):
            raw_basis = spline.transform_new(X_new[:, i])
            blocks.append(raw_basis @ spline.center_mat)
        return np.column_stack(blocks)

    def lpmatrix(self, X_new):
        """Linear predictor matrix for coef vector [intercept, beta]."""
        Z_new = self._build_new_design_matrix(X_new)
        return np.column_stack([np.ones(Z_new.shape[0]), Z_new])

    def predict(self, X_new=None, return_se=False, cov="bayes", type="response"):
        """Predict from the fitted model.

        Parameters
        ----------
        X_new : array-like or None
            New data.  ``None`` → use training data.
        return_se : bool
            If True, return (mu, se) tuple.
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}
            Covariance matrix used for SEs.
        type : {'response', 'terms', 'lpmatrix'}
            What to return.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted")

        Z_new = self.Z if X_new is None else self._build_new_design_matrix(X_new)

        if type == "lpmatrix":
            return np.column_stack([np.ones(Z_new.shape[0]), Z_new])

        if type == "terms":
            terms = np.column_stack(
                [Z_new[:, sl] @ self.coef_[sl] for sl in self.slices]
            )
            if not return_se:
                return terms
            V = self._select_cov(cov)
            ses = []
            for sl in self.slices:
                Xi = Z_new[:, sl]
                Vi = V[sl, sl]
                v = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
                ses.append(np.sqrt(np.maximum(v, 0.0)))
            return terms, np.column_stack(ses)

        mu = self.intercept_ + Z_new @ self.coef_
        if not return_se:
            return mu

        V_full = self._full_coef_cov(cov)
        Xp = np.column_stack([np.ones(Z_new.shape[0]), Z_new])
        var = np.einsum("ij,jk,ik->i", Xp, V_full, Xp)
        se = np.sqrt(np.maximum(var, 0.0))
        return mu, se

    def _select_cov(self, cov):
        if cov == "bayes":
            V = self.Vp_
        elif cov == "freq":
            V = self.Vf_
        elif cov == "kass_steffey":
            V = self.Vp_kass_steffey_
            if V is None:
                raise RuntimeError(
                    "Kass–Steffey covariance not computed; "
                    "call compute_unconditional_covariance(kind='kass_steffey') first"
                )
        elif cov == "wood":
            V = self.Vp_wood_
            if V is None:
                raise RuntimeError(
                    "Wood covariance not computed; "
                    "call compute_unconditional_covariance(kind='wood_full') first"
                )
        else:
            raise ValueError(
                "cov must be 'bayes', 'freq', 'kass_steffey', or 'wood'"
            )
        if V is None:
            raise RuntimeError("Covariance not available; fit model first")
        return V

    def _full_coef_cov(self, cov="bayes", intercept_sigma2=None):
        """Full ``[intercept, beta]`` covariance matrix.

        The intercept variance is ``intercept_sigma2 / n`` (independent of
        spline coefficients; correct for the centered parameterisation where
        ``intercept = mean(y)``).  The smooth-coefficient block comes from
        :meth:`_select_cov`.

        Parameters
        ----------
        cov : str
            Passed to :meth:`_select_cov`.
        intercept_sigma2 : float or None
            Scale estimate used for the intercept variance.  ``None`` →
            ``self.scale_`` (the fitted residual-df scale).  Pass the same
            ``sigma2`` used to build the information matrix so the
            trace(I_hat @ V) term is internally consistent.

        Returns
        -------
        ndarray, shape (1 + n_coef, 1 + n_coef)
        """
        V_smooth = self._select_cov(cov)
        p1 = 1 + self.n_coef_
        V_full = np.zeros((p1, p1), dtype=np.float64)
        sigma2 = self.scale_ if intercept_sigma2 is None else float(intercept_sigma2)
        V_full[0, 0] = sigma2 / self.n_samples_
        V_full[1:, 1:] = V_smooth
        return V_full

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, y=None):
        """Print an honest summary (EDF per term; significance tests not shown)."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")

        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = np.asarray(y, dtype=np.float64).ravel()

        fitted = self.intercept_ + self.Z @ self.coef_
        resid = y - fitted
        rss = float(resid @ resid)
        tss = float(((y - np.mean(y)) ** 2).sum())

        n = len(y)
        edf = float(self.edf_)
        resid_df = n - edf
        scale = rss / max(resid_df, 1.0)

        r2_adj = 1.0 - (rss / max(tss, 1e-15)) * ((n - 1) / max(resid_df, 1.0))
        dev_expl = 1.0 - rss / max(tss, 1e-15)
        method = getattr(self, "_optim_method", "GCV") or "GCV"
        crit_val = self._criterion(y, np.log(self.smoothing_params), method=method)
        gcv = self.gcv_score(y, np.log(self.smoothing_params))

        P = self._assemble_penalty_block(self.smoothing_params)
        A = self.ZTZ + P
        try:
            cA, loA = cho_factor(A, check_finite=False)
            AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        except np.linalg.LinAlgError:
            AinvZTZ = np.linalg.solve(A, self.ZTZ)

        print("Gaussian Additive Model Summary")
        print("=" * 55)
        print(f"Smoothing method   : {method}")
        print(f"Number of samples  : {n}")
        print()
        print("Smooth terms (EDF only; significance tests not shown):")
        print(f"{'term':<20s} {'edf':>8s} {'k':>5s}")
        print("-" * 35)
        for i, sl in enumerate(self.slices):
            edf_i = float(np.trace(AinvZTZ[sl, sl]))
            k_i = sl.stop - sl.start
            print(f"s({self.feature_names[i]:<14s}) {edf_i:8.3f} {k_i:5d}")

        print("-" * 55)
        print(f"Intercept          : {self.intercept_:.6g}")
        print(f"Scale estimate     : {scale:.6g}")
        print(f"EDF (total)        : {edf:.3f}")
        print(f"Residual df        : {resid_df:.3f}")
        print(f"R-sq.(adj)         : {r2_adj:.6f}")
        print(f"Deviance explained : {dev_expl:.2%}")
        print(f"{method} criterion     : {crit_val:.6g}")
        if method != "GCV":
            print(f"GCV (supplementary): {gcv:.6g}")
        print(f"n                  : {n}")

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def confidence_intervals(self, alpha=0.05, cov="bayes", include_intercept=False):
        """Wald-type CIs for spline coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level; CIs are at ``1 - alpha`` coverage.
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}, default='bayes'
            Covariance matrix used for the standard errors.
            ``'kass_steffey'`` and ``'wood'`` require a prior call to
            :meth:`compute_unconditional_covariance`.
        include_intercept : bool, default=False
            If ``True``, prepend a CI for the intercept.

        Returns
        -------
        list of (float, float)
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")

        zcrit = norm.ppf(1.0 - alpha / 2.0)
        V = self._select_cov(cov)
        ses = np.sqrt(np.maximum(np.diag(V), 0.0))

        out = []
        if include_intercept:
            se0 = np.sqrt(max(self.scale_ / self.n_samples_, 0.0))
            out.append((self.intercept_ - zcrit * se0, self.intercept_ + zcrit * se0))

        for b, se in zip(self.coef_, ses):
            out.append((b - zcrit * se, b + zcrit * se))
        return out

    # ------------------------------------------------------------------
    # AIC
    # ------------------------------------------------------------------

    def _gaussian_loglik(self, y, scale="ml"):
        """Gaussian log-likelihood at the fitted values.

        Parameters
        ----------
        scale : {'ml', 'working'}
            ``'ml'``     → sigma^2_ML = RSS / n (appropriate for AIC
            comparability).
            ``'working'`` → ``self.scale_`` (RSS / residual degrees of
            freedom), consistent with the summary.

        Returns
        -------
        loglik : float
            Value of the Gaussian log-likelihood at the fitted values.
        sigma2 : float
            The scale estimate used.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        y = self._resolve_y(y)
        fitted = self.intercept_ + self.Z @ self.coef_
        rss = float(np.sum((y - fitted) ** 2))
        n = float(self.n_samples_)

        if scale == "ml":
            sigma2 = rss / n
        elif scale == "working":
            sigma2 = self.scale_
        else:
            raise ValueError("scale must be 'ml' or 'working'")

        if sigma2 <= 0:
            sigma2 = 1e-15

        loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + rss / (n * sigma2))
        return float(loglik), float(sigma2)

    def _resolve_y(self, y):
        """Return validated y or stored training y."""
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        return self._validate_y(y, self.n_samples_)

    def _observed_information(self, sigma2):
        """Observed information matrix for ``[intercept, beta]``.

        For the Gaussian model with known (or plugged-in) variance the observed
        information is ``X_p.T @ X_p / sigma2`` where ``X_p`` is the full
        linear-predictor matrix including the intercept column.
        """
        Xp = np.column_stack([np.ones(self.n_samples_), self.Z])
        return (Xp.T @ Xp) / sigma2

    def aic_conditional(self, y=None, scale="ml", cov="bayes"):
        """Conventional conditional AIC.

        Computes

            AIC_c = -2 * loglik + 2 * trace(I_hat * V_beta),

        where ``V_beta`` is the conditional covariance (given smoothing
        parameters) and ``I_hat`` is the observed information.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        scale : {'ml', 'working'}
            Which scale estimate to use in the log-likelihood and the
            information matrix.
        cov : {'bayes', 'freq'}
            Which conditional covariance to use for the penalty term.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic`` (= ``tr(I V)``),
        ``scale``.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        loglik, sigma2 = self._gaussian_loglik(y, scale=scale)
        I_hat = self._observed_information(sigma2)
        V_full = self._full_coef_cov(cov, intercept_sigma2=sigma2)
        tau = float(np.trace(I_hat @ V_full))

        return {
            "aic": -2.0 * loglik + 2.0 * tau,
            "loglik": loglik,
            "edf_aic": tau,
            "scale": sigma2,
        }

    def aic_corrected(
        self,
        y=None,
        scale="ml",
        covariance_kind="wood_full",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Wood-style corrected conditional AIC.

        Corrected AIC is computed as:

            AIC_corr = -2 * loglik(beta_hat) + 2 * trace(I_hat @ Vbar_beta)

        where ``Vbar_beta`` is the covariance corrected for
        smoothing-parameter uncertainty (Kass–Steffey or full Wood),
        computed via :meth:`compute_unconditional_covariance` if not
        already available.

        .. note::

            The Wood (2016) corrected AIC is theoretically grounded on the
            Hessian of the **negative marginal likelihood** (REML / LAML)
            with respect to log smoothing parameters. When the model was fitted
            with ``method='GCV'`` the smoothing-parameter uncertainty
            (V_rho) is estimated from the GCV Hessian, which does not have
            the same theoretical justification.  The result is still a
            reasonable heuristic but is **not** the exact Wood corrected AIC.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        scale : {'ml', 'working'}
            Scale estimate for the log-likelihood and information.
        covariance_kind : {'kass_steffey', 'wood_full'}
            Which unconditional covariance approximation to use.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            Passed to :meth:`compute_unconditional_covariance`.
        sp_uncertainty_ridge : float
            Passed to :meth:`compute_unconditional_covariance`.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``,
        ``covariance_kind``, ``heuristic`` (bool — ``True`` when the
        corrected AIC is not theoretically exact, e.g. fitted with GCV).
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        cov_map = {"kass_steffey": "kass_steffey", "wood_full": "wood"}
        if covariance_kind not in cov_map:
            raise ValueError(
                f"covariance_kind must be 'kass_steffey' or 'wood_full', "
                f"got {covariance_kind!r}"
            )
        cov_key = cov_map[covariance_kind]

        # Warn when the fitting criterion is not REML/LAML.
        # Wood's corrected AIC uses V_rho from the marginal-likelihood Hessian;
        # using the GCV Hessian yields a heuristic rather than the exact result.
        optim_method = (self._optim_method or "GCV").upper()
        is_heuristic = optim_method not in {"REML", "LAML", "ML"}
        if is_heuristic:
            warnings.warn(
                f"aic_corrected(): model was fitted with method='{optim_method}'. "
                "Wood's corrected AIC is theoretically grounded on the REML/LAML "
                "Hessian of the negative marginal likelihood w.r.t. log(lambda). "
                "Using a GCV-based smoothing-parameter uncertainty yields a "
                "heuristic approximation, not the exact Wood corrected AIC. "
                "Refit with method='REML' (or 'LAML') for the theoretically "
                "justified result.  The returned dict includes 'heuristic': True.",
                UserWarning,
                stacklevel=2,
            )

        # Compute unconditional covariance if not already present
        attr = (
            self.Vp_kass_steffey_
            if covariance_kind == "kass_steffey"
            else self.Vp_wood_
        )
        if attr is None:
            self.compute_unconditional_covariance(
                y=y,
                kind=covariance_kind,
                sp_uncertainty_regularization=sp_uncertainty_regularization,
                sp_uncertainty_ridge=sp_uncertainty_ridge,
            )

        loglik, sigma2 = self._gaussian_loglik(y, scale=scale)
        I_hat = self._observed_information(sigma2)
        V_full = self._full_coef_cov(cov_key, intercept_sigma2=sigma2)
        tau = float(np.trace(I_hat @ V_full))

        return {
            "aic": -2.0 * loglik + 2.0 * tau,
            "loglik": loglik,
            "edf_aic": tau,
            "scale": sigma2,
            "covariance_kind": covariance_kind,
            "heuristic": is_heuristic,
        }

    # ------------------------------------------------------------------
    # Term-drop tests
    # ------------------------------------------------------------------

    def term_drop_test(self, y=None, term_index=0, method=None):
        """Refit-based approximate term significance test.

        Drops one smooth term, refits (re-optimising smoothing), and
        computes an approximate F-statistic from the change in RSS and
        EDF.  P-values are approximate because smoothing parameters are
        re-estimated in the reduced model.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)
        method = method or self._optim_method or "GCV"

        if not (0 <= term_index < self.n_features_):
            raise IndexError(
                f"term_index must be in [0, {self.n_features_ - 1}], got {term_index}"
            )

        rss_full = self.rss_
        edf_full = self.edf_
        n = self.n_samples_

        keep = [i for i in range(self.n_features_) if i != term_index]
        if not keep:
            raise ValueError("Cannot drop the only term")
        X_red = self.X[:, keep]
        names_red = [self.feature_names[i] for i in keep]

        s_red = np.array([self.smoothing_params[i] for i in keep])

        red = GAM(X_red, k=self.k_, s=s_red, feature_names=names_red)
        red.fit(y, optimize=True, method=method)

        rss_red = red.rss_
        edf_red = red.edf_
        delta_df = max(edf_full - edf_red, 1e-6)
        df_res = max(n - edf_full, 1e-6)
        ms_num = max((rss_red - rss_full) / delta_df, 0.0)
        ms_den = rss_full / df_res
        f_stat = ms_num / max(ms_den, 1e-15)
        pval = float(1.0 - f_dist.cdf(f_stat, delta_df, df_res))

        return {
            "term": self.feature_names[term_index],
            "f_stat": f_stat,
            "p_value": pval,
            "delta_df": delta_df,
            "rss_full": rss_full,
            "rss_reduced": rss_red,
            "edf_full": edf_full,
            "edf_reduced": edf_red,
        }

    # ------------------------------------------------------------------
    # Unconditional covariance (smoothing-parameter uncertainty)
    # ------------------------------------------------------------------

    def compute_unconditional_covariance(
        self,
        y=None,
        method=None,
        kind="kass_steffey",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Covariance corrected for smoothing-parameter uncertainty.

        Two approximation levels are available:

        ``kind='kass_steffey'`` — :math:`\bar V_\beta = V_\beta + J V_\rho J^\top`

            The Kass–Steffey first-order correction.  Fast, often adequate.
            Result is stored in ``Vp_kass_steffey_``; use ``cov='kass_steffey'``
            in :meth:`predict`.

        ``kind='wood_full'`` — :math:`\bar V_\beta = V_\beta + V' + V''`

            The full Wood et al. (2016) correction that also accounts for the
            derivative of the covariance factor :math:`R_\rho` where
            :math:`R_\rho^\top R_\rho = V_\beta`.  More accurate when some
            smoothing parameters are near the boundary of the penalty null
            space.  Result is stored in ``Vp_wood_``; use ``cov='wood'`` in
            :meth:`predict`.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        method : str or None
            Criterion used for the Hessian (``'GCV'``, ``'ML'``, ``'REML'``).
            ``None`` → whatever was used during fitting.
        kind : {'kass_steffey', 'wood_full'}
            Which approximation to compute.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            How to invert the criterion Hessian :math:`H_\rho`:

            - ``'pinv'``:  Moore–Penrose pseudoinverse — eigenvalues below a
              relative tolerance (``1e-10 * max(1, max|evals|)``) are mapped
              to zero inverse, so flat/boundary directions contribute zero
              uncertainty rather than inflated uncertainty.
            - ``'ridge'``: invert ``(H_rho + kappa * I)`` where ``kappa`` is
              automatically raised if needed to keep all shifted eigenvalues
              strictly positive (equivalent to a Gaussian prior on
              :math:`\rho`).
        sp_uncertainty_ridge : float, default=1e-6
            Ridge constant used when ``sp_uncertainty_regularization='ridge'``.

        Returns
        -------
        ndarray, shape (n_coef, n_coef)
            The corrected covariance matrix.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)
        method = method or self._optim_method or "GCV"

        valid_kinds = {"kass_steffey", "wood_full"}
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}")

        sp = self.smoothing_params.copy()
        theta = np.log(sp)
        m = self.n_features_
        p = self.n_coef_

        sol = self._solve_given_smoothing(y, sp, store=False)
        cA, loA = sol["cA"]
        A_inv = cho_solve((cA, loA), np.eye(p), check_finite=False)
        beta = sol["beta"]
        sigma2 = self.scale_
        Vp = self.Vp_

        # D_k = dA/d(rho_k) = lambda_k * S_k  (block-diagonal, only one block nonzero)
        D_blocks = []
        for j in range(m):
            D = np.zeros((p, p), dtype=np.float64)
            sl = self.slices[j]
            D[sl, sl] = sp[j] * self.penalties[j]
            D_blocks.append(D)

        # J = d(beta_hat)/d(rho)  — Gaussian specialisation of implicit deriv.
        # J[:, k] = -A_inv @ D_k @ beta
        J = np.column_stack([-A_inv @ (Dk @ beta) for Dk in D_blocks])

        # V_rho = (regularised) inverse of criterion Hessian w.r.t. log(sp)
        V_rho = self._sp_uncertainty_matrix(
            y, theta, method,
            regularization=sp_uncertainty_regularization,
            ridge=sp_uncertainty_ridge,
        )

        # V' = J V_rho J^T  (Kass–Steffey term)
        V_prime = J @ V_rho @ J.T

        if kind == "kass_steffey":
            V_unc = Vp + V_prime
            self.Vp_kass_steffey_ = 0.5 * (V_unc + V_unc.T)
            return self.Vp_kass_steffey_

        # --- kind == "wood_full": also compute V'' -------------------------

        # Cholesky factor L such that V_beta = L L^T.
        # _cholesky_factor_derivative requires a genuine lower-triangular L.
        # Retry with increasing jitter up to 4 times; if Vp is still not SPD,
        # fall back to Kass–Steffey so we never pass a symmetric square-root
        # into a method that assumes a triangular factor.
        _jitter_schedule = [0.0, 1e-10, 1e-7, 1e-5, 1e-3]
        L = None
        for _jit in _jitter_schedule:
            try:
                Vp_jit = Vp if _jit == 0.0 else Vp + _jit * np.eye(p)
                L = np.linalg.cholesky(Vp_jit)
                if _jit > 0.0:
                    warnings.warn(
                        f"compute_unconditional_covariance(kind='wood_full'): "
                        f"Vp was not numerically SPD; added jitter={_jit:.0e} "
                        f"to obtain a valid Cholesky factor."
                    )
                break
            except np.linalg.LinAlgError:
                continue

        if L is None:
            warnings.warn(
                "compute_unconditional_covariance(kind='wood_full'): "
                "Vp remains non-SPD after jitter retries.  "
                "V'' cannot be computed faithfully; "
                "falling back to Kass–Steffey (V' only).  "
                "Consider using kind='kass_steffey' explicitly.",
                RuntimeWarning,
            )
            V_unc = Vp + V_prime
            self.Vp_kass_steffey_ = 0.5 * (V_unc + V_unc.T)
            # Do not set Vp_wood_ so callers requesting cov="wood" still get
            # an informative error rather than silently incorrect results.
            return self.Vp_kass_steffey_

        # dV_beta/d(rho_k) = -sigma^2 A_inv D_k A_inv
        # (ignoring d(sigma^2)/d(rho_k), consistent with standard mgcv practice)
        dVp = [-sigma2 * A_inv @ Dk @ A_inv for Dk in D_blocks]

        # Cholesky-factor derivatives:  dL_k  via  L dL_k^T + dL_k L^T = dV_k
        # Solved row-by-row (Lyapunov equation for lower-triangular L)
        dL = [self._cholesky_factor_derivative(L, dV) for dV in dVp]

        # V'' = sum_{k,l}  dL_k dL_l^T  V_rho[k,l]
        V_double_prime = np.zeros((p, p), dtype=np.float64)
        for k in range(m):
            for l in range(m):
                if V_rho[k, l] == 0.0:
                    continue
                V_double_prime += V_rho[k, l] * (dL[k] @ dL[l].T)

        V_unc = Vp + V_prime + V_double_prime
        self.Vp_wood_ = 0.5 * (V_unc + V_unc.T)
        return self.Vp_wood_

    # ---- helpers for compute_unconditional_covariance --------------------

    def _sp_uncertainty_matrix(
        self, y, theta, method, regularization="pinv", ridge=1e-6
    ):
        r"""Inverse (or pseudo-inverse) of the criterion Hessian w.r.t. log(sp).

        For ``method`` in ``{'REML', 'LAML'}`` the analytic Hessian from
        :meth:`_laml_objective_gradient_hessian` is used (stable, exact).
        For ``'GCV'`` and ``'ML'`` the adaptive numeric Hessian is the
        fallback.

        Wood notes that when a smoothing parameter is at or near the boundary
        (effectively infinite), the corresponding Hessian eigenvalue tends to
        zero.  The Moore–Penrose pseudoinverse correctly maps these directions
        to zero uncertainty rather than inflating it, while ridge
        regularisation is equivalent to placing a Gaussian prior on
        :math:`\rho`.
        """
        m_upper = method.upper()

        if m_upper in {"REML", "LAML"}:
            intercept = float(np.mean(y))
            y_c = y - intercept
            _, _, H = self._laml_objective_gradient_hessian(y_c, np.asarray(theta))
        else:
            H = self._numeric_hessian(
                lambda th: self._criterion(y, th, method=method), theta
            )

        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)

        if regularization == "pinv":
            tol = 1e-10 * max(1.0, np.max(np.abs(evals)))
            inv_evals = np.where(evals > tol, 1.0 / evals, 0.0)
        elif regularization == "ridge":
            kappa = float(ridge)
            # Ensure the shift is large enough to keep all shifted eigenvalues
            # strictly positive, so V_rho is PSD.
            min_eig = float(evals.min())
            if min_eig + kappa <= 0.0:
                kappa_eff = -min_eig + kappa + 1e-8
                warnings.warn(
                    f"_sp_uncertainty_matrix: ridge={kappa:.2e} insufficient to "
                    f"stabilise Hessian (min eigenvalue {min_eig:.2e}); "
                    f"using effective ridge={kappa_eff:.2e} to ensure PSD V_rho"
                )
            else:
                kappa_eff = kappa
            inv_evals = 1.0 / (evals + kappa_eff)
        else:
            raise ValueError(
                f"sp_uncertainty_regularization must be 'pinv' or 'ridge', "
                f"got {regularization!r}"
            )

        return (evecs * inv_evals) @ evecs.T

    @staticmethod
    def _cholesky_factor_derivative(L, dV):
        """Derivative of a Cholesky factor given :math:`dV = L dL^T + dL L^T`.

        Solves for the lower-triangular :math:`dL` using the standard
        row-by-row algorithm for the triangular Lyapunov equation.
        """
        p = L.shape[0]
        dL = np.zeros_like(L)
        for i in range(p):
            for j in range(i + 1):
                if i == j:
                    s = dV[i, i] - 2.0 * np.dot(dL[i, :j], L[i, :j])
                    dL[i, i] = s / (2.0 * L[i, i]) if abs(L[i, i]) > 1e-15 else 0.0
                else:
                    s = dV[i, j] - (
                        np.dot(dL[i, :j], L[j, :j]) + np.dot(L[i, :j], dL[j, :j])
                    )
                    dL[i, j] = s / L[j, j] if abs(L[j, j]) > 1e-15 else 0.0
        return dL

    @staticmethod
    def _numeric_hessian(func, x, rel_eps=1e-4, abs_eps=1e-4):
        r"""Central-difference Hessian with adaptive per-coordinate step sizes.

        Step for coordinate *i* is ``h_i = rel_eps * max(1, |x_i|) + abs_eps``,
        which scales with the magnitude of :math:`\rho_i` and avoids the
        degeneracy of a fixed step on flat / poorly scaled criteria.

        After computation the result is symmetry-projected and a condition
        diagnostic warning is emitted if the matrix is poorly conditioned.
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        h = rel_eps * np.maximum(1.0, np.abs(x)) + abs_eps

        fx = func(x)
        if not np.isfinite(fx):
            warnings.warn("_numeric_hessian: function value at x is non-finite")
            return np.full((n, n), np.nan)

        H = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = h[i]
            fip = func(x + ei)
            fim = func(x - ei)
            H[i, i] = (fip - 2.0 * fx + fim) / (h[i] * h[i])
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = h[j]
                fpp = func(x + ei + ej)
                fpm = func(x + ei - ej)
                fmp = func(x - ei + ej)
                fmm = func(x - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])

        H = 0.5 * (H + H.T)

        if not np.all(np.isfinite(H)):
            warnings.warn(
                "_numeric_hessian: non-finite entries detected; "
                "criterion may be ill-conditioned at this point"
            )
        else:
            evals = np.linalg.eigvalsh(H)
            emax = np.max(np.abs(evals))
            emin_pos = np.min(evals[evals > 0]) if np.any(evals > 0) else 0.0
            if emax > 0 and emin_pos > 0 and emax / emin_pos > 1e12:
                warnings.warn(
                    f"_numeric_hessian: condition number ~{emax / emin_pos:.1e}; "
                    "Hessian may be unreliable — consider using "
                    "optimizer='outer_newton' for analytic derivatives"
                )

        return H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Concurvity (Wood / mgcv-style)
    # ------------------------------------------------------------------

    @staticmethod
    def _qr_R_nopivot(A):
        """Return the R factor from a no-pivot QR decomposition."""
        # mgcv uses qr(..., LAPACK = FALSE, tol = 0) and then qr.R(...)
        # Here we use SciPy QR without pivoting.
        # mode='economic' is enough for the subsequent slicing.
        if A.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        _, R = scipy_qr(A, mode="economic", pivoting=False, check_finite=False)
        return R

    @staticmethod
    def _safe_ratio(num, den, eps=1e-15):
        """Bounded ratio helper for concurvity measures."""
        num = float(num)
        den = float(den)
        if den <= eps:
            # If the target term has effectively zero norm, concurvity is undefined;
            # return 0 for production robustness (mgcv may yield NaN in degenerate cases).
            return 0.0 if num <= eps else 1.0
        val = num / den
        # Numerical safety: these should be in [0,1] theoretically.
        return float(np.clip(val, 0.0, 1.0))

    def _concurvity_measures_for_pair(self, Xi, Xj, beta_j):
        """Compute (worst, observed, estimate) for dependence of Xj-term on Xi-space.

        This mirrors mgcv's QR-based formulas (see concurvity source in R/mgcv.r).
        """
        # Xi: basis defining the "other" space (or row term in pairwise mode)
        # Xj: basis of the term being assessed (current/full term or column term)
        # beta_j: fitted coefficients for Xj term

        r = Xi.shape[1]
        dj = Xj.shape[1]

        if dj == 0:
            return 0.0, 0.0, 0.0
        if r == 0:
            # No competing space -> no concurvity
            return 0.0, 0.0, 0.0

        # mgcv pattern:
        # R <- qr.R(qr(cbind(Xi, Xj), no pivot))[,-(1:r)]
        Rfull = self._qr_R_nopivot(np.column_stack([Xi, Xj]))
        # With economic QR and n >= p (usual case), shape is (p, p) where p=r+dj.
        # If n < p, shape is (n, p), and concurvity can be unstable/undefined.
        if Rfull.shape[0] < r:
            # Not enough rows to represent Xi block in R in the expected way
            return np.nan, np.nan, np.nan

        R = Rfull[:, r:]  # shape approx (r+dj, dj)

        # Another QR of R:
        # Rt <- qr.R(qr(R, tol=0))
        Rt = self._qr_R_nopivot(R)

        # 1) worst:
        # svd( forwardsolve(t(Rt), t(R[1:r,,drop=FALSE])) )$d[1]^2
        # In Python (0-indexed): R[:r, :]
        R_top = R[:r, :]
        try:
            # solve_triangular on lower-triangular t(Rt)
            M = solve_triangular(
                Rt.T, R_top.T, lower=True, check_finite=False, overwrite_b=False
            )
            svals = np.linalg.svd(M, compute_uv=False)
            worst = float(svals[0] ** 2) if svals.size else 0.0
        except np.linalg.LinAlgError:
            worst = np.nan

        # 2) observed:
        # sum((R[1:r,] %*% beta)^2) / sum((Rt %*% beta)^2)
        num_obs = np.sum((R_top @ beta_j) ** 2)
        den_obs = np.sum((Rt @ beta_j) ** 2)
        observed = self._safe_ratio(num_obs, den_obs)

        # 3) estimate:
        # sum(R[1:r,]^2) / sum(R^2)
        num_est = np.sum(R_top ** 2)
        den_est = np.sum(R ** 2)
        estimate = self._safe_ratio(num_est, den_est)

        return worst, observed, estimate

    def concurvity(self, full=True, include_intercept=False):
        """Wood/mgcv-style concurvity diagnostics.

        Parameters
        ----------
        full : bool, default=True
            If True, compute concurvity of each term with the whole of the rest
            of the model (mgcv full=TRUE style).
            If False, compute pairwise concurvity matrices (mgcv full=FALSE style).
        include_intercept : bool, default=False
            If True, include a 'para' component consisting only of the intercept
            column. In this model class there are no additional parametric terms.

        Returns
        -------
        If full=True:
            dict with keys 'worst', 'observed', 'estimate' each mapping term name -> value,
            and a convenience matrix-like ndarray under key 'matrix' (rows in this order).
        If full=False:
            dict with keys 'worst', 'observed', 'estimate', each an (m x m) ndarray,
            plus 'labels'.

        Notes
        -----
        This follows the mgcv concurvity source's QR-based computations closely,
        adapted to this class's term block structure.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        # Build blocks and labels
        blocks = []
        betas = []
        labels = []

        if include_intercept:
            blocks.append(np.ones((self.n_samples_, 1), dtype=np.float64))
            betas.append(np.array([self.intercept_], dtype=np.float64))
            labels.append("para")

        for i, sl in enumerate(self.slices):
            blocks.append(self.Z[:, sl])
            betas.append(self.coef_[sl])
            labels.append(self.feature_names[i])

        m = len(blocks)
        if m < 1:
            raise ValueError("No terms available for concurvity")

        measure_names = ("worst", "observed", "estimate")

        if full:
            # Each term vs the whole rest of model
            conc = np.zeros((3, m), dtype=np.float64)

            for i in range(m):
                Xj = blocks[i]
                beta_j = betas[i]
                other_blocks = [blocks[k] for k in range(m) if k != i]
                Xi = (
                    np.column_stack(other_blocks)
                    if other_blocks
                    else np.empty((self.n_samples_, 0), dtype=np.float64)
                )

                w, o, e = self._concurvity_measures_for_pair(Xi, Xj, beta_j)
                conc[:, i] = [w, o, e]

            return {
                "matrix": conc,
                "rows": list(measure_names),
                "labels": labels,
                "worst": dict(zip(labels, conc[0])),
                "observed": dict(zip(labels, conc[1])),
                "estimate": dict(zip(labels, conc[2])),
            }

        # Pairwise mode: matrices, mgcv-style
        conc = {
            "worst": np.eye(m, dtype=np.float64),
            "observed": np.eye(m, dtype=np.float64),
            "estimate": np.eye(m, dtype=np.float64),
        }

        # Row i = dependence on term i's space; column j = term j being assessed
        for i in range(m):
            Xi = blocks[i]
            for j in range(m):
                if i == j:
                    continue
                Xj = blocks[j]
                beta_j = betas[j]
                w, o, e = self._concurvity_measures_for_pair(Xi, Xj, beta_j)
                conc["worst"][i, j] = w
                conc["observed"][i, j] = o
                conc["estimate"][i, j] = e

        conc["labels"] = labels
        return conc

    # ------------------------------------------------------------------
    # k-index diagnostic (Wood/mgcv-style simulation test)
    # ------------------------------------------------------------------

    def _term_edf_vector(self):
        """Per-term EDFs from block traces of A^{-1} Z'Z.

        These are the same "display EDFs" printed by :meth:`summary`.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        P = self._assemble_penalty_block(self.smoothing_params)
        A = self.ZTZ + P
        try:
            cA, loA = cho_factor(A, check_finite=False)
            AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        except np.linalg.LinAlgError:
            AinvZTZ = np.linalg.solve(A, self.ZTZ)

        edf = np.zeros(self.n_features_, dtype=np.float64)
        for i, sl in enumerate(self.slices):
            edf[i] = float(np.trace(AinvZTZ[sl, sl]))
        return edf

    def k_check(self, y=None, subsample=5000, n_rep=400, random_state=None):
        """Wood/mgcv-style basis-dimension check (k-index) for 1-D numeric smooths.

        Matches the 1-D branch of ``mgcv::k.check()``: residuals are ordered by
        each smooth's covariate, differenced to estimate local residual variance,
        and the resulting *k-index* is compared to a simulation null obtained by
        reshuffling residuals.

        Parameters
        ----------
        y : array-like or None, default=None
            Response used to form residuals.  ``None`` → stored training y.
        subsample : int, default=5000
            When ``n > subsample`` use a random subsample without replacement
            (matches mgcv's cost-control heuristic).
        n_rep : int, default=400
            Number of residual reshuffles for the simulation p-value.
        random_state : int, np.random.Generator, or None
            Seed / generator for reproducibility.

        Returns
        -------
        dict with keys
            ``labels``       – list of term names (length *m*)
            ``table``        – ndarray, shape (*m*, 4), columns k', edf, k-index, p-value
            ``columns``      – ``["k'", "edf", "k-index", "p-value"]``
            ``subsample_n``  – actual number of observations used
            ``n_rep``        – number of simulations performed

        Notes
        -----
        A k-index **below 1** suggests remaining autocorrelation in the residuals
        at the scale of that smooth's covariate, indicating the basis dimension may
        be too small.  p-values are simulation-based and vary across runs when the
        null is true, as the mgcv documentation notes.

        This implementation covers the 1-D numeric-smooth case that is the only
        smooth type currently supported by this class.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)

        if subsample is None or int(subsample) <= 0:
            raise ValueError("subsample must be a positive integer")
        if int(n_rep) <= 0:
            raise ValueError("n_rep must be a positive integer")

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Response residuals (Gaussian identity link)
        rsd = y - (self.intercept_ + self.Z @ self.coef_)
        n = rsd.shape[0]

        # mgcv-style cost-control subsample
        if n > int(subsample):
            idx_sub = rng.choice(n, size=int(subsample), replace=False)
            X_sub = self.X[idx_sub, :]
            rsd_sub = rsd[idx_sub]
        else:
            X_sub = self.X
            rsd_sub = rsd

        nr = rsd_sub.shape[0]
        if nr < 3:
            raise ValueError("Need at least 3 observations for k_check")

        # Global denominator: mean(rsd^2), used as sigma^2_r
        rsd_var = float(np.mean(rsd_sub ** 2))

        per_edf = self._term_edf_vector()
        m = self.n_features_
        table = np.full((m, 4), np.nan, dtype=np.float64)
        labels = list(self.feature_names)

        # Fill k' and edf unconditionally so they appear even in degenerate cases
        for j, sl in enumerate(self.slices):
            table[j, 0] = float(sl.stop - sl.start)
            table[j, 1] = float(per_edf[j])

        if not np.isfinite(rsd_var) or rsd_var <= 0.0:
            return {
                "labels": labels,
                "table": table,
                "columns": ["k'", "edf", "k-index", "p-value"],
                "subsample_n": int(nr),
                "n_rep": int(n_rep),
            }

        sim_buf = np.empty(int(n_rep), dtype=np.float64)

        for j, sl in enumerate(self.slices):
            xj = X_sub[:, j]

            if not np.issubdtype(xj.dtype, np.number):
                continue
            if not np.isfinite(xj).all():
                continue
            if np.allclose(xj.max(), xj.min()):
                # Constant covariate on the subsample → test undefined
                continue

            # mgcv 1-D branch:  e <- diff(rsd[order(x)])
            order = np.argsort(xj, kind="mergesort")
            e_obs = np.diff(rsd_sub[order])
            v_obs = float(np.mean(e_obs ** 2) / 2.0)

            # Simulation null:  e <- diff(rsd[sample(1:nr, nr)])
            for i in range(int(n_rep)):
                perm = rng.permutation(nr)
                ep = np.diff(rsd_sub[perm])
                sim_buf[i] = float(np.mean(ep ** 2) / 2.0)

            # p = proportion of simulated values *less than* observed (mgcv convention)
            p_value = float(np.mean(sim_buf < v_obs))
            k_index = float(v_obs / rsd_var)

            table[j, 2] = np.clip(k_index, 0.0, np.inf)
            table[j, 3] = np.clip(p_value, 0.0, 1.0)

        return {
            "labels": labels,
            "table": table,
            "columns": ["k'", "edf", "k-index", "p-value"],
            "subsample_n": int(nr),
            "n_rep": int(n_rep),
        }

    # ------------------------------------------------------------------
    # k-refit heuristic (kept as a complement to k_check)
    # ------------------------------------------------------------------

    def k_refit_check(self, y=None, factor=2):
        """Refit-based basis-dimension sensitivity check.

        Doubles (or scales by *factor*) the basis dimension for every smooth
        term, refits the model with freshly optimised smoothing parameters,
        and compares total EDF and the smoothing criterion.  A large increase
        in EDF suggests the current *k* may be too small; the smoothing
        criterion gives a secondary quality signal.

        This is the heuristic that mgcv documentation recommends as a
        sensible follow-up after :meth:`k_check` flags a concern.

        Parameters
        ----------
        y : array-like or None, default=None
            Response.  ``None`` → stored training y.
        factor : int or float, default=2
            ``k_new = max(k + 1, factor * k)``.

        Returns
        -------
        dict with keys ``k_old``, ``k_new``, ``edf_old``, ``edf_new``,
        ``criterion_old``, ``criterion_new``.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = np.asarray(y, dtype=np.float64).ravel()
        method = self._optim_method or "GCV"

        k_current = self.k_
        k_new = int(max(k_current + 1, factor * k_current))

        refit = GAM(
            self.X, k=k_new, s=self.smoothing_params.copy(),
            feature_names=self.feature_names,
        )
        refit.fit(y, optimize=True, method=method)

        return {
            "k_old": k_current,
            "k_new": k_new,
            "edf_old": self.edf_,
            "edf_new": refit.edf_,
            "criterion_old": self._criterion(
                y, np.log(self.smoothing_params), method=method
            ),
            "criterion_new": refit._criterion(
                y, np.log(refit.smoothing_params), method=method
            ),
        }
