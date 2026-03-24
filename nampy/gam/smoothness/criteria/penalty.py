"""Penalty space geometry and stable log-determinants for multi-penalty REML."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def _positive_semidefinite_root(P, *, tol=1e-10):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Penalty root requires a square matrix.")
    if P.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)

    P_sym = 0.5 * (P + P.T)
    evals, U = np.linalg.eigh(P_sym)
    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]
    tol_eff = float(tol) * max(1.0, float(np.max(np.abs(evals))))
    pos_mask = evals > tol_eff
    if not np.any(pos_mask):
        return np.empty((P.shape[0], 0), dtype=np.float64)
    return U[:, pos_mask] * np.sqrt(evals[pos_mask])[np.newaxis, :]


def _eigen_positive_mask(evals, *, tol=1e-10):
    evals = np.asarray(evals, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    # Treat `tol` as an absolute PSD threshold and only add a machine-precision
    # relative term for numerical noise. Using a large purely relative threshold
    # can incorrectly erase genuine small positive eigenvalues for multi-penalty
    # Gaussian REML criteria (e.g. multi-penalty factor-smooth terms).
    tol_eff = max(float(tol), np.finfo(np.float64).eps * scale)
    return evals > tol_eff, float(tol_eff)


def _static_penalty_space(model, *, tol=1e-10):
    cache = getattr(model, "_penalty_subspace_cache_", None)
    if cache is not None:
        return cache

    p_pen = int(model.n_coef_ or 0)
    n_sp = int(model.n_smoothing_params_ or 0)
    if p_pen == 0 or not getattr(model, "penalty_blocks_", None):
        cache = {
            "Y": np.empty((p_pen, 0), dtype=np.float64),
            "Z": np.eye(p_pen, dtype=np.float64),
            "S_groups": [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)],
        }
        setattr(model, "_penalty_subspace_cache_", cache)
        return cache

    St = np.zeros((p_pen, p_pen), dtype=np.float64)
    roots = []
    group_indices = []
    for pb in model.penalty_blocks_:
        S_emb = np.zeros((p_pen, p_pen), dtype=np.float64)
        sl = pb.coef_slice
        S_loc = np.asarray(pb.matrix, dtype=np.float64)
        S_emb[sl, sl] += S_loc
        frob = float(np.sqrt(np.sum(S_emb * S_emb)))
        if frob > 0.0:
            St += S_emb / frob
        roots.append(_positive_semidefinite_root(S_emb, tol=tol))
        group_indices.append(int(pb.smoothing_index))

    if np.allclose(St, 0.0):
        cache = {
            "Y": np.empty((p_pen, 0), dtype=np.float64),
            "Z": np.eye(p_pen, dtype=np.float64),
            "S_groups": [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)],
        }
        setattr(model, "_penalty_subspace_cache_", cache)
        return cache

    es_val, es_vec = np.linalg.eigh(0.5 * (St + St.T))
    idx = np.argsort(es_val)
    es_val = es_val[idx]
    es_vec = es_vec[:, idx]
    tol_eff = max(float(np.max(es_val)), 1.0) * (np.finfo(np.float64).eps ** 0.66)
    range_mask = es_val > tol_eff
    Y = es_vec[:, range_mask]
    Z = es_vec[:, ~range_mask]

    q = int(Y.shape[1])
    S_groups = [np.zeros((q, q), dtype=np.float64) for _ in range(n_sp)]
    if q > 0:
        YT = Y.T
        for root, sp_idx in zip(roots, group_indices):
            if root.shape[1] == 0:
                continue
            Ur = YT @ root
            S_groups[sp_idx] += Ur @ Ur.T

    cache = {"Y": Y, "Z": Z, "S_groups": S_groups}
    setattr(model, "_penalty_subspace_cache_", cache)
    return cache


def _static_penalty_null_dim(model, *, tol=1e-10):
    cache = _static_penalty_space(model, tol=tol)
    return int(np.asarray(cache["Z"], dtype=np.float64).shape[1])


def _static_fixed_and_random_designs(model, X_full, sp, *, tol=1e-10):
    X_full = np.asarray(X_full, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)

    if model.fit_intercept:
        X_intercept = X_full[:, :1]
        X_pen = X_full[:, 1:]
    else:
        X_intercept = np.empty((X_full.shape[0], 0), dtype=np.float64)
        X_pen = X_full

    cache = _static_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    Z = np.asarray(cache["Z"], dtype=np.float64)
    S_groups = cache["S_groups"]

    parts_fix = []
    if X_intercept.shape[1] > 0:
        parts_fix.append(X_intercept)
    if Z.shape[1] > 0:
        parts_fix.append(X_pen @ Z)

    if Y.shape[1] == 0:
        Xf = (
            np.column_stack(parts_fix)
            if parts_fix
            else np.empty((X_full.shape[0], 0), dtype=np.float64)
        )
        return Xf, np.empty((X_full.shape[0], 0), dtype=np.float64), {
            "rank": 0,
            "null_dim": int(Z.shape[1]),
            "logdet_plus": 0.0,
        }

    S_range = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
    for k, Sg in enumerate(S_groups):
        if Sg.size == 0:
            continue
        S_range += float(sp[k]) * np.asarray(Sg, dtype=np.float64)

    evals, U = np.linalg.eigh(0.5 * (S_range + S_range.T))
    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]
    pos_mask, tol_eff = _eigen_positive_mask(evals, tol=tol)
    null_mask = ~pos_mask

    if np.any(null_mask):
        parts_fix.append(X_pen @ (Y @ U[:, null_mask]))

    Xf = (
        np.column_stack(parts_fix)
        if parts_fix
        else np.empty((X_full.shape[0], 0), dtype=np.float64)
    )

    if np.any(pos_mask):
        U1 = U[:, pos_mask]
        d_pos = np.asarray(evals[pos_mask], dtype=np.float64)
        Zr = X_pen @ ((Y @ U1) / np.sqrt(d_pos)[np.newaxis, :])
        logdet_plus = float(np.sum(np.log(d_pos)))
    else:
        Zr = np.empty((X_full.shape[0], 0), dtype=np.float64)
        d_pos = np.empty((0,), dtype=np.float64)
        logdet_plus = 0.0

    return Xf, Zr, {
        "rank": int(d_pos.size),
        "null_dim": int(Z.shape[1] + np.sum(null_mask)),
        "logdet_plus": logdet_plus,
    }


def _static_penalty_summary(model, sp, *, tol=1e-10):
    sp = np.asarray(sp, dtype=np.float64)
    cache = _static_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    Z = np.asarray(cache["Z"], dtype=np.float64)
    S_groups = cache["S_groups"]

    if Y.shape[1] == 0:
        return {
            "rank": 0,
            "null_dim": int(Z.shape[1]),
            "logdet_plus": 0.0,
        }

    S_range = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
    for k, Sg in enumerate(S_groups):
        if Sg.size == 0:
            continue
        S_range += float(sp[k]) * np.asarray(Sg, dtype=np.float64)

    evals = np.linalg.eigvalsh(0.5 * (S_range + S_range.T))
    pos_mask, tol_eff = _eigen_positive_mask(evals, tol=tol)
    d_pos = np.asarray(evals[pos_mask], dtype=np.float64)
    return {
        "rank": int(d_pos.size),
        "null_dim": int(Z.shape[1] + np.sum(~pos_mask)),
        "logdet_plus": float(np.sum(np.log(d_pos)) if d_pos.size > 0 else 0.0),
    }


def _stable_penalty_logdet(model, sp, *, tol=1e-10):
    """
    Stable log-determinant of the penalty for multi-penalty REML (Wood-style).

    Sums log-determinants of each penalty block evaluated at the given smoothing
    parameters, rather than computing the pseudodeterminant of the assembled
    total penalty.  The assembled form is adequate for well-separated penalties
    but can drift near the boundary in grouped multi-penalty Gaussian REML problems.
    """
    sp = np.asarray(sp, dtype=np.float64).ravel()
    cache = _static_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    S_groups = [np.asarray(Sg, dtype=np.float64) for Sg in cache["S_groups"]]
    q = int(Y.shape[1])

    if q == 0 or len(S_groups) == 0:
        return 0.0

    Si = [
        root @ root.T
        for root in (
            _positive_semidefinite_root(0.5 * (Sg + Sg.T), tol=tol) for Sg in S_groups
        )
    ]
    if not any(A.size for A in Si):
        return 0.0

    d_tol = float(np.finfo(np.float64).eps ** 0.3)
    r_tol = float(np.finfo(np.float64).eps ** 0.75)

    S_out = np.zeros((q, q), dtype=np.float64)
    gamma = np.ones(len(Si), dtype=bool)
    K = 0
    Q = q
    iteration = 0

    # `Si_active` always stores the current similarity-transformed penalty
    # blocks in the active lower-right subspace of dimension `Q`.
    Si_active = [np.asarray(A, dtype=np.float64).copy() for A in Si]

    while True:
        iteration += 1

        frob = np.array(
            [
                float(np.linalg.norm(Si_active[i], ord="fro")) if gamma[i] else 0.0
                for i in range(len(Si_active))
            ],
            dtype=np.float64,
        )
        max_frob = max(
            [float(frob[i] * sp[i]) for i in range(len(Si_active)) if gamma[i]] + [0.0]
        )
        if not np.isfinite(max_frob) or max_frob <= 0.0:
            return 0.0

        alpha = np.zeros(len(Si_active), dtype=bool)
        gamma1 = np.zeros(len(Si_active), dtype=bool)
        for i in range(len(Si_active)):
            if not gamma[i]:
                continue
            if float(frob[i] * sp[i]) > max_frob * d_tol:
                alpha[i] = True
            else:
                gamma1[i] = True

        if np.any(gamma1):
            Sb = np.zeros((Q, Q), dtype=np.float64)
            for i, A in enumerate(Si_active):
                if alpha[i]:
                    Sb += A / float(frob[i])
            Sb = 0.5 * (Sb + Sb.T)
            ev = np.linalg.eigvalsh(Sb)
            if ev.size == 0 or ev[-1] <= 0.0:
                r = 0
            else:
                r = 1
                while r < Q and ev[Q - r - 1] > ev[Q - 1] * r_tol:
                    r += 1
        else:
            r = Q

        if Q == r:
            if iteration == 1:
                S_out.fill(0.0)
                for i, A in enumerate(Si_active):
                    S_out += float(sp[i]) * A
            break

        Sb = np.zeros((Q, Q), dtype=np.float64)
        Sg = np.zeros((Q, Q), dtype=np.float64)
        for i, A in enumerate(Si_active):
            if alpha[i]:
                Sb += float(sp[i]) * A
            elif gamma1[i]:
                Sg += float(sp[i]) * A

        Sb = 0.5 * (Sb + Sb.T)
        evals, U = np.linalg.eigh(Sb)
        idx = np.argsort(evals)[::-1]
        evals = np.asarray(evals[idx], dtype=np.float64)
        U = np.asarray(U[:, idx], dtype=np.float64)

        if K > 0:
            B = S_out[:K, K : K + Q] @ U
            S_out[:K, K : K + Q] = B
            S_out[K : K + Q, :K] = B.T

        C = U.T @ Sg @ U
        if r > 0:
            C[np.arange(r), np.arange(r)] += evals[:r]
        S_out[K : K + Q, K : K + Q] = C

        Un = np.asarray(U[:, r:], dtype=np.float64)
        Si_active = [
            Un.T @ A @ Un if gamma1[i] else A for i, A in enumerate(Si_active)
        ]
        K += r
        Q -= r
        gamma = gamma1

    sign, logdet = np.linalg.slogdet(0.5 * (S_out + S_out.T))
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    return float(logdet)


def _stable_penalty_logdet_derivatives(model, sp, *, tol=1e-10, order=2):
    sp = np.asarray(sp, dtype=np.float64).ravel()
    n_sp = int(model.n_smoothing_params_ or 0)
    grad = np.zeros(n_sp, dtype=np.float64)
    hess = np.zeros((n_sp, n_sp), dtype=np.float64)

    cache = _static_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    S_groups = [0.5 * (np.asarray(Sg, dtype=np.float64) + np.asarray(Sg, dtype=np.float64).T) for Sg in cache["S_groups"]]
    q = int(Y.shape[1])

    if q == 0 or len(S_groups) == 0:
        return 0.0, grad, hess

    if not any(Sg.size and np.any(Sg) for Sg in S_groups):
        return 0.0, grad, hess

    d_tol = float(np.finfo(np.float64).eps ** 0.3)
    r_tol = float(np.finfo(np.float64).eps ** 0.75)

    S_out = np.zeros((q, q), dtype=np.float64)
    Qf = np.eye(q, dtype=np.float64)
    gamma = np.ones(len(S_groups), dtype=bool)
    K = 0
    Q = q
    iteration = 0
    Si_active = [Sg.copy() for Sg in S_groups]

    while True:
        iteration += 1
        frob = np.array(
            [
                float(np.linalg.norm(Si_active[i], ord="fro")) if gamma[i] else 0.0
                for i in range(len(Si_active))
            ],
            dtype=np.float64,
        )
        max_frob = max(
            [float(frob[i] * sp[i]) for i in range(len(Si_active)) if gamma[i]] + [0.0]
        )
        if not np.isfinite(max_frob) or max_frob <= 0.0:
            return 0.0, grad, hess

        alpha = np.zeros(len(Si_active), dtype=bool)
        gamma1 = np.zeros(len(Si_active), dtype=bool)
        for i in range(len(Si_active)):
            if not gamma[i]:
                continue
            if float(frob[i] * sp[i]) > max_frob * d_tol:
                alpha[i] = True
            else:
                gamma1[i] = True

        if np.any(gamma1):
            Sb = np.zeros((Q, Q), dtype=np.float64)
            for i, A in enumerate(Si_active):
                if alpha[i]:
                    Sb += A / float(frob[i])
            Sb = 0.5 * (Sb + Sb.T)
            ev = np.linalg.eigvalsh(Sb)
            if ev.size == 0 or ev[-1] <= 0.0:
                r = 0
            else:
                r = 1
                while r < Q and ev[Q - r - 1] > ev[Q - 1] * r_tol:
                    r += 1
        else:
            r = Q

        if Q == r:
            if iteration == 1:
                S_out.fill(0.0)
                for i, A in enumerate(Si_active):
                    S_out += float(sp[i]) * A
            break

        Sb = np.zeros((Q, Q), dtype=np.float64)
        Sg = np.zeros((Q, Q), dtype=np.float64)
        for i, A in enumerate(Si_active):
            if alpha[i]:
                Sb += float(sp[i]) * A
            elif gamma1[i]:
                Sg += float(sp[i]) * A
        Sb = 0.5 * (Sb + Sb.T)
        evals, U = np.linalg.eigh(Sb)
        idx = np.argsort(evals)[::-1]
        evals = np.asarray(evals[idx], dtype=np.float64)
        U = np.asarray(U[:, idx], dtype=np.float64)

        if iteration == 1:
            Qf[:, :Q] = U
        else:
            Qf[:, K : K + Q] = Qf[:, K : K + Q] @ U

        if K > 0:
            B = S_out[:K, K : K + Q] @ U
            S_out[:K, K : K + Q] = B
            S_out[K : K + Q, :K] = B.T

        C = U.T @ Sg @ U
        if r > 0:
            C[np.arange(r), np.arange(r)] += evals[:r]
        S_out[K : K + Q, K : K + Q] = C

        Un = np.asarray(U[:, r:], dtype=np.float64)
        Si_active = [
            Un.T @ A @ Un if gamma1[i] else A for i, A in enumerate(Si_active)
        ]
        K += r
        Q -= r
        gamma = gamma1

    S_out = 0.5 * (S_out + S_out.T)
    try:
        cS, loS = cho_factor(S_out, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf, np.full(n_sp, np.nan), np.full((n_sp, n_sp), np.nan)
    logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(cS)))))
    if not np.isfinite(logdet):
        return np.inf, np.full(n_sp, np.nan), np.full((n_sp, n_sp), np.nan)
    if order <= 0:
        return logdet, grad, hess

    S_inv = cho_solve((cS, loS), np.eye(q), check_finite=False)
    transformed = [
        0.5 * (Qf.T @ Sg @ Qf + (Qf.T @ Sg @ Qf).T)
        for Sg in S_groups
    ]
    SinvSi = [S_inv @ Si for Si in transformed]
    for i, Si in enumerate(transformed):
        if not Si.size or not np.any(Si):
            continue
        grad[i] = float(sp[i] * np.trace(SinvSi[i]))

    if order <= 1:
        return logdet, grad, hess

    for i in range(n_sp):
        if not transformed[i].size or not np.any(transformed[i]):
            continue
        for j in range(i, n_sp):
            if not transformed[j].size or not np.any(transformed[j]):
                continue
            val = -float(sp[i] * sp[j] * np.trace(SinvSi[i] @ SinvSi[j]))
            if i == j:
                val += float(grad[i])
            hess[i, j] = val
            hess[j, i] = val
    return logdet, grad, hess
