"""
GAMLSS fitting utilities: symmetric derivative index arrays and chain-rule transforms.

Mirrors three routines from mgcv/R/gamlss.r:

- ``trind_generator``  → ``trind.generator``
- ``gamlss_etamu``     → ``gamlss.etamu``
- ``gamlss_gH``        → ``gamlss.gH``

All arrays use 0-based Python indexing throughout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Symmetric index arrays  (mgcv: trind.generator)
# ---------------------------------------------------------------------------


def trind_generator(K: int) -> dict[str, np.ndarray]:
    """
    Build upper-triangular symmetric index arrays for K distribution parameters.

    For order-2 derivatives stored in upper-triangular packed form
    (i.e. columns ordered by i<=j, j varies fastest), ``i2[i, j]`` gives the
    0-based column index regardless of whether i<=j or j<i.

    Likewise ``i3[i,j,k]`` for order 3 and ``i4[i,j,k,l]`` for order 4.

    Also returns reverse indices ``i2r``, ``i3r``, ``i4r``: for the m-th packed
    entry, ``i2r[m]`` is the flat index ``j + i*K`` of the (i,j) pair where i<=j.

    Mirrors mgcv ``trind.generator(K)`` with 0-based indexing.
    """
    i4 = np.zeros((K, K, K, K), dtype=np.intp)
    m4 = 0
    for i in range(K):
        for j in range(i, K):
            for k in range(j, K):
                for ll in range(k, K):
                    for pi, pj, pk, pl in _permutations4(i, j, k, ll):
                        i4[pi, pj, pk, pl] = m4
                    m4 += 1

    i3 = np.zeros((K, K, K), dtype=np.intp)
    m3 = 0
    for j in range(K):
        for k in range(j, K):
            for ll in range(k, K):
                for pj, pk, pl in _permutations3(j, k, ll):
                    i3[pj, pk, pl] = m3
                m3 += 1

    i2 = np.zeros((K, K), dtype=np.intp)
    m = 0
    for k in range(K):
        for ll in range(k, K):
            i2[k, ll] = m
            i2[ll, k] = m
            m += 1

    # Reverse indices: for packed column m, give flat index j + i*K (i<=j)
    max_i2 = int(i2[K - 1, K - 1]) + 1
    i2r = np.zeros(max_i2, dtype=np.intp)
    m = 0
    for k in range(K):
        for ll in range(k, K):
            i2r[m] = ll + k * K
            m += 1

    max_i3 = int(i3[K - 1, K - 1, K - 1]) + 1
    i3r = np.zeros(max_i3, dtype=np.intp)
    m = 0
    for j in range(K):
        for k in range(j, K):
            for ll in range(k, K):
                i3r[m] = ll + k * K + j * K * K
                m += 1

    max_i4 = int(i4[K - 1, K - 1, K - 1, K - 1]) + 1
    i4r = np.zeros(max_i4, dtype=np.intp)
    m = 0
    for i in range(K):
        for j in range(i, K):
            for k in range(j, K):
                for ll in range(k, K):
                    i4r[m] = ll + k * K + j * K * K + i * K * K * K
                    m += 1

    return {"i2": i2, "i3": i3, "i4": i4, "i2r": i2r, "i3r": i3r, "i4r": i4r}


def _permutations4(i, j, k, ell):
    seen = set()
    for p in [
        (i, j, k, ell),
        (i, j, ell, k),
        (i, k, j, ell),
        (i, k, ell, j),
        (i, ell, j, k),
        (i, ell, k, j),
        (j, i, k, ell),
        (j, i, ell, k),
        (j, k, i, ell),
        (j, k, ell, i),
        (j, ell, i, k),
        (j, ell, k, i),
        (k, i, j, ell),
        (k, i, ell, j),
        (k, j, i, ell),
        (k, j, ell, i),
        (k, ell, i, j),
        (k, ell, j, i),
        (ell, i, j, k),
        (ell, i, k, j),
        (ell, j, i, k),
        (ell, j, k, i),
        (ell, k, i, j),
        (ell, k, j, i),
    ]:
        if p not in seen:
            seen.add(p)
            yield p


def _permutations3(j, k, ell):
    seen = set()
    for p in [
        (j, k, ell),
        (j, ell, k),
        (k, j, ell),
        (k, ell, j),
        (ell, j, k),
        (ell, k, j),
    ]:
        if p not in seen:
            seen.add(p)
            yield p


# ---------------------------------------------------------------------------
# Chain-rule transform: mu-derivatives → eta-derivatives  (mgcv: gamlss.etamu)
# ---------------------------------------------------------------------------


def gamlss_etamu(
    l1: np.ndarray,
    l2: np.ndarray,
    l3: Any,
    l4: Any,
    ig1: np.ndarray,
    g2: np.ndarray,
    g3: Any,
    g4: Any,
    i2: np.ndarray,
    i3: Any,
    i4: Any = None,  # not used in gamlss_etamu; present for interface parity with mgcv
    deriv: int = 0,
) -> dict[str, Any]:
    """
    Convert log-likelihood derivatives w.r.t. distribution parameters (mu_1..mu_K)
    to derivatives w.r.t. linear predictors (eta_1..eta_K).

    Parameters
    ----------
    l1 : (n, K)
        First derivatives of log-lik w.r.t. each parameter.
    l2 : (n, K*(K+1)//2)
        Second derivatives, packed upper-triangular (column order i<=j, j varies fastest).
    l3 : (n, K*(K+1)*(K+2)//6) or 0
        Third derivatives, similar packing.  Pass 0 if deriv < 1.
    l4 : (n, ...) or 0
        Fourth derivatives.  Pass 0 if deriv < 3.
    ig1 : (n, K)
        d mu_k / d eta_k = 1 / g_k'(mu_k) for each parameter k.
    g2 : (n, K)
        d^2 link_k / d mu_k^2 (second link derivative).
    g3 : (n, K) or 0
        Third link derivative.  Pass 0 if deriv < 1.
    g4 : (n, K) or 0
        Fourth link derivative.  Pass 0 if deriv < 3.
    i2, i3, i4 : index arrays from trind_generator.
    deriv : int
        Controls which output derivatives are computed (0-3 matching mgcv convention).

    Returns
    -------
    dict with keys l1, l2, l3, l4 containing transformed derivatives w.r.t. eta.

    Mirrors mgcv ``gamlss.etamu``.
    """
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    n, K = l1.shape
    del n  # n used implicitly via array operations

    # ---- first derivatives ----
    d1 = l1 * ig1  # (n, K) element-wise

    # ---- second derivatives ----
    d2 = np.empty_like(l2)
    m = 0
    for i in range(K):
        for j in range(i, K):
            if i == j:
                # d^2 l / d eta_i^2 = (d^2 l/d mu_i^2 - dl/dmu_i * g2_i * ig1_i) * ig1_i^2
                d2[:, m] = (l2[:, m] - l1[:, i] * g2[:, i] * ig1[:, i]) * ig1[:, i] ** 2
            else:
                # d^2 l / d eta_i d eta_j = d^2 l / d mu_i d mu_j * ig1_i * ig1_j
                d2[:, m] = l2[:, m] * ig1[:, i] * ig1[:, j]
            m += 1

    # ---- third derivatives ----
    d3: Any = 0
    if deriv > 0:
        l3 = np.asarray(l3, dtype=np.float64)
        g3 = np.asarray(g3, dtype=np.float64)
        d3 = np.empty_like(l3)
        m = 0
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    ord_ = _orders3(i, j, k)
                    mo = max(ord_)
                    if mo == 3:  # pure 3rd: all same index
                        # d^3 / d eta_i^3
                        d3[:, m] = (
                            l3[:, m]
                            - 3.0 * l2[:, i2[i, i]] * g2[:, i] * ig1[:, i]
                            + l1[:, i]
                            * (
                                3.0 * g2[:, i] ** 2 * ig1[:, i] ** 2
                                - g3[:, i] * ig1[:, i]
                            )
                        ) * ig1[:, i] ** 3
                    elif mo == 1:  # all different
                        d3[:, m] = l3[:, m] * ig1[:, i] * ig1[:, j] * ig1[:, k]
                    else:  # 2,1 mixed
                        ii = [i, j, k]
                        k1 = ii[ord_.index(1)]  # index with order 1
                        k2 = ii[ord_.index(2)]  # index with order 2
                        d3[:, m] = (
                            (l3[:, m] - l2[:, i2[k2, k1]] * g2[:, k2] * ig1[:, k2])
                            * ig1[:, k1]
                            * ig1[:, k2] ** 2
                        )
                    m += 1

    # ---- fourth derivatives ----
    d4: Any = 0
    if deriv > 2:
        l4 = np.asarray(l4, dtype=np.float64)
        g4 = np.asarray(g4, dtype=np.float64)
        d4 = np.empty_like(l4)
        m = 0
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    for ll in range(k, K):
                        ord_ = _orders4(i, j, k, ll)
                        mo = max(ord_)
                        ii = [i, j, k, ll]
                        if mo == 4:  # pure 4th: all same
                            d4[:, m] = (
                                l4[:, m]
                                - 6.0 * l3[:, i3[i, i, i]] * g2[:, i] * ig1[:, i]
                                + l2[:, i2[i, i]]
                                * (
                                    15.0 * g2[:, i] ** 2 * ig1[:, i] ** 2
                                    - 4.0 * g3[:, i] * ig1[:, i]
                                )
                                - l1[:, i]
                                * (
                                    15.0 * g2[:, i] ** 3 * ig1[:, i] ** 3
                                    - 10.0 * g2[:, i] * g3[:, i] * ig1[:, i] ** 2
                                    + g4[:, i] * ig1[:, i]
                                )
                            ) * ig1[:, i] ** 4
                        elif mo == 1:  # all different
                            d4[:, m] = (
                                l4[:, m]
                                * ig1[:, i]
                                * ig1[:, j]
                                * ig1[:, k]
                                * ig1[:, ll]
                            )
                        elif mo == 3:  # 3,1
                            k1_idx = ii[ord_.index(1)]
                            k3_idx = ii[ord_.index(3)]
                            d4[:, m] = (
                                (
                                    l4[:, m]
                                    - 3.0
                                    * l3[:, i3[k3_idx, k3_idx, k1_idx]]
                                    * g2[:, k3_idx]
                                    * ig1[:, k3_idx]
                                    + l2[:, i2[k3_idx, k1_idx]]
                                    * (
                                        3.0 * g2[:, k3_idx] ** 2 * ig1[:, k3_idx] ** 2
                                        - g3[:, k3_idx] * ig1[:, k3_idx]
                                    )
                                )
                                * ig1[:, k1_idx]
                                * ig1[:, k3_idx] ** 3
                            )
                        else:
                            counts = {}
                            for idx in ii:
                                counts[idx] = counts.get(idx, 0) + 1
                            if len(counts) == 2:  # 2,2
                                k2s = [x for x, c in counts.items() if c == 2]
                                k2a, k2b = sorted(k2s)
                                d4[:, m] = (
                                    (
                                        l4[:, m]
                                        - l3[:, i3[k2a, k2b, k2b]]
                                        * g2[:, k2a]
                                        * ig1[:, k2a]
                                        - l3[:, i3[k2a, k2a, k2b]]
                                        * g2[:, k2b]
                                        * ig1[:, k2b]
                                        + l2[:, i2[k2a, k2b]]
                                        * g2[:, k2a]
                                        * g2[:, k2b]
                                        * ig1[:, k2a]
                                        * ig1[:, k2b]
                                    )
                                    * ig1[:, k2a] ** 2
                                    * ig1[:, k2b] ** 2
                                )
                            else:  # 2,1,1
                                k2_idx = [x for x, c in counts.items() if c == 2][0]
                                k1s = [x for x, c in counts.items() if c == 1]
                                k1a, k1b = sorted(k1s)
                                d4[:, m] = (
                                    (
                                        l4[:, m]
                                        - l3[:, i3[k2_idx, k1a, k1b]]
                                        * g2[:, k2_idx]
                                        * ig1[:, k2_idx]
                                    )
                                    * ig1[:, k1a]
                                    * ig1[:, k1b]
                                    * ig1[:, k2_idx] ** 2
                                )
                        m += 1

    return {"l1": d1, "l2": d2, "l3": d3, "l4": d4}


def _orders3(i, j, k):
    """Return multiplicities of each index in sorted (i,j,k) with i<=j<=k."""
    ord_ = [1, 1, 1]
    if i == j:
        ord_[0] += 1
        ord_[1] = 0
    if i == k:
        ord_[0] += 1
        ord_[2] = 0
    if ord_[1] and j == k:
        ord_[1] += 1
        ord_[2] = 0
    return ord_


def _orders4(i, j, k, ell):
    """Return multiplicities of each index in sorted (i,j,k,ell) with i<=j<=k<=ell."""
    ord_ = [1, 1, 1, 1]
    if i == j:
        ord_[0] += 1
        ord_[1] = 0
    if i == k:
        ord_[0] += 1
        ord_[2] = 0
    if i == ell:
        ord_[0] += 1
        ord_[3] = 0
    if ord_[1]:
        if j == k:
            ord_[1] += 1
            ord_[2] = 0
        if j == ell:
            ord_[1] += 1
            ord_[3] = 0
    if ord_[2] and k == ell:
        ord_[2] += 1
        ord_[3] = 0
    return ord_


# ---------------------------------------------------------------------------
# Gradient and Hessian of penalized log-lik  (mgcv: gamlss.gH)
# ---------------------------------------------------------------------------


def gamlss_gH(
    X: np.ndarray,
    jj: list[np.ndarray],
    l1: np.ndarray,
    l2: np.ndarray,
    i2: np.ndarray,
    l3: Any = 0,
    i3: Any = 0,
    l4: Any = 0,
    i4: Any = 0,
    d1b: Any = 0,
    d2b: Any = 0,
    deriv: int = 0,
    fh: Any = None,
    D: Any = None,
    sandwich: bool = False,
) -> dict[str, Any]:
    """
    Compute gradient ``lb`` and negative Hessian ``lbb`` of log-likelihood w.r.t.
    all model coefficients, plus first and second derivatives of ``lbb`` w.r.t.
    log-smoothing parameters when ``deriv > 0``.

    Parameters
    ----------
    X : (n, p) model matrix.  ``X[:, jj[k]]`` is the design block for predictor k.
    jj : list of length K, each element an integer index array of column positions.
    l1 : (n, K) first derivatives of log-lik w.r.t. eta (after gamlss_etamu transform).
    l2 : (n, K*(K+1)//2) second derivatives (packed upper-tri, 0-based).
    i2 : (K, K) index array from trind_generator.
    l3 : (n, K*(K+1)*(K+2)//6) or scalar 0.  Third derivatives.
    i3 : (K,K,K) index array or 0.
    l4 : (n, ...) or scalar 0.  Fourth derivatives.
    i4 : (K,K,K,K) or 0.
    d1b : (rank, m) first derivatives of coefficients w.r.t. log-sp.  0 if deriv==0.
    d2b : (rank, m*(m+1)//2) second derivatives.  0 if deriv<3.
    deriv : int
        0 → grad + Hess only;
        1 → also tr(Hp^{-1} dH/drho_j) via supplied ``fh``;
        2 → full list of dH/drho matrices;
        3 → dH/drho matrices + tr(Hp^{-1} d^2H/drho_i drho_j).
    fh : Cholesky factor L (with pivot attr) or None.  Needed for deriv==1,3.
    D : (p,) diagonal preconditioner matching fh.  Needed for deriv==1,3.

    Returns
    -------
    dict with keys: lb, lbb, d1H, trHid2H.

    Mirrors mgcv ``gamlss.gH`` (dense matrix path only; no discrete support).
    """
    X = np.asarray(X, dtype=np.float64)
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    n, p = X.shape
    K = len(jj)

    # --- gradient lb ---
    lb = np.zeros(p, dtype=np.float64)
    for i in range(K):
        lb[jj[i]] += X[:, jj[i]].T @ l1[:, i]

    # --- Hessian lbb ---
    if sandwich:
        l2_sandwich = np.empty((n, K * (K + 1) // 2), dtype=np.float64)
        kk = 0
        for i in range(K):
            for j in range(i, K):
                l2_sandwich[:, kk] = l1[:, i] * l1[:, j]
                kk += 1
        l2 = l2_sandwich

    lbb = np.zeros((p, p), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            A = X[:, jj[i]].T @ (l2[:, i2[i, j], None] * X[:, jj[j]])
            lbb[np.ix_(jj[i], jj[j])] += A
            if j > i:
                lbb[np.ix_(jj[j], jj[i])] += A.T

    d1H: Any = None
    trHid2H: Any = None

    if deriv == 0:
        return {"lb": lb, "lbb": lbb, "d1H": None, "trHid2H": None}

    # --- first derivative of Hessian ---
    d1b = np.asarray(d1b, dtype=np.float64)
    l3 = np.asarray(l3, dtype=np.float64)
    m = d1b.shape[1]

    # Stack d1eta: (n*K, m) where rows n*k : n*(k+1) are for predictor k
    d1eta = np.zeros((n * K, m), dtype=np.float64)
    for ki in range(K):
        d1eta[ki * n : (ki + 1) * n, :] = X[:, jj[ki]] @ d1b[jj[ki], :]

    if deriv == 1:
        # tr(Hp^{-1} dH/drho_j) as vector
        assert fh is not None
        if D is None:
            Hp_inv = np.asarray(fh, dtype=np.float64)
        else:
            L = fh
            piv = getattr(L, "pivot", None)
            if piv is None:
                # assume already pivoted
                Hp_inv = np.linalg.solve(L.T @ L, np.eye(p))
            else:
                ipiv = np.empty_like(piv)
                ipiv[piv] = np.arange(p)
                D_arr = np.asarray(D, dtype=np.float64)
                Hp_inv = (
                    D_arr[:, None]
                    * np.linalg.solve(
                        L.T @ L,
                        np.diag(D_arr)[piv, :],
                    )[ipiv, :]
                )

        d1H = np.zeros(m, dtype=np.float64)
        for i in range(K):
            for j in range(i, K):
                Hpi = Hp_inv[np.ix_(jj[i], jj[j])]
                a = np.einsum("ij,ij->i", X[:, jj[i]] @ Hpi, X[:, jj[j]])
                for ell in range(m):
                    v = np.zeros(n, dtype=np.float64)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n : (q + 1) * n, ell]
                    mult = 1 if i == j else 2
                    d1H[ell] += mult * np.dot(a, v)

    elif deriv >= 2:
        # Full dH/drho matrices
        d1H = []
        for ell in range(m):
            dHl = np.zeros((p, p), dtype=np.float64)
            for i in range(K):
                for j in range(i, K):
                    v = np.zeros(n, dtype=np.float64)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n : (q + 1) * n, ell]
                    A = X[:, jj[i]].T @ (v[:, None] * X[:, jj[j]])
                    dHl[np.ix_(jj[i], jj[j])] += A
                    if j > i:
                        dHl[np.ix_(jj[j], jj[i])] += A.T
            d1H.append(dHl)

        if deriv >= 3:
            # tr(Hp^{-1} d^2H / drho_k drho_l)
            d2b_arr = np.asarray(d2b, dtype=np.float64)
            l4 = np.asarray(l4, dtype=np.float64)
            i4 = np.asarray(i4, dtype=np.intp)

            assert fh is not None and D is not None
            L = fh
            piv = np.asarray(getattr(L, "pivot", np.arange(p)), dtype=np.intp)
            ipiv = np.empty_like(piv)
            ipiv[piv] = np.arange(p)
            D_arr = np.asarray(D, dtype=np.float64)
            Xe = np.zeros((n * K, p), dtype=np.float64)
            for i in range(K):
                Xe[i * n : (i + 1) * n, jj[i]] = X[:, jj[i]]
            rhs = (D_arr[:, None] * Xe.T)[piv, :]
            sol = np.linalg.solve(L.T @ L, rhs)[ipiv, :]
            Xe = (D_arr[:, None] * sol).T

            # Second d eta
            d2eta = np.zeros((n * K, d2b_arr.shape[1]), dtype=np.float64)
            for ki in range(K):
                d2eta[ki * n : (ki + 1) * n, :] = X[:, jj[ki]] @ d2b_arr[jj[ki], :]

            n_sp2 = d2b_arr.shape[1]
            trHid2H = np.zeros(n_sp2, dtype=np.float64)
            VX = np.zeros((n * K, p), dtype=np.float64)
            kk = 0
            for k in range(m):
                for ell in range(k, m):
                    VX.fill(0.0)
                    for i in range(K):
                        for j in range(K):
                            v = np.zeros(n, dtype=np.float64)
                            for q in range(K):
                                v += d2eta[q * n : (q + 1) * n, kk] * l3[:, i3[i, j, q]]
                                for s in range(K):
                                    v += (
                                        d1eta[q * n : (q + 1) * n, k]
                                        * d1eta[s * n : (s + 1) * n, ell]
                                        * l4[:, i4[i, j, q, s]]
                                    )
                            if i == j:
                                rind = slice(i * n, (i + 1) * n)
                                VX[rind, jj[i]] = v[:, None] * X[:, jj[i]]
                            else:
                                rind_i = slice(i * n, (i + 1) * n)
                                rind_j = slice(j * n, (j + 1) * n)
                                VX[rind_j, jj[i]] = v[:, None] * X[:, jj[i]]
                                VX[rind_i, jj[j]] = v[:, None] * X[:, jj[j]]
                    trHid2H[kk] = float(np.sum(Xe * VX))
                    kk += 1

    return {"lb": lb, "lbb": lbb, "d1H": d1H, "trHid2H": trHid2H}
