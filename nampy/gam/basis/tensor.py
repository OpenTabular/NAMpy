import itertools

import numpy as np

from ..penalties.algebra import penalty_eigendecomposition
from ...splines.cubic_basis import cr_exact_null_basis_from_knots


def rowwise_kronecker(matrices):
    mats = [np.asarray(M, dtype=np.float64) for M in matrices]
    if len(mats) == 0:
        raise ValueError("matrices must contain at least one matrix.")
    n = mats[0].shape[0]
    for M in mats:
        if M.ndim != 2 or M.shape[0] != n:
            raise ValueError("All marginal model matrices must be 2D with equal rows.")
    out = mats[0]
    for M in mats[1:]:
        out = np.einsum("ij,ik->ijk", out, M, optimize=True).reshape(n, out.shape[1] * M.shape[1])
    return out


def lifted_tensor_penalty(S, basis_dims, axis):
    S = np.asarray(S, dtype=np.float64)
    basis_dims = [int(d) for d in basis_dims]
    left_dim = int(np.prod(basis_dims[:axis], dtype=np.int64)) if axis > 0 else 1
    right_dim = int(np.prod(basis_dims[axis + 1 :], dtype=np.int64)) if axis + 1 < len(basis_dims) else 1
    out = S
    if left_dim > 1:
        out = np.kron(np.eye(left_dim, dtype=np.float64), out)
    if right_dim > 1:
        out = np.kron(out, np.eye(right_dim, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def tensor_product_penalties(marginal_penalties, basis_dims):
    return [lifted_tensor_penalty(S, basis_dims=basis_dims, axis=j) for j, S in enumerate(marginal_penalties)]


def normalize_tensor_marginal_penalty(S, tol=1e-12):
    S = np.asarray(S, dtype=np.float64)
    if S.shape[0] == 0:
        return S.copy()
    evals = np.linalg.eigvalsh(0.5 * (S + S.T))
    scale = float(np.max(evals))
    if scale <= tol:
        return S.copy()
    return S / scale


def rescale_tensor_penalties_for_fit(B, penalties, tol=1e-12):
    B = np.asarray(B, dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in penalties]
    if len(penalties) == 0:
        return []
    x_scale = float(np.max(np.sum(np.abs(B), axis=1)) ** 2)
    if x_scale <= tol:
        return [S.copy() for S in penalties]
    out = []
    for S in penalties:
        s_scale = float(np.max(np.sum(np.abs(S), axis=0))) / x_scale
        out.append(S.copy() if s_scale <= tol else S / s_scale)
    return out


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)
    dec = penalty_eigendecomposition(S, tol=tol)
    U0, U1, d_pos = dec["U0"], dec["U1"], dec["d_pos"]
    if d_pos.size > 0:
        T_r = U1 / np.sqrt(d_pos)[np.newaxis, :]
        B_r = X @ T_r
    else:
        T_r = np.empty((S.shape[0], 0), dtype=np.float64)
        B_r = np.empty((X.shape[0], 0), dtype=np.float64)
    T_n = U0
    B_n = X @ T_n if T_n.shape[1] > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
    return {
        "B_range": B_r,
        "B_null": B_n,
        "T_range": T_r,
        "T_null": T_n,
        "range_dim": B_r.shape[1],
        "null_dim": B_n.shape[1],
        "rank": dec["rank"],
        "null_space_dim": dec["null_space_dim"],
        "tol_eff": dec["tol_eff"],
    }


def t2_marginal_reparameterization(raw_basis, raw_penalty, tol=1e-10, *, knots=None):
    """
    Reparameterize a marginal smooth for use in t2 tensor products.

    Implements mgcv's ``nat.param(type=3, unit.fnorm=TRUE)``:

    1. Eigendecompose S (descending eigenvalue order).
    2. Scale by sqrt(eigenvalues) for the penalized part; for the null space
       scale so that the Frobenius-normalised column norms match the penalized average.
    3. Rotate the null space columns so that a near-constant vector is last (type=3).
    4. Rescale both penalised and null blocks to unit Frobenius norm (unit.fnorm).

    Returns dict with keys ``B_range``, ``B_null``, ``T_range``, ``T_null``,
    ``range_dim``, ``null_dim``, ``rank``, ``null_space_dim``, ``tol_eff``.
    """
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)
    p = X.shape[1]

    # Step 1: eigendecompose S in descending order (matches R's eigen())
    evals, U = np.linalg.eigh(0.5 * (S + S.T))
    idx = np.argsort(evals)[::-1]
    evals, U = evals[idx], U[:, idx]

    tol_eff = float(np.finfo(np.float64).eps) ** 0.8 * max(1.0, float(np.max(evals)) if evals.size else 1.0)
    rank = int(np.sum(evals > tol_eff))
    null_exists = rank < p

    # Step 2: build E vector, apply eigenvectors, compute column norms
    E = np.ones(p, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    Xp = X @ U
    col_norm = np.sum(Xp ** 2, axis=0) / (E ** 2)
    av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0

    if null_exists:
        for i in range(rank, p):
            if av_norm > 0.0 and col_norm[i] > 0.0:
                E[i] = np.sqrt(col_norm[i] / av_norm)

    # P = U / E  (divide column j by E[j]; matches R's t(t(er$vectors)/E))
    P = U / E[np.newaxis, :]
    Xp = Xp / E[np.newaxis, :]

    # Step 3: type=3 null-space rotation — put near-constant column last
    # Matches R: if (null.exists && type==3 && rank < ncol(X)-1)
    if null_exists and rank < p - 1:
        ind = list(range(rank, p))           # forward  (R: (rank+1):k 1-indexed)
        rind = list(range(p - 1, rank - 1, -1))  # reversed (R: k:(rank+1) 1-indexed)
        Xn = Xp[:, ind].copy()
        n = Xn.shape[0]
        one = np.ones(n, dtype=np.float64)
        # centre columns of Xn
        Xn -= (one[:, None] * (one[None, :] @ Xn)) / n
        # eigendecompose gram matrix (R: eigen gives descending order)
        um_evals, um_vecs = np.linalg.eigh(Xn.T @ Xn)
        desc = np.argsort(um_evals)[::-1]
        um_vecs = um_vecs[:, desc]
        Xp[:, rind] = Xp[:, ind] @ um_vecs
        P[:, rind] = P[:, ind] @ um_vecs

    # Step 4: unit Frobenius norm  (unit.fnorm=TRUE)
    # Penalised block: scale columns of Xp, scale ROWS of P (matches R's P[ind,] *= scale)
    if rank > 0:
        pen_idx = list(range(rank))
        scale = 1.0 / np.sqrt(float(np.mean(Xp[:, pen_idx] ** 2)))
        Xp[:, pen_idx] *= scale
        P[pen_idx, :] *= scale   # rows of P
    else:
        scale = 1.0

    if null_exists:
        null_idx = list(range(rank, p))
        scale_f = 1.0 / np.sqrt(float(np.mean(Xp[:, null_idx] ** 2)))
        Xp[:, null_idx] *= scale_f
        P[null_idx, :] *= scale_f   # rows of P

    B_r = Xp[:, :rank] if rank > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
    B_n = Xp[:, rank:] if null_exists else np.empty((X.shape[0], 0), dtype=np.float64)
    T_r = P[:, :rank] if rank > 0 else np.empty((p, 0), dtype=np.float64)
    T_n = P[:, rank:] if null_exists else np.empty((p, 0), dtype=np.float64)
    return {
        "B_range": B_r,
        "B_null": B_n,
        "T_range": T_r,
        "T_null": T_n,
        "range_dim": int(B_r.shape[1]),
        "null_dim": int(B_n.shape[1]),
        "rank": rank,
        "null_space_dim": int(p - rank),
        "tol_eff": tol_eff,
    }


def _normalize_ord(ord_value, n_marginals):
    if ord_value is None:
        return None
    vals = [int(ord_value)] if np.isscalar(ord_value) else [int(v) for v in ord_value]
    vals = sorted(set(vals))
    for v in vals:
        if v < 0 or v > n_marginals:
            raise ValueError(f"ord entries must lie between 0 and {n_marginals}, got {vals}.")
    return vals


def _mean_constraint_matrix(B):
    B = np.asarray(B, dtype=np.float64)
    if B.shape[1] == 0:
        return np.eye(0, dtype=np.float64)
    c = B.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(c, mode="complete")
    return q[:, 1:]


def build_t2_basis_and_penalties(marginal_decompositions, *, full=False, ord=None, remove_constant_from_null_block=True):
    m = len(marginal_decompositions)
    ord_keep = _normalize_ord(ord, m)
    option_lists = []
    for dec in marginal_decompositions:
        opts = []
        q, r = dec["null_dim"], dec["range_dim"]
        if q > 0:
            if full:
                for j in range(q):
                    opts.append({"kind": "null", "cols": [j], "label": f"n{j+1}"})
            else:
                opts.append({"kind": "null", "cols": list(range(q)), "label": "n"})
        if r > 0:
            opts.append({"kind": "range", "cols": None, "label": "r"})
        option_lists.append(opts)

    specs = []
    for combo in itertools.product(*option_lists):
        order = sum(1 for c in combo if c["kind"] == "range")
        if ord_keep is not None and order not in ord_keep:
            continue
        specs.append({"combo": combo, "order": order, "label": "".join(c["label"] for c in combo)})
    specs.sort(key=lambda spec: -int(spec["order"]))
    penalized_specs = [spec for spec in specs if spec["order"] > 0]
    allnull_specs = [spec for spec in specs if spec["order"] == 0]

    def materialize_component(spec):
        mats = []
        for dec, choice in zip(marginal_decompositions, spec["combo"]):
            mats.append(dec["B_range"] if choice["kind"] == "range" else dec["B_null"][:, choice["cols"]])
        return rowwise_kronecker(mats)

    basis_blocks, block_meta = [], []
    for spec in penalized_specs:
        B = materialize_component(spec)
        if B.shape[1] == 0:
            continue
        basis_blocks.append(B)
        block_meta.append({"order": spec["order"], "label": spec["label"], "penalized": True, "n_cols": B.shape[1]})

    B0_raw = np.column_stack([materialize_component(spec) for spec in allnull_specs]) if len(allnull_specs) > 0 else None
    basis_pre = np.column_stack(basis_blocks + ([B0_raw] if B0_raw is not None and B0_raw.shape[1] > 0 else []))
    penalties_pre, component_slices_pre = [], []
    start = 0
    total_dim_pre = basis_pre.shape[1]
    for meta in block_meta:
        sl = slice(start, start + meta["n_cols"])
        component_slices_pre.append(sl)
        P = np.zeros((total_dim_pre, total_dim_pre), dtype=np.float64)
        P[sl, sl] = np.eye(meta["n_cols"], dtype=np.float64)
        penalties_pre.append(P)
        start += meta["n_cols"]
    allnull_transform = None
    basis, penalties, component_slices = basis_pre, penalties_pre, component_slices_pre
    if B0_raw is not None and B0_raw.shape[1] > 0 and remove_constant_from_null_block:
        C0 = _mean_constraint_matrix(B0_raw)
        allnull_transform = C0
        n_pen = basis_pre.shape[1] - B0_raw.shape[1]
        C_full = np.eye(basis_pre.shape[1], dtype=np.float64)
        C_full = np.column_stack([C_full[:, :n_pen], np.vstack([np.zeros((n_pen, C0.shape[1]), dtype=np.float64), C0])])
        basis = basis_pre @ C_full
        penalties = [0.5 * (C_full.T @ S @ C_full + (C_full.T @ S @ C_full).T) for S in penalties_pre]
        component_slices = component_slices_pre[:-1] + [slice(n_pen, basis.shape[1])]
    return {
        "basis": basis,
        "penalties": penalties,
        "component_slices": component_slices,
        "basis_pre_constraint": basis_pre,
        "penalties_pre_constraint": penalties_pre,
        "allnull_specs": allnull_specs,
        "allnull_transform": allnull_transform,
        "component_specs": block_meta + ([{"order": 0, "label": "null", "penalized": False, "n_cols": 0}] if B0_raw is not None else []),
        "penalized_specs": penalized_specs,
    }


def materialize_t2_newdata(marginal_new_range_null, *, allnull_specs, allnull_transform, penalized_specs):
    def materialize_component(spec):
        mats = []
        for dec, choice in zip(marginal_new_range_null, spec["combo"]):
            mats.append(dec["B_range"] if choice["kind"] == "range" else dec["B_null"][:, choice["cols"]])
        return rowwise_kronecker(mats)

    blocks = []
    for spec in penalized_specs:
        B = materialize_component(spec)
        if B.shape[1] > 0:
            blocks.append(B)
    if len(allnull_specs) > 0:
        B0 = np.column_stack([materialize_component(spec) for spec in allnull_specs])
        if allnull_transform is not None and B0.shape[1] > 0:
            B0 = B0 @ allnull_transform
        if B0.shape[1] > 0:
            blocks.append(B0)
    if len(blocks) == 0:
        n = next(iter(marginal_new_range_null))["B_null"].shape[0]
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack(blocks)

__all__ = [
    "rowwise_kronecker",
    "lifted_tensor_penalty",
    "tensor_product_penalties",
    "normalize_tensor_marginal_penalty",
    "rescale_tensor_penalties_for_fit",
    "marginal_range_null_decomposition",
    "t2_marginal_reparameterization",
    "build_t2_basis_and_penalties",
    "materialize_t2_newdata",
]
