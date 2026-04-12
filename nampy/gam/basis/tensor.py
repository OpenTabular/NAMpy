import numpy as np
from scipy.linalg import eigh

from .._mgcv_constants import EIG_TOL_POWER
from ..penalties.algebra import penalty_eigendecomposition


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
        out = np.einsum("ij,ik->ijk", out, M, optimize=True).reshape(
            n, out.shape[1] * M.shape[1]
        )
    return out


def lifted_tensor_penalty(S, basis_dims, axis):
    S = np.asarray(S, dtype=np.float64)
    basis_dims = [int(d) for d in basis_dims]
    left_dim = int(np.prod(basis_dims[:axis], dtype=np.int64)) if axis > 0 else 1
    right_dim = (
        int(np.prod(basis_dims[axis + 1 :], dtype=np.int64))
        if axis + 1 < len(basis_dims)
        else 1
    )
    out = S
    if left_dim > 1:
        out = np.kron(np.eye(left_dim, dtype=np.float64), out)
    if right_dim > 1:
        out = np.kron(out, np.eye(right_dim, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def tensor_product_penalties(marginal_penalties, basis_dims):
    return [
        lifted_tensor_penalty(S, basis_dims=basis_dims, axis=j)
        for j, S in enumerate(marginal_penalties)
    ]


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


def _eigen_split(raw_basis, raw_penalty, tol=1e-10, *, mode="range_null", knots=None):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)

    if mode == "range_null":
        dec = penalty_eigendecomposition(S, tol=tol)
        U0, U1, d_pos = dec["U0"], dec["U1"], dec["d_pos"]
        if d_pos.size > 0:
            T_r = U1 / np.sqrt(d_pos)[np.newaxis, :]
            B_r = X @ T_r
        else:
            T_r = np.empty((S.shape[0], 0), dtype=np.float64)
            B_r = np.empty((X.shape[0], 0), dtype=np.float64)
        T_n = U0
        B_n = (
            X @ T_n if T_n.shape[1] > 0 else np.empty((X.shape[0], 0), dtype=np.float64)
        )
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

    if mode != "t2":
        raise ValueError(f"Unknown eigen split mode {mode!r}.")

    p = X.shape[1]
    evals, U = eigh(0.5 * (S + S.T), driver="evr")
    idx = np.argsort(evals)[::-1]
    evals, U = evals[idx], U[:, idx]

    tol_eff = float(np.finfo(np.float64).eps) ** EIG_TOL_POWER * max(
        1.0, float(np.max(evals)) if evals.size else 1.0
    )
    rank = int(np.sum(evals > tol_eff))
    null_exists = rank < p

    E = np.ones(p, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    Xp = X @ U
    col_norm = np.sum(Xp**2, axis=0) / (E**2)
    av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0

    if null_exists:
        for i in range(rank, p):
            if av_norm > 0.0 and col_norm[i] > 0.0:
                E[i] = np.sqrt(col_norm[i] / av_norm)

    P = U / E[np.newaxis, :]
    Xp = Xp / E[np.newaxis, :]

    if null_exists and rank < p - 1:
        ind = list(range(rank, p))
        rind = list(range(p - 1, rank - 1, -1))
        Xn = Xp[:, ind].copy()
        n = Xn.shape[0]
        one = np.ones(n, dtype=np.float64)
        Xn -= (one[:, None] * (one[None, :] @ Xn)) / n
        um_evals, um_vecs = eigh(Xn.T @ Xn, driver="evr")
        desc = np.argsort(um_evals)[::-1]
        um_vecs = um_vecs[:, desc]
        Xp[:, rind] = Xp[:, ind] @ um_vecs
        P[:, rind] = P[:, ind] @ um_vecs

    if rank > 0:
        pen_idx = list(range(rank))
        scale = 1.0 / np.sqrt(float(np.mean(Xp[:, pen_idx] ** 2)))
        Xp[:, pen_idx] *= scale
        P[pen_idx, :] *= scale

    if null_exists:
        null_idx = list(range(rank, p))
        scale_f = 1.0 / np.sqrt(float(np.mean(Xp[:, null_idx] ** 2)))
        Xp[:, null_idx] *= scale_f
        P[null_idx, :] *= scale_f

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


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="range_null")


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
    return _eigen_split(raw_basis, raw_penalty, tol=tol, mode="t2", knots=knots)


def _normalize_ord(ord_value, n_marginals):
    if ord_value is None:
        return None
    vals = [int(ord_value)] if np.isscalar(ord_value) else [int(v) for v in ord_value]
    vals = sorted(set(vals))
    for v in vals:
        if v < 0 or v > n_marginals:
            raise ValueError(
                f"ord entries must lie between 0 and {n_marginals}, got {vals}."
            )
    return vals


def _mean_constraint_matrix(B):
    B = np.asarray(B, dtype=np.float64)
    if B.shape[1] == 0:
        return np.eye(0, dtype=np.float64)
    c = B.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(c, mode="complete")
    return q[:, 1:]


def build_t2_basis_and_penalties(
    marginal_decompositions,
    *,
    full=False,
    ord=None,
    remove_constant_from_null_block=True,
):
    m = len(marginal_decompositions)
    ord_keep = _normalize_ord(ord, m)
    if m == 0:
        raise ValueError("marginal_decompositions must contain at least one margin.")

    def _null_labels(n_cols):
        return [f"n{i+1}" for i in range(n_cols)] if full else ["n"]

    def _rowwise_product(A, B):
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.shape[1] == 0 or B.shape[1] == 0:
            return np.empty((A.shape[0], 0), dtype=np.float64)
        return rowwise_kronecker([A, B])

    first = marginal_decompositions[0]
    Z1 = np.asarray(first["B_range"], dtype=np.float64)
    X2_blocks = []
    X2_desc = []
    order_list = []
    label_list = []
    pen2 = [] if full else None
    no_null = False

    if Z1.shape[1] > 0:
        X2_blocks.append(Z1)
        X2_desc.append([{"kind": "range", "cols": None}])
        order_list.append(1)
        label_list.append("r")
        if full:
            pen2.append(True)

    if first["null_dim"] > 0:
        X1_null = np.asarray(first["B_null"], dtype=np.float64)
        if full:
            # mgcv treats each null-space column separately when full=TRUE
            for j, lab in enumerate(_null_labels(X1_null.shape[1])):
                X2_blocks.append(X1_null[:, [j]])
                X2_desc.append([{"kind": "null", "cols": [j]}])
                order_list.append(0)
                label_list.append(lab)
                pen2.append(False)
        else:
            X2_blocks.append(X1_null)
            X2_desc.append([{"kind": "null", "cols": list(range(X1_null.shape[1]))}])
            order_list.append(0)
            label_list.append("n")
    else:
        no_null = True

    for margin_idx in range(1, m):
        dec = marginal_decompositions[margin_idx]
        Zi = np.asarray(dec["B_range"], dtype=np.float64)
        null_exists = int(dec["null_dim"]) > 0
        Xi = np.asarray(dec["B_null"], dtype=np.float64) if null_exists else None

        X1 = X2_blocks
        D1 = X2_desc
        lab1 = label_list
        order1 = order_list
        pen1 = pen2[:] if full else None

        X2_blocks = []
        X2_desc = []
        label_list = []
        order_list = []
        if full:
            pen2 = []

        # Range-space products.
        if Zi.shape[1] > 0:
            for ii, block in enumerate(X1):
                was_pen = True if not full else bool(pen1[ii])
                if (not full) or was_pen:
                    A = _rowwise_product(block, Zi)
                    if A.shape[1] == 0:
                        continue
                    X2_blocks.append(A)
                    X2_desc.append(D1[ii] + [{"kind": "range", "cols": None}])
                    label_list.append(f"{lab1[ii]}r")
                    order_list.append(int(order1[ii]) + 1)
                    if full:
                        pen2.append(True)
                else:
                    for j in range(block.shape[1]):
                        A = _rowwise_product(block[:, [j]], Zi)
                        if A.shape[1] == 0:
                            continue
                        X2_blocks.append(A)
                        X2_desc.append(D1[ii] + [{"kind": "range", "cols": None}])
                        label_list.append(f"{lab1[ii]}r")
                        order_list.append(int(order1[ii]) + 1)
                        pen2.append(True)

        # Null-space products.
        if null_exists and Xi.shape[1] > 0:
            for ii, block in enumerate(X1):
                was_pen = True if not full else bool(pen1[ii])
                if (not full) or (not was_pen):
                    A = _rowwise_product(block, Xi)
                    if A.shape[1] == 0:
                        continue
                    X2_blocks.append(A)
                    X2_desc.append(
                        D1[ii] + [{"kind": "null", "cols": list(range(Xi.shape[1]))}]
                    )
                    label_list.append(f"{lab1[ii]}n")
                    order_list.append(int(order1[ii]))
                    if full:
                        pen2.append(False)
                else:
                    null_labs = _null_labels(Xi.shape[1])
                    for j in range(Xi.shape[1]):
                        A = _rowwise_product(block, Xi[:, [j]])
                        if A.shape[1] == 0:
                            continue
                        X2_blocks.append(A)
                        X2_desc.append(D1[ii] + [{"kind": "null", "cols": [j]}])
                        label_list.append(f"{lab1[ii]}{null_labs[j]}")
                        order_list.append(int(order1[ii]))
                        pen2.append(True)
        else:
            no_null = True

    if ord_keep is not None:
        keep_mask = [int(o) in ord_keep for o in order_list]
        X2_blocks = [blk for blk, keep in zip(X2_blocks, keep_mask) if keep]
        X2_desc = [desc for desc, keep in zip(X2_desc, keep_mask) if keep]
        label_list = [lab for lab, keep in zip(label_list, keep_mask) if keep]
        order_list = [o for o, keep in zip(order_list, keep_mask) if keep]
        if full:
            pen2 = [p for p, keep in zip(pen2, keep_mask) if keep]
        if 0 not in ord_keep:
            no_null = True

    if full and len(X2_blocks) > 0 and pen2 is not None:
        null_idx = [i for i, is_pen in enumerate(pen2) if not bool(is_pen)]
        if len(null_idx) > 1:
            pen_idx = [i for i, is_pen in enumerate(pen2) if bool(is_pen)]
            null_blocks = [X2_blocks[i] for i in null_idx]
            null_desc = [X2_desc[i] for i in null_idx]
            X2_blocks = [X2_blocks[i] for i in pen_idx] + [np.column_stack(null_blocks)]
            X2_desc = [X2_desc[i] for i in pen_idx] + [null_desc]
            label_list = [label_list[i] for i in pen_idx] + ["n"]
            order_list = [order_list[i] for i in pen_idx] + [0]
            pen2 = [True] * len(pen_idx) + [False]

    xc_all = [blk.shape[1] for blk in X2_blocks]
    basis_pre = (
        np.column_stack(X2_blocks)
        if len(X2_blocks) > 0
        else np.empty(
            (marginal_decompositions[0]["B_range"].shape[0], 0), dtype=np.float64
        )
    )

    if not no_null and len(xc_all) > 0:
        pen_col_counts = xc_all[:-1]
        pen_labels = label_list[:-1]
        pen_desc = X2_desc[:-1]
        B0_raw = X2_blocks[-1]
        B0_descs = X2_desc[-1] if full else [X2_desc[-1]]
    else:
        pen_col_counts = xc_all
        pen_labels = label_list
        pen_desc = X2_desc
        B0_raw = None
        B0_descs = []

    penalties_pre = []
    component_slices_pre = []
    block_meta = []
    start = 0
    total_dim_pre = basis_pre.shape[1]
    for lab, order_val, n_cols, desc in zip(
        pen_labels,
        order_list[: len(pen_col_counts)],
        pen_col_counts,
        pen_desc,
    ):
        sl = slice(start, start + n_cols)
        component_slices_pre.append(sl)
        P = np.zeros((total_dim_pre, total_dim_pre), dtype=np.float64)
        P[sl, sl] = np.eye(n_cols, dtype=np.float64)
        penalties_pre.append(P)
        block_meta.append(
            {
                "order": int(order_val),
                "label": str(lab),
                "penalized": True,
                "n_cols": int(n_cols),
                "desc": tuple(desc),
            }
        )
        start += n_cols

    penalized_specs = [
        {"combo": tuple(meta["desc"]), "order": meta["order"], "label": meta["label"]}
        for meta in block_meta
    ]
    allnull_specs = (
        [{"combo": tuple(desc), "order": 0, "label": "null"} for desc in B0_descs]
        if len(B0_descs) > 0
        else []
    )
    allnull_transform = None
    full_constraint_transform = None
    basis, penalties, component_slices = basis_pre, penalties_pre, component_slices_pre
    if B0_raw is not None and B0_raw.shape[1] > 0 and remove_constant_from_null_block:
        C0 = _mean_constraint_matrix(B0_raw)
        allnull_transform = C0
        n_pen = basis_pre.shape[1] - B0_raw.shape[1]
        C_full = np.eye(basis_pre.shape[1], dtype=np.float64)
        C_full = np.column_stack(
            [
                C_full[:, :n_pen],
                np.vstack([np.zeros((n_pen, C0.shape[1]), dtype=np.float64), C0]),
            ]
        )
        full_constraint_transform = C_full
        basis = basis_pre @ C_full
        penalties = [
            0.5 * (C_full.T @ S @ C_full + (C_full.T @ S @ C_full).T)
            for S in penalties_pre
        ]
        component_slices = component_slices_pre[:-1] + [slice(n_pen, basis.shape[1])]
    return {
        "basis": basis,
        "penalties": penalties,
        "component_slices": component_slices,
        "basis_pre_constraint": basis_pre,
        "penalties_pre_constraint": penalties_pre,
        "allnull_specs": allnull_specs,
        "allnull_transform": allnull_transform,
        "full_constraint_transform": full_constraint_transform,
        "component_specs": block_meta
        + (
            [{"order": 0, "label": "null", "penalized": False, "n_cols": 0}]
            if B0_raw is not None
            else []
        ),
        "penalized_specs": penalized_specs,
    }


def materialize_t2_newdata(
    marginal_new_range_null, *, allnull_specs, allnull_transform, penalized_specs
):
    def materialize_component(spec):
        mats = []
        for dec, choice in zip(marginal_new_range_null, spec["combo"]):
            mats.append(
                dec["B_range"]
                if choice["kind"] == "range"
                else dec["B_null"][:, choice["cols"]]
            )
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
