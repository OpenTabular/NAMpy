import numpy as np

from .algebra import rowwise_kronecker


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


__all__ = ["build_t2_basis_and_penalties", "materialize_t2_newdata"]
