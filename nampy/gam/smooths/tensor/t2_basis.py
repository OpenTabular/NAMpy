import numpy as np

from ..algebra import rowwise_kronecker


def _normalize_ord(ord_value, n_marginals):
    if ord_value is None:
        return None
    vals = [int(ord_value)] if np.isscalar(ord_value) else [int(v) for v in ord_value]
    _ = n_marginals
    return list(vals)


def _mean_constraint_matrix(B):
    B = np.asarray(B, dtype=np.float64)
    if B.shape[1] == 0:
        return np.eye(0, dtype=np.float64)
    c = B.mean(axis=0).reshape(-1, 1)
    q, _ = np.linalg.qr(c, mode="complete")
    return q[:, 1:]


def _rowwise_product(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.empty((A.shape[0], 0), dtype=np.float64)
    return rowwise_kronecker([A, B])


def _null_colnames(n_cols):
    return [str(i + 1) for i in range(int(n_cols))]


def _append_block(
    blocks,
    block_matrix,
    desc,
    order_value,
    label,
    penalized,
    col_names=None,
    col_descs=None,
):
    blocks.append(
        {
            "matrix": np.asarray(block_matrix, dtype=np.float64),
            "desc": list(desc),
            "order": int(order_value),
            "label": str(label),
            "penalized": bool(penalized),
            "col_names": None if col_names is None else list(col_names),
            "col_descs": (
                None
                if col_descs is None
                else [list(col_desc) for col_desc in col_descs]
            ),
        }
    )


def build_t2_basis_and_penalties(
    marginal_decompositions,
    *,
    full=False,
    ord=None,
    remove_constant_from_null_block=True,
):
    m = len(marginal_decompositions)
    if m == 0:
        raise ValueError("marginal_decompositions must contain at least one margin.")

    ord_keep = _normalize_ord(ord, m)

    # mgcv::t2.model.matrix is translated directly with its exact block-order
    # bookkeeping (orders, labels, and null-block handling).
    blocks = []
    no_null = True

    first = marginal_decompositions[0]
    Z1 = np.asarray(first["B_range"], dtype=np.float64)
    if Z1.shape[1] > 0:
        _append_block(
            blocks=blocks,
            block_matrix=Z1,
            desc=[{"kind": "range", "cols": None}],
            order_value=1,
            label="r",
            penalized=True,
            col_names=None,
        )

    if first["null_dim"] > 0:
        X1 = np.asarray(first["B_null"], dtype=np.float64)
        if X1.shape[1] > 0:
            _append_block(
                blocks=blocks,
                block_matrix=X1,
                desc=[{"kind": "null", "cols": list(range(X1.shape[1]))}],
                order_value=0,
                label="n",
                penalized=False if full else True,
                col_names=_null_colnames(X1.shape[1]) if full else None,
                col_descs=(
                    [[{"kind": "null", "cols": [j]}] for j in range(X1.shape[1])]
                    if full
                    else None
                ),
            )
        no_null = False

    for margin_idx in range(1, m):
        dec = marginal_decompositions[margin_idx]
        Zi = np.asarray(dec["B_range"], dtype=np.float64)
        null_exists = int(dec["null_dim"]) > 0
        Xi = np.asarray(dec["B_null"], dtype=np.float64) if null_exists else None

        prior_blocks = list(blocks)
        blocks = []
        for prior in prior_blocks:
            # Range-space expansion
            if Zi.shape[1] > 0:
                if (not full) or prior["penalized"]:
                    if prior["matrix"].shape[1] > 0:
                        A = _rowwise_product(prior["matrix"], Zi)
                        if A.shape[1] > 0:
                            _append_block(
                                blocks=blocks,
                                block_matrix=A,
                                desc=prior["desc"] + [{"kind": "range", "cols": None}],
                                order_value=prior["order"] + 1,
                                label=f"{prior['label']}r",
                                penalized=True,
                                col_names=None,
                            )
                else:
                    prior_names = (
                        prior["col_names"]
                        or [prior["label"]] * prior["matrix"].shape[1]
                    )
                    prior_col_descs = prior.get("col_descs")
                    for j, col_name in enumerate(prior_names):
                        A = _rowwise_product(prior["matrix"][:, [j]], Zi)
                        if A.shape[1] > 0:
                            split_label = f"{col_name}r"
                            desc_j = (
                                list(prior_col_descs[j])
                                if prior_col_descs is not None
                                else list(prior["desc"])
                            )
                            _append_block(
                                blocks=blocks,
                                block_matrix=A,
                                desc=desc_j + [{"kind": "range", "cols": None}],
                                order_value=prior["order"] + 1,
                                label=split_label,
                                penalized=True,
                                col_names=[split_label] * A.shape[1] if full else None,
                            )

        # Null-space expansion
        if null_exists and Xi is not None and Xi.shape[1] > 0:
            xnames = _null_colnames(Xi.shape[1]) if full else None
            for prior in prior_blocks:
                if (not full) or (not prior["penalized"]):
                    if prior["matrix"].shape[1] > 0:
                        A = _rowwise_product(prior["matrix"], Xi)
                        if A.shape[1] > 0:
                            prior_names = (
                                prior["col_names"]
                                or [prior["label"]] * prior["matrix"].shape[1]
                            )
                            prior_col_descs = prior.get("col_descs")
                            if full:
                                col_names = [
                                    f"{cn1}{cn2}"
                                    for cn1 in prior_names
                                    for cn2 in xnames
                                ]
                                col_descs = None
                                if prior_col_descs is not None:
                                    col_descs = [
                                        list(prior_desc)
                                        + [{"kind": "null", "cols": [j]}]
                                        for prior_desc in prior_col_descs
                                        for j in range(Xi.shape[1])
                                    ]
                            else:
                                col_names = None
                                col_descs = None
                            _append_block(
                                blocks=blocks,
                                block_matrix=A,
                                desc=prior["desc"]
                                + [{"kind": "null", "cols": list(range(Xi.shape[1]))}],
                                order_value=prior["order"],
                                label=f"{prior['label']}n",
                                penalized=bool(False if full else True),
                                col_names=col_names,
                                col_descs=col_descs,
                            )
                else:
                    # full=TRUE only: split Xi by columns for a penalized prior block.
                    prior_names = (
                        prior["col_names"]
                        or [prior["label"]] * prior["matrix"].shape[1]
                    )
                    for j, cnxi in enumerate(xnames):
                        A = prior["matrix"] * Xi[:, [j]]
                        if A.shape[1] > 0:
                            _append_block(
                                blocks=blocks,
                                block_matrix=A,
                                desc=prior["desc"] + [{"kind": "null", "cols": [j]}],
                                order_value=prior["order"],
                                label=f"{prior['label']}{cnxi}",
                                penalized=True,
                                col_names=None,
                            )
        else:
            no_null = True

    if ord_keep is not None:
        keep_mask = [int(block["order"]) in ord_keep for block in blocks]
        blocks = [block for block, keep in zip(blocks, keep_mask) if keep]
        if 0 not in ord_keep:
            no_null = True

    if len(blocks) == 0:
        n = int(marginal_decompositions[0]["B_range"].shape[0])
        return {
            "basis": np.empty((n, 0), dtype=np.float64),
            "penalties": [],
            "component_slices": [],
            "basis_pre_constraint": np.empty((n, 0), dtype=np.float64),
            "penalties_pre_constraint": [],
            "allnull_specs": [],
            "allnull_transform": None,
            "full_constraint_transform": None,
            "component_specs": [],
            "penalized_specs": [],
        }

    basis_pre = np.column_stack([block["matrix"] for block in blocks])

    allnull_block = None if no_null else blocks[-1]

    penalties_pre = []
    component_slices_pre = []
    component_specs = []
    penalized_specs = []
    total_dim_pre = basis_pre.shape[1]
    start = 0
    for idx, block in enumerate(blocks):
        # mgcv t2: if no all-null block survives (`no_null=True`), every kept
        # block is penalized. Otherwise only the final all-null block is
        # unpenalized and earlier blocks remain penalized.
        is_penalized = True if no_null else (idx != len(blocks) - 1)
        block_n_cols = int(block["matrix"].shape[1])
        block_meta = {
            "order": int(block["order"]),
            "label": str(block["label"]),
            "penalized": bool(is_penalized),
            "n_cols": block_n_cols,
            "desc": tuple(block["desc"]),
            "combo": tuple(block["desc"]),
        }
        component_specs.append(block_meta)

        if is_penalized:
            sl = slice(start, start + block_n_cols)
            component_slices_pre.append(sl)
            P = np.zeros((total_dim_pre, total_dim_pre), dtype=np.float64)
            P[sl, sl] = np.eye(block_n_cols, dtype=np.float64)
            penalties_pre.append(P)
            penalized_specs.append(
                {
                    "combo": tuple(block["desc"]),
                    "order": int(block["order"]),
                    "label": str(block["label"]),
                    "col_descs": (
                        None
                        if block.get("col_descs") is None
                        else tuple(tuple(desc) for desc in block["col_descs"])
                    ),
                }
            )
            start += block_n_cols

    component_slices = component_slices_pre
    penalties = list(penalties_pre)

    allnull_transform = None
    full_constraint_transform = None
    basis = basis_pre

    if (
        allnull_block is not None
        and allnull_block["matrix"].shape[1] > 0
        and remove_constant_from_null_block
    ):
        C0 = _mean_constraint_matrix(allnull_block["matrix"])
        allnull_transform = C0
        n_pen = basis_pre.shape[1] - allnull_block["matrix"].shape[1]
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
        "allnull_specs": (
            [
                {
                    "combo": tuple(allnull_block["desc"]),
                    "order": 0,
                    "label": "null",
                    "col_descs": (
                        None
                        if allnull_block.get("col_descs") is None
                        else tuple(tuple(desc) for desc in allnull_block["col_descs"])
                    ),
                }
            ]
            if allnull_block is not None
            else []
        ),
        "allnull_transform": allnull_transform,
        "full_constraint_transform": full_constraint_transform,
        "component_specs": component_specs,
        "penalized_specs": penalized_specs,
    }


def materialize_t2_newdata(
    marginal_new_range_null, *, allnull_specs, allnull_transform, penalized_specs
):
    def materialize_single_column(desc):
        mats = []
        for dec, choice in zip(marginal_new_range_null, desc):
            if choice["kind"] == "range":
                cols = choice.get("cols")
                if cols is None:
                    raise ValueError(
                        "Column-specific t2 reconstruction requires explicit range columns."
                    )
                mats.append(np.asarray(dec["B_range"], dtype=np.float64)[:, cols])
            else:
                cols = choice.get("cols")
                if cols is None:
                    raise ValueError(
                        "Column-specific t2 reconstruction requires explicit null columns."
                    )
                mats.append(np.asarray(dec["B_null"], dtype=np.float64)[:, cols])
        return rowwise_kronecker(mats)

    def materialize_component(spec):
        col_descs = spec.get("col_descs")
        if col_descs is not None:
            cols = [materialize_single_column(desc) for desc in col_descs]
            return (
                np.column_stack(cols)
                if cols
                else np.empty(
                    (marginal_new_range_null[0]["B_range"].shape[0], 0),
                    dtype=np.float64,
                )
            )
        mats = []
        for dec, choice in zip(marginal_new_range_null, spec["combo"]):
            if choice["kind"] == "range":
                mats.append(np.asarray(dec["B_range"], dtype=np.float64))
            else:
                cols = choice.get("cols")
                if cols is None:
                    mats.append(np.asarray(dec["B_null"], dtype=np.float64))
                else:
                    cols = np.asarray(cols, dtype=np.int64)
                    if cols.size == 0:
                        return np.empty(
                            (marginal_new_range_null[0]["B_range"].shape[0], 0),
                            dtype=np.float64,
                        )
                    mats.append(np.asarray(dec["B_null"], dtype=np.float64)[:, cols])
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
