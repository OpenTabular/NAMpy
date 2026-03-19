import numpy as np
from scipy.linalg import qr as scipy_qr


def reparameterize_smooth(B, P, tol=1e-10):
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


def _penalty_support_mask(P, tol=1e-12):
    P = np.asarray(P, dtype=np.float64)
    if P.size == 0:
        return np.zeros((0,), dtype=bool)
    row_nz = np.any(np.abs(P) > tol, axis=1)
    col_nz = np.any(np.abs(P) > tol, axis=0)
    return np.asarray(row_nz | col_nz, dtype=bool)


def _term_penalty_components(primary, null_space):
    if len(primary) == 0:
        return True, [], {}

    supports = []
    for pb in primary:
        mask = _penalty_support_mask(pb.matrix)
        supports.append((pb, mask))

    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            if np.any(supports[i][1] & supports[j][1]):
                return False, [], {}

    null_map = {}
    if len(null_space) > 0:
        pb0 = null_space[0]
        null_mask = _penalty_support_mask(pb0.matrix)
        hits = []
        for pb, mask in supports:
            if np.any(mask & null_mask):
                hits.append(pb)
        if len(hits) != 1:
            return False, [], {}
        null_map[id(hits[0])] = pb0

    components = []
    for pb, mask in supports:
        components.append(
            {
                "primary": pb,
                "support_mask": mask,
                "null": null_map.get(id(pb)),
            }
        )
    return True, components, null_map


def can_use_simple_ml_reml_structure(model):
    """
    Conservative structural gate for the current ML/REML paths.

    It is enabled only when every penalized smooth term contributes exactly one
    primary smooth penalty, plus at most one null-space penalty acting on the
    same term. This covers the mgcv-style shrinkage/select case while still
    excluding genuinely overlapping multi-penalty structures.
    """
    if model.design_ is None:
        return False

    for tb in model.term_blocks_:
        matches = [pb for pb in model.penalty_blocks_ if pb.coef_slice == tb.coef_slice]
        primary_ids = {id(pb) for pb in matches if pb.kind == "smooth" and not pb.is_null_space_penalty}
        null_ids = {id(pb) for pb in matches if pb.is_null_space_penalty}

        if len(matches) == 0:
            continue

        primary = [
            pb for pb in matches
            if pb.kind == "smooth" and not pb.is_null_space_penalty
        ]
        null_space = [pb for pb in matches if pb.is_null_space_penalty]
        extras = [
            pb for pb in matches
            if id(pb) not in primary_ids and id(pb) not in null_ids
        ]

        if len(primary) < 1:
            return False
        if len(null_space) > 1:
            return False
        if len(extras) > 0:
            return False
        ok, _, _ = _term_penalty_components(primary, null_space)
        if not ok:
            return False

    return True


def can_use_exact_gaussian_ml_reml(model):
    if not model._uses_closed_form_solver():
        return False
    return can_use_simple_ml_reml_structure(model)


def build_penalty_reparameterized_system(model):
    if model.design_ is None:
        return

    fix_blocks = []
    if model.fit_intercept:
        fix_blocks.append(np.ones((model.n_samples_, 1), dtype=np.float64))

    rand_blocks = []
    model._reparam_meta = []
    model._reparam_rand_blocks_ = []
    model._reparam_sp_groups_ = {}
    model.rand_dims_per_term_ = []
    model._primary_reparam_sp_index_per_term_ = []
    model._penalty_ranks = np.empty(len(model.term_blocks_), dtype=np.int64)
    model._penalty_logdet_plus_fixed = np.empty(len(model.term_blocks_), dtype=np.float64)
    rand_start = 0

    for i, tb in enumerate(model.term_blocks_):
        matches = [pb for pb in model.penalty_blocks_ if pb.coef_slice == tb.coef_slice]
        B = tb.basis_train
        primary_ids = {id(pb) for pb in matches if pb.kind == "smooth" and not pb.is_null_space_penalty}
        null_ids = {id(pb) for pb in matches if pb.is_null_space_penalty}

        if len(matches) == 0:
            if B.shape[1] > 0:
                fix_blocks.append(B)

            model._reparam_meta.append(None)
            model.rand_dims_per_term_.append(0)
            model._primary_reparam_sp_index_per_term_.append(None)
            model._penalty_ranks[i] = 0
            model._penalty_logdet_plus_fixed[i] = 0.0
            continue

        primary = [
            pb for pb in matches
            if pb.kind == "smooth" and not pb.is_null_space_penalty
        ]
        null_space = [pb for pb in matches if pb.is_null_space_penalty]
        extras = [
            pb for pb in matches
            if id(pb) not in primary_ids and id(pb) not in null_ids
        ]

        if len(primary) < 1 or len(null_space) > 1 or len(extras) > 0:
            raise NotImplementedError(
                "Current ML/REML reparameterization is enabled only for terms with "
                "at least one primary smooth penalty and disjoint primary supports, "
                "plus at most one null-space penalty per support block. General "
                "overlapping multi-penalty terms are not yet implemented in this path."
            )

        ok, components, _ = _term_penalty_components(primary, null_space)
        if not ok:
            raise NotImplementedError(
                "Current ML/REML reparameterization requires disjoint support for "
                "multiple primary penalties on a term."
            )

        rand_blocks_term = []
        component_meta = []
        total_rank = 0
        total_logdet = 0.0
        primary_sp_indices = []

        covered_mask = np.zeros(B.shape[1], dtype=bool)
        for comp in components:
            pb = comp["primary"]
            pb0 = comp["null"]
            support_mask = np.asarray(comp["support_mask"], dtype=bool)
            if len(components) == 1:
                # Preserve previous mgcv-like singleton behavior: use full term block
                # and let eigen decomposition split null/penalized spaces.
                local_idx = np.arange(B.shape[1], dtype=np.int64)
            else:
                local_idx = np.flatnonzero(support_mask)
            covered_mask[local_idx] = True
            if local_idx.size == 0:
                continue

            B_local = B[:, local_idx]
            P_local = np.asarray(pb.matrix, dtype=np.float64)[np.ix_(local_idx, local_idx)]

            evals = np.linalg.eigvalsh(0.5 * (P_local + P_local.T))
            tol = 1e-10 * max(1.0, np.max(np.abs(evals)))
            pos = evals[evals > tol]

            B0_main, Zr_main, meta = reparameterize_smooth(B_local, P_local)
            extra_meta = None
            B0_use = B0_main
            comp_rank = int(meta["n_pen"])
            comp_logdet = float(np.sum(np.log(pos)) if len(pos) > 0 else 0.0)

            if Zr_main.shape[1] > 0:
                rand_blocks.append(Zr_main)
                rand_blocks_term.append(
                    {
                        "term_index": i,
                        "kind": "smooth",
                        "smoothing_index": int(pb.smoothing_index),
                        "slice": slice(rand_start, rand_start + Zr_main.shape[1]),
                        "n_pen": int(Zr_main.shape[1]),
                        "is_null_space_penalty": False,
                    }
                )
                rand_start += Zr_main.shape[1]

            if pb0 is not None and meta["n_null"] > 0:
                U0 = meta["U0"]
                P0_local = np.asarray(pb0.matrix, dtype=np.float64)[np.ix_(local_idx, local_idx)]
                B_null = B_local @ U0
                P0_null = U0.T @ P0_local @ U0
                B0_extra, Zr_extra, extra_meta = reparameterize_smooth(B_null, P0_null)
                B0_use = B0_extra
                comp_rank += int(extra_meta["n_pen"])
                comp_logdet += float(
                    np.sum(np.log(extra_meta["d_pos"])) if extra_meta["d_pos"].size > 0 else 0.0
                )
                if Zr_extra.shape[1] > 0:
                    rand_blocks.append(Zr_extra)
                    rand_blocks_term.append(
                        {
                            "term_index": i,
                            "kind": str(pb0.kind),
                            "smoothing_index": int(pb0.smoothing_index),
                            "slice": slice(rand_start, rand_start + Zr_extra.shape[1]),
                            "n_pen": int(Zr_extra.shape[1]),
                            "is_null_space_penalty": True,
                        }
                    )
                    rand_start += Zr_extra.shape[1]

            if B0_use.shape[1] > 0:
                fix_blocks.append(B0_use)

            component_meta.append(
                {
                    "primary": meta,
                    "null_space": extra_meta,
                    "support_index": local_idx,
                    "primary_smoothing_index": int(pb.smoothing_index),
                    "null_smoothing_index": None if pb0 is None else int(pb0.smoothing_index),
                }
            )
            primary_sp_indices.append(int(pb.smoothing_index))
            total_rank += comp_rank
            total_logdet += comp_logdet

        # For multi-primary disjoint decomposition, keep any residual columns
        # (outside all penalty supports) as fixed effects.
        if len(components) > 1:
            residual_idx = np.flatnonzero(~covered_mask)
            if residual_idx.size > 0:
                B_resid = B[:, residual_idx]
                if B_resid.shape[1] > 0:
                    fix_blocks.append(B_resid)

        for block in rand_blocks_term:
            model._reparam_rand_blocks_.append(block)
            model._reparam_sp_groups_.setdefault(block["smoothing_index"], []).extend(
                range(block["slice"].start, block["slice"].stop)
            )

        model._reparam_meta.append(
            {
                "primary": component_meta[0]["primary"] if component_meta else None,
                "null_space": component_meta[0]["null_space"] if component_meta else None,
                "components": component_meta,
                "rand_blocks": rand_blocks_term,
                "n_pen": int(sum(block["n_pen"] for block in rand_blocks_term)),
            }
        )
        model.rand_dims_per_term_.append(int(sum(block["n_pen"] for block in rand_blocks_term)))
        model._primary_reparam_sp_index_per_term_.append(
            primary_sp_indices[0] if primary_sp_indices else None
        )
        model._penalty_ranks[i] = total_rank
        model._penalty_logdet_plus_fixed[i] = total_logdet
    if fix_blocks:
        X_fix_raw = np.column_stack(fix_blocks)
        _Q, R, piv = scipy_qr(X_fix_raw, pivoting=True)

        if R.size == 0:
            rank = 0
            keep_cols = np.array([], dtype=int)
        else:
            diag_R = np.abs(np.diag(R[:min(X_fix_raw.shape), :]))
            if diag_R.size == 0:
                rank = 0
                keep_cols = np.array([], dtype=int)
            else:
                rank_tol = (
                    max(X_fix_raw.shape) * np.finfo(float).eps * diag_R[0]
                    if diag_R[0] > 0 else 1e-12
                )
                rank = int(np.sum(diag_R > rank_tol))
                keep_cols = np.sort(piv[:rank])

        model.X_fix_ = (
            X_fix_raw[:, keep_cols]
            if rank > 0
            else np.empty((model.n_samples_, 0), dtype=np.float64)
        )
        model.rank_X_fix_ = rank
        model._fix_pivot_keep = keep_cols
    else:
        model.X_fix_ = np.empty((model.n_samples_, 0), dtype=np.float64)
        model.rank_X_fix_ = 0
        model._fix_pivot_keep = np.array([], dtype=int)

    if rand_blocks:
        model.Z_rand_ = np.column_stack(rand_blocks)
    else:
        model.Z_rand_ = np.empty((model.n_samples_, 0), dtype=np.float64)

    model._reparam_sp_groups_ = {
        int(sp_idx): np.asarray(idxs, dtype=np.int64)
        for sp_idx, idxs in model._reparam_sp_groups_.items()
    }
    model.n_rand_ = model.Z_rand_.shape[1]
    model.ZtZ_rand_ = model.Z_rand_.T @ model.Z_rand_


def build_gaussian_reparameterized_system(model):
    return build_penalty_reparameterized_system(model)
