# gsm/smooths/tensor/ops.py
import itertools
import numpy as np

from ...design.penalties import penalty_eigendecomposition


def rowwise_kronecker(matrices):
    """
    Row-wise Kronecker product of a list of model matrices.

    If matrices[j] has shape (n, k_j), the result has shape
        (n, prod_j k_j)
    and row i is the Kronecker product of the i-th rows of the marginals.
    """
    mats = [np.asarray(M, dtype=np.float64) for M in matrices]
    if len(mats) == 0:
        raise ValueError("matrices must contain at least one matrix.")

    n = mats[0].shape[0]
    for M in mats:
        if M.ndim != 2:
            raise ValueError("Each marginal model matrix must be 2D.")
        if M.shape[0] != n:
            raise ValueError("All marginal model matrices must have the same number of rows.")

    out = mats[0]
    for M in mats[1:]:
        out = np.einsum("ij,ik->ijk", out, M, optimize=True).reshape(
            n, out.shape[1] * M.shape[1]
        )
    return out


def lifted_tensor_penalty(S, basis_dims, axis):
    """
    Lift one marginal penalty into the full tensor-product coefficient space.

    For basis dimensions [k1, k2, ..., km], the lifted penalty for axis j is

        kron(I_left, kron(S_j, I_right))

    with the coefficient ordering matched to rowwise_kronecker().
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Marginal penalty must be a square 2D matrix.")

    basis_dims = [int(d) for d in basis_dims]
    if axis < 0 or axis >= len(basis_dims):
        raise IndexError(f"axis={axis} out of range for {len(basis_dims)} marginal bases.")

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
    """
    Build the full tensor-product penalty list: one lifted penalty per marginal basis.
    """
    penalties = []
    for j, S in enumerate(marginal_penalties):
        penalties.append(lifted_tensor_penalty(S, basis_dims=basis_dims, axis=j))
    return penalties


def marginal_range_null_decomposition(raw_basis, raw_penalty, tol=1e-10):
    """
    Reparameterize one marginal basis into range-space and null-space components.

    If S = U diag(d) U^T with positive eigenvalues d_pos and null-space eigenvectors U0,
    then we use the coefficient reparameterization

        beta_raw = U1 diag(d_pos^{-1/2}) gamma_r + U0 gamma_n

    which yields:
        penalty(gamma) = ||gamma_r||^2

    The corresponding model matrices are:
        B_r = X_raw U1 diag(d_pos^{-1/2})
        B_n = X_raw U0
    """
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)

    dec = penalty_eigendecomposition(S, tol=tol)
    U0 = dec["U0"]
    U1 = dec["U1"]
    d_pos = dec["d_pos"]

    if d_pos.size > 0:
        T_r = U1 / np.sqrt(d_pos)[np.newaxis, :]
        B_r = X @ T_r
    else:
        T_r = np.empty((S.shape[0], 0), dtype=np.float64)
        B_r = np.empty((X.shape[0], 0), dtype=np.float64)

    T_n = U0
    if T_n.shape[1] > 0:
        B_n = X @ T_n
    else:
        B_n = np.empty((X.shape[0], 0), dtype=np.float64)

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


def _normalize_ord(ord_value, n_marginals):
    if ord_value is None:
        return None
    if np.isscalar(ord_value):
        vals = [int(ord_value)]
    else:
        vals = [int(v) for v in ord_value]
    vals = sorted(set(vals))
    for v in vals:
        if v < 0 or v > n_marginals:
            raise ValueError(
                f"ord entries must lie between 0 and {n_marginals}, got {vals}."
            )
    return vals


def _mean_constraint_matrix(B):
    """
    Remove the constant-direction overlap from an unpenalized block by imposing
    a single sum-to-zero side condition on that block only.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")
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
    """
    Construct a t2-style tensor-product basis and its non-overlapping block penalties.

    Parameters
    ----------
    marginal_decompositions : list[dict]
        Output of marginal_range_null_decomposition for each margin.
    full : bool
        If False, each marginal null space is treated as one grouped block.
        If True, each marginal null-space column is treated as a separate null space.
    ord : None, int, or iterable[int]
        Retain only components with this many marginal range spaces.
    remove_constant_from_null_block : bool
        Framework-specific identifiability safeguard: remove one constant direction
        from the combined all-null unpenalized block.

    Returns
    -------
    dict with:
        basis
        penalties
        component_slices
        allnull_specs
        allnull_transform
        component_specs
        penalized_specs
    """
    m = len(marginal_decompositions)
    ord_keep = _normalize_ord(ord, m)

    option_lists = []
    for dec in marginal_decompositions:
        opts = []

        q = dec["null_dim"]
        r = dec["range_dim"]

        if q > 0:
            if full:
                for j in range(q):
                    opts.append(
                        {
                            "kind": "null",
                            "cols": [j],
                            "label": f"n{j+1}",
                        }
                    )
            else:
                opts.append(
                    {
                        "kind": "null",
                        "cols": list(range(q)),
                        "label": "n",
                    }
                )

        if r > 0:
            opts.append(
                {
                    "kind": "range",
                    "cols": None,
                    "label": "r",
                }
            )

        if len(opts) == 0:
            raise ValueError("Each marginal must contribute either null-space or range-space columns.")

        option_lists.append(opts)

    allnull_specs = []
    penalized_specs = []

    for combo in itertools.product(*option_lists):
        order = sum(1 for c in combo if c["kind"] == "range")
        if ord_keep is not None and order not in ord_keep:
            continue

        spec = {
            "combo": combo,
            "order": order,
            "label": "".join(c["label"] for c in combo),
        }
        if order == 0:
            allnull_specs.append(spec)
        else:
            penalized_specs.append(spec)

    def materialize_component(spec):
        mats = []
        for dec, choice in zip(marginal_decompositions, spec["combo"]):
            if choice["kind"] == "range":
                mats.append(dec["B_range"])
            else:
                mats.append(dec["B_null"][:, choice["cols"]])
        return rowwise_kronecker(mats)

    # Build the combined unpenalized all-null block first.
    allnull_transform = None
    basis_blocks = []
    block_meta = []

    if len(allnull_specs) > 0:
        B0 = np.column_stack([materialize_component(spec) for spec in allnull_specs])

        if remove_constant_from_null_block and B0.shape[1] > 0:
            C0 = _mean_constraint_matrix(B0)
            B0 = B0 @ C0
            allnull_transform = C0

        if B0.shape[1] > 0:
            basis_blocks.append(B0)
            block_meta.append(
                {
                    "order": 0,
                    "label": "null",
                    "penalized": False,
                    "n_cols": B0.shape[1],
                }
            )

    # Then build penalized ANOVA components.
    for spec in penalized_specs:
        B = materialize_component(spec)
        if B.shape[1] == 0:
            continue
        basis_blocks.append(B)
        block_meta.append(
            {
                "order": spec["order"],
                "label": spec["label"],
                "penalized": True,
                "n_cols": B.shape[1],
            }
        )

    if len(basis_blocks) == 0:
        basis = np.empty((marginal_decompositions[0]["B_null"].shape[0], 0), dtype=np.float64)
        penalties = []
        component_slices = []
        return {
            "basis": basis,
            "penalties": penalties,
            "component_slices": component_slices,
            "allnull_specs": allnull_specs,
            "allnull_transform": allnull_transform,
            "component_specs": block_meta,
            "penalized_specs": [],
        }

    basis = np.column_stack(basis_blocks)

    penalties = []
    component_slices = []
    start = 0
    total_dim = basis.shape[1]

    for meta in block_meta:
        sl = slice(start, start + meta["n_cols"])
        component_slices.append(sl)
        if meta["penalized"]:
            P = np.zeros((total_dim, total_dim), dtype=np.float64)
            P[sl, sl] = np.eye(meta["n_cols"], dtype=np.float64)
            penalties.append(P)
        start += meta["n_cols"]

    return {
        "basis": basis,
        "penalties": penalties,
        "component_slices": component_slices,
        "allnull_specs": allnull_specs,
        "allnull_transform": allnull_transform,
        "component_specs": block_meta,
        "penalized_specs": penalized_specs,
    }


def materialize_t2_newdata(
    marginal_new_range_null,
    *,
    allnull_specs,
    allnull_transform,
    penalized_specs,
):
    """
    Build the new-data model matrix for a stored t2 decomposition.
    """
    def materialize_component(spec):
        mats = []
        for dec, choice in zip(marginal_new_range_null, spec["combo"]):
            if choice["kind"] == "range":
                mats.append(dec["B_range"])
            else:
                mats.append(dec["B_null"][:, choice["cols"]])
        return rowwise_kronecker(mats)

    blocks = []

    if len(allnull_specs) > 0:
        B0 = np.column_stack([materialize_component(spec) for spec in allnull_specs])
        if allnull_transform is not None and B0.shape[1] > 0:
            B0 = B0 @ allnull_transform
        if B0.shape[1] > 0:
            blocks.append(B0)

    for spec in penalized_specs:
        B = materialize_component(spec)
        if B.shape[1] > 0:
            blocks.append(B)

    if len(blocks) == 0:
        n = next(iter(marginal_new_range_null))["B_null"].shape[0]
        return np.empty((n, 0), dtype=np.float64)

    return np.column_stack(blocks)
