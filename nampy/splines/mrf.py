# splines/mrf.py
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular


def combine_duplicate_polys(polys):
    """
    mgcv-like duplicate-name handling:
    duplicate named polygon entries are combined with NA row separators.

    Accepted input formats
    ----------------------
    - dict[name] -> 2D array
    - list/tuple of (name, 2D array) pairs
    """
    if polys is None:
        return None

    if isinstance(polys, dict):
        items = list(polys.items())
    elif isinstance(polys, (list, tuple)):
        items = list(polys)
    else:
        raise TypeError(
            "polys must be a dict or a list/tuple of (name, polygon_matrix) pairs."
        )

    out = OrderedDict()
    for item in items:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError("Polygon entries must be (name, polygon_matrix) pairs.")
        name, mat = item
        name = str(name)
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != 2:
            raise ValueError(
                f"Polygon for area {name!r} must be a 2-column matrix, got shape {mat.shape}."
            )
        if name not in out:
            out[name] = mat.copy()
        else:
            out[name] = np.vstack([out[name], np.array([[np.nan, np.nan]]), mat])

    return out


def strip_na_rows_and_unique_vertices(poly):
    poly = np.asarray(poly, dtype=np.float64)
    poly = poly[~np.isnan(np.sum(poly, axis=1))]
    if poly.shape[0] == 0:
        return poly

    seen = []
    rows = []
    for row in poly:
        key = (float(row[0]), float(row[1]))
        if key not in seen:
            seen.append(key)
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def polys_to_nb(polys):
    """
    mgcv-style neighbourhood extraction from shared vertices.
    """
    polys = combine_duplicate_polys(polys)
    names = list(polys.keys())
    pcs = [strip_na_rows_and_unique_vertices(polys[nm]) for nm in names]
    n_poly = len(pcs)

    lo1 = np.zeros(n_poly, dtype=np.float64)
    hi1 = np.zeros(n_poly, dtype=np.float64)
    lo2 = np.zeros(n_poly, dtype=np.float64)
    hi2 = np.zeros(n_poly, dtype=np.float64)

    for i, pc in enumerate(pcs):
        lo1[i] = np.min(pc[:, 0])
        hi1[i] = np.max(pc[:, 0])
        lo2[i] = np.min(pc[:, 1])
        hi2[i] = np.max(pc[:, 1])

    nb = OrderedDict((nm, []) for nm in names)

    for k in range(n_poly):
        ol1 = (
            ((lo1[k] <= hi1) & (lo1[k] >= lo1))
            | ((hi1[k] <= hi1) & (hi1[k] >= lo1))
            | ((lo1 <= hi1[k]) & (lo1 >= lo1[k]))
            | ((hi1 <= hi1[k]) & (hi1 >= lo1[k]))
        )
        ol2 = (
            ((lo2[k] <= hi2) & (lo2[k] >= lo2))
            | ((hi2[k] <= hi2) & (hi2[k] >= lo2))
            | ((lo2 <= hi2[k]) & (lo2 >= lo2[k]))
            | ((hi2 <= hi2[k]) & (hi2 >= lo2[k]))
        )
        ol = ol1 & ol2
        ol[k] = False
        cand = np.where(ol)[0]

        cok = pcs[k]
        for j in cand:
            co = np.vstack([pcs[j], cok])
            cou = np.unique(co, axis=0)
            n_shared = co.shape[0] - cou.shape[0]
            if n_shared > 0:
                nb[names[k]].append(names[j])

        nb[names[k]] = sorted(set(nb[names[k]]))

    return nb


def coerce_nb(nb, area_names):
    """
    Normalize a neighbourhood structure to OrderedDict[name] -> list[name].

    Accepted formats
    ----------------
    - dict[name] -> list of neighbor names
    - dict[name] -> list of 1-based or 0-based neighbor indices
    """
    if not isinstance(nb, dict):
        raise TypeError("nb must be a dict mapping area names to neighbors.")

    area_names = list(area_names)
    nb_names = list(nb.keys())

    if set(nb_names) != set(area_names):
        raise ValueError("Mismatch between nb/polys supplied area names and data area names.")

    ordered = OrderedDict()
    for nm in area_names:
        neigh = list(nb[nm])
        if len(neigh) == 0:
            ordered[nm] = []
            continue

        if all(isinstance(v, str) for v in neigh):
            ordered[nm] = list(neigh)
            continue

        if all(isinstance(v, (int, np.integer)) for v in neigh):
            vals = np.asarray(neigh, dtype=int)
            if np.min(vals) >= 1 and np.max(vals) <= len(area_names):
                vals = vals - 1
            if np.min(vals) < 0 or np.max(vals) >= len(area_names):
                raise ValueError("Numeric nb indices are out of bounds.")
            ordered[nm] = [area_names[int(i)] for i in vals.tolist()]
            continue

        raise TypeError(
            "Each nb[name] must be a list of neighbor names or numeric indices."
        )

    return ordered


def laplacian_penalty_from_nb(nb, area_names):
    """
    Default mgcv MRF penalty from a neighbourhood list.
    """
    area_names = list(area_names)
    nb = coerce_nb(nb, area_names)

    p = len(area_names)
    idx = {nm: i for i, nm in enumerate(area_names)}
    S = np.zeros((p, p), dtype=np.float64)

    for nm in area_names:
        i = idx[nm]
        neigh = nb[nm]
        S[i, i] = len(neigh)
        for nn in neigh:
            j = idx[nn]
            if j != i:
                S[i, j] = -1.0

    if not np.allclose(S, S.T, atol=1e-12, rtol=0.0):
        raise ValueError("Auto-constructed MRF penalty is not symmetric.")

    return S, nb


def coerce_penalty_matrix(penalty, area_names):
    """
    Accept either a pandas DataFrame with row/col names or a plain ndarray.

    If a DataFrame is supplied, reorder it to match area_names.
    If a plain ndarray is supplied, it is assumed already ordered as area_names.
    """
    area_names = list(area_names)

    if isinstance(penalty, pd.DataFrame):
        if penalty.shape[0] != penalty.shape[1]:
            raise ValueError("Supplied penalty is not square.")
        cols = list(penalty.columns)
        rows = list(penalty.index)
        if set(cols) != set(area_names) or set(rows) != set(area_names):
            raise ValueError("Penalty row/column names do not match supplied area names.")
        penalty = penalty.loc[area_names, area_names].to_numpy(dtype=np.float64)
    else:
        penalty = np.asarray(penalty, dtype=np.float64)
        if penalty.ndim != 2 or penalty.shape[0] != penalty.shape[1]:
            raise ValueError("Supplied penalty is not square.")
        if penalty.shape[0] != len(area_names):
            raise ValueError("Supplied penalty has the wrong dimension.")

    return 0.5 * (penalty + penalty.T)


def nat_param_type0(X, S, rank=None, tol=None, unit_fnorm=True):
    """
    Python implementation of mgcv::nat.param(X, S, type=0).

    Returns
    -------
    dict with:
        X : transformed model matrix
        D : positive diagonal penalty entries
        P : back-transform to original coefficient parameterization
        rank : penalty rank
    """
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tol = np.finfo(float).eps ** 0.8 if tol is None else float(tol)

    Q, R = scipy_qr(X, mode="economic", pivoting=False)
    if np.linalg.matrix_rank(R) < R.shape[1]:
        raise ValueError("MRF model matrix is not full rank in natural-parameter construction.")

    I = np.eye(R.shape[0], dtype=np.float64)
    invR = solve_triangular(R, I, lower=False, check_finite=False)
    RSR = invR.T @ S @ invR
    RSR = 0.5 * (RSR + RSR.T)

    evals, U = np.linalg.eigh(RSR)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = U[:, idx]

    if rank is None or rank < 1 or rank > S.shape[0]:
        rank = int(np.sum(evals > np.max(evals) * tol))

    D = evals[:rank].copy()
    Xn = Q @ U
    P = invR @ U

    if unit_fnorm:
        if rank > 0:
            ind = np.arange(rank)
            scale = 1.0 / np.sqrt(np.mean(Xn[:, ind] ** 2))
            Xn[:, ind] *= scale
            P[:, ind] *= scale
            D *= scale**2

        if rank < Xn.shape[1]:
            ind = np.arange(rank, Xn.shape[1])
            scalef = 1.0 / np.sqrt(np.mean(Xn[:, ind] ** 2))
            Xn[:, ind] *= scalef
            P[:, ind] *= scalef

    return {
        "X": Xn,
        "D": D,
        "P": P,
        "rank": int(rank),
    }