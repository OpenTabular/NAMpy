"""
Dense matrix reindexing helpers for penalized QR workflows.
"""

from __future__ import annotations

import numpy as np


def drop_columns_dense(a: np.ndarray, drop_sorted: np.ndarray) -> np.ndarray:
    drop_sorted = np.asarray(drop_sorted, dtype=int)
    if drop_sorted.size == 0:
        return np.asarray(a, dtype=np.float64, order="C").copy()
    mask = np.ones(a.shape[1], dtype=bool)
    mask[drop_sorted] = False
    return np.asarray(a, dtype=np.float64)[:, mask].copy()


def drop_rows_dense(a: np.ndarray, drop_sorted: np.ndarray) -> np.ndarray:
    drop_sorted = np.asarray(drop_sorted, dtype=int)
    if drop_sorted.size == 0:
        return np.asarray(a, dtype=np.float64, order="C").copy()
    mask = np.ones(a.shape[0], dtype=bool)
    mask[drop_sorted] = False
    return np.asarray(a, dtype=np.float64)[mask, :].copy()


def restore_dropped_rows(
    a: np.ndarray,
    n_rows_full: int,
    drop_sorted: np.ndarray,
) -> np.ndarray:
    drop_sorted = np.asarray(drop_sorted, dtype=int)
    if drop_sorted.size == 0:
        out = np.zeros((n_rows_full, a.shape[1]), dtype=np.float64)
        out[: a.shape[0], :] = np.asarray(a, dtype=np.float64)
        return out
    n_drop = int(drop_sorted.size)
    ncol = int(a.shape[1])
    if a.shape[0] != n_rows_full - n_drop:
        raise ValueError(
            "restore_dropped_rows: packed row count does not match drop count."
        )
    out = np.zeros((n_rows_full, ncol), dtype=np.float64)
    keep = np.setdiff1d(
        np.arange(n_rows_full, dtype=int), drop_sorted, assume_unique=True
    )
    out[keep, :] = np.asarray(a, dtype=np.float64)
    return out


def permute_columns(
    x: np.ndarray,
    permutation: np.ndarray,
    *,
    reverse: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64, order="C")
    permutation = np.asarray(permutation, dtype=int).ravel()
    if permutation.size != x.shape[1]:
        raise ValueError("permutation length must match number of columns.")
    if reverse:
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(permutation.size, dtype=int)
        return x[:, inv].copy()
    return x[:, permutation].copy()


def permute_rows(
    x: np.ndarray,
    permutation: np.ndarray,
    *,
    reverse: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64, order="C")
    permutation = np.asarray(permutation, dtype=int).ravel()
    if permutation.size != x.shape[0]:
        raise ValueError("permutation length must match number of rows.")
    if reverse:
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(permutation.size, dtype=int)
        return x[inv, :].copy()
    return x[permutation, :].copy()
