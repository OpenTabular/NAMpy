"""Shared ``mgcv`` Sl initial-reparameterization transforms."""

from __future__ import annotations

from typing import Any

import numpy as np


def sl_initial_repara(
    Sl: Any,
    X: np.ndarray,
    *,
    inverse: bool = False,
    both_sides: bool = True,
    cov: bool = True,
) -> np.ndarray:
    """Mirror ``mgcv::Sl.initial.repara`` for implemented linear Sl blocks."""
    X_arr = np.asarray(X, dtype=np.float64).copy()
    if len(Sl) == 0:
        return X_arr

    is_matrix = X_arr.ndim == 2
    for block in Sl:
        if not block.repara:
            continue
        ind = np.arange(block.start0, block.stop0, dtype=int)
        D = np.asarray(block.D, dtype=np.float64)

        if inverse:
            if is_matrix:
                if cov:
                    if D.ndim == 2:
                        if both_sides:
                            X_arr[ind, :] = D @ X_arr[ind, :]
                        X_arr[:, ind] = X_arr[:, ind] @ D.T
                    else:
                        X_arr[:, ind] *= D[np.newaxis, :]
                        if both_sides:
                            X_arr[ind, :] *= D[:, np.newaxis]
                else:
                    if D.ndim == 2:
                        Di = D.T if block.Di is None else np.asarray(block.Di, dtype=np.float64)
                        if both_sides:
                            X_arr[ind, :] = Di.T @ X_arr[ind, :]
                        X_arr[:, ind] = X_arr[:, ind] @ Di
                    else:
                        Di = 1.0 / D
                        X_arr[:, ind] *= Di[np.newaxis, :]
                        if both_sides:
                            X_arr[ind, :] *= Di[:, np.newaxis]
            elif D.ndim == 2:
                X_arr[ind] = D @ X_arr[ind]
            else:
                X_arr[ind] = D * X_arr[ind]
        elif is_matrix:
            if D.ndim == 2:
                if both_sides:
                    X_arr[ind, :] = D.T @ X_arr[ind, :]
                X_arr[:, ind] = X_arr[:, ind] @ D
            else:
                if both_sides:
                    X_arr[ind, :] *= D[:, np.newaxis]
                X_arr[:, ind] *= D[np.newaxis, :]
        elif both_sides:
            if D.ndim == 2:
                X_arr[ind] = D.T @ X_arr[ind]
            else:
                X_arr[ind] = D * X_arr[ind]
        elif D.ndim == 2:
            Di = D.T if block.Di is None else np.asarray(block.Di, dtype=np.float64)
            X_arr[ind] = Di @ X_arr[ind]
        else:
            X_arr[ind] = X_arr[ind] / D

    return np.asarray(X_arr, dtype=np.float64)


def _r_recycle_matrix_assign(rhs: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("R-style matrix recycling requires a 2D target shape.")
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if rhs_arr.shape == shape:
        return rhs_arr
    flat = np.ravel(rhs_arr, order="F") if rhs_arr.ndim == 2 else np.ravel(rhs_arr)
    return np.resize(flat, shape[0] * shape[1]).reshape(shape, order="F")


def sl_inirep(
    Sl: Any,
    X: np.ndarray,
    *,
    l: int = 0,  # noqa: E741
    r: int = 0,
) -> np.ndarray:
    """Mirror ``mgcv::Sl.inirep`` for implemented linear Sl blocks."""
    X_arr = np.asarray(X, dtype=np.float64).copy()
    if len(Sl) == 0 or (not l and not r):
        return X_arr

    is_matrix = X_arr.ndim == 2
    for block in Sl:
        if not block.repara:
            continue
        ind = np.arange(block.start0, block.stop0, dtype=int)
        D = np.asarray(block.D, dtype=np.float64)
        Di = None if block.Di is None else np.asarray(block.Di, dtype=np.float64)
        Di_left = D.T if Di is None else Di
        Di_right = D if Di is None else Di.T
        if l:
            if is_matrix:
                target = X_arr[ind, :]
                if l == 1:
                    value = D @ target
                elif l == 2:
                    value = D.T @ target
                elif l == -1:
                    value = Di_left @ target
                else:
                    value = Di_right @ target
                X_arr[ind, :] = _r_recycle_matrix_assign(value, target.shape)
            elif l == 1:
                X_arr[ind] = D @ X_arr[ind]
            elif l == 2:
                X_arr[ind] = D.T @ X_arr[ind]
            elif l == -1:
                X_arr[ind] = Di_left @ X_arr[ind]
            else:
                X_arr[ind] = Di_right @ X_arr[ind]

        if r:
            if is_matrix:
                target = X_arr[:, ind]
                if l == 1:
                    value = target @ D
                elif l == 2:
                    value = target @ D.T
                elif l == -1:
                    value = target @ Di_left
                else:
                    value = target @ Di_right
                X_arr[:, ind] = _r_recycle_matrix_assign(value, target.shape)
            elif l == 1:
                X_arr[ind] = X_arr[ind] @ D
            elif l == 2:
                X_arr[ind] = X_arr[ind] @ D.T
            elif l == -1:
                X_arr[ind] = X_arr[ind] @ Di_left
            else:
                X_arr[ind] = X_arr[ind] @ Di_right

    return np.asarray(X_arr, dtype=np.float64)


__all__ = ["sl_inirep", "sl_initial_repara"]
