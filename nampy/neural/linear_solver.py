"""Strict solvers for neural architectures that are linear in fixed bases."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.linalg import cg

from .contracts import FixedLinearDesignProvider


def solve_fixed_linear_regression(
    model: FixedLinearDesignProvider,
    *,
    num_features,
    cat_features,
    targets,
    sample_weight=None,
    offset=None,
) -> dict:
    """Fit a weighted ridge model by conjugate gradients without inversion.

    This mirrors GP-NAM's accumulation of ``Phi.T @ Phi`` and ``Phi.T @ y``.
    The intercept column is appended last and excluded from the ridge penalty,
    matching the released Python and MATLAB implementations.
    """
    design = model.linear_design(num_features, cat_features).detach().cpu().numpy()
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, None]
    if response.ndim != 2 or response.shape[0] != design.shape[0]:
        raise ValueError("Fixed-design targets must have shape (n_samples, n_outputs).")

    if offset is not None:
        offset_array = np.asarray(offset, dtype=np.float64)
        if offset_array.ndim == 1:
            offset_array = offset_array[:, None]
        if offset_array.shape[0] != response.shape[0] or offset_array.shape[1] not in {
            1,
            response.shape[1],
        }:
            raise ValueError("offset has an incompatible shape for fixed-design fitting.")
        response = response - offset_array

    if model.intercept is not None:
        design = np.column_stack([design, np.ones(design.shape[0])])

    if sample_weight is None:
        weights = np.ones(design.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != design.shape[0]:
            raise ValueError("sample_weight has an incompatible length.")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight must be finite and non-negative.")
        if float(weights.sum()) <= 0:
            raise ValueError("sample_weight must sum to a positive value.")

    root_weight = np.sqrt(weights)[:, None]
    weighted_design = design * root_weight
    weighted_response = response * root_weight
    normal = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_response

    penalty = np.full(normal.shape[0], float(model.ridge), dtype=np.float64)
    if model.intercept is not None:
        penalty[-1] = 0.0
    normal[np.diag_indices_from(normal)] += penalty

    solutions = []
    iterations = []
    for output in range(rhs.shape[1]):
        iteration_count = 0

        def count_iteration(_):
            nonlocal iteration_count
            iteration_count += 1

        solution, info = cg(
            normal,
            rhs[:, output],
            rtol=float(model.cg_rtol),
            atol=0.0,
            maxiter=model.cg_max_iter,
            callback=count_iteration,
        )
        if info != 0:
            reason = (
                f"did not converge after {info} iterations"
                if info > 0
                else f"failed with illegal-input code {info}"
            )
            raise RuntimeError(f"Conjugate-gradient fixed-design solve {reason}.")
        solutions.append(solution)
        iterations.append(iteration_count)

    coefficients = np.column_stack(solutions)
    if model.intercept is None:
        intercept = None
        linear_coefficients = coefficients
    else:
        intercept = coefficients[-1]
        linear_coefficients = coefficients[:-1]
    parameter = next(model.parameters())
    model.set_linear_coefficients(
        torch.as_tensor(
            linear_coefficients, device=parameter.device, dtype=parameter.dtype
        ),
        None
        if intercept is None
        else torch.as_tensor(intercept, device=parameter.device, dtype=parameter.dtype),
    )
    residual = design @ coefficients - response
    return {
        "solver": "cg",
        "ridge": float(model.ridge),
        "iterations": tuple(iterations),
        "weighted_mse": float(
            np.sum(weights[:, None] * residual**2)
            / (weights.sum() * response.shape[1])
        ),
        "n_rows": int(design.shape[0]),
        "n_columns": int(design.shape[1]),
    }


__all__ = ["solve_fixed_linear_regression"]
