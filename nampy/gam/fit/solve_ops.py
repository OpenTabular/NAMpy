"""Fixed-smoothing solver wrappers."""

from __future__ import annotations


def solve_gaussian_given_smoothing(model, y, smoothing_params):
    from ..engine import solve_gaussian_fit

    return solve_gaussian_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )


def solve_pirls_given_smoothing(model, y, smoothing_params):
    from ..engine import solve_pirls_fit

    return solve_pirls_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )

