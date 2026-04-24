"""Fixed-smoothing solver wrappers."""

from __future__ import annotations

from .solvers.gaussian_exact import solve_gaussian_fit
from .solvers.pirls import solve_pirls_fit


def solve_gaussian_given_smoothing(model, y, smoothing_params):
    return solve_gaussian_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )


def solve_pirls_given_smoothing(model, y, smoothing_params):
    family = getattr(model, "family", None)
    if str(getattr(family, "family_class", "")).lower() == "general":
        from .solvers.general_family_solver import solve_general_family_fit

        return solve_general_family_fit(
            model,
            y,
            smoothing_params,
            weights=model.prior_weights_,
        )

    return solve_pirls_fit(
        model,
        y,
        smoothing_params,
        weights=model.prior_weights_,
    )
