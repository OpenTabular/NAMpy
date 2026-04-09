"""
Fitting backend selection for GAMs.

The fitting backend is determined by the family:
- ``"gaussian_exact"``: closed-form penalized least-squares for Gaussian families.
- ``"pirls"``: penalized IRLS for non-Gaussian families.

:func:`solve_fit` is the single entry point used by the model fitting orchestrator.
"""

from .solvers.gaussian_exact import solve_gaussian_fit
from .solvers.pirls import solve_pirls_fit


def available_fit_backends(model):
    backends = []
    if bool(getattr(model, "_use_stacked_qr", False)):
        backends.append("stacked_qr")
    if bool(getattr(model.family, "supports_closed_form_solve", False)):
        backends.append("gaussian_exact")
    if bool(getattr(model.family, "supports_pirls", False)):
        backends.append("pirls")
    return tuple(backends)


def resolve_fit_backend(model):
    backends = available_fit_backends(model)
    if "stacked_qr" in backends:
        return "stacked_qr"
    if "gaussian_exact" in backends:
        return "gaussian_exact"
    if "pirls" in backends:
        return "pirls"
    raise NotImplementedError(
        f"No supported fitting backend for family={model.family.name!r}."
    )


def solve_fit(model, y, smoothing_params, backend=None, weights=None):
    backend = resolve_fit_backend(model) if backend is None else str(backend).lower()

    if backend == "stacked_qr":
        return solve_gaussian_fit(model, y, smoothing_params, weights=weights)
    if backend == "gaussian_exact":
        return solve_gaussian_fit(model, y, smoothing_params, weights=weights)
    if backend == "pirls":
        return solve_pirls_fit(model, y, smoothing_params, weights=weights)

    raise ValueError(f"Unknown fit backend {backend!r}.")
