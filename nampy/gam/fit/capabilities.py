"""Fit backend capability and smoothing-method helpers."""

from __future__ import annotations


def uses_closed_form_solver(model):
    return bool(getattr(model.family, "supports_closed_form_solve", False))


def uses_pirls_solver(model):
    return bool(getattr(model.family, "supports_pirls", False))


def can_use_exact_gaussian_ml_reml(model):
    from ..smoothing_selection.reparam import can_use_exact_gaussian_ml_reml

    return can_use_exact_gaussian_ml_reml(model)


def can_use_simple_ml_reml_structure(model):
    from ..smoothing_selection.reparam import can_use_simple_ml_reml_structure

    return can_use_simple_ml_reml_structure(model)


def needs_exact_gaussian_reparameterization(model):
    return (
        uses_closed_form_solver(model)
        and can_use_exact_gaussian_ml_reml(model)
        and any(
            bool(getattr(model.family, attr, False))
            for attr in ("supports_ml", "supports_reml", "supports_laml")
        )
    )


def resolve_ml_reml_scoring_backend(model, method="reml"):
    from ..selection import resolve_ml_reml_scoring_backend as _resolve

    return _resolve(model, method=method)


def raise_ml_reml_backend_error(model, method):
    method = str(method).lower()
    backend = resolve_ml_reml_scoring_backend(model, method=method)
    if backend is not None:
        return
    if not bool(getattr(model.family, f"supports_{method}", False)):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={model.family.name!r}."
        )
    if not can_use_simple_ml_reml_structure(model):
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            "currently available for this model configuration. "
            "The current ML/REML backend still rejects penalty layouts "
            "with null-space penalties coupling disconnected primary "
            "penalty components. Use 'fixed', 'gcv', or 'ubre' where "
            "available for those cases."
        )
    raise NotImplementedError(
        f"Automatic smoothing selection with method={method!r} is not "
        f"supported for family={model.family.name!r}."
    )


def supports_smoothing_method(model, method):
    from ..selection import supports_smoothing_method as _supports

    return _supports(model, method)


def resolve_smoothing_method(model, method):
    from ..selection import resolve_smoothing_method as _resolve

    return _resolve(model, method)


def n_free_smoothing_params(model):
    from ..selection import n_free_smoothing_params as _count

    return _count(model)
