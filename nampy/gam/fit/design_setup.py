"""Design compilation and fit-time reparameterization setup."""

from __future__ import annotations

import numpy as np

from .._model_state import _n_smoothing_params, _predictor_designs, _term_blocks_seq
from .capabilities import needs_exact_gaussian_reparameterization
from .smoothing_params import resolve_min_sp, resolve_smoothing_params


def _reject_unsupported_general_family_multi_smooths(model) -> None:
    family_class = str(getattr(model.family, "family_class", "")).lower()
    if family_class != "general":
        return

    smooth_terms = [
        term
        for term in _term_blocks_seq(model)
        if str(getattr(term, "term_type", "")).lower() != "parametric"
        and int(getattr(term, "n_penalties", 0)) > 0
    ]
    if len(smooth_terms) <= 1:
        return

    labels = ", ".join(str(getattr(term, "label", "")) for term in smooth_terms)
    raise NotImplementedError(
        "Multi-smooth general-family models are not supported. "
        "Use at most one penalized smooth term across the general-family "
        f"linear predictors; got {len(smooth_terms)} smooth terms"
        f" ({labels})."
    )


def compile_designs(model, X, feature_names):
    from ..compiler.compile_model import compile_model

    compiled_model = compile_model(
        X=X,
        feature_names=feature_names,
        predictor_specs=model.predictor_specs,
        fit_intercept=model.fit_intercept,
        apply_side_conditions=bool(model.hparams.get("apply_side_conditions", True)),
        side_condition_tol=float(model.hparams.get("side_condition_tol", 1e-10)),
    )
    model.compiled_model_ = compiled_model
    model.side_condition_reports_ = (
        None
        if compiled_model.side_condition_reports is None
        else list(compiled_model.side_condition_reports)
    )
    model.family.validate_predictor_count(len(_predictor_designs(model)))
    _reject_unsupported_general_family_multi_smooths(model)
    model._coef_reduced_to_full_idx = np.asarray(
        compiled_model.coef_reduced_to_full_idx,
        dtype=int,
    )
    model.min_sp_ = resolve_min_sp(model, model.min_sp)
    model.smoothing_params = resolve_smoothing_params(model, _n_smoothing_params(model))

    if needs_exact_gaussian_reparameterization(model):
        from ..smoothing_selection.reparam import (
            assign_exact_reparam_state,
            build_estimate_gam_setup_state,
            build_penalty_reparameterization_state,
        )

        # Exact Gaussian ML/REML mirrors mgcv's `gam.fit3()` canonical
        # reparameterization state rather than the older mixed-model `Sl` view.
        setup = build_estimate_gam_setup_state(model)
        assign_exact_reparam_state(
            model,
            build_penalty_reparameterization_state(
                model,
                np.asarray(setup.X, dtype=np.float64),
                np.asarray(model.smoothing_params, dtype=np.float64),
                deriv=0,
            ),
        )
    else:
        model.reparam_state_ = None
        model.sl_blocks_ = None


def build_gaussian_reparameterized_system(model):
    from ..smoothing_selection.reparam import build_gaussian_reparameterized_system

    return build_gaussian_reparameterized_system(model)


def build_penalty_reparameterized_system(model):
    from ..smoothing_selection.reparam import build_penalty_reparameterized_system

    return build_penalty_reparameterized_system(model)
