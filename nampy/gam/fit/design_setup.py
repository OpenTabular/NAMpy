"""Design compilation and fit-time reparameterization setup."""

from __future__ import annotations

import numpy as np

from ..model_state import _n_smoothing_params, _predictor_designs
from .capabilities import (
    needs_exact_gaussian_reparameterization,
)
from .smoothing_params import resolve_min_sp, resolve_smoothing_params


def compile_designs(model, X, feature_names):
    from ..compiler.compile_model import compile_model

    compiled_model = compile_model(
        X=X,
        feature_names=feature_names,
        predictor_specs=model.predictor_specs,
        fit_intercept=model.fit_intercept,
        apply_side_conditions=bool(model.hparams.get("apply_side_conditions", True)),
        # Fixed internal tolerance: upstream gam.side has no user-facing
        # knob either (mgcv/R/mgcv.r:1266 hard-codes its call-site tol).
        side_condition_tol=1e-10,
    )
    from ..compiler.coefficient_transforms import (
        configure_compiled_coefficient_transforms,
    )

    configure_compiled_coefficient_transforms(
        compiled_model,
        positive_map=str(getattr(model, "positive_transform", "exp")),
        softplus_beta=float(getattr(model, "softplus_beta", 1.0)),
        softplus_threshold=float(getattr(model, "softplus_threshold", 20.0)),
    )
    from ..observations import make_observation_transform

    compiled_model.observation_transform = make_observation_transform(
        size=int(np.asarray(compiled_model.design_matrix).shape[0]),
        ar1_rho=float(getattr(model, "ar1_rho", 0.0)),
        ar_start=getattr(model, "ar_start_", None),
    )
    from ..results import GAMResult

    current = model.gam_result_ or GAMResult()
    model.gam_result_ = current.with_compiled_model(
        compiled_model
    )
    model.family.validate_predictor_count(len(_predictor_designs(model)))
    model.min_sp_ = resolve_min_sp(model, model.min_sp)
    model.smoothing_params = resolve_smoothing_params(model, _n_smoothing_params(model))

    if needs_exact_gaussian_reparameterization(model):
        from .selection.reparam import (
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
