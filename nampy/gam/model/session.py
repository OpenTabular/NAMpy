"""One isolated, transactional invocation of :meth:`GAM.fit`."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..families import clone_gam_family
from ..workspace import FitWorkspace

_FIT_STATE_DEFAULTS = {
    "formula_": None,
    "formula_mode_": False,
    "formula_response_name_": None,
    "formula_preprocess_state_": None,
    "feature_names": None,
    "X_": None,
    "y_": None,
    "prior_weights_": None,
    "offset_train_": None,
    "offset_predict_default_": None,
    "n_samples_": None,
    "smoothing_fixed_mask_": None,
    "smoothing_override_values_": None,
    "smoothing_override_modes_": None,
    "min_sp_": None,
    "reparam_state_": None,
    "sl_blocks_": None,
    "predictor_specs": None,
    "ar_start_": None,
    "ar1_standardized_residuals_": None,
    "_fitted": False,
    "_optim_method": None,
    "_optim_result": None,
    "_optim_trace": None,
    "_optim_used_gradient": None,
    "_optim_used_hessian": None,
    "smoothing_score_": None,
    "gam_result_": None,
}

_DYNAMIC_FIT_STATE = {
    "_gamma_reml_phi_opt_",
    "_gaussian_reml_last_scale_est_",
    "_gaussian_reml_sigma2_opt_",
    "_general_family_outer_derivative_info",
    "_general_family_outer_use_fit5_hessian_",
    "_null_deviance_",
    "_summary_R_",
    "coefficient_transform",
    "observation_transform",
    "positive_coefficient_mask",
}


@dataclass
class FitSession:
    """A private working model whose state is published only after success."""

    working_model: Any

    @classmethod
    def begin(cls, model: Any) -> "FitSession":
        working = copy.copy(model)
        working.__dict__ = dict(model.__dict__)
        working.hparams = copy.deepcopy(dict(model.hparams))

        template = getattr(model, "_family_template", model.family)
        working._family_template = clone_gam_family(template)
        working.family = clone_gam_family(working._family_template)
        working._ws = FitWorkspace()

        # ``smoothing_params`` is both a constructor input and a fitted output.
        # A new fit must start from the configured input, not the previous fit's
        # endpoint (which can even have a different dimension after a formula
        # change).
        working.smoothing_params = copy.deepcopy(
            working.hparams.get("smoothing_params", None)
        )
        for name, value in _FIT_STATE_DEFAULTS.items():
            setattr(working, name, copy.deepcopy(value))
        for name in _DYNAMIC_FIT_STATE:
            working.__dict__.pop(name, None)
        return cls(working_model=working)

    def commit_to(self, model: Any) -> None:
        """Atomically publish the successfully fitted working state."""
        model.__dict__.clear()
        model.__dict__.update(self.working_model.__dict__)


__all__ = ["FitSession"]
