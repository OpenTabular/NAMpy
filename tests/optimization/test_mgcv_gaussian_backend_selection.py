from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from nampy.gam import GAM
from nampy.gam.compiler.structures import CompiledModel
from nampy.gam.smoothing_selection.criteria.ml_reml import (
    resolve_ml_reml_scoring_backend,
)
from tests.mgcv_parity_utils import _make_gaussian_data


def _gaussian_model_with_term(term_type: str):
    family = SimpleNamespace(supports_closed_form_solve=True, supports_pirls=False)
    compiled_model = CompiledModel(
        predictors=(),
        design_matrix=np.empty((0, 0), dtype=np.float64),
        compiled_terms=(SimpleNamespace(term_type=term_type, basis_name=""),),
        compiled_penalties=(),
        metadata={},
        n_coef=0,
        n_smoothing_params=0,
        predictor_full_slices=(),
        coef_reduced_to_full_idx=np.empty((0,), dtype=int),
    )
    return SimpleNamespace(
        family=family,
        compiled_model_=compiled_model,
    )


class TestGaussianRemlBackendSelection:
    """
    Backend-selection checks for parity-sensitive Gaussian REML surfaces that must stay
    on the exact scoring path.
    """
    def test_fs_reml_uses_exact_backend(self):
        """Verify that fs REML uses exact backend."""
        model = _gaussian_model_with_term("factor_smooth_fs")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"

    def test_sz_reml_uses_exact_backend(self):
        """Verify that sz REML uses exact backend."""
        model = _gaussian_model_with_term("factor_smooth_sz")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"

    def test_t2_reml_uses_exact_backend(self):
        """Verify that t2 REML uses exact backend."""
        model = _gaussian_model_with_term("tensor_anova")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"


def test_gaussian_reml_bfgs_uses_profiled_objective_without_joint_sigma2(
    monkeypatch,
):
    """Verify that gaussian REML BFGS uses profiled objective without joint sigma2."""
    from nampy.gam.smoothing_selection.optimize import driver as driver_module
    from nampy.gam.smoothing_selection.optimize.objectives import (
        _GaussianRemlProfiledObjective,
    )

    data = _make_gaussian_data(seed=44, n=80).rename(columns={"x0": "x"})
    data = pd.DataFrame({"y": data["y"], "x": data["x"]})

    captured: dict[str, object] = {}

    def _fake_bfgs(*, objective, x0, bounds, score_type):
        captured["objective_type"] = type(objective)
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["bounds"] = list(bounds)
        captured["score_type"] = score_type
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64).copy(),
            fun=float(objective.fun(x0)),
            success=True,
            status=0,
            message="captured",
            nit=0,
            nfev=1,
            njev=0,
            nhev=0,
        )

    monkeypatch.setattr(driver_module, "_optimize_outer_bfgs_mgcv", _fake_bfgs)
    monkeypatch.setattr(
        driver_module, "_refresh_final_outer_derivatives", lambda *args, **kwargs: None
    )

    model = GAM(
        formula='y ~ s(x, bs="cr", k=8)',
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
    )
    model.fit(data=data)

    assert captured["objective_type"] is _GaussianRemlProfiledObjective
    assert np.asarray(captured["x0"], dtype=np.float64).shape == (
        int(model.compiled_model_.n_smoothing_params),
    )
    assert captured["score_type"] == "reml"
