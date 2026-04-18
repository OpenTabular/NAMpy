from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nampy.gam.compiler.structures import CompiledModel
from nampy.gam.smoothing_selection.criteria.ml_reml import (
    resolve_ml_reml_scoring_backend,
)


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
    def test_fs_reml_uses_exact_backend(self):
        model = _gaussian_model_with_term("factor_smooth_fs")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"

    def test_sz_reml_uses_exact_backend(self):
        model = _gaussian_model_with_term("factor_smooth_sz")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"

    def test_t2_reml_uses_exact_backend(self):
        model = _gaussian_model_with_term("tensor_anova")

        backend = resolve_ml_reml_scoring_backend(model, method="reml")

        assert backend == "gaussian_exact"
