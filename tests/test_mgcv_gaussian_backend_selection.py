from __future__ import annotations

from types import SimpleNamespace

from nampy.gam.smoothing_selection.criteria.ml_reml import (
    resolve_ml_reml_scoring_backend,
)


def _gaussian_model_with_term(term_type: str):
    family = SimpleNamespace(supports_closed_form_solve=True, supports_pirls=False)
    return SimpleNamespace(
        family=family,
        term_blocks_=[SimpleNamespace(term_type=term_type, basis_name="")],
        _can_use_exact_gaussian_ml_reml=lambda: True,
        _can_use_simple_ml_reml_structure=lambda: True,
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
