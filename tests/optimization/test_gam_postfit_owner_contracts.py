from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.smoothing_selection import postfit as postfit_module
from nampy.gam.smoothing_selection.postfit import (
    _postfit_hessian,
    optimizer_endpoint_diagnostics,
)

pytestmark = [
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
]


def test_postfit_hessian_prefers_edge_correct_result_block(monkeypatch):
    """
    Owner-contract coverage verifying that postfit hessian prefers edge correct result
    block.
    """
    model = SimpleNamespace(
        y_=np.array([1.0], dtype=np.float64),
        smoothing_params=np.array([1.0], dtype=np.float64),
        _optim_result=SimpleNamespace(
            hess=np.array([[5.0]], dtype=np.float64),
            outer_info={"hess1": np.array([[2.0]], dtype=np.float64)},
        ),
    )

    monkeypatch.setattr(
        postfit_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "gaussian_exact",
    )
    monkeypatch.setattr(postfit_module, "_n_smoothing_params", lambda model: 1)
    monkeypatch.setattr(
        postfit_module,
        "fit_criterion_hessian",
        lambda *args, **kwargs: np.array([[9.0]], dtype=np.float64),
    )

    H = _postfit_hessian(model, "reml", edge_correct=True)

    np.testing.assert_allclose(H, np.array([[2.0]], dtype=np.float64))


def test_postfit_hessian_recomputes_for_pirls_backends(monkeypatch):
    """
    Owner-contract coverage verifying that postfit hessian recomputes for PIRLS
    backends.
    """
    model = SimpleNamespace(
        y_=np.array([1.0], dtype=np.float64),
        smoothing_params=np.array([1.0], dtype=np.float64),
        _optim_result=SimpleNamespace(hess=np.array([[5.0]], dtype=np.float64)),
    )

    monkeypatch.setattr(
        postfit_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "pirls_laplace",
    )
    monkeypatch.setattr(
        postfit_module,
        "fit_criterion_hessian",
        lambda *args, **kwargs: np.array([[3.0]], dtype=np.float64),
    )

    H = _postfit_hessian(model, "reml", edge_correct=False)

    np.testing.assert_allclose(H, np.array([[3.0]], dtype=np.float64))


def test_postfit_hessian_preserves_exact_pirls_optimizer_hessian(monkeypatch):
    """
    Owner-contract coverage verifying that an exact optimizer Hessian is reused
    for PIRLS backends instead of being recomputed post hoc.
    """
    model = SimpleNamespace(
        y_=np.array([1.0], dtype=np.float64),
        smoothing_params=np.array([1.0], dtype=np.float64),
        _optim_result=SimpleNamespace(
            hess=np.array([[7.0]], dtype=np.float64),
            mgcv_exact_outer_derivatives=True,
        ),
    )

    monkeypatch.setattr(
        postfit_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "pirls_laplace",
    )
    monkeypatch.setattr(postfit_module, "_n_smoothing_params", lambda model: 1)
    monkeypatch.setattr(
        postfit_module,
        "fit_criterion_hessian",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("exact PIRLS Hessian should be reused")
        ),
    )

    H = _postfit_hessian(model, "reml", edge_correct=False)

    np.testing.assert_allclose(H, np.array([[7.0]], dtype=np.float64))


def test_optimizer_endpoint_diagnostics_recomputes_invalid_derivatives_and_projects_bounds(
    monkeypatch,
):
    """
    Owner-contract coverage verifying that optimizer endpoint diagnostics recomputes
    invalid derivatives and projects bounds.
    """
    model = SimpleNamespace(
        _optim_method="reml",
        y_=np.array([1.0], dtype=np.float64),
        smoothing_params=np.array([1.0], dtype=np.float64),
        smoothing_fixed_mask_=None,
        min_sp_=None,
        sp_log_bounds=(0.0, 5.0),
        _optim_result=SimpleNamespace(
            success=True,
            message="ok",
            jac=np.array([np.nan], dtype=np.float64),
            hess=np.array([[np.nan]], dtype=np.float64),
        ),
        smoothing_score_=0.0,
        family=SimpleNamespace(),
    )

    monkeypatch.setattr(postfit_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(postfit_module, "_n_smoothing_params", lambda model: 1)
    monkeypatch.setattr(
        postfit_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "gaussian_exact",
    )
    monkeypatch.setattr(
        postfit_module,
        "criterion_gradient",
        lambda model, y, x, method: np.array([2.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        postfit_module,
        "criterion_hessian",
        lambda model, y, x, method: np.array([[3.0]], dtype=np.float64),
    )

    diag = optimizer_endpoint_diagnostics(model)

    np.testing.assert_allclose(diag["gradient"], np.array([2.0], dtype=np.float64))
    np.testing.assert_allclose(diag["projected_gradient"], np.array([0.0]))
    np.testing.assert_allclose(diag["hessian"], np.array([[3.0]], dtype=np.float64))
    assert diag["criterion_backend"] == "gaussian_exact"
    assert diag["at_lower_bound"] == [True]
    assert diag["at_upper_bound"] == [False]
    assert diag["boundary_limited"] is True
    assert diag["stationary_by_projected_gradient"] is True
    assert diag["stationary_by_raw_gradient"] is False
