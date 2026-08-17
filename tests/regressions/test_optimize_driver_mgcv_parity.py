from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from nampy.gam.fit import orchestrator as orchestrator_module
from nampy.gam.fit import smoothing_params as fit_smoothing_params_module
from nampy.gam.model.api import GAM
from nampy.gam.smoothing_selection.optimize import driver as driver_module
from nampy.gam.smoothing_selection.optimize import objectives as objectives_module
from nampy.gam.smoothing_selection.optimize.objectives import (
    _JointNegbinPirlsRemlObjective,
)


def test_defaults_use_mgcv_outer_newton_optimizer():
    assert (
        inspect.signature(driver_module.optimize_smoothing_params)
        .parameters["optimizer"]
        .default
        == "outer_newton"
    )
    assert (
        inspect.signature(fit_smoothing_params_module.optimize_smoothing_params)
        .parameters["optimizer"]
        .default
        == "outer_newton"
    )
    assert GAM(family="gaussian").smoothing_optimizer == "outer_newton"


def test_efs_forces_reml_before_fixed_sp_return(monkeypatch):
    captured = {}

    family = SimpleNamespace(
        name="poisson",
        family_class="",
        supports_closed_form_solve=False,
        supports_pirls=True,
        supports_reml=True,
        supports_ml=True,
        supports_laml=True,
        supports_gcv=True,
        supports_ubre=False,
        known_scale=None,
    )
    model = SimpleNamespace(
        family=family,
        smoothing_fixed_mask_=None,
        smoothing_params=np.empty((0,), dtype=np.float64),
        min_sp_=None,
        offset_train_=None,
        prior_weights_=None,
    )

    monkeypatch.setattr(driver_module, "_n_smoothing_params", lambda model: 0)
    monkeypatch.setattr(driver_module, "_term_blocks_seq", lambda model: [])
    monkeypatch.setattr(
        driver_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method: "pirls_laplace",
    )

    def _criterion_value(model, y, log_sp, method):
        captured["method"] = method
        captured["log_sp_shape"] = np.asarray(log_sp, dtype=np.float64).shape
        return 12.5

    monkeypatch.setattr(driver_module, "criterion_value", _criterion_value)

    out = driver_module.optimize_smoothing_params(
        model,
        np.array([1.0, 2.0], dtype=np.float64),
        method="gcv",
        optimizer="efs",
    )

    assert out is model
    assert model._optim_method == "reml"
    assert captured == {"method": "reml", "log_sp_shape": (0,)}
    assert model.smoothing_score_ == pytest.approx(12.5, abs=0.0)


@pytest.mark.parametrize("method", ["gcv", "ubre", "aic", "ubreaic"])
def test_general_family_non_reml_methods_force_reml(monkeypatch, method):
    captured = {}

    family = SimpleNamespace(
        name="gaulss",
        family_class="general",
        supports_closed_form_solve=False,
        supports_pirls=False,
        supports_reml=True,
        supports_ml=True,
        supports_laml=True,
        supports_gcv=False,
        supports_ubre=False,
        known_scale=None,
    )
    model = SimpleNamespace(
        family=family,
        smoothing_fixed_mask_=None,
        smoothing_params=np.empty((0,), dtype=np.float64),
        min_sp_=None,
        offset_train_=None,
        prior_weights_=None,
    )

    monkeypatch.setattr(driver_module, "_n_smoothing_params", lambda model: 0)
    monkeypatch.setattr(driver_module, "_term_blocks_seq", lambda model: [])
    monkeypatch.setattr(
        driver_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method: "general_family",
    )

    def _criterion_value(model, y, log_sp, method):
        captured["method"] = method
        captured["log_sp_shape"] = np.asarray(log_sp, dtype=np.float64).shape
        return 8.75

    monkeypatch.setattr(driver_module, "criterion_value", _criterion_value)

    out = driver_module.optimize_smoothing_params(
        model,
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        method=method,
        optimizer="outer_newton",
    )

    assert out is model
    assert model._optim_method == "reml"
    assert captured == {"method": "reml", "log_sp_shape": (0,)}
    assert model.smoothing_score_ == pytest.approx(8.75, abs=0.0)


@pytest.mark.parametrize("method", ["gcv", "ubre", "aic", "ubreaic"])
def test_public_fit_coerces_general_family_method_before_support_check(
    monkeypatch, method
):
    calls = {}

    family = SimpleNamespace(
        name="gaulss",
        family_class="general",
        validate_y=lambda y: np.asarray(y, dtype=np.float64).ravel(),
    )
    model = SimpleNamespace(
        family=family,
        optimize_smoothing=True,
        smoothing_method=method,
        smoothing_optimizer="outer_newton",
        smoothing_params=np.empty((0,), dtype=np.float64),
        smoothing_fixed_mask_=None,
        hparams={},
    )

    monkeypatch.setattr(orchestrator_module, "compile_designs", lambda *args: None)
    monkeypatch.setattr(orchestrator_module, "_n_smoothing_params", lambda model: 0)

    def _supports(model, method):
        calls["support_method"] = method
        return method == "reml"

    def _optimize(model, y, initial_smoothing_params, method, optimizer):
        calls["optimize_method"] = method
        calls["optimizer"] = optimizer
        model._optim_method = method
        model._optim_result = None
        model._optim_trace = []
        model._optim_used_gradient = False
        model._optim_used_hessian = False
        model.smoothing_score_ = 2.5

    monkeypatch.setattr(
        orchestrator_module,
        "supports_smoothing_method",
        _supports,
    )
    monkeypatch.setattr(orchestrator_module, "optimize_smoothing_params", _optimize)
    monkeypatch.setattr(
        orchestrator_module,
        "solve_fit",
        lambda *args, **kwargs: SimpleNamespace(inner_trace=[]),
    )
    monkeypatch.setattr(orchestrator_module, "assign_fit_solution", lambda *args: None)
    monkeypatch.setattr(
        orchestrator_module,
        "build_gam_result",
        lambda model: SimpleNamespace(),
    )

    out = orchestrator_module.fit_model_core(
        model,
        X=np.zeros((3, 1), dtype=np.float64),
        feature_names=["x"],
        y=np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )

    assert out is model
    assert calls == {
        "support_method": "reml",
        "optimize_method": "reml",
        "optimizer": "outer_newton",
    }
    assert model._optim_method == "reml"


def test_all_fixed_smoothing_params_still_optimizes_unknown_gaussian_scale(
    monkeypatch,
):
    captured = {}

    def _deviance(y, mu, weights=None):
        del weights
        return float(np.sum((np.asarray(y) - np.asarray(mu)) ** 2))

    family = SimpleNamespace(
        name="gaussian",
        family_class="",
        supports_closed_form_solve=True,
        supports_pirls=False,
        supports_reml=True,
        supports_ml=True,
        supports_laml=True,
        supports_gcv=True,
        supports_ubre=False,
        known_scale=None,
        deviance=_deviance,
    )
    model = SimpleNamespace(
        family=family,
        smoothing_fixed_mask_=np.array([True]),
        smoothing_params=np.array([2.0], dtype=np.float64),
        min_sp_=None,
        sp_log_bounds=(-80.0, 20.0),
        n_samples_=4,
        score_gamma=1.0,
        offset_train_=None,
        prior_weights_=None,
        hparams={},
    )

    monkeypatch.setattr(driver_module, "_n_smoothing_params", lambda model: 1)
    monkeypatch.setattr(driver_module, "_term_blocks_seq", lambda model: [])
    monkeypatch.setattr(
        driver_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method: "gaussian_dynamic",
    )
    monkeypatch.setattr(
        driver_module, "supports_criterion_gradient", lambda model, method: True
    )
    monkeypatch.setattr(
        driver_module, "supports_criterion_hessian", lambda model, method: True
    )
    monkeypatch.setattr(
        driver_module,
        "_initial_smoothing_params_from_design",
        lambda model, y: np.array([2.0], dtype=np.float64),
    )

    def _fake_newton(*, objective, x0, bounds, **kwargs):
        del objective, kwargs
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["bounds"] = list(bounds)
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64).copy(),
            fun=3.25,
            jac=np.zeros_like(np.asarray(x0, dtype=np.float64)),
            hess=np.eye(np.asarray(x0, dtype=np.float64).size),
            success=True,
            status=0,
            message="captured",
            nit=0,
            nfev=1,
            njev=1,
            nhev=1,
        )

    monkeypatch.setattr(
        driver_module,
        "optimize_outer_newton_indefinite_hessian",
        _fake_newton,
    )

    out = driver_module.optimize_smoothing_params(
        model,
        np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        method="reml",
        optimizer="outer_newton",
    )

    assert out is model
    assert np.asarray(captured["x0"], dtype=np.float64).shape == (1,)
    assert len(captured["bounds"]) == 1
    assert model._optim_result is not None
    assert model.smoothing_params.tolist() == [2.0]


def test_optim_result_fun_is_recomputed_unscaled(monkeypatch):
    class Objective:
        model = None
        y = np.empty((0,), dtype=np.float64)

        def fun(self, x):
            x = np.asarray(x, dtype=np.float64)
            return float(10.0 + np.sum(x**2))

        def jac(self, x):
            return 2.0 * np.asarray(x, dtype=np.float64)

    def _fake_minimize(**kwargs):
        assert kwargs["method"] == "L-BFGS-B"
        return OptimizeResult(
            x=np.array([2.0], dtype=np.float64),
            fun=0.125,
            success=True,
            status=0,
            message="captured",
            nit=1,
            nfev=1,
            njev=1,
        )

    monkeypatch.setattr(driver_module, "minimize", _fake_minimize)

    result = driver_module._optimize_outer_optim_strict(
        objective=Objective(),
        x0=np.array([0.0], dtype=np.float64),
        bounds=[(-10.0, 10.0)],
    )

    assert result.optim_scaled_fun == pytest.approx(0.125, abs=0.0)
    assert result.fun == pytest.approx(14.0, abs=0.0)


def test_negbin_joint_objective_uses_mgcv_theta_first_order(monkeypatch):
    captured = {}

    def _fake_fun(model, y, log_sp, log_theta, method):
        del model, y
        captured["fun"] = (np.asarray(log_sp, dtype=np.float64), log_theta, method)
        return 4.0

    def _fake_grad(model, y, log_sp, log_theta, method):
        del model, y, log_sp, log_theta, method
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def _fake_hess(model, y, log_sp, log_theta, method):
        del model, y, log_sp, log_theta, method
        return np.arange(9.0, dtype=np.float64).reshape(3, 3)

    monkeypatch.setattr(
        objectives_module,
        "criterion_ml_reml_pirls_negbin_joint",
        _fake_fun,
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_gradient_ml_reml_pirls_negbin_joint",
        _fake_grad,
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_hessian_ml_reml_pirls_negbin_joint",
        _fake_hess,
    )

    objective = _JointNegbinPirlsRemlObjective(
        SimpleNamespace(),
        np.empty((0,), dtype=np.float64),
        "REML",
    )
    x = np.array([0.7, -1.0, -2.0], dtype=np.float64)

    assert objective.fun(x) == pytest.approx(4.0, abs=0.0)
    log_sp, log_theta, method = captured["fun"]
    assert log_theta == pytest.approx(0.7, abs=0.0)
    assert log_sp.tolist() == [-1.0, -2.0]
    assert method == "REML"
    assert objective.jac(x).tolist() == [3.0, 1.0, 2.0]
    expected_perm = np.array([2, 0, 1], dtype=np.int64)
    expected_hess = np.arange(9.0, dtype=np.float64).reshape(3, 3)[
        np.ix_(expected_perm, expected_perm)
    ]
    np.testing.assert_array_equal(objective.hess(x), expected_hess)


def test_negbin_reml_native_all_fixed_optimizes_theta_first(monkeypatch):
    captured = {}

    family = SimpleNamespace(
        name="negbin",
        estimate_theta=True,
        theta=2.0,
    )
    model = SimpleNamespace(
        family=family,
        offset_train_=None,
        prior_weights_=None,
        formula="y ~ s(x)",
        smoothing_params=np.array([5.0], dtype=np.float64),
    )

    def _fake_newton(*, objective, x0, bounds, **kwargs):
        del objective, kwargs
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["bounds"] = list(bounds)
        x = np.asarray(x0, dtype=np.float64).copy()
        x[0] += 0.25
        return OptimizeResult(
            x=x,
            fun=8.0,
            jac=np.array([0.0], dtype=np.float64),
            hess=np.array([[1.0]], dtype=np.float64),
            success=True,
            status=0,
            message="captured",
            nit=1,
            nfev=1,
            njev=1,
            nhev=1,
        )

    monkeypatch.setattr(
        driver_module,
        "optimize_outer_newton_indefinite_hessian",
        _fake_newton,
    )

    result = driver_module._optimize_negbin_reml_joint_native(
        model,
        np.array([1.0], dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        np.array([False], dtype=bool),
        "reml",
        [],
        optimizer="outer_newton",
    )

    assert result is not None
    assert captured["x0"].tolist() == pytest.approx([np.log(2.0)], abs=0.0)
    assert len(captured["bounds"]) == 1
    assert result.x.shape == (0,)
    assert result.joint_log_theta == pytest.approx(np.log(2.0) + 0.25, abs=0.0)
    assert result.selected_full_smoothing_params.tolist() == [5.0]
