from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import ml_reml as ml_reml_module
from nampy.gam.fit.selection.optimize import driver as driver_module
from nampy.gam.fit.selection.optimize import objectives as objectives_module
from nampy.gam.fit.selection.optimize.objectives import (
    _CriterionObjective,
    _GaussianRemlJointObjective,
    _GaussianRemlProfiledObjective,
    _JointGammaPirlsRemlObjective,
)
from tests.mgcv_parity_utils import _make_gaussian_data

pytestmark = [
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
]


def _stub_model(*, family_kwargs):
    family_defaults = {
        "name": "stub",
        "supports_closed_form_solve": False,
        "supports_pirls": False,
        "family_class": "",
    }
    family_defaults.update(family_kwargs)
    return SimpleNamespace(family=SimpleNamespace(**family_defaults))


@pytest.mark.parametrize(
    ("family_kwargs", "exact_ok", "simple_ok", "expected"),
    [
        ({"supports_closed_form_solve": True}, True, False, "gaussian_exact"),
        ({"supports_closed_form_solve": True}, False, False, "gaussian_dynamic"),
        ({"supports_pirls": True}, False, True, "pirls_laplace"),
        ({"supports_pirls": True}, False, False, "pirls_laplace_dynamic"),
        ({"family_class": "general"}, False, False, "general_family"),
    ],
    ids=[
        "gaussian_exact",
        "gaussian_dynamic",
        "pirls_laplace",
        "pirls_laplace_dynamic",
        "general_family",
    ],
)
def test_resolve_ml_reml_scoring_backend_covers_owner_matrix(
    monkeypatch, family_kwargs, exact_ok, simple_ok, expected
):
    """Verify that resolve ML REML scoring backend covers owner matrix."""
    monkeypatch.setattr(
        ml_reml_module,
        "can_use_exact_gaussian_ml_reml",
        lambda model: exact_ok,
    )
    monkeypatch.setattr(
        ml_reml_module,
        "can_use_simple_ml_reml_structure",
        lambda model: simple_ok,
    )

    model = _stub_model(family_kwargs=family_kwargs)

    assert ml_reml_module.resolve_ml_reml_scoring_backend(model, method="reml") == (
        expected
    )


def test_criterion_ml_reml_keeps_finite_exact_gaussian_score(monkeypatch):
    """Verify that criterion ML REML keeps finite exact gaussian score."""
    calls: list[str] = []
    monkeypatch.setattr(
        ml_reml_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "gaussian_exact",
    )
    monkeypatch.setattr(
        ml_reml_module,
        "criterion_ml_reml_exact",
        lambda model, y, log_sp, method: 1.25,
    )
    monkeypatch.setattr(
        ml_reml_module,
        "criterion_ml_reml_exact_dynamic",
        lambda model, y, log_sp, method: calls.append("dynamic") or 9.0,
    )

    score = ml_reml_module.criterion_ml_reml(
        SimpleNamespace(),
        np.array([1.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        "REML",
    )

    assert score == pytest.approx(1.25, abs=0.0)
    assert calls == []


def test_criterion_ml_reml_falls_back_to_dynamic_when_exact_is_nonfinite(monkeypatch):
    """Verify that criterion ML REML falls back to dynamic when exact is nonfinite."""
    calls: list[str] = []
    monkeypatch.setattr(
        ml_reml_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "gaussian_exact",
    )
    monkeypatch.setattr(
        ml_reml_module,
        "criterion_ml_reml_exact",
        lambda model, y, log_sp, method: np.inf,
    )
    monkeypatch.setattr(
        ml_reml_module,
        "criterion_ml_reml_exact_dynamic",
        lambda model, y, log_sp, method: calls.append(method) or 2.75,
    )

    score = ml_reml_module.criterion_ml_reml(
        SimpleNamespace(),
        np.array([1.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        "ML",
    )

    assert score == pytest.approx(2.75, abs=0.0)
    assert calls == ["ML"]


def test_criterion_objective_caches_calls_and_merges_trace_rows(monkeypatch):
    """Verify that criterion objective caches calls and merges trace rows."""
    counts = {"fun": 0, "jac": 0, "hess": 0}

    monkeypatch.setattr(
        objectives_module,
        "criterion_value",
        lambda model, y, x, method: counts.__setitem__("fun", counts["fun"] + 1)
        or float(np.sum(x)),
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_gradient",
        lambda model, y, x, method: counts.__setitem__("jac", counts["jac"] + 1)
        or np.asarray(x, dtype=np.float64) + 1.0,
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_hessian",
        lambda model, y, x, method: counts.__setitem__("hess", counts["hess"] + 1)
        or np.eye(np.asarray(x, dtype=np.float64).size, dtype=np.float64) * 3.0,
    )

    obj = _CriterionObjective(
        model=SimpleNamespace(),
        y=np.array([1.0], dtype=np.float64),
        method="reml",
        use_gradient=True,
    )
    x = np.array([0.2, -0.1], dtype=np.float64)

    assert obj.fun(x) == pytest.approx(0.1, abs=0.0)
    np.testing.assert_allclose(obj.jac(x), np.array([1.2, 0.9], dtype=np.float64))
    np.testing.assert_allclose(obj.hess(x), np.eye(2, dtype=np.float64) * 3.0)

    assert obj.fun(x) == pytest.approx(0.1, abs=0.0)
    np.testing.assert_allclose(obj.jac(x), np.array([1.2, 0.9], dtype=np.float64))
    np.testing.assert_allclose(obj.hess(x), np.eye(2, dtype=np.float64) * 3.0)

    assert counts == {"fun": 1, "jac": 1, "hess": 1}
    assert len(obj.trace) == 1
    np.testing.assert_allclose(obj.trace[0]["x"], x)
    assert obj.trace[0]["fun"] == pytest.approx(0.1, abs=0.0)
    np.testing.assert_allclose(obj.trace[0]["grad"], np.array([1.2, 0.9]))
    np.testing.assert_allclose(obj.trace[0]["hess"], np.eye(2) * 3.0)


def test_criterion_objective_refreshes_general_family_score_after_derivatives(
    monkeypatch,
):
    """Verify that criterion objective refreshes general family score after derivatives."""
    model = SimpleNamespace(cached_fun=0.5)
    counts = {"fun": 0, "jac": 0, "hess": 0}

    monkeypatch.setattr(
        objectives_module,
        "resolve_ml_reml_scoring_backend",
        lambda model, method="reml": "general_family",
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_value",
        lambda model, y, x, method: counts.__setitem__("fun", counts["fun"] + 1)
        or float(model.cached_fun),
    )

    def _grad(model, y, x, method):
        counts["jac"] += 1
        model.cached_fun = 1.75
        return np.asarray(x, dtype=np.float64) + 2.0

    def _hess(model, y, x, method):
        counts["hess"] += 1
        model.cached_fun = 2.5
        return np.eye(np.asarray(x, dtype=np.float64).size, dtype=np.float64) * 4.0

    monkeypatch.setattr(objectives_module, "criterion_gradient", _grad)
    monkeypatch.setattr(objectives_module, "criterion_hessian", _hess)

    obj = _CriterionObjective(
        model=model,
        y=np.array([1.0], dtype=np.float64),
        method="ml",
        use_gradient=True,
    )
    x = np.array([0.3, -0.2], dtype=np.float64)

    assert obj.fun(x) == pytest.approx(0.5, abs=0.0)
    np.testing.assert_allclose(obj.jac(x), np.array([2.3, 1.8], dtype=np.float64))
    assert obj._last_fun == pytest.approx(1.75, abs=0.0)
    np.testing.assert_allclose(obj.hess(x), np.eye(2, dtype=np.float64) * 4.0)
    assert obj._last_fun == pytest.approx(2.5, abs=0.0)
    assert obj.trace[0]["fun"] == pytest.approx(2.5, abs=0.0)
    assert counts == {"fun": 3, "jac": 1, "hess": 1}


def test_gaussian_profiled_objective_uses_exact_derivative_terms(monkeypatch):
    """Verify that gaussian profiled objective uses exact derivative terms."""
    calls: dict[str, tuple[np.ndarray, str] | tuple[np.ndarray, str, float]] = {}

    def _profiled(model, y, x, method):
        calls["fun"] = (np.asarray(x, dtype=np.float64).copy(), method)
        return 1.5

    def _derivative_terms(model, y, x, method):
        calls["deriv"] = (np.asarray(x, dtype=np.float64).copy(), method)
        return {
            "valid": True,
            "grad": np.array([4.0, 5.0], dtype=np.float64),
            "hess": np.array([[7.0, 0.0], [0.0, 8.0]], dtype=np.float64),
        }

    monkeypatch.setattr(
        objectives_module,
        "criterion_ml_reml_gaussian_dynamic_profiled",
        _profiled,
    )
    monkeypatch.setattr(
        objectives_module,
        "_gaussian_dynamic_reml_derivative_terms",
        _derivative_terms,
    )

    obj = _GaussianRemlProfiledObjective(
        model=SimpleNamespace(),
        y=np.array([1.0], dtype=np.float64),
        branch_method="LAML",
    )
    x = np.array([0.4, -0.3], dtype=np.float64)

    assert obj.fun(x) == pytest.approx(1.5, abs=0.0)
    np.testing.assert_allclose(obj.jac(x), np.array([4.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(
        obj.hess(x),
        np.array([[7.0, 0.0], [0.0, 8.0]], dtype=np.float64),
    )
    obj.record_iter(x, accepted_step_norm=0.25)

    fun_x, fun_method = calls["fun"]
    deriv_x, deriv_method = calls["deriv"]
    np.testing.assert_allclose(fun_x, x)
    np.testing.assert_allclose(deriv_x, x)
    assert fun_method == "LAML"
    assert deriv_method == "LAML"
    assert obj.accepted_trace[-1]["accepted_step_norm"] == pytest.approx(0.25)


def test_joint_gamma_objective_splits_log_sp_and_log_scale(monkeypatch):
    """Verify that joint gamma objective splits log sp and log scale."""
    calls: dict[str, tuple[np.ndarray, float, str]] = {}

    def _gamma_fun(model, y, log_sp, log_scale, method):
        calls["fun"] = (
            np.asarray(log_sp, dtype=np.float64).copy(),
            float(log_scale),
            method,
        )
        return 2.0

    def _gamma_jac(model, y, log_sp, log_scale, method):
        calls["jac"] = (
            np.asarray(log_sp, dtype=np.float64).copy(),
            float(log_scale),
            method,
        )
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def _gamma_hess(model, y, log_sp, log_scale, method):
        calls["hess"] = (
            np.asarray(log_sp, dtype=np.float64).copy(),
            float(log_scale),
            method,
        )
        return np.eye(3, dtype=np.float64)

    monkeypatch.setattr(
        objectives_module,
        "criterion_ml_reml_pirls_gamma_joint",
        _gamma_fun,
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_gradient_ml_reml_pirls_gamma_joint",
        _gamma_jac,
    )
    monkeypatch.setattr(
        objectives_module,
        "criterion_hessian_ml_reml_pirls_gamma_joint",
        _gamma_hess,
    )

    obj = _JointGammaPirlsRemlObjective(
        model=SimpleNamespace(),
        y=np.array([1.0], dtype=np.float64),
        branch_method="REML",
    )
    x = np.array([0.1, 0.2, -0.4], dtype=np.float64)

    assert obj.fun(x) == pytest.approx(2.0, abs=0.0)
    np.testing.assert_allclose(obj.jac(x), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(obj.hess(x), np.eye(3, dtype=np.float64))
    for key in ("fun", "jac", "hess"):
        log_sp, log_scale, method = calls[key]
        np.testing.assert_allclose(log_sp, np.array([0.1, 0.2], dtype=np.float64))
        assert log_scale == pytest.approx(-0.4, abs=0.0)
        assert method == "REML"


def test_gaussian_reml_outer_newton_uses_joint_objective(monkeypatch):
    """Verify that gaussian REML outer newton uses joint objective."""
    captured: dict[str, object] = {}

    def _fake_newton(*, objective, x0, bounds, **kwargs):
        captured["objective_type"] = type(objective)
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["bounds"] = list(bounds)
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64).copy(),
            fun=0.0,
            success=True,
            status=0,
            message="captured",
            nit=0,
            nfev=1,
            njev=1,
            nhev=1,
            jac=np.zeros_like(np.asarray(x0, dtype=np.float64)),
            hess=np.eye(np.asarray(x0, dtype=np.float64).size, dtype=np.float64),
        )

    monkeypatch.setattr(
        driver_module,
        "optimize_outer_newton_indefinite_hessian",
        _fake_newton,
    )
    monkeypatch.setattr(
        driver_module, "_refresh_final_outer_derivatives", lambda *args, **kwargs: None
    )

    data = _make_gaussian_data(seed=52, n=80)
    model = GAM(
        formula='y ~ s(x0, bs="cr", k=8)',
        family="gaussian",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.array([1.0], dtype=np.float64),
    )
    model.fit(data=data)
    y = model.family.validate_y(model.y_)

    driver_module.optimize_smoothing_params(
        model,
        y,
        method="reml",
        optimizer="outer_newton",
    )

    assert captured["objective_type"] is _GaussianRemlJointObjective
    assert np.asarray(captured["x0"], dtype=np.float64).shape == (
        int(model.gam_result_.compiled_model.n_smoothing_params) + 1,
    )


def test_public_optim_alias_uses_lbfgsb_branch_for_reml(monkeypatch):
    """Verify that public optim alias uses lbfgsb branch for REML."""
    captured: dict[str, object] = {}

    def _fake_minimize(*, fun, x0, method, jac, bounds, options):
        captured["method"] = method
        captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
        captured["bounds"] = list(bounds)
        captured["options"] = dict(options)
        captured["jac"] = jac
        return OptimizeResult(
            x=np.asarray(x0, dtype=np.float64).copy(),
            fun=0.0,
            success=True,
            status=0,
            message="captured optim",
            nit=0,
            nfev=1,
            njev=1,
            jac=np.zeros_like(np.asarray(x0, dtype=np.float64)),
        )

    def _fail_outer_newton(*args, **kwargs):
        raise AssertionError("public optim alias should not route to outer_newton")

    monkeypatch.setattr(driver_module, "minimize", _fake_minimize)
    monkeypatch.setattr(
        driver_module,
        "optimize_outer_newton_indefinite_hessian",
        _fail_outer_newton,
    )

    rng = np.random.default_rng(14)
    x = rng.normal(size=60)
    mu = np.exp(0.2 + 0.3 * x)
    y_obs = rng.poisson(mu).astype(np.float64)
    data = pd.DataFrame({"y": y_obs, "x": x})

    model = GAM(
        formula='y ~ s(x, bs="cr", k=8)',
        family="poisson",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.array([1.0], dtype=np.float64),
    )
    model.fit(data=data)
    y = model.family.validate_y(model.y_)

    driver_module.optimize_smoothing_params(
        model,
        y,
        method="reml",
        optimizer="optim",
    )

    assert captured["method"] == "L-BFGS-B"
    assert np.asarray(captured["x0"], dtype=np.float64).shape == (
        int(model.gam_result_.compiled_model.n_smoothing_params),
    )
    assert callable(captured["jac"])
    assert int(captured["options"]["maxcor"]) == min(
        5, int(model.gam_result_.compiled_model.n_smoothing_params)
    )
    assert float(captured["options"]["ftol"]) == pytest.approx(
        float(np.finfo(np.float64).eps * 1e7),
        abs=0.0,
    )
