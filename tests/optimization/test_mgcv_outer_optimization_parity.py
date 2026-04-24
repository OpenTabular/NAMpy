from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.design_ops import compile_designs
from nampy.gam.fit.solvers.general_family_solver import build_general_family_setup_state
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_from_design_balance,
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.smoothing_selection.optimize.newton import (
    _optimize_outer_newton_indefinite_hessian,
)
from nampy.gam.smoothing_selection.optimize.objectives import _CriterionObjective
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state
from nampy.gam.specs.modeling import prepare_formula_inputs
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.families.test_general_family_mgcv_parity import _gaulss_two_smooth_data
from tests.mgcv_parity_utils import _make_gamma_data, _make_negbin_data

R_SCRIPT = shutil.which("Rscript")
MGCV_OUTER_TRACE_SCRIPT = PARITY_DIR / "mgcv_outer_trace.R"
MGCV_INITIAL_SPG_SCRIPT = PARITY_DIR / "mgcv_initial_spg.R"

pytestmark = [
    pytest.mark.surface_trace,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]

_TRACE_SOURCE_ALIASES = {
    "outer_newton_mgcv": "mgcv_newton",
    "outer_bfgs_mgcv": "mgcv_bfgs",
    "outer_efs_mgcv": "mgcv_efs",
}


def _make_poisson_data(seed=789, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _coerce_optional(value):
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 0:
        return None
    return value


def _coerce_array(value) -> np.ndarray | None:
    value = _coerce_optional(value)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr, dtype=np.float64)


def _run_mgcv_outer_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    optimizer: str,
    *,
    select: bool = False,
    edge_correct: bool = False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "trace.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_OUTER_TRACE_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family,
                method,
                optimizer,
                "true" if select else "false",
                "true" if edge_correct else "false",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_initial_spg(
    data: pd.DataFrame,
    formula,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "initial_spg.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_INITIAL_SPG_SCRIPT),
                str(csv_path),
                str(json_path),
                str(formula),
                family,
                method,
                "true" if select else "false",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _python_newton_edge_correct_result(data: pd.DataFrame, formula: str, family: str):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    y = gam.family.validate_y(gam.y_)
    init = _initial_smoothing_params_from_design_balance(gam, y)
    assert init is not None

    fixed_mask = (
        np.zeros(init.shape[0], dtype=bool)
        if gam.smoothing_fixed_mask_ is None
        else np.asarray(gam.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    x0 = np.log(np.maximum(np.asarray(init[free_mask], dtype=np.float64), 1e-300))

    min_sp = (
        np.zeros_like(init, dtype=np.float64)
        if gam.min_sp_ is None
        else np.asarray(gam.min_sp_, dtype=np.float64)
    )
    bounds = []
    for lower_sp in min_sp[free_mask]:
        lo = (
            float(gam.sp_log_bounds[0])
            if lower_sp <= 0.0
            else max(float(gam.sp_log_bounds[0]), float(np.log(lower_sp)))
        )
        bounds.append((lo, float(gam.sp_log_bounds[1])))

    objective = _CriterionObjective(gam, y, method="reml", use_gradient=True)
    return _optimize_outer_newton_indefinite_hessian(
        objective=objective,
        x0=x0,
        bounds=bounds,
        edge_correct=True,
    )


def _compile_optimization_state(data: pd.DataFrame, formula, family: str, method: str):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    (
        parsed,
        predictor_specs,
        X_np,
        feature_names,
        y_out,
        _used_cols,
        offset_formula,
        preprocess_state,
    ) = prepare_formula_inputs(
        gam,
        data=data,
        formula=formula,
        y=None,
        knots=gam.knots,
        drop_intercept=gam.drop_intercept,
    )
    gam.formula_ = parsed
    gam.formula_mode_ = True
    gam.formula_response_name_ = parsed.response_name
    gam.formula_preprocess_state_ = preprocess_state
    gam.predictor_specs = predictor_specs
    gam.fit_intercept = bool(parsed.predictors[0].intercept)
    gam.X_ = X_np
    gam.feature_names = list(feature_names)
    gam.y_ = gam.family.validate_y(y_out)
    gam.offset_train_ = offset_formula
    gam.n_samples_ = X_np.shape[0]
    gam.prior_weights_ = None
    compile_designs(gam, X_np, gam.feature_names)
    return gam


def _normalize_jsonish(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _normalize_jsonish(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonish(val) for val in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_jsonish(value.item())
        return [_normalize_jsonish(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_jsonish(value.item())
    return value


def _assert_expected_subset_close(actual, expected, *, atol: float):
    actual = _normalize_jsonish(actual)
    expected = _normalize_jsonish(expected)

    if expected is None:
        assert actual is None
        return

    if isinstance(expected, dict):
        if len(expected) == 0:
            return
        assert isinstance(actual, dict)
        assert set(expected) <= set(actual)
        for key, expected_value in expected.items():
            _assert_expected_subset_close(actual.get(key), expected_value, atol=atol)
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        try:
            actual_arr = np.asarray(actual, dtype=np.float64)
            expected_arr = np.asarray(expected, dtype=np.float64)
        except (TypeError, ValueError):
            actual_arr = None
            expected_arr = None
        if (
            actual_arr is not None
            and expected_arr is not None
            and actual_arr.shape == expected_arr.shape
        ):
            np.testing.assert_allclose(
                actual_arr,
                expected_arr,
                atol=atol,
                rtol=0.0,
            )
            return
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_expected_subset_close(actual_value, expected_value, atol=atol)
        return

    if isinstance(expected, bool):
        assert bool(actual) == expected
        return

    if isinstance(expected, int) and not isinstance(expected, bool):
        assert int(actual) == expected
        return

    if isinstance(expected, float):
        np.testing.assert_allclose(float(actual), expected, atol=atol, rtol=0.0)
        return

    if isinstance(expected, str):
        assert _TRACE_SOURCE_ALIASES.get(actual, actual) == _TRACE_SOURCE_ALIASES.get(
            expected, expected
        )
        return

    assert actual == expected


def _assert_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_expected_subset_close(actual, expected, atol=atol)


def _assert_bfgs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_efs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_optim_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_joint_negbin_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_trace_rows_close(actual_rows, expected_rows, *, atol: float):
    assert len(actual_rows) == len(expected_rows) >= 1
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        _assert_trace_row_close(actual_row, expected_row, atol=atol)


def _assert_serialized_trace_matches_mgcv(
    actual_serialized: dict,
    expected: dict,
    *,
    atol: float,
    sp_atol: float | None = None,
):
    _assert_trace_rows_close(
        list(actual_serialized["trace"]),
        list(expected["trace"]),
        atol=atol,
    )
    assert actual_serialized["fit"]["message"] == expected["fit"]["outer_info"]["conv"]
    np.testing.assert_allclose(
        np.log(
            np.asarray(actual_serialized["fit"]["smoothing_params"], dtype=np.float64)
        ),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=atol if sp_atol is None else sp_atol,
        rtol=0.0,
    )
    _assert_expected_subset_close(
        actual_serialized["fit"]["outer_info"],
        expected["fit"]["outer_info"],
        atol=atol,
    )


def _assert_root_gram_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual.T @ actual,
        expected.T @ expected,
        rtol=0.0,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("family", "method", "optimizer", "data_factory"),
    [
        ("poisson", "REML", "newton", _make_poisson_data),
        ("poisson", "REML", "bfgs", _make_poisson_data),
        ("poisson", "GCV.Cp", "optim", _make_poisson_data),
        ("poisson", "REML", "efs", _make_poisson_data),
    ],
    ids=["newton", "bfgs", "optim", "efs"],
)
def test_mgcv_outer_trace_harness_supports_requested_methods(
    family, method, optimizer, data_factory
):
    """Verify that mgcv outer trace harness supports requested methods."""
    data = data_factory(n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    payload = _run_mgcv_outer_trace(data, formula, family, method, optimizer)

    outer = payload["fit"]["outer_info"]
    assert outer["optimizer"] == optimizer
    assert outer["conv"] is not None
    if _coerce_optional(outer["iter"]) is not None:
        assert int(outer["iter"]) >= 1
    elif optimizer == "optim":
        counts = _coerce_array(outer["counts"])
        assert counts is not None
        assert counts.size >= 1
    assert isinstance(payload["trace"], list)
    assert len(payload["trace"]) >= 1

    row0 = payload["trace"][0]
    assert set(row0.keys()) >= {
        "iter",
        "log_sp",
        "criterion",
        "accepted_step_norm",
        "rank_info",
    }
    assert len(row0["log_sp"]) == 2
    assert np.isfinite(float(row0["criterion"]))
    if optimizer == "newton":
        assert _coerce_array(row0["gradient"]) is not None
        assert _coerce_array(row0["hessian"]) is not None
    if optimizer == "bfgs":
        assert row0["rank_info"]["line_search_alpha"] is not None
    if optimizer == "optim":
        assert row0["rank_info"]["n_fun"] >= 1
    if optimizer == "efs":
        assert row0["rank_info"]["mult"] is not None


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_newton_trace_matches_mgcv():
    """Verify that poisson outer newton trace matches mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "newton")
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=5e-7)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["converged"] is True
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=5e-7)


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_newton_indefinite_step_matches_mgcv():
    """Verify that the indefinite-Hessian fallback matches mgcv's REML step."""
    data = _make_poisson_data(seed=23, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "newton")
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=5e-7)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["converged"] is True
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=5e-7)


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_bfgs_trace_matches_mgcv():
    """Verify that poisson outer BFGS trace matches mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "bfgs")
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=2e-5)

    actual_serialized = build_optimizer_trace(gam)
    _assert_serialized_trace_matches_mgcv(
        actual_serialized,
        expected,
        atol=2e-5,
        sp_atol=1e-5,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_efs_trace_matches_mgcv():
    """Verify that poisson outer EFS trace matches mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "efs")
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=2e-5)

    actual_serialized = build_optimizer_trace(gam)
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=2e-5)


@pytest.mark.method_reml
@pytest.mark.family_gaulss
def test_gaulss_outer_efs_trace_matches_mgcv():
    """Verify that gaulss outer EFS trace matches mgcv."""
    data = _gaulss_two_smooth_data(seed=33, n=140)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]

    expected = _run_mgcv_outer_trace(data, str(formula), "gaulss", "REML", "efs")
    gam = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=5e-6)

    actual_outer = dict(getattr(gam._optim_result, "outer_info", {}) or {})
    expected_outer = expected["fit"]["outer_info"]
    _assert_expected_subset_close(actual_outer, expected_outer, atol=5e-6)

    actual_serialized = build_optimizer_trace(gam)
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=5e-6)


@pytest.mark.method_ml
@pytest.mark.family_gaulss
def test_gaulss_initial_spg_matches_mgcv_ml():
    """Verify that gaulss initial.spg matches mgcv under ML."""
    data = _gaulss_two_smooth_data(seed=33, n=140)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_initial_spg(data, formula, "gaulss", "ML")

    gam = _compile_optimization_state(data, formula, "gaulss", "ML")
    y = np.asarray(gam.y_, dtype=np.float64)
    actual = _initial_smoothing_params_mgcv_style(gam, y)

    assert actual is not None
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected["initial_sp"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.method_ml
@pytest.mark.family_gaulss
def test_gaulss_initial_spg_start_and_lbb_match_mgcv_ml():
    """Verify that gaulss initial.spg start and lbb match mgcv under ML."""
    data = _gaulss_two_smooth_data(seed=33, n=140)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_initial_spg(data, formula, "gaulss", "ML")

    gam = _compile_optimization_state(data, formula, "gaulss", "ML")
    y = np.asarray(gam.y_, dtype=np.float64)
    n_sp = int(np.asarray(gam.smoothing_params, dtype=np.float64).size)
    setup = build_general_family_setup_state(
        gam,
        np.ones(n_sp, dtype=np.float64),
        score_type="ML",
    )
    exact_setup = build_estimate_gam_setup_state(gam)
    weights = (
        np.ones_like(y, dtype=np.float64)
        if gam.prior_weights_ is None
        else np.asarray(gam.prior_weights_, dtype=np.float64)
    )
    actual_start = np.asarray(
        gam.family.initialize(
            y,
            setup.X_initial,
            setup.jj,
            offset=setup.offset_list,
            weights=weights,
            E=exact_setup.Eb,
        ),
        dtype=np.float64,
    )
    actual_lbb = np.asarray(
        gam.family.ll(
            y,
            setup.X_initial,
            setup.jj,
            actual_start,
            weights,
            offset=setup.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        np.asarray(setup.X_initial, dtype=np.float64),
        np.asarray(expected["X_initial"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )
    _assert_root_gram_equal(
        np.asarray(exact_setup.Eb, dtype=np.float64),
        np.asarray(expected["Eb"], dtype=np.float64),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        actual_start,
        np.asarray(expected["start"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual_lbb,
        np.asarray(expected["lbb"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.method_ml
@pytest.mark.family_gaulss
def test_gaulss_outer_newton_trace_matches_mgcv_ml():
    """Verify that gaulss outer newton trace matches mgcv under ML."""
    data = _gaulss_two_smooth_data(seed=33, n=140)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]

    expected = _run_mgcv_outer_trace(data, str(formula), "gaulss", "ML", "newton")
    gam = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="ML",
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=1e-6)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["converged"] is True
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=1e-6)


@pytest.mark.method_reml
def test_gamma_outer_newton_joint_scale_history_matches_mgcv():
    """Verify that gamma outer newton joint scale history matches mgcv."""
    data = _make_gamma_data(seed=123, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(data, formula, "gamma", "REML", "newton")
    gam = GAM(
        family="gamma",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    result = getattr(gam, "_optim_result", None)
    assert result is not None
    assert bool(getattr(result, "joint_gamma_reml_outer", False)) is True

    actual_outer = dict(getattr(result, "outer_info", {}) or {})
    expected_outer = expected["fit"]["outer_info"]
    np.testing.assert_allclose(
        np.asarray(actual_outer["score_hist"], dtype=np.float64),
        np.asarray(expected_outer["score_hist"], dtype=np.float64),
        atol=5e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(result.joint_log_phi),
        float(expected["trace"][-1]["log_scale"]),
        atol=5e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.log(np.asarray(gam.smoothing_params, dtype=np.float64)),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=5e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.log(float(gam._gamma_reml_phi_opt_)),
        float(expected["trace"][-1]["log_scale"]),
        atol=5e-7,
        rtol=0.0,
    )


@pytest.mark.method_reml
def test_negbin_est_outer_newton_trace_matches_mgcv_joint_theta():
    """Verify that negative-binomial est outer newton trace matches mgcv joint theta."""
    data = _make_negbin_data(seed=93, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 1.8, "estimate_theta": True}

    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "negbin_est:1.8",
        "REML",
        "newton",
    )
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    actual_serialized = build_optimizer_trace(gam)
    actual_trace = list(actual_serialized["trace"])
    expected_trace = list(expected["trace"])

    _assert_trace_rows_close(actual_trace, expected_trace, atol=2e-5)
    _assert_serialized_trace_matches_mgcv(actual_serialized, expected, atol=2e-5)
    np.testing.assert_allclose(
        float(np.log(gam.family.theta)),
        float(expected_trace[-1]["log_theta"]),
        atol=2e-5,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6], fx=[True, False])',
        'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6], fx=[True, False], mc=[True, False])',
    ],
    ids=["te_fx_vector", "ti_fx_vector_mc"],
)
def test_poisson_tensor_vector_fx_outer_newton_trace_matches_mgcv(formula):
    """Verify that poisson tensor vector fx outer newton trace matches mgcv."""
    data = _make_poisson_data(seed=904, n=220)
    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "newton")

    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    )
    gam.fit(data=data)
    actual_serialized = build_optimizer_trace(gam)

    _assert_serialized_trace_matches_mgcv(
        actual_serialized,
        expected,
        atol=5e-6,
        sp_atol=2e-5,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_newton_edge_correction_matches_mgcv():
    """Verify that poisson outer newton edge correction matches mgcv."""
    data = _make_poisson_data(seed=789, n=180)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "poisson",
        "REML",
        "newton",
        edge_correct=True,
    )
    actual = _python_newton_edge_correct_result(data, formula, "poisson")

    outer = expected["fit"]["outer_info"]
    assert bool(outer["edge_correct"]) is True
    assert bool(actual.mgcv_edge_correct) is True
    assert bool(actual.mgcv_edge_correct_applied) is True

    np.testing.assert_allclose(
        np.asarray(actual.lsp1, dtype=np.float64),
        np.asarray(outer["lsp1"], dtype=np.float64),
        atol=5e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual.hess1, dtype=np.float64),
        np.asarray(outer["hess1"], dtype=np.float64),
        atol=5e-7,
        rtol=0.0,
    )
    assert actual.db_drho1 is not None
    assert actual.dw_drho1 is not None
    assert np.asarray(actual.db_drho1, dtype=np.float64).ndim == 2
    assert np.asarray(actual.db_drho1, dtype=np.float64).shape[1] == actual.x.size
    assert np.asarray(actual.dw_drho1, dtype=np.float64).shape == (
        len(data),
        actual.x.size,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_optim_endpoint_and_metadata_match_mgcv():
    """Verify that poisson outer optim endpoint and metadata match mgcv."""
    data = _make_poisson_data(seed=789, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    expected = _run_mgcv_outer_trace(data, formula, "poisson", "REML", "optim")
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="optim",
        sp_log_bounds=(-80.0, 25.0),
    )
    gam.fit(data=data)

    actual_trace = list(getattr(gam, "_optim_trace", []) or [])
    expected_trace = list(expected["trace"])

    assert len(actual_trace) == len(expected_trace) >= 1
    _assert_optim_trace_row_close(actual_trace[0], expected_trace[0], atol=5e-7)
    np.testing.assert_allclose(
        np.asarray(actual_trace[-1]["log_sp"], dtype=np.float64),
        np.asarray(expected_trace[-1]["log_sp"], dtype=np.float64),
        atol=2e-1,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual_trace[-1]["criterion"]),
        float(expected_trace[-1]["criterion"]),
        atol=5e-7,
        rtol=0.0,
    )

    result = getattr(gam, "_optim_result", None)
    assert result is not None
    actual_outer = dict(getattr(result, "outer_info", {}) or {})
    expected_outer = expected["fit"]["outer_info"]
    assert actual_outer["conv"] == expected_outer["conv"]
    assert int(actual_outer["convergence"]) == int(expected_outer["convergence"])
    np.testing.assert_array_equal(
        np.asarray(actual_outer["counts"], dtype=np.int64),
        np.asarray(expected_outer["counts"], dtype=np.int64),
    )
    assert "FACTR*EPSMCH" in str(actual_outer["message"])

    actual_serialized = build_optimizer_trace(gam)
    _assert_serialized_trace_matches_mgcv(
        actual_serialized,
        expected,
        atol=5e-7,
        sp_atol=2e-1,
    )
