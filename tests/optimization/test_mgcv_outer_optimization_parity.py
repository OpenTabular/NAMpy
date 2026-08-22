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
from nampy.gam.fit.backends import solve_fit
from nampy.gam.fit.design_setup import compile_designs
from nampy.gam.fit.selection.criteria.pirls.derivatives import (
    _gdi1_kernel,
    _serialize_pirls_postproc_derivatives,
)
from nampy.gam.fit.selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.fit.selection.optimize.newton import (
    _optimize_outer_newton_indefinite_hessian,
)
from nampy.gam.fit.selection.optimize.objectives import _CriterionObjective
from nampy.gam.fit.solvers.general_family import newton as general_newton
from nampy.gam.fit.state import assign_fit_solution
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.specs.modeling import prepare_formula_inputs
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _run_mgcv_snapshot,
)

R_SCRIPT = shutil.which("Rscript")
MGCV_OUTER_TRACE_SCRIPT = PARITY_DIR / "mgcv_outer_trace.R"

pytestmark = [
    pytest.mark.surface_trace,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]

_TRACE_SOURCE_ALIASES = {
    "outer_newton_strict": "outer_newton_strict",
    "mgcv_newton": "outer_newton_strict",
    "outer_bfgs_strict": "outer_bfgs_strict",
    "mgcv_bfgs": "outer_bfgs_strict",
    "outer_efs_strict": "outer_efs_strict",
    "mgcv_efs": "outer_efs_strict",
    "outer_optim_strict": "outer_optim_strict",
    "mgcv_optim": "outer_optim_strict",
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
    weights_column: str | None = None,
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
                weights_column or "",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _python_newton_edge_correct_state(
    data: pd.DataFrame, formula: str, family: str
):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    y = gam.family.validate_y(gam.y_)
    init = _initial_smoothing_params_from_design(gam, y)
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
        lo = float(np.log(lower_sp)) if lower_sp > 0.0 else -np.inf
        bounds.append((lo, np.inf))

    objective = _CriterionObjective(gam, y, method="reml", use_gradient=True)
    result = _optimize_outer_newton_indefinite_hessian(
        objective=objective,
        x0=x0,
        bounds=bounds,
        edge_correct=True,
    )
    return gam, result


def _python_newton_edge_correct_result(data: pd.DataFrame, formula: str, family: str):
    _gam, result = _python_newton_edge_correct_state(data, formula, family)
    return result


def _finalize_python_edge_correct_fit(
    data: pd.DataFrame, formula: str, family: str
):
    """Run the real final solve/post-process at an edge-corrected endpoint."""
    gam, result = _python_newton_edge_correct_state(data, formula, family)
    fixed_mask = (
        np.zeros_like(gam.smoothing_params, dtype=bool)
        if gam.smoothing_fixed_mask_ is None
        else np.asarray(gam.smoothing_fixed_mask_, dtype=bool)
    )
    endpoint_sp = np.asarray(gam.smoothing_params, dtype=np.float64).copy()
    endpoint_sp[~fixed_mask] = np.exp(np.asarray(result.x, dtype=np.float64))
    gam.smoothing_params = endpoint_sp
    gam._optim_method = "reml"
    gam._optim_result = result
    gam.smoothing_score_ = float(result.fun)
    sol = solve_fit(
        gam,
        gam.family.validate_y(gam.y_),
        endpoint_sp,
        weights=gam.prior_weights_,
    )
    assign_fit_solution(gam, sol)
    return gam, result


def _capture_vb_corr_calls(monkeypatch):
    """Capture the exact production calls to the Vb.corr port."""
    calls = []
    original = general_newton._vb_corr_root

    def wrapped(X_root, **kwargs):
        correction = original(X_root, **kwargs)
        calls.append(
            {
                "rho": np.asarray(kwargs["rho"], dtype=np.float64).copy(),
                "Vr": np.asarray(kwargs["Vr"], dtype=np.float64).copy(),
                "scale_estimated": bool(kwargs.get("scale_est", False)),
                "correction_unscaled": np.asarray(
                    correction, dtype=np.float64
                ).copy(),
            }
        )
        return correction

    monkeypatch.setattr(general_newton, "_vb_corr_root", wrapped)
    return calls


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


def _assert_expected_subset_close(
    actual,
    expected,
    *,
    atol: float,
    field_atols: dict[str, float] | None = None,
    field_name: str | None = None,
):
    actual = _normalize_jsonish(actual)
    expected = _normalize_jsonish(expected)
    effective_atol = (
        field_atols.get(field_name, atol)
        if field_atols is not None and field_name is not None
        else atol
    )

    if expected is None:
        assert actual is None
        return

    if isinstance(expected, dict):
        if len(expected) == 0:
            return
        assert isinstance(actual, dict)
        assert set(expected) <= set(actual)
        for key, expected_value in expected.items():
            _assert_expected_subset_close(
                actual.get(key),
                expected_value,
                atol=atol,
                field_atols=field_atols,
                field_name=str(key),
            )
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
                atol=effective_atol,
                rtol=0.0,
            )
            return
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_expected_subset_close(
                actual_value,
                expected_value,
                atol=atol,
                field_atols=field_atols,
                field_name=field_name,
            )
        return

    if isinstance(expected, bool):
        assert bool(actual) == expected
        return

    if isinstance(expected, int) and not isinstance(expected, bool):
        assert int(actual) == expected
        return

    if isinstance(expected, float):
        np.testing.assert_allclose(
            float(actual),
            expected,
            atol=effective_atol,
            rtol=0.0,
        )
        return

    if isinstance(expected, str):
        assert _TRACE_SOURCE_ALIASES.get(actual, actual) == _TRACE_SOURCE_ALIASES.get(
            expected, expected
        )
        return

    assert actual == expected


def _assert_trace_row_close(
    actual: dict,
    expected: dict,
    *,
    atol: float,
    field_atols: dict[str, float] | None = None,
):
    _assert_expected_subset_close(
        actual,
        expected,
        atol=atol,
        field_atols=field_atols,
    )


def _assert_bfgs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_efs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_joint_negbin_trace_row_close(actual: dict, expected: dict, *, atol: float):
    _assert_trace_row_close(actual, expected, atol=atol)


def _assert_trace_rows_close(
    actual_rows,
    expected_rows,
    *,
    atol: float,
    field_atols: dict[str, float] | None = None,
):
    assert len(actual_rows) == len(expected_rows) >= 1
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        _assert_trace_row_close(
            actual_row,
            expected_row,
            atol=atol,
            field_atols=field_atols,
        )


def _assert_serialized_trace_matches_mgcv(
    actual_serialized: dict,
    expected: dict,
    *,
    atol: float,
    sp_atol: float | None = None,
):
    trace_field_atols = None
    if sp_atol is not None:
        trace_field_atols = {
            "accepted_step_norm": sp_atol,
            "log_sp": sp_atol,
        }
    _assert_trace_rows_close(
        list(actual_serialized["trace"]),
        list(expected["trace"]),
        atol=atol,
        field_atols=trace_field_atols,
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


@pytest.mark.parametrize(
    (
        "family",
        "optimizer",
        "data_factory",
        "formula",
        "sp_atol",
        "fit_atol",
    ),
    [
        pytest.param(
            "gaussian",
            "bfgs",
            lambda: _make_gaussian_data(seed=127, n=140),
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
            3e-4,
            3e-5,
            id="gaussian_reml_bfgs",
        ),
        pytest.param(
            "gaussian",
            "efs",
            lambda: _make_gaussian_data(seed=127, n=140),
            'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
            3e-4,
            3e-5,
            id="gaussian_reml_efs",
        ),
        pytest.param(
            "binomial",
            "optim",
            lambda: _make_binomial_data(seed=461, n=160),
            'y ~ s(x0, bs="cr", k=8)',
            3e-4,
            3e-4,
            id="binomial_reml_optim",
        ),
    ],
)
def test_additional_optimizer_family_endpoints_match_mgcv_behaviorally(
    family,
    optimizer,
    data_factory,
    formula,
    sp_atol,
    fit_atol,
):
    """Cover missing optimizer/family cells without requiring identical paths."""
    data = data_factory()
    expected_trace = _run_mgcv_outer_trace(
        data, formula, family, "REML", optimizer
    )
    expected = _run_mgcv_snapshot(
        data=data,
        formula=formula,
        family=family,
        method="REML",
        optimizer=optimizer,
        allow_live_run=True,
    )
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data)
    actual_trace = build_optimizer_trace(gam)

    actual_rows = list(actual_trace["trace"])
    expected_rows = list(expected_trace["trace"])
    assert len(actual_rows) >= 1
    assert len(expected_rows) >= 1
    assert actual_trace["fit"]["message"] == expected_trace["fit"]["outer_info"][
        "conv"
    ]
    rank_info = dict(actual_rows[0]["rank_info"] or {})
    if optimizer == "bfgs":
        assert rank_info.get("line_search_alpha") is not None
    elif optimizer == "efs":
        assert rank_info.get("mult") is not None
    else:
        assert optimizer == "optim"
        assert int(rank_info.get("n_fun", 0)) >= 1
    # Compare the optimizer-owned endpoint without constraining the route or
    # number of accepted steps used to reach it.
    np.testing.assert_allclose(
        float(actual_rows[-1]["criterion"]),
        float(expected_rows[-1]["criterion"]),
        rtol=0.0,
        atol=fit_atol,
    )
    if family == "gaussian" and optimizer == "efs":
        np.testing.assert_allclose(
            float(actual_rows[-1]["log_scale"]),
            float(expected_rows[-1]["log_scale"]),
            rtol=0.0,
            atol=fit_atol,
        )
        permuted = data.sample(frac=1.0, random_state=90210).reset_index(drop=True)
        permuted_gam = GAM(
            family=family,
            formula=formula,
            optimize_smoothing=True,
            smoothing_method="REML",
            smoothing_optimizer=optimizer,
        ).fit(data=permuted)
        permuted_rows = list(build_optimizer_trace(permuted_gam)["trace"])
        # This is a behavioral invariant, with room for reduction-order noise.
        # The previously profiled criterion changed by O(10^2) under this
        # permutation even though the fitted GAM was unchanged.
        np.testing.assert_allclose(
            np.log(np.asarray(permuted_gam.smoothing_params, dtype=np.float64)),
            np.log(np.asarray(gam.smoothing_params, dtype=np.float64)),
            rtol=0.0,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            float(permuted_rows[-1]["criterion"]),
            float(actual_rows[-1]["criterion"]),
            rtol=0.0,
            atol=5e-7,
        )
        np.testing.assert_allclose(
            float(permuted_rows[-1]["log_scale"]),
            float(actual_rows[-1]["log_scale"]),
            rtol=0.0,
            atol=5e-7,
        )

    actual_fit = actual["fit"]
    expected_fit = expected["fit"]
    np.testing.assert_allclose(
        np.asarray(actual_fit["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected_fit["log_smoothing_params"], dtype=np.float64),
        rtol=0.0,
        atol=sp_atol,
    )
    for key in ("edf_total", "edf_by_term", "deviance"):
        np.testing.assert_allclose(
            np.asarray(actual_fit[key], dtype=np.float64),
            np.asarray(expected_fit[key], dtype=np.float64),
            rtol=0.0,
            atol=fit_atol,
        )

    for key in ("response", "link", "se_response", "se_link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            rtol=fit_atol,
            atol=fit_atol,
        )


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
        sp_atol=2e-5,
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


@pytest.mark.method_reml
def test_negbin_est_identity_outer_newton_trace_matches_mgcv_joint_theta():
    """Verify that negbin identity-link joint theta Newton trace matches mgcv."""
    data = _make_negbin_data(seed=910, n=220, theta=1.4)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {
        "name": "negbin",
        "theta": 1.4,
        "estimate_theta": True,
        "link": "identity",
    }

    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "negbin_est:1.4:identity",
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


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_unconditional_covariance_components_match_mgcv(monkeypatch):
    """Check Vb, Vc1, Vc2 and their derivative inputs independently."""
    data = _make_poisson_data(seed=806, n=160)
    formula = 'y ~ s(x0, bs="cr", k=7) + s(x1, bs="cr", k=7)'
    expected = _run_mgcv_outer_trace(
        data,
        formula,
        "poisson",
        "REML",
        "newton",
    )

    actual_calls = _capture_vb_corr_calls(monkeypatch)
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)

    expected_calls = list(expected["fit"]["vb_corr_calls"])
    expected_postproc = list(expected["fit"]["postproc_calls"])
    assert len(actual_calls) == len(expected_calls) == 1
    assert len(expected_postproc) == 1

    actual_call = actual_calls[0]
    expected_call = expected_calls[0]
    np.testing.assert_allclose(
        actual_call["rho"], expected_call["rho"], atol=5e-7, rtol=0.0
    )
    np.testing.assert_allclose(
        actual_call["Vr"], expected_call["Vr"], atol=5e-7, rtol=0.0
    )
    assert actual_call["scale_estimated"] == bool(
        expected_call["scale_estimated"]
    )

    fit_result = gam.gam_result_.fit_core_solution.fit_result
    actual_vb = np.asarray(fit_result.cov_bayes, dtype=np.float64)
    actual_vc = np.asarray(fit_result.cov_unconditional, dtype=np.float64)
    actual_vc2 = float(fit_result.scale) * np.asarray(
        actual_call["correction_unscaled"], dtype=np.float64
    )
    actual_vc1 = actual_vc - actual_vb - actual_vc2

    expected_vb = np.asarray(expected["fit"]["Vp"], dtype=np.float64)
    expected_vc = np.asarray(expected["fit"]["Vc"], dtype=np.float64)
    expected_vc2 = float(expected["fit"]["scale"]) * np.asarray(
        expected_call["correction_unscaled"], dtype=np.float64
    )
    expected_vc1 = expected_vc - expected_vb - expected_vc2

    for actual_component, expected_component in (
        (actual_vb, expected_vb),
        (actual_vc1, expected_vc1),
        (actual_vc2, expected_vc2),
        (actual_vc, expected_vc),
    ):
        np.testing.assert_allclose(
            actual_component,
            expected_component,
            atol=8e-7,
            rtol=2e-5,
        )

    kernel = _gdi1_kernel(
        gam,
        gam.y_,
        gam.gam_result_.fit_core_solution,
        np.asarray(gam.smoothing_params, dtype=np.float64),
        method="REML",
    )
    derivatives = _serialize_pirls_postproc_derivatives(kernel)
    np.testing.assert_allclose(
        np.asarray(derivatives["dbeta"], dtype=np.float64),
        np.asarray(expected_postproc[0]["db_drho"], dtype=np.float64),
        atol=8e-7,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(derivatives["dW_obs"], dtype=np.float64),
        np.asarray(expected_postproc[0]["dw_drho"], dtype=np.float64),
        atol=8e-7,
        rtol=2e-5,
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
    assert bool(actual.edge_correction_requested) is True
    assert bool(actual.edge_correction_applied) is True

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
    np.testing.assert_allclose(
        np.asarray(actual.db_drho1, dtype=np.float64),
        np.asarray(outer["db_drho1"], dtype=np.float64),
        atol=5e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual.dw_drho1, dtype=np.float64),
        np.asarray(outer["dw_drho1"], dtype=np.float64),
        atol=5e-7,
        rtol=0.0,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_edge_corrected_final_vc_and_fitted_edf2_match_mgcv(monkeypatch):
    """Edge derivatives replace final Vc while EDF2 stays at the fitted model."""
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

    actual_calls = _capture_vb_corr_calls(monkeypatch)
    gam, result = _finalize_python_edge_correct_fit(data, formula, "poisson")
    expected_calls = list(expected["fit"]["vb_corr_calls"])
    assert bool(result.edge_correction_applied) is True
    assert len(actual_calls) == len(expected_calls) == 2

    fit_result = gam.gam_result_.fit_core_solution.fit_result
    actual_vc = np.asarray(fit_result.cov_unconditional, dtype=np.float64)
    actual_edf2 = np.asarray(fit_result.edf2, dtype=np.float64)
    np.testing.assert_allclose(
        actual_vc,
        np.asarray(expected["fit"]["Vc"], dtype=np.float64),
        atol=8e-7,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        actual_edf2,
        np.asarray(expected["fit"]["edf2"], dtype=np.float64),
        atol=8e-7,
        rtol=2e-5,
    )

    scale = float(fit_result.scale)
    for actual_call, expected_call in zip(
        actual_calls, expected_calls, strict=True
    ):
        np.testing.assert_allclose(
            actual_call["rho"], expected_call["rho"], atol=5e-7, rtol=0.0
        )
        np.testing.assert_allclose(
            actual_call["Vr"], expected_call["Vr"], atol=8e-7, rtol=2e-5
        )
        np.testing.assert_allclose(
            scale * actual_call["correction_unscaled"],
            float(expected["fit"]["scale"])
            * np.asarray(expected_call["correction_unscaled"], dtype=np.float64),
            atol=8e-7,
            rtol=2e-5,
        )

    # The second Vb.corr call is the edge-corrected Vc2.  This assertion also
    # guards against accidentally retaining the fitted-model correction.
    assert not np.allclose(
        actual_calls[0]["correction_unscaled"],
        actual_calls[1]["correction_unscaled"],
        atol=1e-10,
        rtol=1e-8,
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
    trace_atol = 2e-5
    trace_field_atols = {"accepted_step_norm": 2e-1, "log_sp": 2e-1}

    # The REML criterion is essentially flat in the second log(sp) coordinate
    # near the optimum (gradient ~1e-7 while sp heads to effective infinity),
    # so L-BFGS-B's trailing line-search decisions hinge on last-bit gradient
    # differences between BLAS builds. The exact number of evaluations before
    # the FACTR*EPSMCH stop is therefore platform-dependent (observed 27 vs 25
    # on criterion values agreeing to ~1e-8). Require the same path row-by-row
    # up to the final row of the shorter trace, and only bounded length slack.
    assert min(len(actual_trace), len(expected_trace)) >= 2
    assert abs(len(actual_trace) - len(expected_trace)) <= 4
    n_common = min(len(actual_trace), len(expected_trace)) - 1
    for actual_row, expected_row in zip(
        actual_trace[:n_common], expected_trace[:n_common], strict=True
    ):
        _assert_trace_row_close(
            actual_row,
            expected_row,
            atol=trace_atol,
            field_atols=trace_field_atols,
        )

    np.testing.assert_allclose(
        float(actual_trace[-1]["criterion"]),
        float(expected_trace[-1]["criterion"]),
        atol=5e-6,
        rtol=0.0,
    )
    actual_end_sp = np.asarray(actual_trace[-1]["log_sp"], dtype=np.float64)
    expected_end_sp = np.asarray(expected_trace[-1]["log_sp"], dtype=np.float64)
    # A coordinate still climbing towards the +25 bound at convergence is
    # "effectively infinite" smoothing: its endpoint depends on how many flat
    # trailing steps the platform's line search took, so only require that both
    # implementations agree it is effectively infinite.
    effectively_infinite = expected_end_sp > 15.0
    np.testing.assert_allclose(
        actual_end_sp[~effectively_infinite],
        expected_end_sp[~effectively_infinite],
        atol=2e-1,
        rtol=0.0,
    )
    assert np.all(actual_end_sp[effectively_infinite] > 15.0)

    result = getattr(gam, "_optim_result", None)
    assert result is not None
    actual_outer = dict(getattr(result, "outer_info", {}) or {})
    expected_outer = expected["fit"]["outer_info"]
    assert actual_outer["conv"] == expected_outer["conv"]
    assert int(actual_outer["convergence"]) == int(expected_outer["convergence"])
    actual_counts = np.asarray(actual_outer["counts"], dtype=np.int64)
    expected_counts = np.asarray(expected_outer["counts"], dtype=np.int64)
    assert actual_counts.shape == expected_counts.shape
    # Evaluation counts inherit the platform-dependent trailing-step slack.
    assert np.all(np.abs(actual_counts - expected_counts) <= 4)
    assert "FACTR*EPSMCH" in str(actual_outer["message"])

    actual_serialized = build_optimizer_trace(gam)
    serialized_rows = list(actual_serialized["trace"])
    assert abs(len(serialized_rows) - len(expected_trace)) <= 4
    _assert_trace_rows_close(
        serialized_rows[:n_common],
        expected_trace[:n_common],
        atol=trace_atol,
        field_atols=trace_field_atols,
    )
    assert actual_serialized["fit"]["message"] == expected["fit"]["outer_info"]["conv"]
    actual_final_log_sp = np.log(
        np.asarray(actual_serialized["fit"]["smoothing_params"], dtype=np.float64)
    )
    expected_final_log_sp = np.log(
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
    )
    np.testing.assert_allclose(
        actual_final_log_sp[~effectively_infinite],
        expected_final_log_sp[~effectively_infinite],
        atol=2e-1,
        rtol=0.0,
    )
    assert np.all(actual_final_log_sp[effectively_infinite] > 15.0)
