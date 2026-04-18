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
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_from_design_balance,
)
from nampy.gam.smoothing_selection.optimize.objectives import _CriterionObjective
from nampy.gam.smoothing_selection.optimize.outer import (
    _optimize_outer_newton_indefinite_hessian,
)
from tests._paths import PARITY_DIR, REPO_ROOT

R_SCRIPT = shutil.which("Rscript")
MGCV_OUTER_TRACE_SCRIPT = PARITY_DIR / "mgcv_outer_trace.R"

pytestmark = [
    pytest.mark.surface_trace,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]


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


def _assert_trace_row_close(actual: dict, expected: dict, *, atol: float):
    np.testing.assert_allclose(
        np.asarray(actual["log_sp"], dtype=np.float64),
        np.asarray(expected["log_sp"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["criterion"]),
        float(expected["criterion"]),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["gradient"], dtype=np.float64),
        np.asarray(expected["gradient"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["hessian"], dtype=np.float64),
        np.asarray(expected["hessian"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    assert (
        actual["rank_info"]["step_halving_count"]
        == expected["rank_info"]["step_halving_count"]
    )
    assert bool(actual["rank_info"]["converged_here"]) == bool(
        expected["rank_info"]["converged_here"]
    )


def _assert_bfgs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    np.testing.assert_allclose(
        np.asarray(actual["log_sp"], dtype=np.float64),
        np.asarray(expected["log_sp"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["criterion"]),
        float(expected["criterion"]),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["gradient"], dtype=np.float64),
        np.asarray(expected["gradient"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["rank_info"]["line_search_alpha"]),
        float(expected["rank_info"]["line_search_alpha"]),
        atol=atol,
        rtol=0.0,
    )
    assert bool(actual["rank_info"]["converged_here"]) == bool(
        expected["rank_info"]["converged_here"]
    )


def _assert_efs_trace_row_close(actual: dict, expected: dict, *, atol: float):
    np.testing.assert_allclose(
        np.asarray(actual["log_sp"], dtype=np.float64),
        np.asarray(expected["log_sp"], dtype=np.float64),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["criterion"]),
        float(expected["criterion"]),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["rank_info"]["mult"]),
        float(expected["rank_info"]["mult"]),
        atol=atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(actual["rank_info"]["max_step"]),
        float(expected["rank_info"]["max_step"]),
        atol=atol,
        rtol=0.0,
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

    assert len(actual_trace) == len(expected_trace) >= 1
    for actual_row, expected_row in zip(actual_trace, expected_trace):
        _assert_trace_row_close(actual_row, expected_row, atol=5e-7)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["converged"] is True
    assert actual_serialized["fit"]["message"] == expected["fit"]["outer_info"]["conv"]
    np.testing.assert_allclose(
        np.log(
            np.asarray(actual_serialized["fit"]["smoothing_params"], dtype=np.float64)
        ),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=5e-7,
        rtol=0.0,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_bfgs_trace_matches_mgcv():
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

    assert len(actual_trace) == len(expected_trace) >= 1
    for actual_row, expected_row in zip(actual_trace, expected_trace):
        _assert_bfgs_trace_row_close(actual_row, expected_row, atol=2e-5)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["message"] == expected["fit"]["outer_info"]["conv"]
    np.testing.assert_allclose(
        np.log(
            np.asarray(actual_serialized["fit"]["smoothing_params"], dtype=np.float64)
        ),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_efs_trace_matches_mgcv():
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

    assert len(actual_trace) == len(expected_trace) >= 1
    for actual_row, expected_row in zip(actual_trace, expected_trace):
        _assert_efs_trace_row_close(actual_row, expected_row, atol=2e-5)

    actual_serialized = build_optimizer_trace(gam)
    assert actual_serialized["fit"]["message"] == expected["fit"]["outer_info"]["conv"]
    np.testing.assert_allclose(
        np.log(
            np.asarray(actual_serialized["fit"]["smoothing_params"], dtype=np.float64)
        ),
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        atol=2e-5,
        rtol=0.0,
    )


@pytest.mark.method_reml
@pytest.mark.family_poisson
def test_poisson_outer_newton_edge_correction_matches_mgcv():
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


@pytest.mark.method_reml
@pytest.mark.family_poisson
@pytest.mark.parametrize("optimizer", ["optim"])
def test_requested_outer_optimizers_raise_explicitly_in_python(optimizer):
    data = _make_poisson_data(seed=789, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    gam = GAM(
        family="poisson",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
    )

    with pytest.raises(NotImplementedError, match="smoothing_optimizer"):
        gam.fit(data=data)
