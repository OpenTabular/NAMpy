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
from nampy.gam.model_state import _fit_workspace
from nampy.gam.parity import build_optimizer_trace
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.mgcv_parity_utils import _family_specs

R_SCRIPT = shutil.which("Rscript")
MGCV_TRACE_SCRIPT = PARITY_DIR / "mgcv_trace.R"
MGCV_NEGBIN_INNER_TRACE_SCRIPT = PARITY_DIR / "mgcv_negbin_inner_trace.R"


def _make_gaussian_data(seed=321, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.1 * x0) + 0.35 * x1**2 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_binomial_data(seed=456, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) - 0.45 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=n).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_poisson_data(seed=789, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_linked_id_univariate_data(seed=1501, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.8, 1.8, size=n)
    y = np.sin(1.1 * x0) + 0.4 * np.cos(0.8 * x1) + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_linked_id_cyclic_data(seed=1502, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x1 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    y = np.sin(x0) + 0.35 * np.cos(1.5 * x1) + rng.normal(scale=0.08, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


LINKED_ID_TRACE_CASES = [
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")',
        False,
        1e-12,
        1e-8,
        id="linked_cr",
    ),
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="ps", k=8, m=[2, 3], id="g")'
        ' + s(x1, bs="ps", k=8, m=[2, 3], id="g")',
        False,
        1e-12,
        1e-7,
        id="linked_ps_m_ordered",
    ),
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="tp", k=8, id="g") + s(x1, bs="tp", k=8, id="g")',
        False,
        1e-9,
        1e-6,
        id="linked_tp",
    ),
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="ts", k=8, id="g") + s(x1, bs="ts", k=8, id="g")',
        False,
        1e-2,
        5e-5,
        id="linked_ts",
    ),
    pytest.param(
        _make_linked_id_cyclic_data,
        'y ~ s(x0, bs="cc", k=6, id="g") + s(x1, bs="cc", k=6, id="g")',
        False,
        1e-11,
        1e-8,
        id="linked_cc",
    ),
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")',
        True,
        1e-12,
        1e-7,
        id="linked_cr_select_true",
    ),
    pytest.param(
        _make_linked_id_univariate_data,
        'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=8, id="g")',
        False,
        1e-12,
        1e-7,
        id="linked_cr_incompatible_k",
    ),
]


def _run_mgcv_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "trace.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_TRACE_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
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


def _fit_nampy_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    gam = GAM(
        family=family,
        formula=formula,
        select=select,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return build_optimizer_trace(gam)


def _fit_nampy_model_and_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    gam = GAM(
        family=family,
        formula=formula,
        select=select,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return gam, build_optimizer_trace(gam)


def _run_mgcv_negbin_inner_trace(
    data: pd.DataFrame,
    formula: str,
    family,
):
    _family_obj, family_token = _family_specs(family)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "trace.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_NEGBIN_INNER_TRACE_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family_token,
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_negbin_inner_trace(data: pd.DataFrame, formula: str, family):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    return list(_fit_workspace(gam).get("pirls_last_inner_trace", []) or []), gam


def _criterion_series(trace_obj):
    out = []
    for row in trace_obj.get("trace", []):
        c = row.get("criterion", None)
        if c is not None:
            out.append(float(c))
    return np.asarray(out, dtype=np.float64)


def _tail_criterion_series(trace_obj, n_tail: int):
    crit = _criterion_series(trace_obj)
    if crit.size <= n_tail:
        return crit
    return crit[-int(n_tail) :]


def _assert_strict_score_hist_exact(model, expected, *, atol=1e-12, rtol=0.0):
    expected_scores = np.asarray(
        expected["fit"]["outer_info"]["score_hist"], dtype=np.float64
    )
    actual_result = getattr(model, "_optim_result", None)

    assert actual_result is not None
    assert hasattr(actual_result, "strict_score_hist")

    actual_scores = np.asarray(actual_result.strict_score_hist, dtype=np.float64)

    assert actual_scores.shape == expected_scores.shape
    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=rtol,
        atol=atol,
    )
