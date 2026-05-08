"""Historically noisy requested parity scenarios kept visible in one place.

Model-fit parity for most cases in this module is now green; downstream output /
diagnostic gaps are tracked separately. Keep this module strict so regressions
in the core fit surface become loud again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._mgcv_parity_requested_shared import (
    CaseSpec,
    _assert_requested_parity,
)
from tests._mgcv_parity_requested_shared import (
    _fit_nampy_snapshot as _fit_requested_nampy_snapshot,
)
from tests._mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_snapshot,
    _make_gaussian_data_3col,
    _make_negbin_data,
    _make_sz_data_3x3,
    _run_mgcv_snapshot,
)


def _rename_univariate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "x0" in out.columns:
        out = out.rename(columns={"x0": "x"})
    return out


def _data_gaussian_tensor() -> pd.DataFrame:
    df = _make_gaussian_data_3col(seed=105, n=50)
    return pd.DataFrame({"y": df["y"], "x1": df["x0"], "x2": df["x1"]})


def _data_sz_interaction() -> pd.DataFrame:
    return _make_sz_data_3x3(seed=112).copy()


def _data_negbin_theta_estimation() -> pd.DataFrame:
    return _rename_univariate(_make_negbin_data(seed=113, n=300, theta=1.4))[["y", "x"]]


GAUSSIAN_TI_MC_CASE = CaseSpec(
    case_id="gaussian_ti_mc",
    formula='y ~ ti(x1, x2, bs=["cr", "cr"], k=[4, 4], mc=c(True, True))',
    family="gaussian",
    data_factory=_data_gaussian_tensor,
)


REQUESTED_PARITY_TRACKED_MODEL_CASES: list[CaseSpec] = [
    CaseSpec(
        case_id="gaussian_te_full_false",
        formula='y ~ te(x1, x2, bs=["cr", "cr"], k=[8, 8])',
        family="gaussian",
        data_factory=_data_gaussian_tensor,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="factor_smooth_sz",
        formula='y ~ s(f1, f2, x, bs="sz", k=6)',
        family="gaussian",
        data_factory=_data_sz_interaction,
        # Exact `sz` 3x3 fit is underdetermined (`n < p`): constructor surface,
        # predictions and criterion match mgcv tightly, but the raw
        # coefficient/covariance representative still drifts on the covariance
        # surface that also drives the explicit unconditional-SE xfails tracked
        # elsewhere.
        skip_coef_comparison=True,
        se_tol_scale=2e-3,
    ),
]

# Backward-compatible alias for existing imports.
REQUESTED_PARITY_FAILING_OR_WARNING_CASES = REQUESTED_PARITY_TRACKED_MODEL_CASES


def _parametrize_requested_parity_failing_cases():
    for c in REQUESTED_PARITY_TRACKED_MODEL_CASES:
        yield c


@pytest.mark.parametrize(
    "case",
    _parametrize_requested_parity_failing_cases(),
    ids=[c.case_id for c in REQUESTED_PARITY_TRACKED_MODEL_CASES],
)
def test_requested_mgcv_parity_tracked_model_cases(case: CaseSpec):
    """
    Verify the tracked failing-or-warning parity cases so they remain isolated and
    explicitly documented against mgcv.
    """
    data = case.data_factory()
    actual = _fit_requested_nampy_snapshot(case, data)
    expected = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )
    _assert_requested_parity(case, actual, expected)


def test_gaussian_ti_mc_reml_matches_mgcv():
    """Tracked parity coverage verifying that gaussian ti mc REML matches mgcv."""
    data = GAUSSIAN_TI_MC_CASE.data_factory()
    actual = _fit_requested_nampy_snapshot(GAUSSIAN_TI_MC_CASE, data)
    expected = _run_mgcv_snapshot(
        data=data,
        formula=GAUSSIAN_TI_MC_CASE.formula,
        family=GAUSSIAN_TI_MC_CASE.family,
        method="REML",
        select=GAUSSIAN_TI_MC_CASE.select,
        weights_column=GAUSSIAN_TI_MC_CASE.weights_column,
    )
    _assert_requested_parity(GAUSSIAN_TI_MC_CASE, actual, expected)


class TestAdditionalScenarioParityFailingOrWarning:
    test_gaussian_fs_select_reml_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_fs_select_reml_matches_mgcv
    )


def test_negbin_theta_estimation_reml_matches_mgcv():
    """
    Tracked parity coverage verifying that negative-binomial theta estimation REML
    matches mgcv.
    """
    data = _data_negbin_theta_estimation()
    formula = 'y ~ s(x, bs="tp", k=12)'
    family = {"name": "negbin", "theta": 1.0, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=5e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["family_theta"], dtype=np.float64),
        np.asarray(expected["fit"]["family_theta"], dtype=np.float64),
        atol=5e-5,
        rtol=0.0,
    )


def test_strict_factor_by_link_parity():
    """Tracked parity coverage verifying that strict factor by link parity."""
    rng = np.random.default_rng(31)
    n = 80
    x = rng.normal(size=n)
    fac = np.array(["p", "q"] * (n // 2), dtype=object)
    y = np.sin(x) + 0.4 * (fac == "q").astype(float) + rng.normal(0, 0.15, n)
    data = pd.DataFrame({"y": y, "x": x, "fac": fac})
    formula = 'y ~ s(x, by=fac, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=5e-10,
        rtol=0.0,
    )
