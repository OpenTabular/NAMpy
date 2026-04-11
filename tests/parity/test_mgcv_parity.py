"""Targeted mgcv parity checks across 20 representative GAM models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_gaussian_data_3col,
    _make_mrf_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data_noisy,
    _make_sz_data_3x3,
    _run_mgcv_snapshot,
)

from nampy.gam import GAM


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    formula: str
    family: str | dict
    data_factory: callable
    select: bool = False
    weights_column: str | None = None
    # tp eigenvector signs are LAPACK-implementation-dependent; compare predictions instead.
    skip_coef_comparison: bool = False


def _rename_univariate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "x0" in out.columns:
        out = out.rename(columns={"x0": "x"})
    return out


def _data_gaussian_univariate() -> pd.DataFrame:
    return _rename_univariate(_make_gaussian_data(seed=101, n=240))[["y", "x"]]


def _data_binomial_univariate() -> pd.DataFrame:
    return _rename_univariate(_make_binomial_data(seed=102, n=260))[["y", "x"]]


def _data_poisson_univariate() -> pd.DataFrame:
    return _rename_univariate(_make_poisson_data(seed=103, n=260))[["y", "x"]]


def _data_gamma_univariate() -> pd.DataFrame:
    return _rename_univariate(_make_gamma_data(seed=104, n=260))[["y", "x"]]


def _data_gaussian_tensor() -> pd.DataFrame:
    # Use x0 as x1 and x1 as x2, ignoring x2, to avoid duplicate column names from rename.
    df = _make_gaussian_data_3col(seed=105, n=260)
    return pd.DataFrame({"y": df["y"], "x1": df["x0"], "x2": df["x1"]})


def _data_random_intercept() -> pd.DataFrame:
    return _make_random_effect_data_noisy(seed=106, n_draws=72, sigma=0.35).rename(
        columns={"f": "g"}
    )[["y", "g"]]


def _data_gaussian_by_factor() -> pd.DataFrame:
    rng = np.random.default_rng(107)
    n = 240
    x = rng.uniform(-2.0, 2.0, size=n)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    shifts = {"a": 0.6, "b": -0.35, "c": 0.1}
    y = np.sin(1.3 * x) + np.array([shifts[v] for v in f]) + rng.normal(
        0.0, 0.12, size=n
    )
    return pd.DataFrame({"y": y, "x": x, "f": f})


def _data_gaussian_fs_by_factor() -> pd.DataFrame:
    rng = np.random.default_rng(108)
    n = 240
    x = rng.uniform(-2.0, 2.0, size=n)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    y = (
        np.sin(1.2 * x)
        + np.array([{"a": 0.4, "b": -0.2, "c": 0.3}[v] for v in f])
        + rng.normal(0.0, 0.15, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "f": f})


def _data_binomial_separation() -> pd.DataFrame:
    rng = np.random.default_rng(109)
    n = 260
    x = rng.normal(size=n)
    y = (x > 0.0).astype(int)
    return pd.DataFrame({"y": y, "x": x})


def _data_gaussian_weights() -> pd.DataFrame:
    rng = np.random.default_rng(110)
    n = 240
    x = rng.uniform(-2.0, 2.0, size=n)
    w = rng.uniform(0.4, 2.5, size=n)
    y = np.sin(1.4 * x) + rng.normal(0.0, 0.18 / np.sqrt(w), size=n)
    return pd.DataFrame({"y": y, "x": x, "w": w})


def _data_gaussian_offset() -> pd.DataFrame:
    rng = np.random.default_rng(111)
    n = 240
    x = rng.uniform(-2.0, 2.0, size=n)
    off = 0.3 + 0.4 * np.cos(0.8 * x)
    y = off + np.sin(1.1 * x) + rng.normal(0.0, 0.12, size=n)
    return pd.DataFrame({"y": y, "x": x, "off": off})


def _data_mrf_lattice() -> pd.DataFrame:
    return _make_mrf_data().copy()


def _data_sz_interaction() -> pd.DataFrame:
    return _make_sz_data_3x3(seed=112).copy()


def _data_negbin_theta_estimation() -> pd.DataFrame:
    return _rename_univariate(_make_negbin_data(seed=113, n=300, theta=1.4))[["y", "x"]]


CASES: list[CaseSpec] = [
    CaseSpec(
        case_id="gaussian_cr",
        formula='y ~ s(x, bs="cr", k=10)',
        family="gaussian",
        data_factory=_data_gaussian_univariate,
    ),
    CaseSpec(
        case_id="gaussian_tp_k20",
        formula='y ~ s(x, bs="tp", k=20)',
        family="gaussian",
        data_factory=_data_gaussian_univariate,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="binomial_logit",
        formula='y ~ s(x, bs="tp", k=12)',
        family="binomial",
        data_factory=_data_binomial_univariate,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="binomial_probit",
        formula='y ~ s(x, bs="tp", k=12)',
        family={"name": "binomial", "link": "probit"},
        data_factory=_data_binomial_univariate,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="poisson",
        formula='y ~ s(x, bs="tp", k=12)',
        family="poisson",
        data_factory=_data_poisson_univariate,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="gamma_log",
        formula='y ~ s(x, bs="tp", k=12)',
        family="gamma",
        data_factory=_data_gamma_univariate,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="gaussian_te",
        formula='y ~ te(x1, x2, bs=["cr", "cr"], k=[8, 8])',
        family="gaussian",
        data_factory=_data_gaussian_tensor,
    ),
    CaseSpec(
        case_id="gaussian_ti_mc",
        formula='y ~ ti(x1, x2, bs=["cr", "cr"], k=[8, 8], mc=c(True, True))',
        family="gaussian",
        data_factory=_data_gaussian_tensor,
    ),
    CaseSpec(
        case_id="gaussian_t2_full_false",
        formula='y ~ t2(x1, x2, bs=["cr", "cr"], k=[8, 8], full=False)',
        family="gaussian",
        data_factory=_data_gaussian_tensor,
    ),
    CaseSpec(
        case_id="gaussian_random_intercept_re",
        formula='y ~ s(g, bs="re")',
        family="gaussian",
        data_factory=_data_random_intercept,
    ),
    CaseSpec(
        case_id="gaussian_by_factor",
        formula='y ~ f + s(x, by=f, bs="cr", k=10)',
        family="gaussian",
        data_factory=_data_gaussian_by_factor,
    ),
    CaseSpec(
        case_id="gaussian_fs_by_factor",
        formula='y ~ s(x, bs="fs", by=f, k=8)',
        family="gaussian",
        data_factory=_data_gaussian_fs_by_factor,
        skip_coef_comparison=True,
    ),
    CaseSpec(
        case_id="gaussian_select_true",
        formula='y ~ s(x, bs="cr", k=10)',
        family="gaussian",
        data_factory=_data_gaussian_univariate,
        select=True,
    ),
    CaseSpec(
        case_id="binomial_separation",
        formula='y ~ s(x, bs="tp", k=12)',
        family="binomial",
        data_factory=_data_binomial_separation,
    ),
    CaseSpec(
        case_id="gaussian_weights",
        formula='y ~ s(x, bs="cr", k=10)',
        family="gaussian",
        data_factory=_data_gaussian_weights,
        weights_column="w",
    ),
    CaseSpec(
        case_id="gaussian_formula_offset",
        formula='y ~ offset(off) + s(x, bs="cr", k=10)',
        family="gaussian",
        data_factory=_data_gaussian_offset,
    ),
    CaseSpec(
        case_id="mrf_lattice",
        formula='y ~ s(region, bs="mrf", k=3, xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))',
        family="gaussian",
        data_factory=_data_mrf_lattice,
    ),
    CaseSpec(
        case_id="gaussian_tensor_cr_tp",
        formula='y ~ te(x1, x2, bs=["cr", "tp"], k=[8, 8])',
        family="gaussian",
        data_factory=_data_gaussian_tensor,
    ),
    CaseSpec(
        case_id="factor_smooth_sz",
        formula='y ~ s(f1, f2, x, bs="sz", k=6)',
        family="gaussian",
        data_factory=_data_sz_interaction,
    ),
    CaseSpec(
        case_id="negbin_theta_estimation",
        formula='y ~ s(x, bs="tp", k=12)',
        family={"name": "negbin", "theta": 1.0, "estimate_theta": True},
        data_factory=_data_negbin_theta_estimation,
    ),
]


def _fit_nampy_snapshot(case: CaseSpec, data: pd.DataFrame):
    model = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    sample_weight = None
    if case.weights_column is not None:
        sample_weight = np.asarray(data[case.weights_column], dtype=np.float64)
    model.fit(data=data, sample_weight=sample_weight)
    return model.parity_snapshot(X=data, include_covariances=True)


def _assert_requested_parity(
    case: CaseSpec,
    actual_snapshot: dict,
    expected_snapshot: dict,
) -> None:
    if case.skip_coef_comparison:
        link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
        link_expected = np.asarray(
            expected_snapshot["predictions"]["link"], dtype=np.float64
        )
        link_tol = 1e-4 * (1.0 + np.abs(link_actual))
        link_err = np.abs(link_actual - link_expected)
        assert np.all(link_err <= link_tol), (
            f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
            f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
        )
    else:
        beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
        beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
        assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
        beta_tol = 1e-6 * (1.0 + np.abs(beta))
        beta_err = np.abs(beta - beta_mgcv)
        assert np.all(beta_err <= beta_tol), (
            f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
            f"max_tol={beta_tol.max():.3e}"
        )

    edf = float(actual_snapshot["fit"]["edf_total"])
    edf_mgcv = float(expected_snapshot["fit"]["edf_total"])
    assert abs(edf - edf_mgcv) < 1e-4, (
        f"{case.case_id}: |edf-edf_mgcv|={abs(edf - edf_mgcv):.3e} >= 1e-4"
    )

    reml = float(actual_snapshot["fit"]["criterion_value"])
    reml_mgcv = float(expected_snapshot["fit"]["criterion_value"])
    assert (
        abs(reml - reml_mgcv) < 1e-4
    ), f"{case.case_id}: |REML-REML_mgcv|={abs(reml - reml_mgcv):.3e} >= 1e-4"

    cov = np.asarray(actual_snapshot["fit"]["cov_bayes"], dtype=np.float64)
    cov_mgcv = np.asarray(expected_snapshot["fit"]["cov_bayes"], dtype=np.float64)
    assert cov.shape == cov_mgcv.shape, f"{case.case_id}: covariance shape mismatch"
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    se_mgcv = np.sqrt(np.clip(np.diag(cov_mgcv), 0.0, None))
    se_tol = 1e-6 * (1.0 + np.abs(se))
    se_err = np.abs(se - se_mgcv)
    assert np.all(se_err <= se_tol), (
        f"{case.case_id}: |se-se_mgcv| exceeded tolerance; max_err={se_err.max():.3e}, "
        f"max_tol={se_tol.max():.3e}"
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_requested_mgcv_parity_20_models(case: CaseSpec):
    data = case.data_factory()

    actual = _fit_nampy_snapshot(case, data)
    expected = _run_mgcv_snapshot(
        data=data,
        formula=case.formula,
        family=case.family,
        method="REML",
        select=case.select,
        weights_column=case.weights_column,
    )

    _assert_requested_parity(case, actual, expected)
