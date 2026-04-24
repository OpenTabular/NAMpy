"""Targeted mgcv parity checks across a representative GAM model matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._mgcv_parity_requested_shared import (
    CaseSpec,
    _assert_requested_parity,
    _fit_nampy_snapshot,
)
from tests.mgcv_parity_utils import (
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_gaussian_data_3col,
    _make_mrf_data,
    _make_poisson_data,
    _make_random_effect_data_noisy,
    _run_mgcv_snapshot,
)


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
    y = (
        np.sin(1.3 * x)
        + np.array([shifts[v] for v in f])
        + rng.normal(0.0, 0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "f": f})


def _data_binomial_separation() -> pd.DataFrame:
    rng = np.random.default_rng(109)
    n = 260
    x = rng.normal(size=n)
    eta = 8.0 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=n)
    return pd.DataFrame({"y": y, "x": x})


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
        case_id="binomial_separation",
        formula='y ~ s(x, bs="tp", k=12)',
        family="binomial",
        data_factory=_data_binomial_separation,
        se_tol_scale=3e-6,
    ),
    CaseSpec(
        case_id="gaussian_te",
        formula='y ~ te(x1, x2, bs=["cr", "cr"], k=[8, 8])',
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
        criterion_atol=5e-1,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_requested_mgcv_parity_models(case: CaseSpec):
    """
    Verify the requested end-to-end parity model matrix by comparing each configured
    NAMpy fit and output surface against mgcv.
    """
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
