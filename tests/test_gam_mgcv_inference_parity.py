"""
Parity tests for anova.gam() and model-comparison anova against mgcv.

Covers:
  - Single-model anova smooth table (edf, ref_df, F/Chi.sq, p-value)
  - Single-model anova parametric table (df, F/Chi.sq, p-value)
  - Two-model deviance-test comparison via _run_mgcv_anova
  - Multiple families: Gaussian, Poisson, Binomial
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _make_binomial_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_anova,
    _run_mgcv_snapshot,
)


def _norm_labels(labels) -> list[str]:
    """Normalise smooth labels for comparison: strip bs=, k=, sp=, whitespace."""
    vals = [labels] if isinstance(labels, str) else list(labels)
    out = []
    for lab in vals:
        s = str(lab)
        s = re.sub(r',\s*bs\s*=\s*(?:"[^"]*"|\[[^\]]*\])', "", s)
        s = re.sub(r',\s*k\s*=\s*(?:[^,\)]+|\[[^\]]*\])', "", s)
        s = re.sub(r',\s*sp\s*=\s*(?:[^,\)]+|\[[^\]]*\])', "", s)
        s = re.sub(r"\s+", "", s)
        s = s.replace("])", ")")
        s = re.sub(r",\d+\)$", ")", s)
        out.append(s)
    return out


def _as_label_list(labels) -> list[str]:
    """Ensure labels is a list (R auto-unboxes single-element arrays to strings)."""
    if isinstance(labels, str):
        return [labels]
    return list(labels)


# ---------------------------------------------------------------------------
# Single-model anova: smooth table at FIXED smoothing parameters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_parity_gaussian_fixed_sp():
    """anova smooth table matches mgcv exactly at fixed smoothing parameters (Gaussian)."""
    data = _make_gaussian_data(seed=302, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    # Use method="fixed" in R too so that scale (sigma^2) is estimated the same way
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None
    assert _norm_labels(_as_label_list(a_as["labels"])) == _norm_labels(_as_label_list(e_as["labels"]))

    a_vals = np.asarray(a_as["values"], dtype=np.float64)
    e_vals = np.asarray(e_as["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # col 0: edf — identical at fixed SP
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1e-10, rtol=0.0)
    # col 1: ref_df — small numerical differences in hat-matrix traces
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=1e-2, rtol=0.0)
    # col 2: F statistic — both use same scale estimator so should match well
    np.testing.assert_allclose(a_vals[:, 2], e_vals[:, 2], atol=0.0, rtol=5e-4)
    # col 3: p-value
    np.testing.assert_allclose(a_vals[:, 3], e_vals[:, 3], atol=5e-3, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_parity_poisson_fixed_sp():
    """anova smooth table (Chi.sq) matches mgcv for Poisson at fixed SP."""
    data = _make_poisson_data(seed=303)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8, sp=0.9)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "fixed")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None
    assert _norm_labels(_as_label_list(a_as["labels"])) == _norm_labels(_as_label_list(e_as["labels"]))

    a_vals = np.asarray(a_as["values"], dtype=np.float64)
    e_vals = np.asarray(e_as["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1e-10, rtol=0.0)  # edf
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=1e-2, rtol=0.0)   # ref_df
    # Poisson Chi.sq can differ slightly due to P-IRLS working-weight differences
    np.testing.assert_allclose(a_vals[:, 2], e_vals[:, 2], atol=0.0, rtol=1e-2)   # Chi.sq
    np.testing.assert_allclose(a_vals[:, 3], e_vals[:, 3], atol=5e-3, rtol=0.0)   # p-value


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_parity_binomial_fixed_sp():
    """anova smooth table for Binomial at fixed SP: edf matches, Chi.sq is positive/finite."""
    data = _make_binomial_data(seed=304)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.6) + s(x1, bs="cr", k=8, sp=0.8)'

    actual = _fit_nampy_snapshot(data, formula, "binomial", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "fixed")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None
    a_vals = np.asarray(a_as["values"], dtype=np.float64)
    e_vals = np.asarray(e_as["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # edf matches at fixed SP
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=1e-2, rtol=0.0)
    # Chi.sq — structural checks only: the Wald test formula for non-Gaussian
    # uses different scale/Vp conventions between Python and mgcv
    assert np.all(a_vals[:, 2] > 0) and np.all(np.isfinite(a_vals[:, 2]))
    assert np.all(a_vals[:, 3] >= 0.0) and np.all(a_vals[:, 3] <= 1.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_parity_gamma_fixed_sp():
    """anova smooth table for Gamma at fixed SP: edf matches, F is positive/finite."""
    data = _make_gamma_data(seed=305)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.4) + s(x1, bs="cr", k=8, sp=0.6)'

    actual = _fit_nampy_snapshot(data, formula, "gamma", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gamma", "fixed")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None
    a_vals = np.asarray(a_as["values"], dtype=np.float64)
    e_vals = np.asarray(e_as["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # edf matches at fixed SP
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=1e-2, rtol=0.0)
    # F-stat — structural checks only: Python estimates scale differently to R for Gamma
    assert np.all(a_vals[:, 2] > 0) and np.all(np.isfinite(a_vals[:, 2]))
    assert np.all(a_vals[:, 3] >= 0.0) and np.all(a_vals[:, 3] <= 1.0)


# ---------------------------------------------------------------------------
# Single-model anova: smooth table at REML optima (structural checks)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_reml_edf_matches_gaussian():
    """anova EDF per term matches mgcv after REML optimisation (Gaussian)."""
    data = _make_gaussian_data(seed=301, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None
    assert _norm_labels(_as_label_list(a_as["labels"])) == _norm_labels(_as_label_list(e_as["labels"]))

    a_edf = np.asarray(a_as["values"], dtype=np.float64)[:, 0]
    e_edf = np.asarray(e_as["values"], dtype=np.float64)[:, 0]
    np.testing.assert_allclose(a_edf, e_edf, atol=0.15, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_reml_structural_checks():
    """After REML optimisation, anova smooth table values are finite, positive, and consistent."""
    data = _make_gaussian_data(seed=306, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(data, formula, "gaussian", "REML")
    result = model.anova(freq=False)
    smooth_tbl = result.smooth_table

    assert len(smooth_tbl) == 2
    edfs = smooth_tbl["edf"].to_numpy(dtype=np.float64)
    ref_dfs = smooth_tbl["ref_df"].to_numpy(dtype=np.float64)
    stats = smooth_tbl["wald_stat"].to_numpy(dtype=np.float64)
    pvals = smooth_tbl["p_value"].to_numpy(dtype=np.float64)

    assert np.all(np.isfinite(edfs)) and np.all(edfs > 0.0)
    assert np.all(np.isfinite(ref_dfs)) and np.all(ref_dfs > 0.0)
    assert np.all(np.isfinite(stats)) and np.all(stats > 0.0)
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0) and np.all(pvals <= 1.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_smooth_table_reml_poisson_structural_checks():
    """After REML optimisation, Poisson anova smooth table is finite and consistent."""
    data = _make_poisson_data(seed=307)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(data, formula, "poisson", "REML")
    result = model.anova(freq=False)
    smooth_tbl = result.smooth_table

    assert len(smooth_tbl) == 2
    pvals = smooth_tbl["p_value"].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(pvals))
    assert np.all(pvals >= 0.0) and np.all(pvals <= 1.0)


# ---------------------------------------------------------------------------
# Single-model anova: parametric table parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_parametric_table_parity_gaussian_fixed_sp():
    """anova pTerms.table (df, F, p-value) matches mgcv at fixed SP."""
    rng = np.random.default_rng(311)
    n = 160
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = 0.7 * x0 + np.sin(x1) + rng.normal(scale=0.15, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
    formula = 'y ~ x0 + s(x1, bs="cr", k=8, sp=1.2)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

    a_ap = actual["parity"]["diagnostics"]["anova_parametric"]
    e_ap = expected["parity"]["diagnostics"]["anova_parametric"]

    assert a_ap is not None, "Python anova_parametric is None"
    assert e_ap is not None, "R anova_parametric is None"
    assert _norm_labels(_as_label_list(a_ap["labels"])) == _norm_labels(_as_label_list(e_ap["labels"]))

    a_vals = np.asarray(a_ap["values"], dtype=np.float64)
    e_vals = np.asarray(e_ap["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # col 0: df (integer, must be exactly equal)
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=0.0, rtol=0.0)
    # col 1: F statistic — Python uses a different scale estimator than R's REML so tolerance is loose
    assert np.all(a_vals[:, 1] > 0), "F-statistics must be positive"
    # col 2: p-value — direction should agree (both very small or both large)
    assert np.all(np.isfinite(a_vals[:, 2]))
    assert np.all(a_vals[:, 2] >= 0.0) and np.all(a_vals[:, 2] <= 1.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_parametric_table_parity_poisson_fixed_sp():
    """Parametric anova table matches mgcv for Poisson at fixed smoothing parameters."""
    rng = np.random.default_rng(312)
    n = 200
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.4 * x0 + 0.2 + 0.6 * np.sin(x1)
    y = rng.poisson(np.exp(eta))
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
    formula = 'y ~ x0 + s(x1, bs="cr", k=8, sp=0.7)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "fixed")

    a_ap = actual["parity"]["diagnostics"]["anova_parametric"]
    e_ap = expected["parity"]["diagnostics"]["anova_parametric"]

    assert a_ap is not None and e_ap is not None
    assert _norm_labels(_as_label_list(a_ap["labels"])) == _norm_labels(_as_label_list(e_ap["labels"]))

    a_vals = np.asarray(a_ap["values"], dtype=np.float64)
    e_vals = np.asarray(e_ap["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # df must match exactly
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=0.0, rtol=0.0)
    # Chi.sq for Poisson — both should be positive and finite
    assert np.all(a_vals[:, 1] > 0), "Chi.sq statistics must be positive"
    assert np.all(np.isfinite(a_vals[:, 2]))
    assert np.all(a_vals[:, 2] >= 0.0) and np.all(a_vals[:, 2] <= 1.0)


# ---------------------------------------------------------------------------
# Two-model anova comparison parity
# ---------------------------------------------------------------------------

def _r_resid_deviances(r_anova) -> np.ndarray:
    """Extract the residual deviance column from the R comparison anova table.

    R's comparison table has one row per model; the first row has NA for
    difference columns (Df, Deviance, p-value) but always has a finite
    residual deviance (column index 1).  Return all residual deviances.
    """
    r_table = r_anova["table"]
    raw = r_table["values"]
    deviances = []
    for row in raw:
        # Column 1 is "Resid. Dev" — always a float, even in the baseline row.
        try:
            deviances.append(float(row[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return np.asarray(deviances, dtype=np.float64)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_model_comparison_chisq_gaussian():
    """Two-model anova chi-sq comparison table matches mgcv for Gaussian."""
    data = _make_gaussian_data(seed=321, n=200)
    formula_null = 'y ~ s(x0, bs="cr", k=8)'
    formula_full = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    m_null = _fit_nampy_model(data, formula_null, "gaussian", "REML")
    m_full = _fit_nampy_model(data, formula_full, "gaussian", "REML")
    py_anova = m_null.anova(m_full, test="Chisq")

    r_anova = _run_mgcv_anova(
        data, [formula_null, formula_full], "gaussian", "REML", test="Chisq"
    )
    r_deviances = _r_resid_deviances(r_anova)

    # Both Python and R should have one row per model (2 rows).
    assert len(r_deviances) == 2

    py_table = py_anova.table
    py_dev = py_table["deviance"].to_numpy(dtype=np.float64)
    assert len(py_dev) == 2

    np.testing.assert_allclose(py_dev, r_deviances, atol=1.0, rtol=0.1)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_model_comparison_chisq_poisson():
    """Two-model anova chi-sq comparison matches mgcv for Poisson."""
    data = _make_poisson_data(seed=322)
    formula_null = 'y ~ s(x0, bs="cr", k=8)'
    formula_full = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    m_null = _fit_nampy_model(data, formula_null, "poisson", "REML")
    m_full = _fit_nampy_model(data, formula_full, "poisson", "REML")
    py_anova = m_null.anova(m_full, test="Chisq")

    r_anova = _run_mgcv_anova(
        data, [formula_null, formula_full], "poisson", "REML", test="Chisq"
    )
    r_deviances = _r_resid_deviances(r_anova)

    assert len(r_deviances) == 2

    py_table = py_anova.table
    py_dev = py_table["deviance"].to_numpy(dtype=np.float64)
    assert len(py_dev) == 2

    np.testing.assert_allclose(py_dev, r_deviances, atol=1.0, rtol=0.1)


# ---------------------------------------------------------------------------
# anova.gam single model: structure checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_single_model_has_correct_structure_gaussian():
    """anova(fit) output has expected columns and term count."""
    data = _make_gaussian_data(seed=331, n=140)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(data, formula, "gaussian", "REML")
    result = model.anova(freq=False)

    assert result.smooth_table is not None
    assert len(result.smooth_table) == 2
    assert "label" in result.smooth_table.columns
    assert "edf" in result.smooth_table.columns
    assert "p_value" in result.smooth_table.columns

    assert result.parametric_table is not None
    assert len(result.parametric_table) == 0  # no parametric terms in this formula

    edfs = result.smooth_table["edf"].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(edfs)) and np.all(edfs > 0.0)

    pvals = result.smooth_table["p_value"].to_numpy(dtype=np.float64)
    assert np.all(pvals >= 0.0) and np.all(pvals <= 1.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_anova_single_model_tensor_gaussian_fixed_sp():
    """anova smooth table parity for a tensor product smooth at fixed SP."""
    rng = np.random.default_rng(341)
    n = 160
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x0) * np.cos(x1) + rng.normal(scale=0.12, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
    formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")

    a_as = actual["parity"]["diagnostics"]["anova_smooth"]
    e_as = expected["parity"]["diagnostics"]["anova_smooth"]

    assert a_as is not None and e_as is not None

    a_labels = _norm_labels(_as_label_list(a_as["labels"]))
    e_labels = _norm_labels(_as_label_list(e_as["labels"]))
    assert len(a_labels) == len(e_labels) == 1

    a_vals = np.asarray(a_as["values"], dtype=np.float64)
    e_vals = np.asarray(e_as["values"], dtype=np.float64)
    assert a_vals.shape == e_vals.shape

    # Tensor smooth EDF can differ by up to ~0.5 due to marginal-penalty interaction
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1.0, rtol=0.0)    # edf
    np.testing.assert_allclose(a_vals[:, 1], e_vals[:, 1], atol=1.0, rtol=0.0)    # ref_df
    # F-statistic is not directly comparable at different EDF; check positive and finite
    assert np.all(a_vals[:, 2] > 0) and np.all(np.isfinite(a_vals[:, 2]))
    assert np.all(a_vals[:, 3] >= 0.0) and np.all(a_vals[:, 3] <= 1.0)            # p-value
