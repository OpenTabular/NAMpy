from __future__ import annotations

import importlib

import matplotlib
import numpy as np
import pandas as pd
import pytest

import nampy
import nampy.gam as gam_package
from nampy.gam import GAM
from nampy.gam._model_state import _term_blocks_seq
from tests.mgcv_parity_utils import (
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)

matplotlib.use("Agg")

plots_module = importlib.import_module("nampy.gam.diagnostics.plots")
diagnostics_pkg = importlib.import_module("nampy.gam.diagnostics")
smoothing_pkg = importlib.import_module("nampy.gam.smoothing_selection")
model_api_module = importlib.import_module("nampy.gam.model.api")


def test_public_gam_package_exports_contract():
    """Keep package exports aligned with the repository public-API rule."""
    assert gam_package.__all__ == [
        "GAM",
        "fit_model_core",
        "solve_fit",
        "FitCoreSolution",
    ]
    assert gam_package.GAM is GAM
    assert not hasattr(nampy, "GAM")


pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_plot_gam_port_prepares_and_renders_real_model_terms():
    """The plot.gam port prepares per-term data and renders on a real fit."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    rng = np.random.default_rng(41)
    n = 70
    data = pd.DataFrame(
        {
            "x0": rng.uniform(-1.5, 1.5, n),
            "x1": rng.uniform(-1.0, 1.0, n),
        }
    )
    data["y"] = (
        np.sin(1.3 * data["x0"]) + 0.3 * data["x1"] ** 2 + rng.normal(0, 0.1, n)
    )
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0],
    ).fit(data=data)

    out = gam.plot(residuals=True)
    assert [P["kind"] for P in out["pd"]] == ["1d", "1d"]
    for P in out["pd"]:
        assert P["plot_me"] and P["plot_ci"]
        assert np.asarray(P["fit"]).shape == (100,)
        assert np.asarray(P["se"]).shape == (100,)
        assert "p_resid" in P
    fig = out["figures"][0]
    assert fig.axes[0].get_xlabel() == "x0"
    assert fig.axes[1].get_xlabel() == "x1"
    plots_module.plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    plots_module.plt.close(fig)


def test_gam_public_wrappers_delegate_to_underlying_modules(monkeypatch):
    """Verify that gam public wrappers delegate to underlying modules."""
    calls: dict[str, object] = {}

    def _print_summary(model, *, dispersion=None, freq=False, re_test=True):
        calls["summary"] = model
        return "summary-ok"

    def _concurvity(model, full=True):
        calls["concurvity"] = (model, full)
        return {"ok": True, "full": full}

    def _sp_vcov(model, edge_correct=True, reg=1e-3):
        calls["sp_vcov"] = (model, edge_correct, reg)
        return np.eye(1, dtype=np.float64)

    def _one_se_rule(model, candidate_indices=None):
        calls["one_se_rule"] = (model, candidate_indices)
        return np.array([2.0], dtype=np.float64)

    def _plot_gam(model, **kwargs):
        calls["plot"] = (model, dict(kwargs))
        return "plot-ok"

    def _gam_check(model, *, type="deviance", k_sample=5000, k_rep=200, seed=None):
        calls["gam_check"] = (model, type, k_sample, k_rep, seed)
        return {"ok": True, "type": type}

    monkeypatch.setattr(diagnostics_pkg, "print_summary", _print_summary)
    monkeypatch.setattr(diagnostics_pkg, "concurvity", _concurvity)
    monkeypatch.setattr(diagnostics_pkg, "plot_gam", _plot_gam)
    monkeypatch.setattr(diagnostics_pkg, "gam_check", _gam_check)
    monkeypatch.setattr(smoothing_pkg, "sp_vcov", _sp_vcov)
    monkeypatch.setattr(smoothing_pkg, "one_se_rule", _one_se_rule)
    monkeypatch.setattr(
        model_api_module,
        "coerce_X",
        lambda model, X: (np.asarray(X, dtype=np.float64) + 10.0, ["x0"]),
    )

    gam = GAM(family="gaussian")
    gam._fitted = True
    gam.formula_mode_ = False

    assert gam.summary() == "summary-ok"
    assert gam.concurvity(full=False) == {"ok": True, "full": False}
    assert gam.gam_check(type="pearson", k_sample=12, k_rep=7, seed=123) == {
        "ok": True,
        "type": "pearson",
    }
    np.testing.assert_allclose(
        gam.sp_vcov(edge_correct=False, reg=0.25),
        np.eye(1, dtype=np.float64),
    )
    np.testing.assert_allclose(
        gam.one_se_rule(candidate_indices=[0]),
        np.array([2.0], dtype=np.float64),
    )
    assert gam.plot(select=1, se_with_mean=True) == "plot-ok"

    assert calls["summary"] is gam
    assert calls["concurvity"] == (gam, False)
    assert calls["gam_check"] == (gam, "pearson", 12, 7, 123)
    assert calls["sp_vcov"] == (gam, False, 0.25)
    assert calls["one_se_rule"] == (gam, [0])
    plot_model, plot_kwargs = calls["plot"]
    assert plot_model is gam
    assert plot_kwargs == {"select": 1, "se_with_mean": True}


def _small_formula_offset_data(seed=702, n=48):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, n)
    off = 0.15 + 0.08 * np.cos(x)
    y = 0.7 + off + np.sin(1.2 * x) + rng.normal(scale=0.04, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "off": off,
            "unused": rng.normal(size=n),
        }
    )


def test_gam_bic_uses_public_loglik_and_effective_df_formula():
    """Verify BIC follows the mgcv-style effective-df formula."""
    gam = GAM(family="gaussian")
    gam.y_ = np.zeros(9, dtype=np.float64)
    gam.loglik = lambda: -3.25
    gam._loglik_effective_df = lambda: 2.5

    assert gam.bic() == pytest.approx(-2.0 * (-3.25) + np.log(9.0) * 2.5)


def test_predict_terms_matches_prediction_terms_response_intercept_and_offset():
    """Verify predict_terms decomposes the public prediction surfaces."""
    data = _small_formula_offset_data(seed=704, n=54)
    y_counts = np.maximum(
        0,
        np.round(np.exp(0.2 + 0.5 * np.sin(data["x"]) + data["off"])).astype(int),
    )
    data = data.assign(y=y_counts)
    api_offset = np.full(len(data), 0.03, dtype=np.float64)
    gam = GAM(
        family="poisson",
        formula='y ~ offset(off) + s(x, bs="cr", k=6, sp=0.6)',
    )
    gam.fit(data=data)

    values = gam.predict_terms(data, offset=api_offset)
    terms = gam.predict(data, type="terms", offset=api_offset)

    np.testing.assert_allclose(
        values["output"],
        gam.predict(data, type="link", offset=api_offset),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        values["response"],
        gam.predict(data, type="response", offset=api_offset),
        atol=1e-12,
        rtol=1e-12,
    )
    for j, term in enumerate(_term_blocks_seq(gam)):
        np.testing.assert_allclose(
            values[term.term_id],
            terms[:, j],
            atol=1e-12,
            rtol=1e-12,
        )
    assert "intercept" in values
    np.testing.assert_allclose(
        values["offset"],
        np.asarray(data["off"], dtype=np.float64) + api_offset,
        atol=1e-12,
        rtol=1e-12,
    )


def test_gam_lpmatrix_wrapper_matches_predict_lpmatrix_for_formula_newdata():
    """Verify the direct lpmatrix wrapper follows the public prediction surface."""
    data = _small_formula_offset_data(seed=706, n=50)
    gam = GAM(
        family="gaussian",
        formula='y ~ offset(off) + s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)
    newdata = data.iloc[:9].copy()

    np.testing.assert_allclose(
        gam.lpmatrix(newdata),
        gam.predict(newdata, type="lpmatrix"),
        atol=1e-12,
        rtol=1e-12,
    )


def test_tensor_numeric_by_scalar_sp_is_invalid_for_mgcv_multi_penalty_te():
    """mgcv requires one term-level sp per tensor penalty."""
    rng = np.random.default_rng(713)
    n = 72
    x0 = rng.uniform(-1.2, 1.2, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(0.4, 1.5, size=n)
    y = z * (np.sin(x0) + 0.25 * x1**2) + rng.normal(scale=0.04, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1, "z": z})
    gam = GAM(
        family="gaussian",
        formula='y ~ te(x0, x1, by=z, bs=["cr", "cr"], k=[5, 5], sp=0.8)',
    )
    # mgcv/R/mgcv.r::gam.setup stops when length(spi) != ncol(Li).
    with pytest.raises(NotImplementedError, match="one value per penalty"):
        gam.fit(data=data)


def test_formula_metadata_tracks_transformed_terms_offsets_and_ignores_unused_columns():
    """Verify transformed formula preprocessing keeps only relevant source columns."""
    data = _small_formula_offset_data(seed=715, n=52)
    gam = GAM(
        family="gaussian",
        formula='y ~ I(x**2) + offset(off) + s(I(x + 0.2), bs="cr", k=6, sp=0.8)',
    )
    gam.fit(data=data)

    used = gam.formula_used_columns_
    assert used is not None
    assert "x" in used
    assert "off" in used
    assert "unused" not in used
    assert gam.formula_offset_name_ == "off"
    assert gam.formula_offset_names_ == ("off",)


def test_gam_check_report_contains_mgcv_comparable_and_local_blocks():
    """Verify gam_check exposes comparable residual/k-check and local optimizer blocks."""
    data = _small_formula_offset_data(seed=723, n=58)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    report = gam.gam_check(type="deviance", k_sample=30, k_rep=3, seed=42)

    assert set(report) >= {"mgcv_comparable", "nampy_specific"}
    assert report["mgcv_comparable"]["residual_type"] == "deviance"
    assert len(report["mgcv_comparable"]["residuals"]) == len(data)
    assert report["mgcv_comparable"]["k_check"] is not None
    assert "convergence" in report["nampy_specific"]


def test_gam_check_rejects_unknown_residual_type():
    """Verify gam_check unsupported residual types fail explicitly."""
    data = _small_formula_offset_data(seed=724, n=42)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    with pytest.raises(ValueError, match="residual type"):
        gam.gam_check(type="working")


def test_summary_contains_representative_family_term_and_fit_statistics():
    """Verify the print.summary.gam-shaped text and returned GAMSummary."""
    data = _small_formula_offset_data(seed=725, n=46)
    gam = GAM(
        family="gaussian",
        formula='y ~ offset(off) + s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    summary = gam.summary()
    text = str(summary)

    assert "Family: gaussian" in text
    assert "Link function: identity" in text
    assert "Formula:" in text
    assert "offset(off)" in text
    assert "Approximate significance of smooth terms:" in text
    assert "Scale est. =" in text
    assert f"n = {gam.n_samples_}" in text
    assert summary.r_sq is not None
    assert summary.dev_expl is not None
    assert summary.null_deviance is not None


def test_predict_iterms_preserves_term_contributions_and_returns_distinct_se():
    """Verify iterms differs from terms only through mean-uncertainty SE rows."""
    data = _small_formula_offset_data(seed=727, n=42)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    terms, terms_se = gam.predict(data, type="terms", return_se=True)
    iterms, iterms_se = gam.predict(data, type="iterms", return_se=True)

    np.testing.assert_allclose(iterms, terms, rtol=0.0, atol=0.0)
    assert iterms_se.shape == terms_se.shape
    assert np.all(np.isfinite(iterms_se))
    assert not np.allclose(iterms_se, terms_se, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("pred_type", "terms", "exclude", "return_se"),
    [
        ("link", None, ["s(x)"], True),
        ("response", ["(Intercept)", "s(x)"], None, True),
        ("terms", ["s(x)"], None, True),
        ("iterms", ["s(x)"], None, True),
        ("lpmatrix", None, ["unused"], False),
    ],
)
def test_predict_terms_and_exclude_filters_match_mgcv(
    pred_type, terms, exclude, return_se
):
    """Verify predict.gam term filters across values, SEs, and lpmatrix blocks."""
    data = _small_formula_offset_data(seed=728, n=42)
    formula = 'y ~ unused + s(x, bs="cr", k=6, sp=0.7)'
    gam = GAM(
        family="gaussian",
        formula=formula,
    )
    gam.fit(data=data)
    newdata = data.iloc[::3].copy()

    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type=pred_type,
        return_se=return_se,
        terms=terms,
        exclude=exclude,
        allow_live_run=True,
    )
    actual = gam.predict(
        newdata,
        type=pred_type,
        return_se=return_se,
        terms=terms,
        exclude=exclude,
    )

    if return_se:
        actual_fit, actual_se = actual
        expected_fit = np.asarray(expected["pred"], dtype=np.float64)
        expected_se = np.asarray(expected["se"], dtype=np.float64)
        if pred_type not in {"terms", "iterms"}:
            expected_fit = expected_fit.ravel()
            expected_se = expected_se.ravel()
        np.testing.assert_allclose(
            np.asarray(actual_fit, dtype=np.float64),
            expected_fit,
            atol=2e-10,
            rtol=2e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual_se, dtype=np.float64),
            expected_se,
            atol=2e-10,
            rtol=2e-10,
        )
    else:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected["pred"], dtype=np.float64),
            atol=2e-10,
            rtol=2e-10,
        )


def test_predict_unknown_term_filter_mirrors_mgcv_warning_and_zeroing_order():
    """Verify unknown terms warn after upstream-style design-block filtering."""
    data = _small_formula_offset_data(seed=729, n=42)
    gam = GAM(
        family="gaussian",
        formula='y ~ unused + s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    with pytest.warns(UserWarning, match="non-existent terms requested - ignoring"):
        values = gam.predict(data, type="terms", terms=["missing"])

    assert values.shape == (len(data), 2)
    np.testing.assert_allclose(values, 0.0, atol=0.0, rtol=0.0)


def test_predict_terms_handles_factor_smooth_terms():
    """Verify factor smooth term effects match mgcv predict(type="terms")."""
    rng = np.random.default_rng(732)
    n = 72
    x = rng.uniform(-1.3, 1.3, size=n)
    f = pd.Categorical(rng.choice(np.array(["a", "b", "c"], dtype=object), size=n))
    y = np.sin(x) + np.array([{"a": 0.2, "b": -0.3, "c": 0.45}[str(v)] for v in f])
    y = y + rng.normal(scale=0.05, size=n)
    data = pd.DataFrame({"y": y, "x": x, "f": f})
    formula = 'y ~ s(x, f, bs="fs", k=5, xt="cr", sp=[0.8, 0.8, 0.8])'
    gam = GAM(family="gaussian", formula=formula)
    gam.fit(data=data)

    values = gam.predict_terms(data)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
    expected_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    for j, term in enumerate(_term_blocks_seq(gam)):
        assert term.term_id in values
        np.testing.assert_allclose(
            values[term.term_id],
            expected_terms[:, j],
            atol=1e-8,
            rtol=1e-8,
        )


def test_mixed_fixed_and_free_smoothing_parameters_fit_and_predict():
    """Verify mixed fixed/free smoothing parameter formulas match mgcv."""
    rng = np.random.default_rng(733)
    n = 80
    x0 = rng.uniform(-1.4, 1.4, size=n)
    x1 = rng.uniform(-1.1, 1.1, size=n)
    y = np.sin(x0) + 0.2 * x1**2 + rng.normal(scale=0.05, size=n)
    data = pd.DataFrame({"y": y, "x0": x0, "x1": x1})
    formula = 'y ~ s(x0, bs="cr", k=6, sp=0.7) + s(x1, bs="cr", k=6)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    np.testing.assert_allclose(
        gam.predict(data, type="link"),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    fixed_mask = np.asarray(gam.smoothing_fixed_mask_, dtype=bool)
    np.testing.assert_allclose(
        np.asarray(gam.smoothing_params, dtype=np.float64)[fixed_mask],
        np.array([0.7], dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(gam.smoothing_params, dtype=np.float64)[~fixed_mask],
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64),
        atol=1e-6,
        rtol=1e-6,
    )


def test_predict_unknown_exclude_filter_mirrors_mgcv_warning_and_is_ignored():
    """Unknown exclude= labels warn and leave the prediction untouched.

    Companion to the terms= branch above; the exclude branch
    (predict/predictions.py) previously had no test reaching its warning.
    """
    data = _small_formula_offset_data(seed=731, n=42)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=6, sp=0.7)',
    )
    gam.fit(data=data)

    baseline = gam.predict(data, type="terms")
    with pytest.warns(UserWarning, match="non-existent exclude terms requested"):
        values = gam.predict(data, type="terms", exclude=["missing"])
    np.testing.assert_array_equal(
        np.asarray(values, dtype=np.float64),
        np.asarray(baseline, dtype=np.float64),
    )
