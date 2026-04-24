from __future__ import annotations

import importlib
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from nampy.gam import GAM

matplotlib.use("Agg")

plots_module = importlib.import_module("nampy.gam.diagnostics.plots")
diagnostics_pkg = importlib.import_module("nampy.gam.diagnostics")
smoothing_pkg = importlib.import_module("nampy.gam.smoothing_selection")
model_api_module = importlib.import_module("nampy.gam.model.api")

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_plot_gam_terms_handles_uni_bi_and_high_dim_terms(monkeypatch):
    """Verify that plot gam terms handles uni bi and high dim terms."""
    monkeypatch.setattr(plots_module, "_require_fitted", lambda model: None)

    X_plot = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.2],
            [2.0, 1.0, 0.4],
            [3.0, 1.5, 0.6],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        plots_module,
        "_coerce_feature_matrix",
        lambda model, X, none_is_training=True: X_plot,
    )
    monkeypatch.setattr(
        plots_module,
        "_term_blocks_seq",
        lambda model: (
            SimpleNamespace(
                term_id="u",
                label="s(x0)",
                feature_info=SimpleNamespace(
                    feature_indices=[0],
                    feature_names=["x0"],
                ),
            ),
            SimpleNamespace(
                term_id="b",
                label="te(x0,x1)",
                feature_info=SimpleNamespace(
                    feature_indices=[0, 1],
                    feature_names=["x0", "x1"],
                ),
            ),
            SimpleNamespace(
                term_id="h",
                label="t3(x0,x1,x2)",
                feature_info=SimpleNamespace(
                    feature_indices=[0, 1, 2],
                    feature_names=["x0", "x1", "x2"],
                ),
            ),
        ),
    )

    model = SimpleNamespace(
        predict_feature_vals=lambda X: {
            "u": np.array([0.0, 1.0, 0.5, 1.5], dtype=np.float64),
            "b": np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64),
            "h": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        }
    )

    fig = plots_module.plot_gam_terms(model, X=None, n_cols=2)

    assert fig.axes[0].get_title() == "s(x0)"
    assert fig.axes[0].get_xlabel() == "x0"
    assert fig.axes[0].get_ylabel() == "term effect"
    assert fig.axes[1].get_title() == "te(x0,x1)"
    assert "Plot not implemented" in fig.axes[2].texts[0].get_text()
    assert fig.axes[2].axison is False
    assert fig.axes[3].axison is False
    plots_module.plt.close(fig)


def test_gam_public_wrappers_delegate_to_underlying_modules(monkeypatch):
    """Verify that gam public wrappers delegate to underlying modules."""
    calls: dict[str, object] = {}

    def _print_summary(model):
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

    def _plot_gam_terms(model, X=None, n_cols=2, figsize=None):
        calls["plot"] = (model, X.copy(), n_cols, figsize)
        return "plot-ok"

    monkeypatch.setattr(diagnostics_pkg, "print_summary", _print_summary)
    monkeypatch.setattr(diagnostics_pkg, "concurvity", _concurvity)
    monkeypatch.setattr(diagnostics_pkg, "plot_gam_terms", _plot_gam_terms)
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
    np.testing.assert_allclose(
        gam.sp_vcov(edge_correct=False, reg=0.25),
        np.eye(1, dtype=np.float64),
    )
    np.testing.assert_allclose(
        gam.one_se_rule(candidate_indices=[0]),
        np.array([2.0], dtype=np.float64),
    )
    assert (
        gam.plot(X=np.array([[1.0], [2.0]], dtype=np.float64), n_cols=3, figsize=(4, 5))
        == "plot-ok"
    )

    assert calls["summary"] is gam
    assert calls["concurvity"] == (gam, False)
    assert calls["sp_vcov"] == (gam, False, 0.25)
    assert calls["one_se_rule"] == (gam, [0])
    plot_model, plot_X, plot_n_cols, plot_figsize = calls["plot"]
    assert plot_model is gam
    np.testing.assert_allclose(plot_X, np.array([[11.0], [12.0]], dtype=np.float64))
    assert plot_n_cols == 3
    assert plot_figsize == (4, 5)

