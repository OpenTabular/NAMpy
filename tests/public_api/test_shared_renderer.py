"""The shared term-plot renderer consumes plain prepared dicts, no model."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nampy.plotting import render_term_plots


def _prepared_1d(n_terms=2, n=50, with_ci=False):
    rng = np.random.default_rng(0)
    pd_list = []
    for index in range(n_terms):
        x = np.sort(rng.uniform(size=n))
        entry = {
            "kind": "1d",
            "plot_me": True,
            "x": x,
            "fit": np.sin((index + 1) * x),
            "raw": x,
            "xlab": f"x{index}",
            "ylab": f"f(x{index})",
            "main": "",
            "scheme": 0,
        }
        if with_ci:
            entry["plot_ci"] = True
            entry["ll"] = (entry["fit"] - 0.2)[:, None]
            entry["ul"] = (entry["fit"] + 0.2)[:, None]
        pd_list.append(entry)
    return {
        "pd": pd_list,
        "ylim": None,
        "partial_resids": False,
        "by_resids": False,
        "shift": 0.0,
        "trans": lambda values: values,
        "jit": False,
        "select": None,
        "scale": False,
        "rug_default": True,
    }


def test_renderer_draws_hand_built_1d_terms():
    figures = render_term_plots(_prepared_1d())
    try:
        assert len(figures) == 1
        assert len(figures[0].axes) >= 2
    finally:
        plt.close("all")


def test_renderer_ci_and_rug_branches():
    figures = render_term_plots(_prepared_1d(with_ci=True), rug=True)
    try:
        assert len(figures) == 1
    finally:
        plt.close("all")


def test_renderer_rejects_empty_plot_list():
    prepared = _prepared_1d()
    for entry in prepared["pd"]:
        entry["plot_me"] = False
    with pytest.raises(ValueError, match="No terms to plot"):
        render_term_plots(prepared)


def test_neural_plot_terms_uses_shared_renderer(tmp_path):
    from nampy.models.linreg import LinRegRegressor

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})
    y = 2.0 * X["x"].to_numpy() - X["z"].to_numpy()

    estimator = LinRegRegressor(numerical_preprocessing="standardization")
    estimator.fit(X, y, max_epochs=2, patience=1, checkpoint_path=str(tmp_path))

    figures = estimator.plot_terms(X)
    try:
        assert len(figures) == 1
        assert len(figures[0].axes) >= 2
    finally:
        plt.close("all")


def test_prepared_from_contributions_term_features_mapping():
    from nampy.plotting import prepared_from_contributions

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x0": rng.uniform(size=30), "x3": rng.normal(size=30)})
    terms = {
        "gam:s(x0, k=6)": np.sin(frame["x0"].to_numpy()),
        "nn:x3": frame["x3"].to_numpy() * 0.5,
        "gam:unmappable": np.zeros(30),
    }

    prepared = prepared_from_contributions(
        frame,
        terms,
        term_features={"gam:s(x0, k=6)": "x0", "nn:x3": "x3"},
    )

    labels = [entry["xlab"] for entry in prepared["pd"]]
    assert "gam:s(x0, k=6)" in labels
    assert "nn:x3" in labels
    assert "gam:unmappable" not in labels

    mapped = next(
        entry for entry in prepared["pd"] if entry["xlab"] == "gam:s(x0, k=6)"
    )
    np.testing.assert_array_equal(mapped["raw"], frame["x0"].to_numpy())
