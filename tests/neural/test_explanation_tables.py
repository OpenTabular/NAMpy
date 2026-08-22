import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nampy.contracts import AdditivePrediction
from nampy.explanations import (
    center_additive_prediction,
    explain_additive_prediction,
    term_importance_table,
)
from nampy.models._plotting import plot_interaction_effects


def _prediction():
    x = np.array([-1.0, -0.5, 0.5, 1.0])
    z = np.array([0.0, 0.0, 1.0, 1.0])
    terms = {"x": x, "x:z": x * z}
    link = 0.25 + terms["x"] + terms["x:z"]
    return AdditivePrediction(
        response=link,
        link=link,
        terms=terms,
        intercept=0.25,
        backend="neural",
    )


def test_explanation_table_has_counts_values_and_importance():
    X = pd.DataFrame(
        {"x": [-1.0, -0.5, 0.5, 1.0], "z": [0.0, 0.0, 1.0, 1.0]}
    )
    table = explain_additive_prediction(X, _prediction(), max_bins=2)

    assert set(table["term"]) == {"x", "x:z"}
    assert table.groupby(["term", "output"])["count"].sum().eq(len(X)).all()
    interaction = table.loc[table["term"] == "x:z"]
    assert interaction["value_2"].notna().all()
    assert interaction["term_type"].eq("interaction").all()


def test_term_importance_is_mean_absolute_link_contribution():
    table = term_importance_table(_prediction()).set_index("term")
    assert table.loc["x", "importance"] == np.mean(
        np.abs(_prediction().terms["x"])
    )
    assert table.loc["x:z", "term_type"] == "interaction"


def test_explanation_table_preserves_arbitrary_order_feature_values():
    X = pd.DataFrame(
        {
            "x": [-1.0, -0.5, 0.5, 1.0],
            "z": [0.0, 0.0, 1.0, 1.0],
            "w": [1.0, 2.0, 3.0, 4.0],
        }
    )
    contribution = X["x"].to_numpy() * X["z"].to_numpy() * X["w"].to_numpy()
    prediction = AdditivePrediction(
        response=contribution,
        link=contribution,
        terms={"x:z:w": contribution},
        intercept=0.0,
        backend="neural",
    )
    table = explain_additive_prediction(X, prediction)
    assert table["value_3"].notna().all()


def test_higher_order_interaction_plot_uses_conditioned_slices(monkeypatch):
    X = pd.DataFrame(
        {
            "x": np.tile(np.linspace(-1.0, 1.0, 12), 3),
            "z": np.repeat([-1.0, 0.0, 1.0], 12),
            "w": np.repeat([0.0, 1.0, 2.0], 12),
        }
    )
    contribution = (X["x"] * X["z"] * X["w"]).to_numpy()[:, None]
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_interaction_effects(
        "x:z:w",
        contribution,
        X_train_scaled=X,
        num_bins=4,
        slice_bins=3,
    )
    assert plt.get_fignums()
    plt.close("all")


def test_centered_prediction_has_zero_mean_terms_and_preserves_reconstruction():
    centered = center_additive_prediction(_prediction())
    for values in centered.terms.values():
        assert np.mean(values) == pytest.approx(0.0)
    centered.validate_additive_reconstruction()
    np.testing.assert_allclose(centered.link, _prediction().link)
