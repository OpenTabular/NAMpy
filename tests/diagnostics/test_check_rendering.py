"""Public QQ/model-check rendering and fitted-state preservation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.check_plots import prepare_gam_check_data
from nampy.gam.diagnostics.qq import prepare_qq_gam_data


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(145)
    x = np.linspace(0.1, 1, 50)
    frame = pd.DataFrame({"x": x, "y": np.sin(4 * x) + rng.normal(0, 0.2, len(x))})
    return GAM(
        formula="y~s(x,bs='cr',k=6)",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.4],
    ).fit(data=frame)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("level", [0, 0.8, 1, 2])
def test_public_qq_envelopes_traces_reference_and_limits(model, level):
    data = model.qq(rep=5, level=level, seed=17, figsize=(5, 4))
    expected = prepare_qq_gam_data(model, rep=5, level=level, seed=17)
    np.testing.assert_array_equal(data["theoretical"], expected["theoretical"])
    np.testing.assert_array_equal(data["residuals"], expected["residuals"])
    assert len(data["figures"]) == 1
    fig = data["figures"][0]
    np.testing.assert_array_equal(fig.get_size_inches(), [5, 4])
    ax = fig.axes[0]
    assert ax.get_xlabel() == data["xlab"]
    assert ax.get_ylabel() == data["ylab"]
    np.testing.assert_allclose(ax.get_ylim(), data["ylim"])
    assert len(ax.lines) == (6 if level >= 1 else 1)
    np.testing.assert_array_equal(ax.lines[-1].get_xdata(), ax.lines[-1].get_ydata())
    assert len(ax.collections) == (2 if 0 < level < 1 else 1)
    np.testing.assert_allclose(
        ax.collections[-1].get_offsets(),
        np.column_stack([data["theoretical"], data["residuals"]]),
    )
    if level >= 1:
        for i, line in enumerate(ax.lines[:-1]):
            np.testing.assert_array_equal(line.get_xdata(), data["theoretical"])
            np.testing.assert_array_equal(
                line.get_ydata(), data["simulated_residuals"][:, i]
            )


@pytest.mark.parametrize("old_style", [False, True])
@pytest.mark.parametrize("type", ["deviance", "pearson", "response"])
def test_public_check_panel_coordinates_histogram_and_old_style(model, old_style, type):
    data = model.plot_check(old_style=old_style, type=type, seed=12)
    expected = prepare_gam_check_data(model, old_style=old_style, type=type, seed=12)
    assert len(data["figures"]) == 1
    axes = data["figures"][0].axes
    assert len(axes) == 4
    assert len(axes[0].lines) == (0 if old_style else 1)
    for index, key in [(1, "residual_vs_linear"), (3, "response_vs_fitted")]:
        panel = expected[key]
        np.testing.assert_allclose(
            axes[index].collections[0].get_offsets(),
            np.column_stack([panel["x"], panel["y"]]),
        )
        assert axes[index].get_title() == panel["title"]
    hist = expected["histogram"]
    np.testing.assert_allclose(
        [p.get_height() for p in axes[2].patches], hist["counts"]
    )
    np.testing.assert_allclose(
        [p.get_x() for p in axes[2].patches], hist["breaks"][:-1]
    )
    np.testing.assert_allclose(
        [p.get_width() for p in axes[2].patches], np.diff(hist["breaks"])
    )


def test_display_preserves_model_state_and_canonical_predictions(model):
    from nampy.gam.model_state import _coef_full

    coefficients = _coef_full(model).copy()
    response = model.y_.copy()
    from nampy.gam.diagnostics.residuals import _prior_weights

    weights = _prior_weights(model).copy()
    prediction = model.predict().copy()
    from nampy.gam.model_state import _fit_result, _fit_scale

    scale = _fit_scale(model)
    state = _fit_result(model)
    model.qq(seed=37, rep=4)
    model.plot_check(seed=37, rep=4)
    np.testing.assert_array_equal(_coef_full(model), coefficients)
    np.testing.assert_array_equal(model.y_, response)
    np.testing.assert_array_equal(_prior_weights(model), weights)
    np.testing.assert_array_equal(model.predict(), prediction)
    assert _fit_result(model) is state
    assert _fit_scale(model) == scale


@pytest.mark.parametrize("classification", [False, True])
def test_sklearn_adapter_qq_and_check_delegation(classification):
    from nampy.models import GAMClassifier, GAMRegressor

    rng = np.random.default_rng(29)
    x = np.linspace(0.1, 1, 50).reshape(-1, 1)
    y = (
        (rng.random(len(x)) < 0.5 + 0.3 * np.sin(4 * x[:, 0]))
        if classification
        else np.sin(4 * x[:, 0]) + rng.normal(0, 0.2, len(x))
    )
    cls = GAMClassifier if classification else GAMRegressor
    adapter = cls(k=6, optimize_smoothing=False).fit(x, y)
    before = adapter.predict(x).copy()
    qq = adapter.qq(seed=31)
    check = adapter.plot_check(seed=31)
    np.testing.assert_array_equal(qq["theoretical"], check["qq"]["theoretical"])
    assert len(qq["figures"]) == len(check["figures"]) == 1
    np.testing.assert_array_equal(adapter.predict(x), before)


def test_unfitted_public_diagnostics_fail():
    for method in ["qq", "plot_check"]:
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(GAM(), method)()
