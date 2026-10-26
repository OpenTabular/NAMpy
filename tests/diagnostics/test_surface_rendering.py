"""Public surface displays and fit-state preservation contracts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.plotting import render_prediction_surface


@pytest.fixture
def prepared():
    x = np.linspace(0, 1, 8)
    xx, yy = np.meshgrid(x, x)
    fit = np.sin(xx).ravel() + yy.ravel() ** 2
    return {
        "x": x,
        "y": x,
        "fit": fit,
        "se_fit": np.full(64, 0.1),
        "lower": fit - 0.2,
        "upper": fit + 0.2,
        "se": -1,
        "view": ["x", "y"],
        "zlim": (-0.3, 2.5),
        "zlab": "linear predictor",
    }


@pytest.mark.parametrize("plot_type", ["persp", "contour"])
@pytest.mark.parametrize("color", ["heat", "topo", "cm", "terrain", "gray", "bw"])
def test_surface_renderer(prepared, plot_type, color):
    figs = render_prediction_surface(prepared, plot_type=plot_type, color=color)
    assert len(figs) == 1
    assert figs[0].axes[0].get_xlabel() == "x"
    assert figs[0].axes[0].get_ylabel() == "y"
    plt.close("all")


def test_surface_uncertainty_and_contour_warning(prepared):
    prepared["se"] = 2
    with pytest.warns(UserWarning, match="No contour"):
        figures = render_prediction_surface(prepared, plot_type="contour")
    ax = figures[0].axes[0]
    assert ax.name == "3d"
    assert len(ax.collections) == 3
    plt.close("all")


@pytest.mark.parametrize(
    "kwargs", [{"plot_type": "invalid"}, {"color": "invalid"}, {"n_col": 1}]
)
def test_surface_renderer_rejects_invalid_options(prepared, kwargs):
    with pytest.raises(ValueError):
        render_prediction_surface(prepared, **kwargs)


def test_vis_adapter_and_prediction_contract():
    from nampy.models import GAMRegressor

    rng = np.random.default_rng(771)
    X = rng.uniform(0.1, 0.9, (60, 2))
    model = GAMRegressor(k=6, optimize_smoothing=False).fit(
        X, np.sin(X[:, 0] * 3) + X[:, 1] ** 2
    )
    result = model.vis(view=["x0", "x1"], n_grid=8, plot_type="contour")
    np.testing.assert_allclose(
        result["fit"], model.gam_.predict(result["newdata"], type="link")
    )
    assert len(result["figures"]) == 1
    plt.close("all")


def test_raw_vis_inputs_survive_transforms_and_failed_refit(tmp_path):
    rng = np.random.default_rng(272)
    data = pd.DataFrame({"x": rng.uniform(1, 3, 60), "z": rng.uniform(-1, 1, 60)})
    data["y"] = np.log(data.x) + data.z**2
    model = GAM(formula="y ~ s(log(x),bs='ps',k=6) + s(z,bs='cr',k=6)").fit(data=data)
    captured = model.vis_training_data_.copy(deep=True)
    data.loc[:, "x"] = 100
    pd.testing.assert_frame_equal(model.vis_training_data_, captured)
    with pytest.raises(KeyError):
        model.fit(data=data.drop(columns="z"))
    pd.testing.assert_frame_equal(model.vis_training_data_, captured)
    path = tmp_path / "model.pkl"
    model.save_model(path)
    loaded = GAM.load_model(path)
    pd.testing.assert_frame_equal(loaded.vis_training_data_, captured)
    result = loaded.vis(view=["x", "z"], n_grid=8)
    assert np.max(result["x"]) < 3
    plt.close("all")


@pytest.mark.parametrize("scheme", [0, 1, 2, 3, 4, 5])
def test_sos_plot_rendering(scheme):
    rng = np.random.default_rng(56)
    lat = rng.uniform(-70, 70, 60)
    lon = rng.uniform(-170, 170, 60)
    data = pd.DataFrame(
        {"lat": lat, "lon": lon, "y": np.sin(np.deg2rad(lat)) + np.cos(np.deg2rad(lon))}
    )
    model = GAM(formula="y ~ s(lat,lon,bs='sos',k=10)").fit(data=data)
    result = model.plot(scheme=scheme, n2=12, theta=20, phi=40)
    assert len(result["figures"]) == 1
    if scheme <= 1:
        assert result["pd"][0]["kind"] == "sos"
        assert not result["figures"][0].axes[0].axison
    else:
        assert result["pd"][0]["kind"] == "2d"
        assert result["pd"][0]["scheme"] == scheme - 2
    plt.close("all")


def test_sos_preparation_preserves_no_rug_for_separate_rendering():
    from nampy.gam.diagnostics import prepare_plot_gam_data, render_plot_gam

    rng = np.random.default_rng(61)
    data = pd.DataFrame(
        {"lat": rng.uniform(-60, 60, 40), "lon": rng.uniform(-150, 150, 40)}
    )
    data["y"] = np.sin(np.deg2rad(data.lat))
    model = GAM(formula="y ~ s(lat,lon,bs='sos',k=8)").fit(data=data)
    prepared = prepare_plot_gam_data(model, rug=False, n2=12)
    assert prepared["rug_default"] is False
    figs = render_plot_gam(model, prepared)
    assert len(figs) == 1
    plt.close("all")
