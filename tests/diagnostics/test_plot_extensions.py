"""Static mgcv references for maps and derivative curves; renderer contracts."""

import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.derivatives import prepare_derivative_plot
from nampy.gam.diagnostics.plots import prepare_plot_gam_data
from nampy.plotting.render import _polygon_path
from tests.reference_fixtures import load_reference, reference_key, save_reference

R_DRIVER = r"""
library(mgcv)
library(jsonlite)
a <- commandArgs(TRUE)
p <- fromJSON(a[1], simplifyVector=FALSE)
d <- as.data.frame(lapply(p$data, unlist))
if (p$mode == 'mrf') {
 d$region <- factor(d$region)
 polys <- lapply(p$polygons, function(x) do.call(rbind, lapply(x, unlist)))
 b <- gam(y ~ s(region,bs='mrf',k=p$k,xt=list(polys=polys)),data=d,sp=.7)
 png(tempfile(fileext='.png'))
 pd <- plot(b)[[1]]
 dev.off()
 out <- list(fit=as.numeric(pd$fit),version=as.character(packageVersion('mgcv')))
} else {
 b <- gam(as.formula(p$formula),data=d,sp=.7)
 h <- 1e-4
 xp <- xm <- d
 xp$x <- d$x+h; xm$x <- d$x-h
 Xp <- predict(b,xp,type='lpmatrix'); Xm <- predict(b,xm,type='lpmatrix')
 X <- predict(b,d,type='lpmatrix')
 D <- if(p$order==1) (Xp-Xm)/(2*h) else (Xp-2*X+Xm)/h^2
 out <- list(fit=as.numeric(D%*%coef(b)),se=sqrt(rowSums((D%*%b$Vp)*D)),version=as.character(packageVersion('mgcv')))
}
write_json(out,a[2],auto_unbox=TRUE,digits=16)
"""


def reference(payload):
    key = reference_key("plot_extensions_v1", payload)
    cached = load_reference("mgcv_plot_extensions", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmp:
        script, source, dest = [
            Path(tmp) / name for name in ("run.R", "input.json", "out.json")
        ]
        script.write_text(R_DRIVER)
        source.write_text(json.dumps(payload))
        run = subprocess.run(
            ["Rscript", str(script), str(source), str(dest)],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        result = json.loads(dest.read_text())
    save_reference("mgcv_plot_extensions", key, result)
    return result


@pytest.mark.parametrize("basis", ["ps", "bs"])
@pytest.mark.parametrize("order", [1, 2])
def test_derivative_plot_matches_mgcv(basis, order):
    rng = np.random.default_rng(182)
    x = rng.uniform(0.05, 0.95, 90)
    data = pd.DataFrame({"x": x, "y": np.sin(3 * x) + rng.normal(0, 0.08, len(x))})
    formula = f'y ~ s(x,bs="{basis}",k=7)'
    model = GAM(formula=formula, smoothing_params=[0.7]).fit(data=data)
    expected = reference(
        {
            "mode": "derivative",
            "formula": formula,
            "order": order,
            "data": data.to_dict("list"),
        }
    )
    p = prepare_derivative_plot(model, deriv=order)["pd"][0]
    sort = np.argsort(x, kind="stable")
    np.testing.assert_allclose(
        p["fit"], np.asarray(expected["fit"])[sort], atol=3e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        p["se_fit"], np.asarray(expected["se"])[sort], atol=3e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        (p["ul"] - p["ll"])[:, 0], 2 * 1.959963984540054 * p["se_fit"]
    )
    figures = model.plot_derivative(deriv=order)
    assert len(figures) == 1
    plt.close("all")


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("scheme", [0, 1])
def test_mrf_map_matches_mgcv(k, scheme):
    polygons = {
        name: [[i, 0], [i + 1, 0], [i + 1, 1], [i, 1], [i, 0]]
        for name, i in [("d", 3), ("b", 1), ("a", 0), ("c", 2)]
    }
    region = np.tile(["a", "b", "c", "d"], 20)
    data = pd.DataFrame(
        {
            "region": pd.Categorical(region),
            "y": np.tile([1, 2, -1, 0], 20) + np.sin(np.arange(80)) * 0.2,
        }
    )
    formula = f'y ~ s(region,bs="mrf",k={k},xt={{"polys":{polygons!r}}})'
    model = GAM(formula=formula, smoothing_params=[0.7]).fit(data=data)
    expected = reference(
        {"mode": "mrf", "k": k, "polygons": polygons, "data": data.to_dict("list")}
    )
    p = prepare_plot_gam_data(model, scheme=scheme)["pd"][0]
    assert p["region_names"] == list(polygons)
    np.testing.assert_allclose(p["fit"], expected["fit"], atol=1e-8)
    figs = model.plot(scheme=scheme)["figures"]
    assert len(figs) == 1
    assert len(figs[0].axes) == 2
    np.testing.assert_allclose(figs[0].axes[0].collections[0].get_array(), p["fit"])
    plt.close("all")


def test_nested_polygon_winding_and_disjoint_loops():
    outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    hole = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
    island = np.array([[1.5, 1.5], [2, 1.5], [2, 2], [1.5, 2]])
    separate = outer + [6, 0]
    path = _polygon_path(
        np.vstack(
            [
                outer,
                [[np.nan, np.nan]],
                hole,
                [[np.nan, np.nan]],
                island,
                [[np.nan, np.nan]],
                separate,
            ]
        )
    )
    starts = np.flatnonzero(path.codes == 1)
    signs = []
    for part in np.split(path.vertices, starts[1:]):
        signs.append(
            np.sign(
                np.sum(
                    part[:, 0] * np.roll(part[:, 1], -1)
                    - part[:, 1] * np.roll(part[:, 0], -1)
                )
            )
        )
    assert signs == [1, -1, 1, 1]


@pytest.mark.parametrize("level", [0, 1, -1, float("nan")])
def test_invalid_derivative_level(level):
    with pytest.raises(ValueError, match="level"):
        prepare_derivative_plot(None, level=level)


@pytest.mark.parametrize("order", [1, 2])
def test_derivative_new_rows_and_adapter(order):
    from nampy.models import GAMRegressor

    x = np.linspace(0, 1, 60)
    model = GAMRegressor(basis="ps", k=7, optimize_smoothing=False).fit(
        x[:, None], np.sin(x * 3)
    )
    new = np.array([[0.8], [0.2], [0.5]])
    result = model.derivative(new, deriv=order)
    figures = model.plot_derivative(new, deriv=order, level=0.8)
    line = figures[0].axes[0].lines[-1]
    np.testing.assert_allclose(line.get_xdata(), np.sort(new[:, 0]))
    np.testing.assert_allclose(line.get_ydata(), result.derivative[[1, 2, 0]])
    plt.close("all")


@pytest.mark.parametrize("basis", ["cr", "gp", "sos"])
def test_derivative_plot_preserves_unsupported_basis_boundary(basis):
    data = pd.DataFrame({"x": np.linspace(0.1, 0.9, 50)})
    data["y"] = np.sin(data.x * 3)
    formula = f'y ~ s(x, bs="{basis}", k=7)'
    if basis == "sos":
        data["z"] = np.sin(np.arange(50)) * 30
        formula = 'y ~ s(x,z,bs="sos",k=7)'
    model = GAM(formula=formula).fit(data=data)
    with pytest.raises(NotImplementedError):
        model.plot_derivative()


def test_mrf_without_geometry_has_no_map():
    data = pd.DataFrame(
        {"r": pd.Categorical(np.tile(["a", "b", "c"], 12)), "y": np.sin(np.arange(36))}
    )
    model = GAM(
        formula="y ~ s(r,bs='mrf',xt={'nb':{'a':['b'],'b':['a','c'],'c':['b']}})"
    ).fit(data=data)
    assert not prepare_plot_gam_data(model)["pd"][0]["plot_me"]
    with pytest.raises(ValueError, match="No terms"):
        model.plot()


def test_derivative_transformed_axis_uses_formula_expression():
    data = pd.DataFrame({"x": np.linspace(1, 3, 60)})
    data["y"] = np.log(data.x) ** 2
    model = GAM(formula='y ~ s(log(x),bs="ps",k=7)').fit(data=data)
    p = prepare_derivative_plot(model)["pd"][0]
    assert p["xlab"] == "log(x)"
    np.testing.assert_allclose(p["x"], np.log(data.x))
