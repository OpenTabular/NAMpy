"""Real vis.gam grid and prediction snapshots; normal execution needs no R."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.vis import prepare_vis_gam_data
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression


def _case(**options):
    rng = np.random.default_rng(2456)
    x, z = rng.uniform(0.1, 1, (2, 80))
    w = np.linspace(-0.6, 0.9, 80)
    g = np.asarray(["c", "a", "b"] * 27)[:80]
    mean = np.sin(4 * x) + z * z + 0.2 * w + (g == "a") * 0.4
    y = mean + rng.normal(0, 0.25, len(x))
    family = options.pop("family", "gaussian")
    transform = options.pop("transform", False)
    unused_factor = options.pop("unused_factor", False)
    with_offset = options.pop("with_offset", False)
    transformed_by = options.pop("transformed_by", False)
    if family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-mean)))
    smoothx = "log(x)" if transform else "x"
    formula = f"y ~ s({smoothx},bs='cr',k=6) + s(z,bs='cr',k=6) + w + g"
    if family == "gaulss":
        formula = [f"y ~ s({smoothx},bs='cr',k=6)+w+g", "~ s(z,bs='cr',k=6)"]
    if transformed_by:
        formula = formula.replace("s(z,", "s(z,by=I(w*w),")
    if with_offset:
        formula += " + offset(off)"
    return {
        "data": {
            **({"off": (0.05 * x + 0.1 * z).tolist()} if with_offset else {}),
            "x": x.tolist(),
            "z": z.tolist(),
            "w": w.tolist(),
            "g": g.tolist(),
            "y": y.tolist(),
        },
        "levels": ["c", "a", "b"] + (["unused"] if unused_factor else []),
        "formula": formula,
        "family": family,
        "options": {"view": ["x", "z"], "n_grid": 9, **options},
    }


_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
d<-as.data.frame(lapply(p$data,unlist));d$g<-factor(d$g,levels=unlist(p$levels))
f<-if(is.list(p$formula)) lapply(p$formula,as.formula) else as.formula(p$formula)
b<-gam(f,data=d,family=get(p$family,mode='function')(),sp=c(.4,.7),method='REML')
# Observe the actual upstream function at exit, including its factor axes,
# unmasked conditioning grid and uncertainty bounds. A temporary PDF discards drawing.
trace('vis.gam',where=asNamespace('mgcv'),print=FALSE,exit=quote({
 if(exists('fv',inherits=FALSE)) assign('.vis_snapshot',list(x=m1,y=m2,fit=as.numeric(fv$fit),se_fit=as.numeric(fv$se.fit),
 newdata=newd,view=view,zlim=c(min.z,max.z),lp=lp),envir=.GlobalEnv)
}))
args<-p$options;names(args)<-gsub('_','.',names(args))
for(nm in c('view','zlim')) if(!is.null(args[[nm]])) args[[nm]]<-unlist(args[[nm]])
if(!is.null(args$cond)) args$cond<-lapply(args$cond,unlist)
pdf(file=a[3]);tryCatch(do.call(get('vis.gam',envir=asNamespace('mgcv')),c(list(x=b),args)),
 error=function(e) assign('.vis_snapshot',list(error=conditionMessage(e)),envir=.GlobalEnv));dev.off()
untrace('vis.gam',where=asNamespace('mgcv'))
write_json(.vis_snapshot,a[2],digits=17,auto_unbox=TRUE,null='null',na='null',dataframe='columns')
"""


def _reference(case):
    key = reference_key("vis.gam-v1", case, normalize_floats=True)
    cached = load_reference("mgcv_vis_gam", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmp:
        script, inp, out, pdf = (
            Path(tmp) / name
            for name in ("ref.R", "case.json", "out.json", "discard.pdf")
        )
        script.write_text(_SCRIPT)
        inp.write_text(json.dumps(case))
        result = subprocess.run(
            _build_r_command(script, str(inp), str(out), str(pdf)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        cached = json.loads(out.read_text())
    save_reference("mgcv_vis_gam", key, cached)
    return cached


def _fit(case):
    frame = pd.DataFrame(case["data"])
    frame["g"] = pd.Categorical(frame.g, categories=case["levels"])
    return GAM(
        formula=case["formula"],
        family=case["family"],
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=[0.4, 0.7],
    ).fit(data=frame)


CASES = [
    {},
    {"view": None},
    {"view": ["g", "x"], "n_grid": 8},
    {"view": ["x", "g"], "n_grid": 8},
    {"view": ["g", "x"], "n_grid": 2},
    {"cond": {"w": 0.23, "g": "b"}, "n_grid": 7},
    {"too_far": 0.09},
    {"too_far": 0.09, "se": 2},
    {"se": 1.5, "zlim": [-1, 3]},
    {"family": "binomial", "type": "response", "se": 2},
    {"family": "binomial", "type": "link"},
    {"family": "gaulss", "lp": 2, "se": 1},
    {"family": "gaulss", "lp": 2.7},
    {"family": "gaulss", "lp": -1},
    {"family": "gaulss", "type": "response", "lp": 2, "se": 2},
    {"transform": True},
    {"transformed_by": True},
    {"unused_factor": True},
    {"unused_factor": True, "view": ["g", "x"], "n_grid": 8},
    {"with_offset": True},
    {"with_offset": True, "cond": {"off": 0.4}, "se": 2},
]


@pytest.mark.parametrize("options", CASES, ids=lambda p: str(p))
def test_vis_gam_grid_fit_se_and_limits_match_actual_mgcv(options):
    case = _case(**options)
    expected = _reference(case)
    if "error" in expected:
        assert "replacement length" in expected["error"]
        with pytest.raises(ValueError, match="unobserved factor levels"):
            prepare_vis_gam_data(_fit(case), **case["options"])
        return
    actual = prepare_vis_gam_data(_fit(case), **case["options"])
    assert actual["view"] == expected["view"]
    for key in ("x", "y", "fit", "se_fit", "zlim"):
        np.testing.assert_allclose(
            actual[key],
            np.asarray(expected[key], dtype=float),
            rtol=2e-6,
            atol=2e-7,
            equal_nan=True,
        )
    for column in actual["newdata"]:
        if column == "g":
            assert actual["newdata"][column].tolist() == expected["newdata"][column]
        else:
            np.testing.assert_allclose(
                actual["newdata"][column],
                expected["newdata"][column],
                atol=1e-14,
                rtol=1e-14,
            )
    if actual["se"] > 0:
        np.testing.assert_allclose(
            actual["lower"],
            actual["fit"] - actual["se"] * actual["se_fit"],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual["upper"],
            actual["fit"] + actual["se"] * actual["se_fit"],
            equal_nan=True,
        )


@pytest.fixture(scope="module")
def model():
    return _fit(_case())


@pytest.mark.parametrize(
    "options,match",
    [
        ({"n_grid": 1}, "n_grid"),
        ({"n_grid": 2.5}, "n_grid"),
        ({"view": ["x"]}, "view variables"),
        ({"view": ["x", "absent"]}, "view variables"),
        ({"type": "terms"}, "type must"),
        ({"zlim": [1, 0]}, "zlim"),
        ({"zlim": [1]}, "zlim"),
        ({"zlim": [0, np.nan]}, "zlim"),
        ({"cond": {"w": [1, 2]}}, "single value"),
        ({"view": ["x", "g"], "too_far": 0.1}, "numeric view"),
    ],
)
def test_vis_argument_boundaries(model, options, match):
    with pytest.raises(ValueError, match=match):
        prepare_vis_gam_data(model, **options)


def test_vis_fails_unfitted_and_insufficient_view():
    with pytest.raises(RuntimeError, match="not fitted"):
        prepare_vis_gam_data(GAM())
    frame = pd.DataFrame({"x": np.linspace(0, 1, 30), "y": np.linspace(1, 2, 30)})
    one = GAM(formula="y~x", optimize_smoothing=False).fit(data=frame)
    with pytest.raises(ValueError, match="enough terms"):
        prepare_vis_gam_data(one)


def test_vis_conditioning_and_grid_do_not_mutate_training_state(model):
    before = model.vis_training_data_.copy(deep=True)
    result = prepare_vis_gam_data(
        model, view=["x", "z"], cond={"g": "a", "w": 0.1}, n_grid=5
    )
    pd.testing.assert_frame_equal(before, model.vis_training_data_)
    np.testing.assert_array_equal(
        result["newdata"].x, np.tile(result["grid_values"][0], 5)
    )
    np.testing.assert_array_equal(
        result["newdata"].z, np.repeat(result["grid_values"][1], 5)
    )
