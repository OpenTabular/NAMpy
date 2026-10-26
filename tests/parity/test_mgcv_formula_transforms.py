"""R model-frame transformations retain fitted state when predicting."""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import _build_r_command, _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = [pytest.mark.surface_regression]


def _data():
    rng = np.random.default_rng(913)
    x = np.linspace(-1.8, 2.1, 90)
    frame = pd.DataFrame(
        {"x": x, "g": np.resize([1, 2, 3], len(x)), "h": np.resize(["b", "a"], len(x))}
    )
    frame["y"] = (
        0.4 + 0.7 * x - 0.2 * x * x + 0.2 * frame.g + rng.normal(0, 0.3, len(x))
    )
    new = pd.DataFrame(
        {"x": [-2.2, -0.13, 0.47, 2.4], "g": [3, 1, 2, 3], "h": ["a", "b", "b", "a"]}
    )
    return frame, new


FORMULAS = [
    "y ~ factor(g)",
    "y ~ factor(g,levels=[4,3,1,2]) + x",
    "y ~ factor(g,levels=[3,1,2],labels=['high','low','mid'])",
    "y ~ ordered(g,levels=[3,1,2])",
    "y ~ factor(cut(x,breaks=[-3,0,3],labels=False)) + g",
    "y ~ interaction(factor(g),factor(h)) + x",
    "y ~ poly(exp(x),degree=3)",
    "y ~ factor(g, levels=[3,1,2]) + x",
    "y ~ ordered(g) + x",
    "y ~ ordered(g) * x",
    "y ~ factor(g) * factor(h)",
    "y ~ interaction(g,h)",
    "y ~ interaction(g,h,drop=True)",
    "y ~ cut(x,breaks=[-3,0,3],labels=['low','high']) + g",
    "y ~ cut(x,breaks=[-3,0,3],labels=False) + g",
    "y ~ poly(x,3)",
    "y ~ poly(x,degree=3,raw=True)",
    "y ~ poly(x,3) * factor(g)",
    "y ~ s(x,k=7,bs='cr') + factor(g)",
    "y ~ s(x,k=7,bs='cr',by=ordered(g))",
]


def _reference(formula, data, new):
    payload = {
        "formula": formula,
        "data": data.to_dict("list"),
        "new": new.to_dict("list"),
    }
    key = reference_key("rhs-transforms-v1", payload, normalize_floats=True)
    result = load_reference("mgcv_formula_transforms", key)
    if result is not None:
        return result
    code = """
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);nd<-as.data.frame(p$new)
f<-as.formula(p$formula);b<-gam(f,data=d,sp=if(grepl("s(",p$formula,fixed=TRUE)) rep(.4,if(grepl("by",p$formula)) 2 else 1) else NULL,method="REML")
pr<-predict(b,nd,se.fit=TRUE)
write_json(list(X=unname(predict(b,type="lpmatrix")),Xp=unname(predict(b,nd,type="lpmatrix")),
fit=unname(pr$fit),se=unname(pr$se.fit),Vp=unname(b$Vp),version=as.character(packageVersion("mgcv"))),a[2],digits=17)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "reference.R"
        inp = root / "input.json"
        out = root / "out.json"
        script.write_text(code)
        inp.write_text(
            json.dumps({**payload, "formula": _normalize_python_formula_text(formula)})
        )
        process = subprocess.run(
            _build_r_command(script, str(inp), str(out)), capture_output=True, text=True
        )
        assert process.returncode == 0, process.stderr
        result = json.loads(out.read_text())
    save_reference("mgcv_formula_transforms", key, result)
    return result


@pytest.mark.parametrize("formula", FORMULAS)
def test_fitted_formula_transforms_match_mgcv(formula):
    data, new = _data()
    expected = _reference(formula, data, new)
    model = GAM(
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=0.4,
    )
    model.fit(data=data)
    np.testing.assert_allclose(
        model.predict(type="lpmatrix"), expected["X"], atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        model.predict(new, type="lpmatrix"), expected["Xp"], atol=2e-10, rtol=2e-10
    )
    fit, se = model.predict(new, return_se=True)
    np.testing.assert_allclose(fit, expected["fit"], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(se, expected["se"], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(model.vcov(), expected["Vp"], atol=2e-9, rtol=2e-9)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(new), fit)
    np.testing.assert_allclose(model.predict(new.iloc[[2]]), fit[[2]], atol=1e-12)


@pytest.mark.parametrize("expr", ["factor(g)", "ordered(g)", "interaction(g,h)"])
def test_formula_factor_transform_rejects_unseen_prediction_levels(expr):
    data, new = _data()
    model = GAM(formula=f"y ~ {expr}", optimize_smoothing=False).fit(data=data)
    new.loc[0, "g"] = 4
    with pytest.raises(ValueError, match="unseen|missing"):
        model.predict(new)


@pytest.mark.parametrize(
    "expr,match",
    [
        ("cut(x,breaks=1)", "invalid number"),
        ("cut(x,breaks=[-3,0,0,3])", "unique"),
        ("poly(x,100)", "unique points"),
        ("poly(x,2,3)", "same length"),
        ("factor(g,exclude=None)", "Unsupported factor"),
    ],
)
def test_unsupported_formula_transform_options_fail_explicitly(expr, match):
    data, _ = _data()
    with pytest.raises((ValueError, NotImplementedError), match=match):
        GAM(formula=f"y ~ {expr}", optimize_smoothing=False).fit(data=data)


@pytest.mark.parametrize("expr", ["ordered(g)", "factor(g)", "poly(x,3)"])
def test_formula_transform_missing_prediction_rows_follow_na_pass(expr):
    data, new = _data()
    model = GAM(formula=f"y ~ {expr}", optimize_smoothing=False).fit(data=data)
    expected = model.predict(new)
    column = "x" if expr.startswith("poly") else "g"
    new[column] = new[column].astype(float)
    new.loc[1, column] = np.nan
    prediction = model.predict(new)
    assert np.isnan(prediction[1])
    np.testing.assert_allclose(prediction[[0, 2, 3]], expected[[0, 2, 3]], atol=1e-12)


@pytest.mark.parametrize("expr", ["factor(g)", "ordered(g)"])
def test_single_level_formula_factors_cannot_supply_contrasts(expr):
    data, _ = _data()
    data["g"] = 1
    with pytest.raises(ValueError, match="at least two levels"):
        GAM(formula=f"y ~ {expr}", optimize_smoothing=False).fit(data=data)
