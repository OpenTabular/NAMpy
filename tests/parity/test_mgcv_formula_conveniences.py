"""Committed R references for merged factors, cut and multivariate poly."""

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
from nampy.gam.formula.transforms import evaluate_transform
from tests.mgcv_parity_utils import _build_r_command, _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = [pytest.mark.surface_regression]


def _r_expression(text):
    return (
        _normalize_python_formula_text(text)
        .replace("include_lowest", "include.lowest")
        .replace("dig_lab", "dig.lab")
        .replace("lex_order", "lex.order")
    )


def _reference(payload):
    key = reference_key("formula-conveniences-v1", payload, normalize_floats=True)
    result = load_reference("mgcv_formula_conveniences", key)
    if result is not None:
        return result
    code = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
if(p$mode=='expression') {
  x<-eval(parse(text=p$expression),d)
  ans<-list(values=if(is.factor(x)) as.character(x) else unname(x),levels=levels(x),ordered=is.ordered(x))
} else {
  b<-gam(as.formula(p$formula),data=d,method='REML');nd<-as.data.frame(p$new)
  pr<-tryCatch(predict(b,nd,se.fit=TRUE),error=function(e) conditionMessage(e))
  one<-tryCatch(predict(b,nd[2,,drop=FALSE]),error=function(e) conditionMessage(e))
  ans<-list(X=unname(predict(b,type='lpmatrix')),Vp=unname(b$Vp),
            fit=if(is.character(pr)) NULL else unname(pr$fit),
            se=if(is.character(pr)) NULL else unname(pr$se.fit),
            Xp=if(is.character(pr)) NULL else unname(predict(b,nd,type='lpmatrix')),
            error=if(is.character(pr)) pr else NULL,
            one=if(is.character(one)) NULL else unname(one),
            one_error=if(is.character(one)) one else NULL)
}
ans$version<-as.character(packageVersion('mgcv'))
write_json(ans,a[2],digits=17,auto_unbox=FALSE,null='null')
"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        script = base / "ref.R"
        inputs = base / "in.json"
        output = base / "out.json"
        script.write_text(code)
        rpayload = dict(payload)
        for name in ("formula", "expression"):
            if name in rpayload:
                rpayload[name] = _r_expression(rpayload[name])
        inputs.write_text(json.dumps(rpayload))
        process = subprocess.run(
            _build_r_command(script, str(inputs), str(output)),
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr
        result = json.loads(output.read_text())
    save_reference("mgcv_formula_conveniences", key, result)
    return result


def _data():
    rng = np.random.default_rng(817)
    data = pd.DataFrame(
        {
            "x": np.linspace(-2, 2, 120),
            "z": rng.uniform(-1, 1, 120),
            "w": rng.normal(size=120),
            "g": np.resize([1, 2, 3, 4], 120),
        }
    )
    data["y"] = (
        0.3
        + 0.4 * data.x
        - 0.2 * data.x**2
        + 0.5 * data.z
        + 0.2 * data.x * data.z
        + rng.normal(0, 0.2, 120)
    )
    new = pd.DataFrame(
        {
            "x": [-2, -0.4, 0.8, 2],
            "z": [0.6, -0.2, 0.5, -0.7],
            "w": [-1, 0.2, 0.7, -0.4],
            "g": [4, 2, 3, 1],
        }
    )
    return data, new


FORMULAS = [
    "y ~ factor(g,labels=['a','a','b','b']) + x",
    "y ~ ordered(g,labels=['low','mid','mid','high']) + x",
    "y ~ factor(g,levels=[4,3,2,1],labels=['a','b','b','a']) * x",
    "y ~ factor(g,labels='group') + x",
    "y ~ cut(x,breaks=3)",
    "y ~ cut(x,breaks=3,labels=False) + z",
    "y ~ cut(x,breaks=3,labels=['a','b','c']) + z",
    "y ~ cut(x,breaks=[-3,0,3]) + z",
    "y ~ cut(x,breaks=[-3,0,3],right=False) * z",
    "y ~ cut(x,breaks=[-3,-1,1,3],ordered_result=True) + z",
    "y ~ poly(x,z,degree=2)",
    "y ~ poly(x,z,degree=3,raw=True)",
    "y ~ poly(x,z,w,degree=3)",
    "y ~ poly(cbind(x,z),degree=3)",
    "y ~ poly(cbind(x,z),2,raw=True)",
    "y ~ poly(cbind(x,z,w),degree=2)",
    "y ~ poly(exp(x),z,degree=2) * factor(g)",
]


@pytest.mark.parametrize("formula", FORMULAS)
def test_formula_convenience_fits_prediction_covariance_and_saved_state(formula):
    data, new = _data()
    reference = _reference(
        {
            "mode": "model",
            "formula": formula,
            "data": data.to_dict("list"),
            "new": new.to_dict("list"),
        }
    )
    model = GAM(formula=formula, optimize_smoothing=False).fit(data=data)
    np.testing.assert_allclose(
        model.predict(type="lpmatrix"), reference["X"], atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(model.vcov(), reference["Vp"], atol=2e-9, rtol=2e-9)
    if reference["error"] is not None:
        with pytest.raises(ValueError, match="unseen|missing|labels|degree|columns"):
            model.predict(new)
        return
    prediction, se = model.predict(new, return_se=True)
    np.testing.assert_allclose(
        model.predict(new, type="lpmatrix"), reference["Xp"], atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(prediction, reference["fit"], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(se, reference["se"], atol=2e-9, rtol=2e-9)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(new), prediction)
    if reference["one_error"] is not None:
        with pytest.raises(ValueError, match="unseen|missing|labels|degree|columns"):
            model.predict(new.iloc[[1]])
    else:
        np.testing.assert_allclose(
            model.predict(new.iloc[[1]]), reference["one"], atol=2e-9, rtol=2e-9
        )
    # Subset evaluation must not replace the training polynomial/factor state.
    np.testing.assert_array_equal(model.predict(new), prediction)


EXPRESSION_CASES = [
    (
        "cut(x,breaks=[1,1.00000000000001,1.00000000000002])",
        [1.000000000000005, 1.000000000000015],
    ),
    ("cut(x,breaks=[1.23456,2.34567,3.45678],dig_lab=5)", [1.5, 3.0]),
    ("factor(x,labels=[1,1,2])", [1, 2, 3, 1]),
    ("factor(x,labels=[1,'1','two'])", [1, 2, 3, 1]),
    ("factor(x)", [1, 2, 3, 1]),
    ("cut(x,breaks=3,labels=[1,1,2])", [1.0, 2.0, 3.0, 4.0]),
    ("cut(x,breaks=3)", [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ("cut(x,breaks=4)", [0.0, 0.0, 0.0]),
    ("cut(x,breaks=4)", [-2.0, -2.0, -2.0]),
    ("cut(x,breaks=4.7)", [1.0, 2.0, 3.0, 4.0]),
    ("cut(x,breaks=[0,1,2],include_lowest=True)", [0.0, 0.2, 1.0, 2.0]),
    ("cut(x,breaks=[0,1,2],right=False,include_lowest=True)", [0.0, 0.2, 1.0, 2.0]),
    ("cut(x,breaks=[10000,20000,30000])", [10001.0, 19000.0, 20000.0, 25000.0]),
    ("cut(x,breaks=[0.000001,0.000002,0.000003])", [0.0000011, 0.0000025]),
    ("cut(x,breaks=[1,1.00001,1.00002])", [1.000005, 1.000015]),
    ("factor(x,labels=['same','same','last'])", [1, 2, 3, 1]),
    ("ordered(x,labels=['same','same','last'])", [1, 2, 3, 1]),
    ("cut(x,breaks=3,labels=['same','same','last'])", [1.0, 2.0, 3.0, 4.0]),
]


@pytest.mark.parametrize("expression,values", EXPRESSION_CASES)
def test_cut_labels_boundaries_and_merged_factor_levels_match_r(expression, values):
    data = pd.DataFrame({"x": values})
    reference = _reference(
        {"mode": "expression", "expression": expression, "data": data.to_dict("list")}
    )
    result, _ = evaluate_transform(expression, data)
    if isinstance(result, pd.Categorical):
        assert result.tolist() == reference["values"]
        assert list(result.categories) == reference["levels"]
        assert result.ordered == reference["ordered"][0]
    else:
        np.testing.assert_allclose(result, reference["values"], atol=2e-12, rtol=2e-12)
