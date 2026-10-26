"""Scalar Gaussian ML/REML scores retain prior-weight semantics.

Fixed smoothing still optimizes the source likelihood variance. Its Pearson
scale is a different reported quantity; score comparisons use the profiled
criterion rather than plugging that Pearson scale into the likelihood.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import criterion_value
from nampy.gam.fit.selection.criteria.gaussian import criterion_ml_reml_exact
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_weighted_gaussian_criterion"
SCRIPT = r'''
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
b<-gam(y~s(x,bs="cr",k=7)+offset(off),data=d,family=gaussian(),weights=p$weights,sp=.7,method=p$method)
P<-predict(b,type="lpmatrix")
out<-list(version=as.character(packageVersion("mgcv")),score=unname(b$gcv.ubre),criterion_scale=b$reml.scale,pearson_scale=b$sig2,fitted=unname(fitted(b)),prediction_kernel=unname(P%*%b$Vp%*%t(P)),edf=sum(b$edf),prior_weights=b$prior.weights,outer=b$outer.info)
write_json(out,a[2],digits=17,auto_unbox=TRUE)
'''


def inputs():
    return json.loads((ROOT / "inputs.json").read_text())


def payload(method, weight_case):
    frozen = inputs()
    n = len(frozen["data"]["y"])
    weight = {"none": None, "ones": [1.] * n, "constant": [2.] * n, "nonuniform_zero": frozen["weights"]}[weight_case]
    return {"data": frozen["data"], "weights": weight, "method": method, "weight_case": weight_case}


def reference(p):
    key = reference_key("weighted-gaussian-criterion-v1", {"payload": p, "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest()})
    result = load_reference("mgcv_weighted_gaussian_criterion", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            script, inp, out = [Path(tmp) / name for name in ("source.R", "in.json", "out.json")]
            script.write_text(SCRIPT)
            inp.write_text(json.dumps(p))
            run = subprocess.run(["Rscript", str(script), str(inp), str(out)], capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_weighted_gaussian_criterion", key, result)
    return result


@pytest.mark.parametrize("method", ["ML", "REML"])
@pytest.mark.parametrize("weight_case", ["none", "ones", "constant", "nonuniform_zero"])
def test_scalar_gaussian_criterion_matches_mgcv_at_profiled_variance(method, weight_case):
    p = payload(method, weight_case)
    expected = reference(p)
    assert expected["version"] == "1.9.4"
    data = pd.DataFrame(p["data"])
    model = GAM(formula="y~s(x,bs='cr',k=7)+offset(off)", family="gaussian", smoothing_params=[.7], optimize_smoothing=False, smoothing_method="fixed").fit(data=data, sample_weight=p["weights"])
    log_sp = np.log(model.smoothing_params)
    np.testing.assert_allclose(criterion_value(model, data.y.to_numpy(), log_sp, method=method.lower()), expected["score"], rtol=2e-7, atol=2e-7)
    # A direct caller must not enter the unweighted Schur-complement algebra
    # simply because the model's penalty structure supports that algebra.
    np.testing.assert_allclose(criterion_ml_reml_exact(model, data.y.to_numpy(), log_sp, method), expected["score"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(model.predict(data), expected["fitted"], rtol=2e-7, atol=2e-7)
    P = model.lpmatrix(data)
    np.testing.assert_allclose(P @ model.vcov() @ P.T, expected["prediction_kernel"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(model.fit_result().scale, expected["pearson_scale"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(model.fit_result().edf_total, expected["edf"], rtol=2e-7, atol=2e-7)


def test_gaussian_random_design_preserves_unit_penalty_operator():
    from nampy.gam.fit.selection.reparam import build_penalty_reparameterization_state

    data = pd.DataFrame(inputs()["data"])
    model = GAM(formula="y~s(x,bs='cr',k=7)+offset(off)", smoothing_params=[.7], optimize_smoothing=False, smoothing_method="fixed").fit(data=data)
    state = build_penalty_reparameterization_state(model, model.lpmatrix(data), model.smoothing_params)
    q = state.X_range.shape[1]
    penalty = state.St[:q, :q]
    coefficient_map = np.linalg.lstsq(state.X_range, state.Z_rand, rcond=None)[0]
    np.testing.assert_allclose(state.X_range @ coefficient_map, state.Z_rand, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(coefficient_map.T @ penalty @ coefficient_map, np.eye(q), rtol=2e-11, atol=2e-11)
    # The random-effect covariance kernel is independent of coordinates.
    kernel = state.X_range @ np.linalg.solve(penalty, state.X_range.T)
    np.testing.assert_allclose(state.Z_rand @ state.Z_rand.T, kernel, rtol=2e-11, atol=2e-11)
