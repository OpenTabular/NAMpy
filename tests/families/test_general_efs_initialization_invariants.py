"""Static source evidence for sign-sensitive initial.spg and common-start EFS."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.shash import ShashFamily
from nampy.gam.families.gamlss.twlss import TwlssFamily
from nampy.gam.fit.solvers.general_family import fixed_smoothing
from tests.families._general_efs_invariants import INITIAL_SP
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures"
SOURCE = r"""
library(mgcv);library(jsonlite)
pen.reg<-mgcv:::pen.reg
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
base.setup<-mgcv:::Sl.setup;base.initial<-mgcv:::initial.spg
recorded.initial<-NULL
assignInNamespace('initial.spg',function(...) {
 result<-base.initial(...);recorded.initial<<-result;result
},ns='mgcv')
ans<-list()
for(common in c(FALSE,TRUE)) for(flip in c(FALSE,TRUE)) {
 assignInNamespace('Sl.setup',function(...) {
  result<-base.setup(...)
  if(flip) {
   result[[1]]$D[,3]<- -result[[1]]$D[,3]
   if(!is.null(result[[1]]$Di))result[[1]]$Di[3,]<- -result[[1]]$Di[3,]
  };result
 },ns='mgcv')
 form<-if(p$family=='shash')list(y~s(x,bs='cr',k=6),~s(x,bs='cr',k=5),~1,~1) else
                           list(y~s(x,bs='cr',k=6),~1,~s(x,bs='cr',k=5))
 fam<-if(p$family=='shash')shash() else twlss()
 args<-list(formula=form,data=d,family=fam,method='REML',optimizer='efs',
  control=gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10)))
 if(common)args$in.out<-list(sp=unlist(p$initial_sp),scale=1)
 fit<-do.call(gam,args)
 penalties<-lapply(fit$smooth,function(sm){
  z<-matrix(0,length(coef(fit)),length(coef(fit)))
  j<-sm$first.para:sm$last.para;z[j,j]<-sm$S[[1]];z
 })
 ans[[length(ans)+1]]<-list(initial=recorded.initial,sp=fit$sp,
  design=predict(fit,d,type='lpmatrix'),penalties=penalties,fitted=fitted(fit),
  score=as.numeric(fit$gcv.ubre),edf=sum(fit$edf),iter=fit$iter)
}
write_json(ans,a[2],digits=17,auto_unbox=TRUE)
"""


def source_reference(name):
    data = json.loads((ROOT / f"mgcv_{name}" / "inputs.json").read_text())
    payload = {"family": name, "data": data, "initial_sp": INITIAL_SP}
    key = reference_key("general-efs-sign-v1", {"payload": payload, "source": SOURCE})
    result = load_reference("mgcv_general_efs_invariants", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [Path(directory) / name for name in ("run.R", "in.json", "out.json")]
            source.write_text(SOURCE)
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(_build_r_command(source, str(inputs), str(output)), capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_general_efs_invariants", key, result)
    return data, result


@pytest.mark.parametrize("name", ["shash", "twlss"])
def test_source_default_efs_is_sign_sensitive_but_common_start_is_invariant(name):
    _, captures = source_reference(name)
    original, flipped, common, common_flipped = captures
    # Source physical model/penalties stay identical under the arbitrary sign.
    for capture in captures[1:]:
        np.testing.assert_allclose(capture["design"], original["design"], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(capture["penalties"], original["penalties"], atol=2e-12, rtol=2e-12)
    assert np.max(np.abs(np.asarray(original["initial"]) - flipped["initial"])) > 1
    assert np.max(np.abs(np.asarray(original["fitted"]) - flipped["fitted"])) > 1e-7
    for key in ("fitted", "score", "edf", "sp"):
        np.testing.assert_allclose(common[key], common_flipped[key], atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("name", ["shash", "twlss"])
def test_common_start_efs_prediction_and_penalty_invariants_under_sign_transport(name, monkeypatch):
    from tests.families import test_shash_family_contracts as shash
    from tests.families import test_twlss_family_contracts as twlss
    from tests.families._general_efs_invariants import assert_efs_function_space

    module = shash if name == "shash" else twlss
    data = json.loads((module.ROOT / "inputs.json").read_text())
    payload = {"data": data, "kind": "efs", "initial_sp": INITIAL_SP}
    payload.update({"b": .01, "phi_pen": .001} if name == "shash" else {"link": "log", "a": 1.01, "b": 1.99})
    reference = module.reference(payload)
    original = fixed_smoothing._sl_single_penalty_block

    def flipped(*args, **kwargs):
        block = original(*args, **kwargs)
        if block.start == 2 and block.D.ndim == 2:
            block.D[:, 2] *= -1
            block.Di[2, :] *= -1
        return block

    monkeypatch.setattr(fixed_smoothing, "_sl_single_penalty_block", flipped)
    formula = ["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)", "~1", "~1"] if name == "shash" else ["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"]
    model = GAM(formula=formula, family=ShashFamily() if name == "shash" else TwlssFamily(),
                smoothing_params=INITIAL_SP, smoothing_optimizer="efs", smoothing_method="REML",
                optimize_smoothing=True, control={"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}}).fit(pd.DataFrame(data))
    newdata = pd.DataFrame({"x": np.linspace(-1.4, 1.4, 17)})
    assert_efs_function_space(model, newdata, reference)
    np.testing.assert_allclose(model.predict(newdata, type="response"), reference["prediction"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(model.smoothing_params, reference["sp"], atol=2e-7, rtol=2e-7)


def test_general_efs_sign_capture_provenance():
    path = ROOT / "mgcv_general_efs_invariants"
    manifest = json.loads((path / "manifest.json").read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest["generator_sha256"]
    for name, expected in manifest["fixture_sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
