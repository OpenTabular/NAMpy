"""Censored-logistic fitting references from installed mgcv 1.9-4."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.families.clog import CensoredLogisticFamily
from tests.families.test_mgcv_cnorm_workflows import SCRIPT, _assert_fit, _case, _data
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

SOURCE = SCRIPT.replace("cnorm", "clog").replace(
    "sp=p$sp,method='REML'", "sp=p$sp,method=p$method,scale=p$scale,control=if(is.null(p$control))gam.control() else do.call(gam.control,p$control)"
).replace("pr<-predict(b,nd", "if(!is.null(d$z)) nd$z<-seq(.9,-.9,length.out=19)\n pr<-predict(b,nd", 1)


def reference(case, source_text=SOURCE):
    key = reference_key("clog-workflow-v1", {"case": case, "source_sha256": hashlib.sha256(source_text.encode()).hexdigest()})
    result = load_reference("mgcv_clog_workflows", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            source, inputs, output = [Path(tmp) / n for n in ("run.R", "in.json", "out.json")]
            source.write_text(source_text)
            inputs.write_text(json.dumps(case))
            run = subprocess.run(_build_r_command(source, str(inputs), str(output)), capture_output=True, text=True, timeout=120)
            result = json.loads(output.read_text()) if run.returncode == 0 else {
                "accepted": False, "subprocess_returncode": run.returncode, "stderr": run.stderr
            }
            save_reference("mgcv_clog_workflows", key, result)
    return result


def assert_clog_fit(case, ref, model, *, linear_limit=False):
    _assert_fit(case, ref, model)
    np.testing.assert_allclose(model.family.getTheta(), ref["theta"], rtol=2e-7, atol=2e-7)
    if model.smoothing_score_ is not None:
        np.testing.assert_allclose(model.smoothing_score_, ref["score"], rtol=2e-7, atol=2e-7)
    source_sp = ref["sp"]
    if source_sp is not None and np.asarray(source_sp).size == 0:
        source_sp = ref["full_sp"]
    if linear_limit:
        # This captured nlm fit reaches the smooth's linear null space.
        # Huge raw SP values no longer identify a distinct fitted effect;
        # retain strict prediction/covariance/score checks above and EDF here.
        from nampy.gam.model_state import _edf_total

        np.testing.assert_allclose(np.sum(ref["edf"]), 2, rtol=0, atol=2e-7)
        np.testing.assert_allclose(_edf_total(model), 2, rtol=0, atol=2e-7)
    elif source_sp is not None:
        np.testing.assert_allclose(model.smoothing_params, np.atleast_1d(source_sp), rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("theta,selected", [(1.0, False), (2.0, False), (None, False), (1.0, True), (None, True)])
def test_clog_fitting_matches_mgcv(response, theta, selected):
    case = _case(response, offset=True)
    case.update(theta=theta, sp=None if selected else [0.7], method="REML", scale=1)
    # Freeze every bound as a full-precision string before jsonlite can
    # coerce mixed numeric/string JSON arrays with its default formatting.
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    ref = reference(case)
    assert ref["accepted"], ref
    assert ref["converged"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=theta),
                smoothing_params=case["sp"], optimize_smoothing=selected,
                smoothing_method="REML").fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)
    np.testing.assert_allclose(model.family.getTheta(), ref["theta"], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("method", ["ML", "REML"])
@pytest.mark.parametrize("theta", [1.0, None])
@pytest.mark.parametrize("variant", ["precision", "log", "sqrt", "scale2", "two_smooths", "parametric"])
def test_clog_selection_and_structure(method, theta, variant):
    case = _case("mixed", offset=True)
    case.update(theta=theta, sp=None, method=method, scale=2 if variant == "scale2" else 1)
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    if variant == "precision":
        case["data"]["w"] = np.linspace(0.5, 2, len(case["data"]["y"])).tolist()
        if theta is None:
            case["control"] = {"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}}
    if variant in {"log", "sqrt"}:
        case["link"] = variant
    if variant == "two_smooths":
        case["data"]["z"] = np.cos(np.arange(len(case["data"]["y"])) * 0.731).tolist()
        case["formula"] += "+s(z,bs='cr',k=5)"
    if variant == "parametric":
        case["formula"] = "cbind(y,bound)~x+offset(off)"
    ref = reference(case)
    if theta is None and variant == "precision":
        assert ref["accepted"] and ref["converged"]
        with pytest.raises(NotImplementedError, match="nonunit precision"):
            GAM(formula=case["formula"], family="clog", smoothing_method=method).fit(_data(case), sample_weight="w")
        return
    if theta is None and variant in {"log", "sqrt"}:
        if not ref["accepted"]:
            assert ref["subprocess_returncode"] != 0
            assert "caught segfault" in ref["stderr"]
        with pytest.raises(NotImplementedError, match="estimated.*nonidentity"):
            GAM(formula=case["formula"], family=CensoredLogisticFamily(link=case["link"]),
                smoothing_method=method).fit(_data(case))
        return
    assert ref["accepted"] and ref["converged"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=theta, link=case["link"]),
                smoothing_method=method, optimize_smoothing=True, scale=case["scale"], control=case.get("control")).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)
    from nampy.gam.model_state import _edf2

    if ref["Vc"] is not None:
        np.testing.assert_allclose(model.vcov(unconditional=True), ref["Vc"], rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(_edf2(model), ref["edf2"], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("capture", ["first", "last"])
def test_clog_weighted_off_optimum_joint_derivatives(capture):
    from nampy.gam.fit.selection.criteria.pirls.family_vector_theta import (
        criterion_gradient_ml_reml_pirls_vector_theta_joint,
        criterion_hessian_ml_reml_pirls_vector_theta_joint,
        criterion_ml_reml_pirls_vector_theta_joint,
    )
    from nampy.gam.model_state import _fit_workspace
    from tests.families.test_mgcv_cnorm_workflows import SELECTION_SCRIPT

    source = SELECTION_SCRIPT.replace("cnorm", "clog")
    if capture == "last":
        source = source.replace("&&!exists('.clog_joint',envir=.GlobalEnv,inherits=FALSE)", "")
    case = _case("mixed", offset=True)
    case.update(theta=None, method="REML")
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    case["data"]["w"] = np.linspace(.5, 2, len(case["data"]["y"])).tolist()
    ref = reference(case, source)["joint"]
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=1), smoothing_method="REML",
                optimize_smoothing=True).fit(_data(case), sample_weight="w")
    # The lower-level derivative contract includes theta coordinates even
    # though the public weighted ancillary-optimization route remains guarded.
    model.family.n_theta = 1
    workspace = _fit_workspace(model)
    for name in ("pirls_coef_start", "pirls_eta_start", "pirls_mu_start", "pirls_eval_start", "pirls_eval_eta_start", "pirls_eval_mu_start"):
        setattr(workspace, name, None)
    workspace.pirls_lock_start = True
    parameters = np.asarray(ref["parameters"]).ravel()
    args = model, model.response_, parameters[1:], parameters[:1], "REML"
    for function, key in ((criterion_ml_reml_pirls_vector_theta_joint, "score"),
                          (criterion_gradient_ml_reml_pirls_vector_theta_joint, "gradient"),
                          (criterion_hessian_ml_reml_pirls_vector_theta_joint, "hessian")):
        np.testing.assert_allclose(function(*args), ref[key], rtol=2e-7, atol=2e-7, err_msg=key)


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("theta", [1.0, None])
def test_clog_alternative_optimizer_endpoints(optimizer, theta):
    source = SOURCE.replace("method=p$method,", "method=p$method,optimizer=c('outer',p$optimizer),")
    case = _case("mixed")
    case.update(theta=theta, sp=None, method="REML", scale=1, optimizer=optimizer)
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    if optimizer == "optim":
        case["control"] = {"epsilon": 1e-10, "optim": {"factr": 10}}
    if optimizer == "nlm":
        case["control"] = {"epsilon": 1e-10, "nlm": {"gradtol": 1e-10, "steptol": 1e-10}}
    ref = reference(case, source)
    if (optimizer == "optim" and theta is not None) or (optimizer == "nlm" and theta is None):
        assert ref["accepted"]
        if optimizer == "optim":
            assert not ref["converged"]
        with pytest.raises(NotImplementedError, match="optimizer/ancillary"):
            GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=theta),
                smoothing_optimizer=optimizer, smoothing_method="REML").fit(_data(case))
        return
    assert ref["accepted"] and ref["converged"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=theta),
                smoothing_method="REML", smoothing_optimizer=optimizer,
                control=case.get("control"), optimize_smoothing=True).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model, linear_limit=optimizer == "nlm")
