"""Censored normal workflows captured from installed CRAN mgcv 1.9-4."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_cnorm_workflows"


def _case(response="mixed", offset=False):
    data = json.loads((ROOT / "inputs.json").read_text())
    if response == "vector":
        data["y"] = data["vector"]
    lhs = "cbind(y,bound)" if response == "mixed" else "y"
    return {
        "data": data,
        "response": response,
        "offset": offset,
        "formula": lhs + "~s(x,bs='cr',k=7)" + ("+offset(off)" if offset else ""),
        "theta": 1.0,
        "link": "identity",
        "sp": [0.7],
    }


SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);d$bound<-as.numeric(d$bound)
warnings<-character();warning_calls<-character()
result<-tryCatch(withCallingHandlers({
 b<-gam(as.formula(p$formula),data=d,family=do.call(cnorm,list(theta=p$theta,link=p$link)),weights=d$w,sp=p$sp,method='REML')
 nd<-data.frame(x=seq(-.93,.93,length.out=19),off=seq(-.08,.08,length.out=19))
 pr<-predict(b,nd,type='response',se.fit=TRUE);pl<-predict(b,nd,type='link',se.fit=TRUE)
 X<-predict(b,type='lpmatrix');P<-predict(b,nd,type='lpmatrix');su<-summary(b)
 rr<-lapply(c('deviance','working','response','pearson'),function(t)tryCatch(residuals(b,type=t),error=function(e)list(error=conditionMessage(e))))
 list(accepted=TRUE,version=as.character(packageVersion('mgcv')),coef=coef(b),X=X,P=P,
 theta=b$family$getTheta(),physical_theta=b$family$getTheta(TRUE),sp=b$sp,full_sp=b$full.sp,
 edf=b$edf,edf1=b$edf1,Vp=b$Vp,Ve=b$Ve,Vc=b$Vc,edf2=b$edf2,
 eta=b$linear.predictors,mu=fitted(b),deviance=deviance(b),null_deviance=b$null.deviance,
 loglik=as.numeric(logLik(b)),loglik_df=attr(logLik(b),'df'),aic=AIC(b),bic=BIC(b),
 score=as.numeric(b$gcv.ubre),scale=b$sig2,working_weights=b$working.weights,
 fisher_weights=b$weights,working_response=b$linear.predictors-b$offset+b$residuals,
 residuals=rr,p_table=su$p.table,s_table=su$s.table,r_sq=su$r.sq,dev_expl=su$dev.expl,
 prediction=pr$fit,se=pr$se.fit,link_prediction=pl$fit,link_se=pl$se.fit,
 raw_deviance=b$family$dev.resids(b$y,fitted(b),b$prior.weights),
 ls=b$family$ls(b$y,b$prior.weights,b$family$getTheta(),1),
 family_aic=b$family$aic(b$y,fitted(b),wt=b$prior.weights),
 covariance_kernel=P%*%b$Vp%*%t(P),converged=b$converged,iter=b$iter)
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));warning_calls<<-c(warning_calls,paste(deparse(conditionCall(w)),collapse=" "));invokeRestart('muffleWarning')}),
error=function(e)list(accepted=FALSE,error=conditionMessage(e),version=as.character(packageVersion('mgcv'))))
result$warnings<-warnings;result$warning_calls<-warning_calls;write_json(result,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case, *, script=SCRIPT):
    key = reference_key(
        "cnorm-workflows-v1",
        {"case": case, "script_sha256": hashlib.sha256(script.encode()).hexdigest()},
    )
    result = load_reference("mgcv_cnorm_workflows", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [
                Path(directory) / n for n in ("ref.R", "input.json", "out.json")
            ]
            source.write_text(script)
            inputs.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(source, str(inputs), str(output)),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_cnorm_workflows", key, result)
    return result


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("offset", [False, True])
def test_cnorm_fixed_sp_upstream_inventory(response, offset):
    reference = _reference(_case(response, offset))
    assert reference["accepted"], reference
    assert reference["version"] == "1.9.4"
    assert reference["converged"]
    warnings = reference["warnings"]
    warnings = [warnings] if isinstance(warnings, str) else warnings
    expected = [
        "Pearson residuals not available for this family - returning deviance residuals"
    ]
    if response == "mixed":
        expected = [
            "number of items to replace is not a multiple of replacement length"
        ] * 2 + expected
    assert warnings == expected
    assert np.isfinite(reference["coef"]).all()
    assert reference["physical_theta"] == 1


def _data(case):
    import pandas as pd

    data = pd.DataFrame(case["data"])
    data["bound"] = np.asarray(data["bound"], dtype=float)
    return data


def _newdata():
    import pandas as pd

    return pd.DataFrame(
        {
            "x": np.linspace(-0.93, 0.93, 19),
            "off": np.linspace(-0.08, 0.08, 19),
            "z": np.linspace(0.9, -0.9, 19),
        }
    )


def _fit(case, *, explicit=False):
    from nampy.gam import GAM
    from nampy.gam.families.cnorm import CensoredNormalFamily

    data = _data(case)
    options = {
        "family": CensoredNormalFamily(theta=case["theta"], link=case["link"]),
        "smoothing_params": case["sp"]
        if case.get("method", "fixed") == "fixed"
        else None,
        "optimize_smoothing": case.get("method", "fixed") != "fixed",
        "smoothing_method": case.get("method", "fixed"),
        "scale": case.get("scale", 0),
    }
    if not explicit:
        return GAM(formula=case["formula"], **options).fit(data=data, sample_weight="w")
    response = (
        data[["y", "bound"]].to_numpy()
        if case["response"] == "mixed"
        else data["y"].to_numpy()
    )
    # Formula response is supplied explicitly; the RHS remains identical.
    return GAM(formula=case["formula"], **options).fit(
        data=data, y=response, sample_weight="w"
    )


def _assert_fit(case, reference, model):
    from nampy.gam.inference.null_deviance import null_deviance
    from nampy.gam.model_state import _coef_full, _deviance, _edf_total, _fit_state

    checks = {
        "coef": _coef_full(model),
        "Vp": model.vcov(),
        "Ve": model.vcov(freq=True),
        "eta": model.predict(type="link"),
        "mu": model.predict(type="response"),
        "deviance": _deviance(model),
        "null_deviance": null_deviance(model),
        "aic": model.aic(),
        "bic": model.bic(),
        "loglik": model.loglik(),
    }
    for name, actual in checks.items():
        np.testing.assert_allclose(
            actual, reference[name], atol=2e-7, rtol=2e-7, err_msg=name
        )
    np.testing.assert_allclose(
        _edf_total(model), np.sum(reference["edf"]), atol=2e-7, rtol=2e-7
    )
    state = _fit_state(model)
    for name in ("working_response", "working_weights", "fisher_weights"):
        if reference[name] is not None:
            np.testing.assert_allclose(
                getattr(state, name),
                reference[name],
                atol=2e-7,
                rtol=2e-7,
                err_msg=name,
            )
    for kind, key in (("response", "prediction"), ("link", "link_prediction")):
        predicted, se = model.predict(_newdata(), type=kind, return_se=True)
        np.testing.assert_allclose(predicted, reference[key], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(
            se,
            reference["se" if kind == "response" else "link_se"],
            atol=2e-7,
            rtol=2e-7,
        )
    for index, kind in enumerate(("deviance", "working", "response")):
        expected = reference["residuals"][index]
        if isinstance(expected, dict):
            continue
        np.testing.assert_allclose(
            model.residuals(type=kind), expected, atol=2e-7, rtol=2e-7, err_msg=kind
        )
    P = model.predict(_newdata(), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, reference["covariance_kernel"], atol=2e-7, rtol=2e-7
    )


def _reordered_case(kind):
    case = _case("mixed", True)
    n = len(case["data"]["y"])
    if kind == "permutation":
        order = np.r_[np.arange(0, n, 2), np.arange(1, n, 2)][::-1]
    elif kind == "subset":
        order = np.flatnonzero(np.arange(n) % 3 != 0)
    elif kind == "duplicate":
        order = np.r_[np.arange(n), [5, 6, 7, 8, 9] * 3]
    else:
        raise ValueError(kind)
    case["data"] = {
        key: [values[i] for i in order] for key, values in case["data"].items()
    }
    return case


@pytest.mark.parametrize("kind", ["permutation", "subset", "duplicate"])
def test_cnorm_upstream_aligned_response_rows(kind):
    reference = _reference(_reordered_case(kind))
    assert reference["accepted"], reference
    assert reference["converged"]
    assert np.isfinite(reference["coef"]).all()
    if kind == "permutation":
        baseline = _reference(_case("mixed", True))
        for name in ("coef", "Vp", "Ve", "prediction", "aic", "deviance"):
            np.testing.assert_allclose(
                reference[name], baseline[name], atol=2e-10, rtol=2e-10
            )


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("offset", [False, True])
def test_cnorm_fixed_sp_matches_mgcv(response, offset):
    case = _case(response, offset)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("offset", [False, True])
def test_cnorm_explicit_response_matches_mgcv(response, offset):
    case = _case(response, offset)
    _assert_fit(case, _reference(case), _fit(case, explicit=True))


@pytest.mark.parametrize("kind", ["permutation", "subset", "duplicate"])
def test_cnorm_aligned_response_rows_match_mgcv(kind):
    case = _reordered_case(kind)
    _assert_fit(case, _reference(case), _fit(case, explicit=True))


def test_cnorm_response_survives_pickle_and_failed_refit():
    import pickle

    case = _case("mixed", False)
    model = _fit(case, explicit=True)
    restored = pickle.loads(pickle.dumps(model))
    _assert_fit(case, _reference(case), restored)
    data = _data(case)
    malformed = data[["y", "bound"]].to_numpy()
    malformed[7, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        restored.fit(data=data, y=malformed, sample_weight="w")
    _assert_fit(case, _reference(case), restored)


@pytest.mark.parametrize("first", ["vector", "mixed"])
def test_cnorm_refit_replaces_response_metadata(first):
    case = _case(first, False)
    model = _fit(case, explicit=True)
    second = "mixed" if first == "vector" else "vector"
    new_case = _case(second, False)
    data = _data(new_case)
    response = (
        data[["y", "bound"]].to_numpy() if second == "mixed" else data["y"].to_numpy()
    )
    model.fit(data=data, y=response, sample_weight="w")
    _assert_fit(new_case, _reference(new_case), model)


SELECTION_SCRIPT = (
    SCRIPT.replace(
        " b<-gam(",
        """
 trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,
  tracer=quote(.cnorm_inputsp<-sp),exit=quote({rv<-returnValue()
   if(deriv==2&&!is.null(rv$REML2)&&!exists('.cnorm_joint',envir=.GlobalEnv,inherits=FALSE))
    assign('.cnorm_joint',list(parameters=.cnorm_inputsp,score=rv$REML,
      gradient=rv$REML1,hessian=rv$REML2,coef=rv$coefficients,
      mu=rv$fitted.values,eta=rv$linear.predictors,method=scoreType),envir=.GlobalEnv)
  }))
 b<-gam(""",
    )
    .replace(
        "sp=p$sp,method='REML'",
        "sp=if(p$method=='fixed')p$sp else NULL,method=if(p$method=='fixed')'REML' else p$method",
    )
    .replace(
        "list(accepted=TRUE,",
        "list(joint=if(exists('.cnorm_joint',envir=.GlobalEnv,inherits=FALSE)).cnorm_joint else NULL,accepted=TRUE,",
    )
)


def _selection_case(
    response="mixed", *, theta=None, method="REML", offset=False, link="identity"
):
    case = _case(response, offset)
    case.update(theta=theta, method=method, link=link)
    return case


def _selection_reference(case):
    return _reference(case, script=SELECTION_SCRIPT)


SELECTION_CASES = (
    [
        {"response": response, "theta": 2.0, "method": "fixed"}
        for response in ("vector", "mixed")
    ]
    + [
        {"response": response, "theta": None, "method": method}
        for response in ("vector", "mixed")
        for method in ("fixed", "REML", "ML")
    ]
    + [{"response": "mixed", "theta": None, "method": "REML", "offset": True}]
)


@pytest.mark.parametrize("options", SELECTION_CASES)
def test_cnorm_upstream_scale_selection_inventory(options):
    case = _selection_case(**options)
    reference = _selection_reference(case)
    assert reference["accepted"], reference
    assert reference["converged"]
    assert np.isfinite(reference["coef"]).all()
    assert np.isfinite(reference["physical_theta"]).all()
    if options["theta"] is None:
        assert reference["joint"] is not None
        assert np.isfinite(reference["joint"]["hessian"]).all()


@pytest.mark.parametrize("options", SELECTION_CASES)
def test_cnorm_scale_selection_matches_mgcv(options):
    from nampy.gam.model_state import _edf2

    case = _selection_case(**options)
    reference = _selection_reference(case)
    model = _fit(case)
    _assert_fit(case, reference, model)
    np.testing.assert_allclose(
        model.family.getTheta(), np.atleast_1d(reference["theta"]), atol=2e-7, rtol=2e-7
    )
    if case["method"] != "fixed":
        np.testing.assert_allclose(
            model.smoothing_params, np.atleast_1d(reference["sp"]), atol=2e-7, rtol=2e-7
        )
    if model._optim_result is not None:
        np.testing.assert_allclose(
            model._optim_result.fun, reference["score"], atol=2e-7, rtol=2e-7
        )
    if reference["Vc"] is not None:
        np.testing.assert_allclose(
            model.vcov(unconditional=True), reference["Vc"], atol=2e-7, rtol=2e-7
        )
        np.testing.assert_allclose(
            _edf2(model), reference["edf2"], atol=2e-7, rtol=2e-7
        )


EXPANSION_CASES = (
    [
        {"response": response, "link": link, "weights": "unit"}
        for response in ("vector", "mixed")
        for link in ("log", "sqrt")
    ]
    + [
        {"response": response, "link": "identity", "weights": "precision"}
        for response in ("vector", "mixed")
    ]
    + [{"response": "mixed", "link": "identity", "weights": "zero"}]
)


def _expansion_case(response="mixed", link="identity", weights="unit"):
    case = _case(response, True)
    case["link"] = link
    if weights != "unit":
        n = len(case["data"]["y"])
        case["data"]["w"] = (0.5 + np.arange(n) % 4 * 0.5).tolist()
        if weights == "zero":
            for i in (0, 6, 7, 8):
                case["data"]["w"][i] = 0.0
    return case


@pytest.mark.parametrize("options", EXPANSION_CASES)
def test_cnorm_upstream_link_precision_inventory(options):
    reference = _reference(_expansion_case(**options))
    assert reference["version"] == "1.9.4"
    if options["weights"] != "zero":
        assert reference["accepted"], reference
        assert reference["converged"]
        assert np.isfinite(reference["coef"]).all()
    else:
        assert not reference["accepted"]
        assert reference["error"] == "inner loop 1; can't correct step size"


@pytest.mark.parametrize("options", EXPANSION_CASES[:-1])
def test_cnorm_link_precision_matches_mgcv(options):
    case = _expansion_case(**options)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("options", SELECTION_CASES[2:])
def test_cnorm_off_optimum_joint_derivatives_match_mgcv(options):
    from nampy.gam.fit.selection.criteria.pirls.family_vector_theta import (
        criterion_gradient_ml_reml_pirls_vector_theta_joint,
        criterion_hessian_ml_reml_pirls_vector_theta_joint,
        criterion_ml_reml_pirls_vector_theta_joint,
    )
    from nampy.gam.model_state import _fit_workspace

    case = _selection_case(**options)
    reference = _selection_reference(case)["joint"]
    # Build a free-SP map to compare gam.fit4's full derivative arrays,
    # including SP coordinates subsequently fixed by the outer caller.
    model_case = dict(
        case, method="REML" if case["method"] == "fixed" else case["method"]
    )
    model = _fit(model_case)
    parameters = np.asarray(reference["parameters"]).ravel()
    workspace = _fit_workspace(model)
    for name in (
        "pirls_coef_start",
        "pirls_eta_start",
        "pirls_mu_start",
        "pirls_eval_start",
        "pirls_eval_eta_start",
        "pirls_eval_mu_start",
    ):
        setattr(workspace, name, None)
    workspace.pirls_lock_start = True
    args = (
        model,
        model.response_,
        parameters[1:],
        parameters[:1],
        "ML" if case["method"] == "ML" else "REML",
    )
    for function, key in (
        (criterion_ml_reml_pirls_vector_theta_joint, "score"),
        (criterion_gradient_ml_reml_pirls_vector_theta_joint, "gradient"),
        (criterion_hessian_ml_reml_pirls_vector_theta_joint, "hessian"),
    ):
        np.testing.assert_allclose(
            function(*args), reference[key], atol=2e-7, rtol=2e-7, err_msg=key
        )
    np.testing.assert_allclose(
        workspace.pirls_last_coef, reference["coef"], atol=2e-7, rtol=2e-7
    )


def test_cnorm_upstream_vector_zero_precision_inventory():
    reference = _reference(_expansion_case(response="vector", weights="zero"))
    assert reference["version"] == "1.9.4"


@pytest.mark.parametrize("method", ["REML", "ML"])
def test_cnorm_upstream_fixed_sd_selection_inventory(method):
    reference = _selection_reference(_selection_case(theta=2.0, method=method))
    assert reference["accepted"], reference
    assert reference["converged"]
    assert reference["physical_theta"] == 2


@pytest.mark.parametrize("method", ["REML", "ML"])
def test_cnorm_fixed_sd_selection_matches_mgcv(method):
    test_cnorm_scale_selection_matches_mgcv({"theta": 2.0, "method": method})


HARD_SCRIPT = SELECTION_SCRIPT.replace(
    "weights=d$w,", "weights=d$w,scale=if(is.null(p$scale))0 else p$scale,"
)
HARD_CASES = [
    {"kind": kind, "estimated": estimated}
    for kind in ("two_smooths", "no_intercept", "no_smooths", "rank", "scale2")
    for estimated in (False, True)
]


def _hard_case(kind, estimated=False):
    case = _selection_case(theta=None if estimated else 1.0, method="REML", offset=True)
    if kind == "two_smooths":
        case["data"] = json.loads((ROOT / "extension_inputs.json").read_text())
        case["formula"] = "cbind(y,bound)~s(x,bs='cr',k=7)+s(z,bs='cr',k=6)+offset(off)"
        case["sp"] = [0.7, 0.8]
    elif kind == "no_intercept":
        case["formula"] += "-1"
    elif kind == "no_smooths":
        case["formula"] = "cbind(y,bound)~x+offset(off)"
        case["sp"] = []
    elif kind == "rank":
        case["data"]["z"] = list(case["data"]["x"])
        case["formula"] = "cbind(y,bound)~x+z+s(x,bs='cr',k=7)+offset(off)"
        if not estimated:
            case["method"] = "fixed"
    elif kind == "scale2":
        case["scale"] = 2.0
    else:
        raise ValueError(kind)
    return case


def _hard_reference(case):
    script = HARD_SCRIPT
    if "z" in case["data"]:
        script = script.replace(
            "off=seq(-.08,.08,length.out=19))",
            "off=seq(-.08,.08,length.out=19),z=seq(.9,-.9,length.out=19))",
        )
    return _reference(case, script=script)


@pytest.mark.parametrize("options", HARD_CASES)
def test_cnorm_upstream_structural_scale_inventory(options):
    reference = _hard_reference(_hard_case(**options))
    assert reference["version"] == "1.9.4"


def test_cnorm_estimated_endpoint_persistence():
    import pickle

    case = _selection_case(offset=True)
    reference = _selection_reference(case)
    model = pickle.loads(pickle.dumps(_fit(case)))
    _assert_fit(case, reference, model)
    np.testing.assert_allclose(
        model.family.getTheta(), np.atleast_1d(reference["theta"]), atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        model.vcov(unconditional=True), reference["Vc"], atol=2e-7, rtol=2e-7
    )
    assert not model.response_.values.flags.writeable
    assert not model.response_.bounds.flags.writeable


@pytest.mark.parametrize("options", HARD_CASES)
def test_cnorm_structural_scale_matches_mgcv(options):
    from nampy.gam.model_state import _edf2

    case = _hard_case(**options)
    reference = _hard_reference(case)
    assert reference["accepted"], reference
    if options["kind"] == "scale2" and options["estimated"]:
        with pytest.raises(NotImplementedError, match="nonunit external dispersion"):
            _fit(case)
        return
    if options["kind"] == "rank" and options["estimated"]:
        with pytest.raises(NotImplementedError, match="rank-deficient"):
            _fit(case)
        return
    model = _fit(case)
    _assert_fit(case, reference, model)
    np.testing.assert_allclose(
        model.family.getTheta(), np.atleast_1d(reference["theta"]), atol=2e-7, rtol=2e-7
    )
    if case["method"] != "fixed":
        np.testing.assert_allclose(
            model.smoothing_params, np.atleast_1d(reference["sp"]), atol=2e-7, rtol=2e-7
        )
    if model._optim_result is not None:
        np.testing.assert_allclose(
            model._optim_result.fun, reference["score"], atol=2e-7, rtol=2e-7
        )
    if reference["Vc"] is not None:
        np.testing.assert_allclose(
            model.vcov(unconditional=True), reference["Vc"], atol=2e-7, rtol=2e-7
        )
        np.testing.assert_allclose(
            _edf2(model), reference["edf2"], atol=2e-7, rtol=2e-7
        )


@pytest.mark.parametrize("method", ["GCV.Cp", "P-ML"])
def test_cnorm_upstream_criterion_alias_inventory(method):
    reference = _selection_reference(_selection_case(method=method))
    baseline = _selection_reference(_selection_case(method="REML"))
    assert reference["accepted"], reference
    for key in ("coef", "Vp", "theta", "sp", "score", "Vc"):
        np.testing.assert_array_equal(reference[key], baseline[key])


@pytest.mark.parametrize("method", ["GCV.Cp", "P-ML"])
def test_cnorm_criterion_alias_matches_mgcv(method):
    test_cnorm_scale_selection_matches_mgcv({"method": method})


def test_cnorm_laml_alias_matches_reml():
    case = _selection_case(method="REML")
    reference = _selection_reference(case)
    case["method"] = "LAML"
    model = _fit(case)
    _assert_fit(case, reference, model)
    np.testing.assert_allclose(
        model.smoothing_params, np.atleast_1d(reference["sp"]), atol=2e-7, rtol=2e-7
    )
    np.testing.assert_allclose(
        model.family.getTheta(), np.atleast_1d(reference["theta"]), atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("link", ["log", "sqrt"])
def test_cnorm_upstream_nonidentity_estimated_inventory(link):
    reference = _selection_reference(_selection_case(link=link))
    assert reference["version"] == "1.9.4"


@pytest.mark.parametrize("link", ["log", "sqrt"])
def test_cnorm_nonidentity_estimated_matches_mgcv(link):
    test_cnorm_scale_selection_matches_mgcv({"link": link})


@pytest.mark.parametrize(
    "index,filename", [(0, "inputs.json"), (1, "extension_inputs.json")]
)
def test_cnorm_source_json_preserves_exact_censor_pairs(index, filename):
    """Actual jsonlite roundtrip retains exact rows, not tiny intervals."""
    data = json.loads((ROOT / filename).read_text())
    reference = json.loads((ROOT / "input_roundtrip_audit.json").read_text())[index]
    # jsonlite recognizes these uppercase spellings as numeric infinities.
    assert all(
        value in {"Inf", "-Inf"} for value in data["bound"] if isinstance(value, str)
    )
    bounds = np.asarray(data["bound"], dtype=float)
    np.testing.assert_array_equal(reference["y"], data["y"])
    np.testing.assert_array_equal(
        reference["finite_bound"], bounds[np.isfinite(bounds)]
    )
    np.testing.assert_array_equal(
        reference["exact"], np.flatnonzero(bounds == data["y"]) + 1
    )


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("estimated", [False, True])
def test_cnorm_summary_matches_mgcv(response, estimated):
    case = _selection_case(response=response) if estimated else _case(response)
    reference = _selection_reference(case) if estimated else _reference(case)
    summary = _fit(case).summary()
    for key, table in (("p_table", summary.p_table), ("s_table", summary.s_table)):
        values = (
            table[["edf", "ref_df", "wald_stat", "p_value"]]
            if key == "s_table"
            else table
        )
        np.testing.assert_allclose(
            values.to_numpy(float), reference[key], atol=2e-7, rtol=2e-7, err_msg=key
        )
    np.testing.assert_allclose(summary.r_sq, reference["r_sq"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(
        summary.dev_expl, reference["dev_expl"], atol=2e-7, rtol=2e-7
    )


ANOVA_SCRIPT = SELECTION_SCRIPT.replace(
    "coef=coef(b),X=X",
    "anova=lapply(c(FALSE,TRUE),function(fr){a<-anova(b,freq=fr);list(parametric=a$pTerms.table,smooth=a$s.table)}),coef=coef(b),X=X",
).replace(
    "off=seq(-.08,.08,length.out=19))",
    "off=seq(-.08,.08,length.out=19),z=seq(.9,-.9,length.out=19))",
)


def _anova_case(estimated):
    case = _selection_case(
        theta=None if estimated else 1.0,
        method="REML" if estimated else "fixed",
        offset=True,
    )
    case["data"] = json.loads((ROOT / "extension_inputs.json").read_text())
    case["formula"] = "cbind(y,bound)~z+s(x,bs='cr',k=7)+offset(off)"
    return case


@pytest.mark.parametrize("estimated", [False, True])
def test_cnorm_upstream_anova_inventory(estimated):
    reference = _reference(_anova_case(estimated), script=ANOVA_SCRIPT)
    assert reference["accepted"], reference
    assert reference["converged"]
    assert len(reference["anova"]) == 2


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("freq", [False, True])
def test_cnorm_anova_matches_mgcv(estimated, freq):
    case = _anova_case(estimated)
    reference = _reference(case, script=ANOVA_SCRIPT)["anova"][int(freq)]
    result = _fit(case).anova(freq=freq)
    np.testing.assert_allclose(
        result.parametric_table[["df", "wald_stat", "p_value"]].to_numpy(float),
        reference["parametric"],
        atol=2e-7,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        result.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(float),
        reference["smooth"],
        atol=2e-7,
        rtol=2e-7,
    )
