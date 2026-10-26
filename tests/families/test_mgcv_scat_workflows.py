"""Actual mgcv scaled-t workflow snapshots; ordinary runs never require R."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression


def _case(
    link="identity",
    method="fixed",
    estimated=False,
    zero=False,
    multiple=False,
    rank_deficient=False,
    theta=None,
    min_df=3,
    joint=False,
    scale=None,
    tight=False,
):
    # Freeze reference doubles: platform libm differences must not alter keys
    # or perturb an observed-curvature denominator near zero.
    data = json.loads(
        (
            Path(__file__).parents[1]
            / "reference_fixtures"
            / "mgcv_scat_workflows"
            / "inputs.json"
        ).read_text()
    )
    if zero:
        for index in [0, 19, 52]:
            data["w"][index] = 0.0
    formula = "y~s(x,bs='cr',k=7)+offset(off)"
    if multiple:
        formula += "+s(z,bs='cr',k=6)"
    if rank_deficient:
        formula += "+z+I(2*z)"
    case = {
        "link": link,
        "method": method,
        "theta": theta
        if theta is not None
        else ([-5, -0.28] if estimated else [5, 0.28]),
        "min_df": min_df,
        "formula": formula,
        "sp": [0.4, 0.7] if multiple else [0.4],
        "data": data,
    }

    if tight:
        case["control"] = {"epsilon": 1e-10, "newton": {"conv_tol": 1e-10}}
    if scale is not None:
        case["scale"] = scale
    if joint:
        case["joint"] = True
    return case


_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
warnings<-character(); result<-tryCatch(withCallingHandlers({
fam<-do.call(scat,list(theta=p$theta,link=p$link,min.df=p$min_df))
initial<-list(theta=fam$getTheta(),transformed=fam$getTheta(TRUE),n_theta=fam$n.theta,
 min_df=get('.min.df',environment(fam$getTheta)))
args<-list(formula=as.formula(p$formula),data=d,family=fam,weights=d$w,
 method=if(p$method %in% c('fixed','LAML')) 'REML' else if(p$method=='UBRE') 'GCV.Cp' else p$method)
if(p$method=='fixed') args$sp<-p$sp
if(!is.null(p$scale)) args$scale<-p$scale
if(isTRUE(p$joint)) trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,exit=quote({
 rv<-returnValue()
 if(deriv==2 && !is.null(rv$REML2) && !exists('.scat_joint',envir=.GlobalEnv,inherits=FALSE))
 assign('.scat_joint',list(parameters=sp,value=rv$REML,gradient=rv$REML1,hessian=rv$REML2,
 coefficients=rv$coefficients,score_type=scoreType),envir=.GlobalEnv)
}))
b<-do.call(gam,args)
nd<-data.frame(x=seq(-.9,.9,length.out=15),z=seq(.8,-.8,length.out=15),off=.13)
pr<-predict(b,nd,se.fit=TRUE,type='response');P<-predict(b,nd,type='lpmatrix')
su<-summary(b)
list(accepted=TRUE,version=as.character(packageVersion('mgcv')),initial=initial,
 theta=b$family$getTheta(),transformed_theta=b$family$getTheta(TRUE),
 n_theta=b$family$n.theta,method=b$method,coef=coef(b),edf=sum(b$edf),
 scale=b$sig2,sp=b$sp,Vp=b$Vp,Ve=b$Ve,covariance_kernel=P%*%b$Vp%*%t(P),
 prediction=pr$fit,se=pr$se.fit,link_prediction=predict(b,nd),
 fitted=fitted(b),deviance=deviance(b),loglik=as.numeric(logLik(b)),
 loglik_df=attr(logLik(b),'df'),aic=AIC(b),bic=BIC(b),score=as.numeric(b$gcv.ubre),
 p_table=su$p.table,s_table=su$s.table,
 residuals=lapply(c('deviance','pearson','working','response'),function(t) residuals(b,type=t)),
 outer=b$outer.info,joint=if(isTRUE(p$joint)) .scat_joint else NULL)
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
error=function(e)list(accepted=FALSE,error=conditionMessage(e),version=as.character(packageVersion('mgcv'))))
result$warnings<-warnings
write_json(result,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key(
        "scat-joint-entry-v2" if case.get("joint") else "scat-workflow-v1",
        case,
        normalize_floats=True,
    )
    result = load_reference("mgcv_scat_workflows", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            script, inp, out = [Path(tmp) / s for s in ["ref.R", "in.json", "out.json"]]
            script_code = _SCRIPT
            if case.get("fixed_sp"):
                script_code = script_code.replace(
                    "b<-do.call(gam,args)", "args$sp<-p$sp;b<-do.call(gam,args)"
                )
            if case.get("joint"):
                script_code = script_code.replace(
                    "print=FALSE,exit=",
                    "print=FALSE,tracer=quote(.scat_inputsp<-sp),exit=",
                ).replace("parameters=sp", "parameters=.scat_inputsp")
            if case.get("control"):
                script_code = script_code.replace(
                    "b<-do.call(gam,args)",
                    "args$control<-gam.control(epsilon=p$control$epsilon,newton=list(conv.tol=p$control$newton$conv_tol));b<-do.call(gam,args)",
                )
            script.write_text(script_code)
            inp.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(script, str(inp), str(out)),
                capture_output=True,
                text=True,
            )
            if run.returncode != 0:
                assert "caught segfault" in run.stderr, run.stderr
                result = {
                    "accepted": False,
                    "version": "1.9.4",
                    "upstream_crash": True,
                    "returncode": run.returncode,
                    "error": run.stderr[:1400],
                }
            else:
                result = json.loads(out.read_text())
        save_reference("mgcv_scat_workflows", key, result)
    return result


CORE = [
    (link, method, estimated)
    for link in ["identity", "log", "inverse"]
    for method in ["fixed", "REML", "ML", "UBRE"]
    for estimated in [False, True]
]
EXTRA = [
    {"link": link, "method": method, "estimated": True, "zero": True, "multiple": True}
    for link in ["identity", "log", "inverse"]
    for method in ["fixed", "REML", "ML"]
]
EXTRA += [
    {"method": "REML", "estimated": True, "rank_deficient": True},
    {"theta": [0, 0.28]},
    {"theta": [-5, 0.28]},
    {"theta": [2, 0.28]},
]


@pytest.mark.parametrize("link,method,estimated", CORE)
def test_scat_upstream_workflow_inventory(link, method, estimated):
    result = _reference(_case(link, method, estimated))
    assert result["version"] == "1.9.4"
    assert result["accepted"], result
    assert result["method"] == ("ML" if method == "ML" else "REML")
    if method == "UBRE":
        # Extended-family GCV.Cp requests are rerouted by mgcv to REML.
        # These references do not claim an independently supported UBRE score.
        reml = _reference(_case(link, "REML", estimated))
        np.testing.assert_allclose(result["coef"], reml["coef"], atol=1e-13)
        np.testing.assert_allclose(result["score"], reml["score"], atol=1e-13)


@pytest.mark.parametrize("options", EXTRA)
def test_scat_upstream_difficult_workflow_inventory(options):
    result = _reference(_case(**options))
    assert result["version"] == "1.9.4"
    if options.get("zero") and options.get("multiple") and options["method"] == "ML":
        assert result.get("upstream_crash") and result["returncode"] == -11
    else:
        assert result["accepted"], result


ISOLATION = [
    dict(link=link, method="ML", estimated=True, **variant)
    for link in ["identity", "log", "inverse"]
    for variant in [{"zero": True}, {"multiple": True}]
]


@pytest.mark.parametrize("options", ISOLATION)
def test_scat_upstream_ml_weight_rank_inventory(options):
    result = _reference(_case(**options))
    assert result["version"] == "1.9.4"
    if options.get("zero"):
        assert result.get("upstream_crash") and result["returncode"] == -11
    else:
        assert result["accepted"], result


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_upstream_off_optimum_joint_reference(link):
    result = _reference(_case(link=link, method="REML", estimated=True, joint=True))
    assert result["accepted"], result
    assert np.shape(result["joint"]["hessian"]) == (3, 3)


def _fit(case, method=None, family_override=None):
    from nampy.gam import GAM
    from nampy.gam.families.scat import ScaledTFamily

    method = case["method"] if method is None else method
    return GAM(
        formula=case["formula"],
        family=(
            ScaledTFamily(theta=case["theta"], link=case["link"], min_df=case["min_df"])
            if family_override is None
            else family_override
        ),
        smoothing_method=method,
        scale=case.get("scale", 0),
        control=case.get("control"),
        optimize_smoothing=method != "fixed" and not case.get("fixed_sp", False),
        smoothing_params=(
            case["sp"] if method == "fixed" or case.get("fixed_sp") else None
        ),
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


def _newdata():
    return pd.DataFrame(
        {"x": np.linspace(-0.9, 0.9, 15), "z": np.linspace(0.8, -0.8, 15), "off": 0.13}
    )


def _assert_fit(case, reference, model):
    from nampy.gam.model_state import _coef_full, _edf_total, _fit_scale

    tolerance = (
        3e-6 if case["method"] == "fixed" and reference["n_theta"] == 0 else 5e-5
    )
    for name, actual, expected in [
        ("coefficients", _coef_full(model), reference["coef"]),
        ("EDF", _edf_total(model), reference["edf"]),
        ("scale", _fit_scale(model), reference["scale"]),
        ("Bayesian covariance", model.vcov(), reference["Vp"]),
        ("frequentist covariance", model.vcov(freq=True), reference["Ve"]),
        ("fitted response", model.predict(), reference["fitted"]),
        ("log likelihood", model.loglik(), reference["loglik"]),
        ("AIC", model.aic(), reference["aic"]),
        ("BIC", model.bic(), reference["bic"]),
    ]:
        np.testing.assert_allclose(
            actual, expected, atol=tolerance, rtol=tolerance, err_msg=name
        )
    if case.get("scale") == 2 and reference["n_theta"] > 0:
        # These source fits reach df=min.df: R's positive gaps are8.45e-7
        # and2.40e-6. The log-gap has no finite identifiable boundary value;
        # compare df and sigma themselves without changing fit tolerances.
        np.testing.assert_allclose(
            reference["transformed_theta"][0],
            reference["initial"]["min_df"],
            atol=5e-5,
            rtol=0,
        )
        np.testing.assert_allclose(
            model.family.getTheta(True),
            reference["transformed_theta"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="physical theta at df boundary",
        )
    else:
        np.testing.assert_allclose(
            model.family.getTheta(),
            reference["theta"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="working theta",
        )
    prediction, se = model.predict(_newdata(), type="response", return_se=True)
    np.testing.assert_allclose(
        prediction, reference["prediction"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(se, reference["se"], atol=tolerance, rtol=tolerance)
    P = model.predict(_newdata(), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T,
        reference["covariance_kernel"],
        atol=tolerance,
        rtol=tolerance,
    )
    linear_limit = False
    if case["method"] != "fixed" and not case.get("fixed_sp"):
        expected_sp = np.atleast_1d(reference["sp"])
        if (
            case["method"] in {"ML", "REML"}
            and case["link"] == "identity"
            and expected_sp.size == 2
        ):
            linear_limit = True
            # The second smooth reaches the linear limit: R stops at7.695e7,
            # Python at1.301e8, with R term EDF1.00013077 and log-SP gradient
            # -1.40e-5. Finite representatives of infinite SP are not identified.
            # Retain strict fitted/covariance/EDF/criterion checks and check the
            # vanishing boundary contribution rather than that representative.
            np.testing.assert_allclose(
                model.smoothing_params[0], expected_sp[0], atol=5e-5, rtol=5e-4
            )
            np.testing.assert_allclose(
                model._optim_result.fun,
                reference["score"],
                atol=tolerance,
                rtol=tolerance,
            )
            source_edf = np.asarray(reference["s_table"])[1, 0]
            actual_edf = model.summary().s_table["edf"].iloc[1]
            assert abs(actual_edf - 1) <= max(tolerance, 2 * abs(source_edf - 1))
            stopping = case.get("control", {}).get("newton", {}).get("conv_tol", 1e-6)
            assert abs(
                np.asarray(reference["outer"]["grad"]).ravel()[-1]
            ) < stopping * abs(reference["score"])
            assert abs(
                np.asarray(model._optim_result.jac).ravel()[-1]
            ) < stopping * abs(reference["score"])
        else:
            np.testing.assert_allclose(
                model.smoothing_params, expected_sp, atol=5e-5, rtol=5e-4
            )
    for kind, expected in zip(
        ["deviance", "pearson", "working", "response"],
        reference["residuals"],
        strict=True,
    ):
        actual = model.residuals(type=kind)
        if kind == "working" and reference["n_theta"] > 0:
            from nampy.gam.families.scat import ScaledTFamily
            from nampy.gam.fit.selection.criteria.pirls.derivatives import (
                _family_ddeta_logtheta,
            )

            # Optimized endpoints differ within the upstream Newton stopping
            # tolerance. Deta2 can approach zero, amplifying that difference:
            # identity REML row72 has residual -102.30674 vs -102.28743 but
            # scores differ only 1.15e-6. Compare the source working-score
            # invariant, keeping raw residual checks at pinned R endpoints.
            family = ScaledTFamily(
                theta=case["theta"], link=case["link"], min_df=case["min_df"]
            )
            family.putTheta(reference["theta"])
            y, weights = np.asarray(case["data"]["y"]), np.asarray(case["data"]["w"])
            pinned = _family_ddeta_logtheta(
                family, y, np.asarray(reference["fitted"]), weights, deriv=0
            )
            current = _family_ddeta_logtheta(
                model.family, y, model.predict(), weights, deriv=0
            )
            # gam.fit4 working residuals are undefined (0/0) at zero
            # observation weights; preserve those NaNs and compare the
            # score invariant on the source good rows only.
            good = weights != 0
            np.testing.assert_allclose(
                actual[~good],
                np.asarray(expected, dtype=float)[~good],
                atol=0,
                rtol=0,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                0.5 * current["Deta2"][good] * actual[good],
                -0.5 * pinned["Deta"][good],
                atol=tolerance,
                rtol=tolerance,
                err_msg="working score",
            )
        else:
            np.testing.assert_allclose(
                actual, expected, atol=tolerance, rtol=tolerance, err_msg=kind
            )
    summary = model.summary()
    np.testing.assert_allclose(
        summary.p_table.to_numpy(dtype=float),
        np.asarray(reference["p_table"], dtype=float),
        atol=tolerance,
        rtol=tolerance,
    )
    actual_summary = summary.s_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=float)
    expected_summary = np.asarray(reference["s_table"], dtype=float)
    checked = np.ones(actual_summary.shape, dtype=bool)
    if linear_limit:
        # Ref.df also tends to one as this smooth becomes linear. Its finite
        # departure depends on the same unidentifiable huge-SP representative.
        # Exact summary arithmetic is checked separately at the pinned R point.
        assert abs(actual_summary[1, 1] - 1) <= max(
            tolerance, 2 * abs(expected_summary[1, 1] - 1)
        )
        checked[1, 1] = False
    np.testing.assert_allclose(
        actual_summary[checked],
        expected_summary[checked],
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("link,method,estimated", [c for c in CORE if c[1] != "UBRE"])
def test_scat_public_fit_selection_inference(link, method, estimated):
    case = _case(link, method, estimated)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_laml_is_reml_alias(link):
    case = _case(link, "REML", True)
    _assert_fit(case, _reference(case), _fit(case, method="LAML"))


@pytest.mark.parametrize("options", [p for p in EXTRA if p.get("method") != "ML"])
def test_scat_difficult_fits(options):
    case = _case(**options)
    if options.get("rank_deficient"):
        assert _reference(case)["accepted"]
        with pytest.raises(NotImplementedError, match="rank-deficient.*portable"):
            _fit(case)
        return
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("estimated", [False, True])
def test_scat_persistence(tmp_path, estimated):
    from nampy.gam import GAM

    case = _case(estimated=estimated)
    model = _fit(case)
    path = tmp_path / "scat.model"
    model.save_model(path)
    loaded = GAM.load_model(path)
    np.testing.assert_array_equal(loaded.family.getTheta(), model.family.getTheta())
    np.testing.assert_allclose(
        loaded.predict(_newdata()), model.predict(_newdata()), atol=1e-13
    )
    _assert_fit(case, _reference(case), loaded)


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_multiple_penalty_ml_fit(link):
    case = _case(link=link, method="ML", estimated=True, multiple=True)
    _assert_fit(case, _reference(case), _fit(case))


def test_scat_constructor_theta_reference_semantics():
    zero = _reference(_case(theta=[0, 0.28]))["initial"]
    mixed = _reference(_case(theta=[-5, 0.28]))["initial"]
    reset = _reference(_case(theta=[2, 0.28]))
    assert zero["n_theta"] == mixed["n_theta"] == 2
    np.testing.assert_array_equal(zero["theta"], [-2, -1])
    np.testing.assert_allclose(mixed["transformed"], [5, 0.28], atol=1e-15)
    assert reset["initial"]["n_theta"] == 0
    np.testing.assert_allclose(reset["initial"]["min_df"], 1.8, atol=1e-15)
    assert "Supplied df below min.df. min.df reset" in reset["warnings"]


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_upstream_fixed_theta_ml_zero_weights(link):
    result = _reference(_case(link=link, method="ML", zero=True))
    assert result.get("upstream_crash") and result["returncode"] == -11


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_exact_off_optimum_joint_derivatives(link):
    from nampy.gam.fit.selection.criteria.pirls.family_scat import (
        criterion_gradient_ml_reml_pirls_scat_joint,
        criterion_hessian_ml_reml_pirls_scat_joint,
        criterion_ml_reml_pirls_scat_joint,
    )

    case = _case(link=link, method="REML", estimated=True, joint=True)
    reference = _reference(case)["joint"]
    model = _fit(case)
    parameters = np.asarray(reference["parameters"]).ravel()
    args = (model, model.y_, parameters[2:], parameters[:2], "REML")
    np.testing.assert_allclose(
        criterion_ml_reml_pirls_scat_joint(*args),
        reference["value"],
        atol=2e-6,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        criterion_gradient_ml_reml_pirls_scat_joint(*args),
        reference["gradient"],
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_hessian_ml_reml_pirls_scat_joint(*args),
        reference["hessian"],
        atol=3e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_public_family_dict(link):
    case = _case(link=link)
    family = {"name": "scat", "link": link, "theta": [5, 0.28], "min_df": 3}
    _assert_fit(case, _reference(case), _fit(case, family_override=family))


def test_scat_public_family_string():
    case = _case(estimated=True)
    _assert_fit(case, _reference(case), _fit(case, family_override="scat"))


_UNCERTAINTY_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
fam<-do.call(scat,list(theta=p$theta,link=p$link,min.df=p$min_df))
b<-gam(as.formula(p$formula),data=d,family=fam,weights=d$w,method=p$method)
nd<-data.frame(x=seq(-.9,.9,length.out=15),z=seq(.8,-.8,length.out=15),off=.13)
pr<-predict(b,nd,se.fit=TRUE,type='response',unconditional=TRUE)
pl<-predict(b,nd,se.fit=TRUE,type='link',unconditional=TRUE)
P<-predict(b,nd,type='lpmatrix')
result<-list(version=as.character(packageVersion('mgcv')),Vc=b$Vc,edf2=b$edf2,
 Vsp=b$V.sp,sp_vcov=sp.vcov(b),Vp=b$Vp,Ve=b$Ve,coef=coef(b),edf=b$edf,
 theta=b$family$getTheta(),transformed_theta=b$family$getTheta(TRUE),sp=b$sp,
 unconditional_kernel=P%*%b$Vc%*%t(P),prediction=pr$fit,se=pr$se.fit,
 link_prediction=pl$fit,link_se=pl$se.fit,aic=AIC(b),bic=BIC(b),
 loglik=as.numeric(logLik(b)),loglik_df=attr(logLik(b),'df'))
write_json(result,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _uncertainty_reference(case, availability=False):
    import hashlib

    script_code = _UNCERTAINTY_SCRIPT
    if case.get("control"):
        script_code = script_code.replace(
            "method=p$method)",
            "method=p$method,control=gam.control(epsilon=p$control$epsilon,newton=list(conv.tol=p$control$newton$conv_tol)))",
        )
    if availability:
        script_code = script_code.replace(
            "b<-gam(as.formula(p$formula),data=d,family=fam,weights=d$w,method=p$method)",
            "args<-list(formula=as.formula(p$formula),data=d,family=fam,weights=d$w,method='REML');"
            "if(grepl('s(',p$formula,fixed=TRUE)) args$sp<-p$sp;b<-do.call(gam,args)",
        ).replace("P%*%b$Vc%*%t(P)", "if(is.null(b$Vc)) NULL else P%*%b$Vc%*%t(P)")
    key = reference_key(
        "scat-uncertainty-availability-v1" if availability else "scat-uncertainty-v1",
        {
            "case": case,
            "script_sha256": hashlib.sha256(script_code.encode()).hexdigest(),
        },
        normalize_floats=True,
    )
    result = load_reference("mgcv_scat_workflows", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            script, inp, out = [
                Path(tmp) / s for s in ["uncertainty.R", "in.json", "out.json"]
            ]
            script.write_text(script_code)
            inp.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(script, str(inp), str(out)),
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_scat_workflows", key, result)
    return result


UNCERTAINTY_CASES = [
    {"link": link, "method": "REML", "estimated": estimated}
    for link in ["identity", "log", "inverse"]
    for estimated in [False, True]
]
UNCERTAINTY_CASES += [
    {"link": link, "method": "REML", "estimated": True, "zero": True, "multiple": True}
    for link in ["identity", "log"]
]


@pytest.mark.parametrize("options", UNCERTAINTY_CASES)
def test_scat_upstream_uncertainty_inventory(options):
    result = _uncertainty_reference(_case(**options))
    assert result["version"] == "1.9.4"
    assert np.shape(result["Vc"]) == np.shape(result["Vp"])
    assert np.isfinite(result["loglik_df"])


@pytest.mark.parametrize("options", UNCERTAINTY_CASES)
def test_scat_unconditional_covariance_and_edf(options):
    from nampy.gam.model_state import _edf2

    # The two zero-weight/two-smooth fits have flat outer directions.
    # Default stopping leaves EDF2 different by 2.55e-4 (identity) and
    # SP covariance by 6.92e-4 (log). Match tighter controls in both R and
    # Python for these uncertainty comparisons; retain original references
    # and default-control full-fit tests without relaxing any tolerance.
    case = _case(**options, tight=bool(options.get("zero")))
    reference = _uncertainty_reference(case)
    model = _fit(case)
    np.testing.assert_allclose(
        model.vcov(unconditional=True), reference["Vc"], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(_edf2(model), reference["edf2"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(
        model.sp_vcov(), reference["sp_vcov"], atol=5e-5, rtol=5e-5
    )
    prediction, se = model.predict(
        _newdata(), type="response", return_se=True, unconditional=True
    )
    np.testing.assert_allclose(
        prediction, reference["prediction"], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(se, reference["se"], atol=5e-5, rtol=5e-5)
    P = model.predict(_newdata(), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov(unconditional=True) @ P.T,
        reference["unconditional_kernel"],
        atol=5e-5,
        rtol=5e-5,
    )


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_scat_upstream_uncertainty_availability(estimated, smooth):
    case = _case(estimated=estimated)
    if not smooth:
        case["formula"] = "y~x+offset(off)"
        case["sp"] = []
    reference = _uncertainty_reference(case, availability=True)
    assert reference["version"] == "1.9.4"


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_scat_uncertainty_availability_and_no_smooths(estimated, smooth):
    from nampy.gam.model_state import _cov_unconditional, _edf2

    case = _case(estimated=estimated)
    if not smooth:
        case["formula"] = "y~x+offset(off)"
        case["sp"] = []
    reference = _uncertainty_reference(case, availability=True)
    model = _fit(case)
    if not estimated:
        assert reference["Vc"] is reference["edf2"] is None
        assert _cov_unconditional(model) is _edf2(model) is None
        assert model.sp_vcov() is None
    else:
        np.testing.assert_allclose(
            model.vcov(unconditional=True), reference["Vc"], atol=5e-5, rtol=5e-5
        )
        np.testing.assert_allclose(
            _edf2(model), reference["edf2"], atol=5e-5, rtol=5e-5
        )
        np.testing.assert_allclose(
            model.sp_vcov(), reference["sp_vcov"], atol=5e-5, rtol=5e-5
        )
    np.testing.assert_allclose(
        model.loglik(), reference["loglik"], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(model.aic(), reference["aic"], atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_scat_upstream_explicit_scale(estimated, method):
    reference = _reference(_case(method=method, estimated=estimated, scale=2))
    assert reference["accepted"], reference
    assert reference["scale"] == 2


@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_scat_explicit_scale_fit(estimated, method):
    case = _case(method=method, estimated=estimated, scale=2)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("method", ["fixed", "REML", "ML"])
def test_scat_working_residual_at_reference_endpoint(link, method):
    from nampy.gam.families.scat import ScaledTFamily
    from nampy.gam.fit.selection.criteria.pirls.derivatives import (
        _family_ddeta_logtheta,
    )

    case = _case(link, method, True)
    reference = _reference(case)
    family = ScaledTFamily(theta=case["theta"], link=link)
    family.putTheta(reference["theta"])
    derivatives = _family_ddeta_logtheta(
        family,
        np.asarray(case["data"]["y"]),
        np.asarray(reference["fitted"]),
        np.asarray(case["data"]["w"]),
        deriv=0,
    )
    np.testing.assert_allclose(
        -derivatives["Deta"] / derivatives["Deta2"],
        reference["residuals"][2],
        atol=2e-7,
        rtol=2e-7,
    )


@pytest.mark.parametrize(
    "link,method", [("identity", "REML"), ("identity", "ML"), ("log", "REML")]
)
def test_scat_optimized_working_score_invariant(link, method):
    from nampy.gam.families.scat import ScaledTFamily
    from nampy.gam.fit.selection.criteria.pirls.derivatives import (
        _family_ddeta_logtheta,
    )

    case = _case(link, method, True)
    reference = _reference(case)
    model = _fit(case)
    family = ScaledTFamily(theta=case["theta"], link=link)
    family.putTheta(reference["theta"])
    y, w = np.asarray(case["data"]["y"]), np.asarray(case["data"]["w"])
    expected = _family_ddeta_logtheta(
        family, y, np.asarray(reference["fitted"]), w, deriv=0
    )
    actual = _family_ddeta_logtheta(model.family, y, model.predict(), w, deriv=0)
    working = model.residuals(type="working")
    source_working = np.asarray(reference["residuals"][2])
    index = int(np.argmax(np.abs(working - source_working)))
    print(
        {
            "index": index,
            "curvature_expected": float(expected["Deta2"][index]),
            "curvature_actual": float(actual["Deta2"][index]),
            "score_expected": float(-0.5 * expected["Deta"][index]),
            "score_actual": float(-0.5 * actual["Deta"][index]),
            "residual_expected": float(source_working[index]),
            "residual_actual": float(working[index]),
        }
    )
    np.testing.assert_allclose(
        0.5 * actual["Deta2"] * working, -0.5 * expected["Deta"], atol=5e-5, rtol=5e-5
    )


def _structural_joint_case(kind):
    case = _case(
        method="REML",
        estimated=True,
        joint=True,
        zero=kind == "zero_two_smooths",
        multiple=kind == "zero_two_smooths",
    )
    if kind == "no_smooths":
        case["formula"] = "y~x+offset(off)"
        case["sp"] = []
    return case


@pytest.mark.parametrize("kind,dimension", [("zero_two_smooths", 4), ("no_smooths", 2)])
def test_scat_upstream_structural_joint_reference(kind, dimension):
    reference = _reference(_structural_joint_case(kind))
    assert reference["accepted"], reference
    assert np.shape(reference["joint"]["hessian"]) == (dimension, dimension)


@pytest.mark.parametrize("kind", ["zero_two_smooths", "no_smooths"])
def test_scat_structural_joint_derivatives(kind):
    from nampy.gam.fit.selection.criteria.pirls.family_scat import (
        criterion_gradient_ml_reml_pirls_scat_joint,
        criterion_hessian_ml_reml_pirls_scat_joint,
        criterion_ml_reml_pirls_scat_joint,
    )

    case = _structural_joint_case(kind)
    reference = _reference(case)["joint"]
    parameters = np.asarray(reference["parameters"]).ravel()
    # Compile and solve at the captured R point without first running an outer
    # optimizer. This isolates the derivative kernel from convergence behavior.
    prepared = dict(case)
    prepared["method"] = "fixed"
    prepared["theta"] = [np.exp(parameters[0]) + case["min_df"], np.exp(parameters[1])]
    prepared["sp"] = np.exp(parameters[2:]).tolist()
    model = _fit(prepared)
    model.family.n_theta = 2
    args = (model, model.y_, parameters[2:], parameters[:2], "REML")
    np.testing.assert_allclose(
        criterion_ml_reml_pirls_scat_joint(*args),
        reference["value"],
        atol=2e-6,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        criterion_gradient_ml_reml_pirls_scat_joint(*args),
        reference["gradient"],
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_hessian_ml_reml_pirls_scat_joint(*args),
        reference["hessian"],
        atol=3e-6,
        rtol=1e-5,
    )


def test_scat_linear_limit_summary_at_reference_endpoint():
    case = _case(method="ML", estimated=True, multiple=True)
    reference = _reference(case)
    parameters = np.asarray(reference["theta"])
    prepared = dict(case)
    prepared["method"] = "fixed"
    prepared["theta"] = [np.exp(parameters[0]) + case["min_df"], np.exp(parameters[1])]
    prepared["sp"] = np.atleast_1d(reference["sp"]).tolist()
    model = _fit(prepared)
    summary = model.summary()
    np.testing.assert_allclose(
        summary.p_table.to_numpy(dtype=float),
        np.asarray(reference["p_table"], dtype=float),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        summary.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=float
        ),
        reference["s_table"],
        atol=2e-6,
        rtol=2e-6,
    )


TIGHT_CASES = [
    {
        "link": link,
        "method": "REML",
        "estimated": True,
        "zero": True,
        "multiple": True,
        "tight": True,
    }
    for link in ["identity", "log"]
]
TIGHT_CASES += [
    {
        "link": "inverse",
        "method": "ML",
        "estimated": True,
        "multiple": True,
        "tight": True,
    }
]


@pytest.mark.parametrize("options", TIGHT_CASES)
def test_scat_upstream_tight_endpoint_reference(options):
    case = _case(**options)
    reference = _reference(case)
    assert reference["accepted"], reference
    if options.get("zero"):
        uncertainty = _uncertainty_reference(case)
        assert uncertainty["Vc"] is not None


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("method", ["ML", "REML", "LAML"])
def test_scat_fixed_smoothing_explicit_method_reference(link, method):
    case = _case(link=link, method=method, estimated=True)
    case["fixed_sp"] = True
    reference = _reference(case)
    assert reference["accepted"], reference
    assert reference["method"] == ("ML" if method == "ML" else "REML")
    assert np.asarray(reference["sp"]).size == 0


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("method", ["ML", "REML", "LAML"])
def test_scat_fixed_smoothing_preserves_explicit_method(link, method):
    case = _case(link=link, method=method, estimated=True)
    case["fixed_sp"] = True
    reference = _reference(case)
    model = _fit(case)
    _assert_fit(case, reference, model)
    np.testing.assert_allclose(model.smoothing_params, case["sp"], atol=0, rtol=0)
    assert model._optim_result.fun == pytest.approx(
        reference["score"], abs=5e-5, rel=5e-5
    )


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("estimated", [False, True])
def test_scat_fixed_smoothing_ml_zero_weight_guard(link, estimated):
    case = _case(link=link, method="ML", estimated=estimated, zero=True)
    case["fixed_sp"] = True
    if estimated:
        with pytest.raises(NotImplementedError, match="mgcv 1.9-4.*crashes"):
            _fit(case)
    else:
        # No outer criterion is evaluated when both theta and SP are fixed.
        # This conditional operation does not claim ML selection on zero rows.
        model = _fit(case)
        assert np.isfinite(model.predict()).all()
        np.testing.assert_allclose(model.family.getTheta(True), case["theta"])


def test_scat_rank_deficient_conditional_fit_without_outer_uncertainty():
    case = _case(method="fixed", estimated=False, rank_deficient=True)
    _assert_fit(case, _reference(case), _fit(case))
