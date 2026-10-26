"""Static actual-mgcv ziP workflow contracts; no R during ordinary testing."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression
ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_zip_workflows"


def _generate_inputs():
    """Development-only data recipe; tests load frozen doubles instead."""
    rng = np.random.default_rng(261026)
    x = np.linspace(-1, 1, 180)
    gamma = 0.2 + 1.2 * np.sin(3 * x) + 0.2 * x
    lam = np.exp(gamma)
    presence = rng.uniform(size=x.size) < -np.expm1(-np.exp(-0.5 + 0.8 * gamma))
    positive = rng.poisson(lam)
    while np.any(positive == 0):
        zero = positive == 0
        positive[zero] = rng.poisson(lam[zero])
    return {
        "x": x.tolist(),
        "y": (presence * positive).tolist(),
        "w": np.ones(x.size).tolist(),
        "off": np.zeros(x.size).tolist(),
    }


def _case(method="fixed", estimated=False, weights="unit", scale=None):
    data = json.loads((ROOT / "inputs.json").read_text())
    if weights != "unit":
        data["w"] = (0.5 + 0.5 * (np.arange(len(data["y"])) % 4)).tolist()
        if weights == "zero":
            for index in (0, 19, 52):
                data["w"][index] = 0.0
    case = {
        "data": data,
        "formula": "y~s(x,bs='cr',k=7)",
        "theta": None if estimated else [-0.5, -0.6931471805599453],
        "b": 0.3,
        "sp": [0.4],
        "method": method,
        "weights": weights,
    }
    if scale is not None:
        case["scale"] = scale
    return case


SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);warnings<-character()
result<-tryCatch(withCallingHandlers({
fam<-ziP(theta=p$theta,b=p$b)
args<-list(formula=as.formula(p$formula),data=d,family=fam,weights=d$w,
 method=if(p$method %in% c('fixed','LAML')) 'REML' else p$method)
if(p$method=='fixed')args$sp<-p$sp
if(!is.null(p$scale))args$scale<-p$scale
trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,
 tracer=quote(.zip_inputsp<-sp),exit=quote({
 rv<-returnValue()
 if(deriv==2&&!is.null(rv$REML2)&&!exists('.zip_joint',envir=.GlobalEnv,inherits=FALSE))
 assign('.zip_joint',list(parameters=.zip_inputsp,value=rv$REML,gradient=rv$REML1,
 hessian=rv$REML2,coefficients=rv$coefficients,score_type=scoreType),envir=.GlobalEnv)
}))
b<-do.call(gam,args);nd<-data.frame(x=seq(-.95,.95,length.out=17),off=0)
P<-predict(b,nd,type='lpmatrix');X<-predict(b,type='lpmatrix')
pr<-predict(b,nd,type='response',se.fit=TRUE);pl<-predict(b,nd,type='link',se.fit=TRUE)
pu<-predict(b,nd,type='response',se.fit=TRUE,unconditional=TRUE);su<-summary(b)
r<-lapply(c('deviance','working','response','pearson'),function(type)
 tryCatch(residuals(b,type=type),error=function(e)list(error=conditionMessage(e))))
list(accepted=TRUE,version=as.character(packageVersion('mgcv')),method=b$method,
 theta=b$family$getTheta(),transformed_theta=b$family$getTheta(TRUE),n_theta=b$family$n.theta,
 coef=coef(b),sp=b$sp,full_sp=b$full.sp,score=as.numeric(b$gcv.ubre),outer=b$outer.info,
 X=X,S=lapply(b$smooth,function(s)s$S),edf=b$edf,edf1=b$edf1,edf2=b$edf2,
 Vp=b$Vp,Ve=b$Ve,Vc=b$Vc,Vsp=b$V.sp,sp_cov=sp.vcov(b),
 prediction=pr$fit,se=pr$se.fit,link_prediction=pl$fit,link_se=pl$se.fit,
 unconditional_prediction=pu$fit,unconditional_se=pu$se.fit,
 covariance_kernel=P%*%b$Vp%*%t(P),fitted_latent=fitted(b),response_fitted=predict(b,type='response'),
 deviance=deviance(b),raw_deviance=b$family$dev.resids(b$y,fitted(b),b$prior.weights),
 saturated=b$family$saturated.ll(b$y,b$family,b$prior.weights),null_deviance=b$null.deviance,
 scale=b$sig2,loglik=as.numeric(logLik(b)),loglik_df=attr(logLik(b),'df'),aic=AIC(b),bic=BIC(b),
 residuals=r,p_table=su$p.table,s_table=su$s.table,r_sq=su$r.sq,dev_expl=su$dev.expl,
 joint=if(exists('.zip_joint',envir=.GlobalEnv,inherits=FALSE)) .zip_joint else NULL)
},warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
error=function(e)list(accepted=FALSE,error=conditionMessage(e),version=as.character(packageVersion('mgcv'))))
result$warnings<-warnings;write_json(result,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key(
        "zip-workflow-v1",
        {"case": case, "script_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest()},
        normalize_floats=True,
    )
    result = load_reference("mgcv_zip_workflows", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            script, inp, out = [
                Path(tmp) / name for name in ("probe.R", "input.json", "output.json")
            ]
            script_code = SCRIPT
            if "z" in case["data"]:
                script_code = script_code.replace(
                    "nd<-data.frame(x=seq(-.95,.95,length.out=17),off=0)",
                    "nd<-data.frame(x=seq(-.95,.95,length.out=17),z=seq(.8,-.8,length.out=17),off=0)",
                )
            if case.get("fixed_sp"):
                script_code = script_code.replace(
                    "b<-do.call(gam,args)", "args$sp<-p$sp;b<-do.call(gam,args)"
                )
            script.write_text(script_code)
            inp.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(script, str(inp), str(out)),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_zip_workflows", key, result)
    return result


@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_initial_upstream_workflow(method, estimated):
    result = _reference(_case(method=method, estimated=estimated))
    assert result["accepted"], result
    assert result["version"] == "1.9.4"
    assert result["n_theta"] == (2 if estimated else 0)
    assert np.isfinite(result["coef"]).all()
    assert not np.allclose(result["fitted_latent"], result["response_fitted"])


@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("estimated", [False, True])
@pytest.mark.parametrize(
    "weights,scale",
    [
        ("nonunit", None),
        ("zero", None),
        ("unit", -1),
        ("unit", 0),
        ("unit", 1),
        ("unit", 2),
    ],
)
def test_zip_upstream_weight_scale_inventory(method, estimated, weights, scale):
    result = _reference(_case(method, estimated, weights, scale))
    assert result["version"] == "1.9.4"
    # This inventory records upstream behavior; it does not advertise Python
    # support for weighted or externally scaled fits.
    assert result["accepted"], result
    warnings = result["warnings"]
    warnings = [warnings] if isinstance(warnings, str) else warnings
    step_failure = any("step failure" in warning for warning in warnings)
    assert step_failure == (weights != "unit" and (estimated or method == "REML"))
    if weights == "unit" and scale in (-1, 0, 1):
        baseline = _reference(_case(method, estimated))
        for key in ("coef", "theta", "Vp", "score"):
            np.testing.assert_array_equal(result[key], baseline[key])


def _fit(case):
    import pandas as pd

    from nampy.gam import GAM
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    return GAM(
        family=ZeroInflatedPoissonFamily(theta=case["theta"], b=case["b"]),
        formula=case["formula"],
        smoothing_method=case["method"],
        smoothing_params=(
            case["sp"] if case["method"] == "fixed" or case.get("fixed_sp") else None
        ),
        optimize_smoothing=case["method"] != "fixed"
        and not case.get("fixed_sp", False),
        scale=case.get("scale", 0),
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


def _newdata():
    import pandas as pd

    return pd.DataFrame(
        {"x": np.linspace(-0.95, 0.95, 17), "z": np.linspace(0.8, -0.8, 17), "off": 0.0}
    )


def _assert_fit(case, reference, model):
    from nampy.gam.model_state import _coef_full, _deviance, _edf2, _edf_total

    tolerance = 5e-5 if reference["n_theta"] else 3e-6
    checks = [
        ("coef", _coef_full(model)),
        ("theta", model.family.getTheta()),
        ("Vp", model.vcov()),
        ("Ve", model.vcov(freq=True)),
        ("fitted_latent", model.predict(type="link")),
        ("response_fitted", model.predict(type="response")),
        ("deviance", _deviance(model)),
        ("loglik", model.loglik()),
        ("aic", model.aic()),
        ("bic", model.bic()),
    ]
    for key, value in checks:
        np.testing.assert_allclose(
            value, reference[key], atol=tolerance, rtol=tolerance, err_msg=key
        )
    assert _edf_total(model) == pytest.approx(
        np.sum(reference["edf"]), abs=tolerance, rel=tolerance
    )
    if case["method"] != "fixed" and not case.get("fixed_sp"):
        np.testing.assert_allclose(
            model.smoothing_params,
            np.atleast_1d(reference["sp"]),
            atol=tolerance,
            rtol=tolerance,
        )
    if model._optim_result is not None:
        assert model._optim_result.fun == pytest.approx(
            reference["score"], abs=tolerance, rel=tolerance
        )
    for kind, key, sekey in [
        ("response", "prediction", "se"),
        ("link", "link_prediction", "link_se"),
    ]:
        prediction, se = model.predict(_newdata(), type=kind, return_se=True)
        np.testing.assert_allclose(
            prediction, reference[key], atol=tolerance, rtol=tolerance, err_msg=key
        )
        np.testing.assert_allclose(
            se, reference[sekey], atol=tolerance, rtol=tolerance, err_msg=sekey
        )
    P = model.predict(_newdata(), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T,
        reference["covariance_kernel"],
        atol=tolerance,
        rtol=tolerance,
    )
    if reference["Vc"] is not None:
        np.testing.assert_allclose(
            model.vcov(unconditional=True),
            reference["Vc"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="Vc",
        )
        np.testing.assert_allclose(
            _edf2(model),
            reference["edf2"],
            atol=tolerance,
            rtol=tolerance,
            err_msg="edf2",
        )
        _, se = model.predict(
            _newdata(), type="response", return_se=True, unconditional=True
        )
        np.testing.assert_allclose(
            se, reference["unconditional_se"], atol=tolerance, rtol=tolerance
        )
    for kind, expected in zip(
        ["deviance", "working", "response"], reference["residuals"][:3], strict=True
    ):
        np.testing.assert_allclose(
            model.residuals(type=kind),
            expected,
            atol=tolerance,
            rtol=tolerance,
            err_msg=kind + " residuals",
        )
    summary = model.summary()
    for table, key in [(summary.p_table, "p_table"), (summary.s_table, "s_table")]:
        if reference[key] is None:
            assert table.empty
            continue
        values = (
            table[["edf", "ref_df", "wald_stat", "p_value"]]
            if key == "s_table"
            else table
        )
        np.testing.assert_allclose(
            values.to_numpy(dtype=float),
            np.asarray(reference[key], dtype=float),
            atol=tolerance,
            rtol=tolerance,
            err_msg=key,
        )
    assert summary.null_deviance == pytest.approx(
        reference["null_deviance"], abs=tolerance, rel=tolerance
    )
    assert summary.dev_expl == pytest.approx(
        reference["dev_expl"], abs=tolerance, rel=tolerance
    )


@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_initial_workflow_fit(method, estimated):
    case = _case(method=method, estimated=estimated)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("method", ["fixed", "REML", "ML", "LAML"])
def test_zip_off_optimum_joint_derivatives(method):
    from nampy.gam.fit.selection.criteria.pirls.family_vector_theta import (
        criterion_gradient_ml_reml_pirls_vector_theta_joint,
        criterion_hessian_ml_reml_pirls_vector_theta_joint,
        criterion_ml_reml_pirls_vector_theta_joint,
    )

    case = _case(method=method, estimated=True)
    reference = _reference(case)["joint"]
    point = np.asarray(reference["parameters"]).ravel()
    conditional = _case(method="fixed", estimated=False)
    conditional["theta"] = point[:2].tolist()
    conditional["sp"] = np.exp(point[2:]).tolist()
    model = _fit(conditional)
    model.family.n_theta = 2
    args = (model, model.y_, point[2:], point[:2], "ML" if method == "ML" else "REML")
    assert criterion_ml_reml_pirls_vector_theta_joint(*args) == pytest.approx(
        reference["value"], abs=2e-6, rel=2e-7
    )
    np.testing.assert_allclose(
        criterion_gradient_ml_reml_pirls_vector_theta_joint(*args),
        np.asarray(reference["gradient"]).ravel(),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_hessian_ml_reml_pirls_vector_theta_joint(*args),
        reference["hessian"],
        atol=3e-6,
        rtol=1e-5,
    )


def _structural_case(structure, estimated):
    case = _case(method="REML", estimated=estimated)
    extra = json.loads((ROOT / "extension_inputs.json").read_text())
    if structure in {"ML", "LAML"}:
        case["method"] = structure
    elif structure in {"fixed_ML", "fixed_REML", "fixed_LAML"}:
        case["method"] = structure.removeprefix("fixed_")
        case["fixed_sp"] = True
    elif structure == "offset":
        case["data"]["off"] = extra["off"]
        case["formula"] += "+offset(off)"
    elif structure == "multiple":
        case["data"]["z"] = extra["z"]
        case["formula"] += "+s(z,bs='cr',k=6)"
        case["sp"] = [0.4, 0.7]
    elif structure == "rank":
        case["data"]["z"] = extra["z"]
        case["formula"] += "+z+I(2*z)"
    elif structure == "no_intercept":
        case["formula"] += "-1"
    elif structure == "no_smooths":
        case["formula"] = "y~x"
        case["sp"] = []
    else:
        raise ValueError(structure)
    return case


STRUCTURES = [
    "ML",
    "LAML",
    "fixed_ML",
    "fixed_REML",
    "fixed_LAML",
    "offset",
    "multiple",
    "rank",
    "no_intercept",
    "no_smooths",
]


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_structural_upstream_reference(structure, estimated):
    result = _reference(_structural_case(structure, estimated))
    assert result["accepted"], result
    assert np.isfinite(result["coef"]).all()


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_structural_workflow_fit(structure, estimated):
    case = _structural_case(structure, estimated)
    if structure == "rank":
        with pytest.raises(
            NotImplementedError, match="rank-deficient.*portable mgcv parity"
        ):
            _fit(case)
        return
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_external_scale_two_fit(method, estimated):
    case = _case(method=method, estimated=estimated, scale=2)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_persistence(tmp_path, estimated):
    from nampy.gam import GAM

    case = _case(method="REML", estimated=estimated)
    model = _fit(case)
    path = tmp_path / "zip.model"
    model.save_model(path)
    loaded = GAM.load_model(path)
    _assert_fit(case, _reference(case), loaded)


@pytest.mark.parametrize("estimated,b", [(False, 0.0), (True, 0.0), (True, 0.9)])
def test_zip_default_and_slope_boundary_reference(estimated, b):
    case = _case(method="REML", estimated=estimated)
    case["b"] = b
    reference = _reference(case)
    assert reference["accepted"], reference
    assert np.isfinite(reference["coef"]).all()


@pytest.mark.parametrize("estimated,b", [(False, 0.0), (True, 0.0), (True, 0.9)])
def test_zip_default_and_slope_boundary_fit(estimated, b):
    case = _case(method="REML", estimated=estimated)
    case["b"] = b
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_rank_postfit_at_upstream_endpoint(estimated):
    from nampy.gam.fit.backends import solve_pirls_given_smoothing
    from nampy.gam.fit.postprocess.extended_uncertainty import (
        extended_unconditional_postfit,
    )
    from nampy.gam.fit.state import FitCoreSolution

    case = _structural_case("rank", estimated)
    reference = _reference(case)
    from types import SimpleNamespace

    conditional = _structural_case("rank", False)
    conditional["method"] = "fixed"
    conditional["theta"] = reference["theta"]
    conditional["sp"] = np.atleast_1d(reference["sp"]).tolist()
    model = _fit(conditional)
    model.family.n_theta = 2 if estimated else 0
    model._optim_method = "REML"
    model._optim_result = SimpleNamespace(
        outer_info={"hess": np.atleast_2d(reference["outer"]["hess"])}
    )
    solution = solve_pirls_given_smoothing(model, model.y_, model.smoothing_params)
    core = (
        solution
        if isinstance(solution, FitCoreSolution)
        else FitCoreSolution.from_dict(solution)
    )
    for actual, key in [
        (core.fit_result.coef_full, "coef"),
        (core.fit_result.cov_bayes, "Vp"),
    ]:
        np.testing.assert_allclose(actual, reference[key], atol=2e-7, rtol=2e-7)
    # Actual R returned a correction, but identical rank-deficient H has
    # platform-dependent Cholesky success. Pinned fixed-theta prediction
    # covariance differed .00484641; this is not only a null-coordinate issue.
    assert np.isfinite(reference["Vc"]).all()
    with pytest.raises(
        NotImplementedError, match="rank-deficient.*portable mgcv parity"
    ):
        extended_unconditional_postfit(model, solution, core.fit_result, core.fit_state)


@pytest.mark.parametrize("name", ["ziP", "zip"])
def test_zip_default_public_string(name):
    import pandas as pd

    from nampy.gam import GAM

    case = _case(method="REML", estimated=True)
    case["b"] = 0.0
    model = GAM(
        family=name,
        formula=case["formula"],
        smoothing_method="REML",
        optimize_smoothing=True,
    ).fit(data=pd.DataFrame(case["data"]))
    _assert_fit(case, _reference(case), model)


def test_zip_default_regressor_adapter():
    import pandas as pd

    from nampy.models.gam import GAMRegressor

    case = _case(method="REML", estimated=True)
    case["b"] = 0.0
    model = GAMRegressor(
        family="ziP",
        formula=case["formula"],
        smoothing_method="REML",
        optimize_smoothing=True,
    ).fit(pd.DataFrame(case["data"]))
    reference = _reference(case)
    _assert_fit(case, reference, model.gam_)
    prediction = model.predict(_newdata())
    se = model.standard_errors(_newdata())
    components = model.predict_components(_newdata())
    components.validate_additive_reconstruction()
    np.testing.assert_allclose(components.response, prediction, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        components.link, reference["link_prediction"], atol=5e-5, rtol=5e-5
    )
    assert not np.allclose(components.response, components.link)
    np.testing.assert_allclose(
        prediction, reference["prediction"], atol=5e-5, rtol=5e-5
    )
    np.testing.assert_allclose(se, reference["se"], atol=5e-5, rtol=5e-5)


def _rank_conditional_case():
    case = _structural_case("rank", False)
    case["method"] = "fixed"
    return case


def test_zip_rank_conditional_upstream_reference():
    reference = _reference(_rank_conditional_case())
    assert reference["accepted"], reference
    assert reference["Vc"] is None


def test_zip_rank_conditional_fit():
    case = _rank_conditional_case()
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_gcv_cp_upstream_reml_alias(estimated):
    case = _case(method="GCV.Cp", estimated=estimated)
    reference = _reference(case)
    assert reference["accepted"], reference
    assert reference["method"] == "REML"
    reml = _reference(_case(method="REML", estimated=estimated))
    for key in ("coef", "theta", "sp", "score", "Vp"):
        np.testing.assert_array_equal(reference[key], reml[key])


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_gcv_cp_uses_reml_fit(estimated):
    case = _case(method="GCV.Cp", estimated=estimated)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_p_ml_upstream_reml_alias(estimated):
    reference = _reference(_case(method="P-ML", estimated=estimated))
    assert reference["accepted"], reference
    assert reference["method"] == "REML"
    reml = _reference(_case(method="REML", estimated=estimated))
    for key in ("coef", "theta", "sp", "score", "Vp"):
        np.testing.assert_array_equal(reference[key], reml[key])


@pytest.mark.parametrize("estimated", [False, True])
def test_zip_p_ml_uses_reml_fit(estimated):
    case = _case(method="P-ML", estimated=estimated)
    _assert_fit(case, _reference(case), _fit(case))


@pytest.mark.parametrize("method", ["NCV", "unknown-method"])
def test_zip_fixed_sp_estimated_theta_rejects_unsupported_method(method):
    case = _case(method=method, estimated=True)
    case["fixed_sp"] = True
    with pytest.raises((NotImplementedError, ValueError), match="method"):
        _fit(case)
