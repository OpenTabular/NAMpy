"""Acceptance of previously guarded clog routes against captured R states."""

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.families import CensoredLogisticFamily
from tests.families.test_mgcv_clog_workflows import SOURCE, assert_clog_fit, reference
from tests.families.test_mgcv_cnorm_workflows import _case, _data

TRACE_SOURCE = SOURCE.replace(" b<-gam(", """
.clog_calls <- list()
trace('gam.fit4',where=asNamespace('mgcv'),print=FALSE,
 tracer=quote(.clog_par<-sp),exit=quote({rv<-returnValue()
 .clog_calls[[length(.clog_calls)+1]] <<- list(params=.clog_par,deriv=deriv,
   coef=rv$coefficients,score=rv$REML,gradient=rv$REML1,hessian=rv$REML2)
 }))
 b<-gam(""").replace("list(accepted=TRUE,", "list(calls=.clog_calls,outer=b$outer.info,accepted=TRUE,")


def acceptance_case(kind, method="REML"):
    case = _case("mixed", offset=True)
    case.update(theta=None, sp=None, method=method, scale=1)
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    if kind == "precision":
        case["data"]["w"] = np.linspace(.5, 2, len(case["data"]["y"])).tolist()
        case["control"] = {"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}}
    elif kind in {"log", "sqrt"}:
        case["link"] = kind
    elif kind == "nlm":
        case["control"] = {"epsilon": 1e-10, "nlm": {"gradtol": 1e-10, "steptol": 1e-10}}
    elif kind == "optim":
        case["theta"] = 1.0
        case["control"] = {"epsilon": 1e-10, "optim": {"factr": 10}}
    return case


@pytest.mark.parametrize("kind,method", [("precision", "ML"), ("precision", "REML"), ("nlm", "REML"), ("optim", "REML")])
def test_previously_guarded_clog_endpoints(kind, method):
    case = acceptance_case(kind, method)
    source = TRACE_SOURCE
    optimizer = kind if kind in {"nlm", "optim"} else "outer_newton"
    if optimizer != "outer_newton":
        source = source.replace("method=p$method,", f"method=p$method,optimizer=c('outer','{optimizer}'),")
    ref = reference(case, source)
    assert ref["accepted"], ref
    if kind == "precision":
        assert_failed_newton_state(ref)
        with pytest.raises(RuntimeError, match="selection failed"):
            GAM(formula=case["formula"], family="clog", smoothing_method=method,
                control=case["control"], optimize_smoothing=True).fit(_data(case), sample_weight="w")
        return
    if kind == "nlm":
        assert ref["outer"]["code"] == 3
        with pytest.warns(RuntimeWarning, match="did not converge"):
            model = GAM(formula=case["formula"], family="clog", smoothing_method=method,
                smoothing_optimizer=optimizer, control=case["control"],
                optimize_smoothing=True).fit(_data(case), sample_weight="w")
        assert not model._optim_result.success
        return
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=case["theta"]),
                smoothing_method=method, smoothing_optimizer=optimizer, control=case.get("control"),
                optimize_smoothing=True).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model, linear_limit=kind == "optim")


def assert_failed_newton_state(ref):
    """Prove returned R coefficients do not belong to its reported params."""
    assert ref["outer"]["conv"] == "step failed"
    parameters = np.r_[np.atleast_1d(ref["theta"]), np.log(np.atleast_1d(ref["sp"]))]
    owners = [call for call in ref["calls"] if call["coef"] is not None
              and np.array_equal(call["coef"], ref["coef"])]
    assert owners
    assert all(np.max(np.abs(np.asarray(call["params"]).ravel() - parameters)) > 1e-8 for call in owners)


@pytest.mark.parametrize("link", ["log", "sqrt"])
def test_clog_estimated_nonidentity_reml(link):
    case = acceptance_case(link)
    ref = reference(case, TRACE_SOURCE)
    assert ref["accepted"], ref
    assert_failed_newton_state(ref)
    with pytest.raises(RuntimeError, match="selection failed"):
        GAM(formula=case["formula"], family=CensoredLogisticFamily(link=link),
            smoothing_method="REML", optimize_smoothing=True).fit(_data(case), sample_weight="w")


@pytest.mark.parametrize("optimizer", ["bfgs", "optim"])
def test_clog_estimated_precision_alternative_optimizer(optimizer):
    case = acceptance_case("precision")
    source = TRACE_SOURCE.replace("method=p$method,", f"method=p$method,optimizer=c('outer','{optimizer}'),")
    ref = reference(case, source)
    assert ref["accepted"], ref
    model = GAM(formula=case["formula"], family="clog", smoothing_method="REML",
                smoothing_optimizer=optimizer, optimize_smoothing=True,
                control=case["control"]).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)


def test_failed_newton_refit_preserves_prior_clog_fit():
    case = acceptance_case("precision")
    model = GAM(formula=case["formula"], family="clog", smoothing_method="REML",
                smoothing_optimizer="optim", control=case["control"],
                optimize_smoothing=True).fit(_data(case), sample_weight="w")
    prediction = model.predict()
    theta = model.family.getTheta()
    # Select the captured failed Newton route for a subsequent fit. FitSession
    # must roll back family state and coefficients if the fit raises.
    model.smoothing_optimizer = "outer_newton"
    with pytest.raises(RuntimeError, match="selection failed"):
        model.fit(_data(case), sample_weight="w")
    np.testing.assert_array_equal(model.predict(), prediction)
    np.testing.assert_array_equal(model.family.getTheta(), theta)


def test_clog_fixed_scale_efs():
    case = acceptance_case("optim")
    case.pop("control", None)
    source = TRACE_SOURCE.replace("method=p$method,", "method=p$method,optimizer='efs',")
    ref = reference(case, source)
    assert ref["accepted"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(theta=1),
                smoothing_method="REML", smoothing_optimizer="efs", optimize_smoothing=True).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)


@pytest.mark.parametrize("method", ["ML", "REML"])
def test_clog_successful_estimated_nlm(method):
    case = _case("vector", offset=True)
    case.update(theta=None, sp=None, method=method, scale=1,
                control={"epsilon": 1e-10, "nlm": {"gradtol": 1e-10, "steptol": 1e-10}})
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    source = TRACE_SOURCE.replace("method=p$method,", "method=p$method,optimizer=c('outer','nlm'),")
    ref = reference(case, source)
    assert ref["accepted"], ref
    assert ref["outer"]["code"] in {1, 2}, ref["outer"]
    model = GAM(formula=case["formula"], family="clog", smoothing_method=method,
                smoothing_optimizer="nlm", optimize_smoothing=True,
                control=case["control"]).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)


@pytest.mark.parametrize("response", ["vector", "mixed"])
def test_clog_estimated_scale_efs(response):
    case = _case(response, offset=True)
    case.update(theta=None, sp=None, method="REML", scale=1)
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    source = TRACE_SOURCE.replace("method=p$method,", "method=p$method,optimizer='efs',")
    ref = reference(case, source)
    assert ref["accepted"], ref
    model = GAM(formula=case["formula"], family="clog", smoothing_method="REML",
                smoothing_optimizer="efs", optimize_smoothing=True).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)
    from nampy.gam.fit.selection.postfit import sp_vcov
    assert sp_vcov(model) is None


@pytest.mark.parametrize("method", ["ML", "REML"])
@pytest.mark.parametrize("link", ["log", "sqrt"])
@pytest.mark.parametrize("optimizer", ["bfgs", "optim"])
def test_clog_nonidentity_alternative_optimizers(method, link, optimizer):
    case = acceptance_case(link, method)
    case["control"] = {"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}, "optim": {"factr": 10}}
    source = TRACE_SOURCE.replace("method=p$method,", f"method=p$method,optimizer=c('outer','{optimizer}'),")
    ref = reference(case, source)
    if method == "ML" and optimizer == "bfgs":
        assert not ref["accepted"] and ref["subprocess_returncode"] != 0
        assert "caught segfault" in ref["stderr"]
        with pytest.raises(NotImplementedError, match="subprocess crashes"):
            GAM(formula=case["formula"], family=CensoredLogisticFamily(link=link),
                smoothing_method=method, smoothing_optimizer=optimizer).fit(_data(case))
        return
    assert ref["accepted"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(link=link),
                smoothing_method=method, smoothing_optimizer=optimizer, optimize_smoothing=True,
                control=case["control"]).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)


@pytest.mark.parametrize("variant", ["precision", "log", "sqrt", "scale2", "fixed_sp"])
def test_clog_efs_expanded_scope(variant):
    case = acceptance_case(variant)
    case["theta"] = None
    if variant == "scale2":
        case["scale"] = 2
    if variant == "fixed_sp":
        case["sp"] = [0.7]
    source = TRACE_SOURCE.replace("method=p$method,", "method=p$method,optimizer='efs',")
    ref = reference(case, source)
    if variant == "fixed_sp":
        assert not ref["accepted"]
        assert "NA/NaN/Inf in foreign function call" in ref["error"]
        with pytest.raises(NotImplementedError, match="requires a free smoothing"):
            GAM(formula=case["formula"], family="clog", smoothing_method="REML",
                smoothing_optimizer="efs", optimize_smoothing=False,
                smoothing_params=case["sp"]).fit(_data(case))
        return
    assert ref["accepted"], ref
    model = GAM(formula=case["formula"], family=CensoredLogisticFamily(link=case["link"]),
                smoothing_method="REML", smoothing_optimizer="efs", optimize_smoothing=True,
                smoothing_params=case["sp"], scale=case["scale"], control=case.get("control")).fit(_data(case), sample_weight="w")
    assert_clog_fit(case, ref, model)
