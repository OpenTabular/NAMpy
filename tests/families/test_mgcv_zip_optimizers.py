"""ziP alternative optimizers, captured from actual mgcv 1.9-4."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _coef_full, _edf2, _edf_total
from tests.families.test_mgcv_zip_workflows import SCRIPT, _case
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference


def _optimizer_case(optimizer, estimated):
    case = _case(method="REML", estimated=estimated)
    case["optimizer"] = optimizer
    if optimizer == "optim":
        case["control"] = {"epsilon": 1e-10, "optim": {"factr": 10.0}}
    return case


def _reference(case):
    source = SCRIPT.replace(
        "b<-do.call(gam,args)",
        "args$optimizer<-c('outer',p$optimizer);"
        "if(!is.null(p$control))args$control<-do.call(gam.control,p$control);"
        "b<-do.call(gam,args)",
    )
    key = reference_key(
        "zip-optimizers-v1",
        {"case": case, "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
    )
    result = load_reference("mgcv_zip_optimizers", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            script, inputs, output = [
                Path(directory) / n for n in ("ref.R", "in.json", "out.json")
            ]
            script.write_text(source)
            inputs.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(script, str(inputs), str(output)),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_zip_optimizers", key, result)
    return result


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_upstream_optimizer_inventory(optimizer, estimated):
    ref = _reference(_optimizer_case(optimizer, estimated))
    assert ref["accepted"], ref
    assert ref["version"] == "1.9.4"
    assert (ref["Vc"] is not None) == (optimizer == "bfgs")
    assert (ref["edf2"] is not None) == (optimizer == "bfgs")


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("estimated", [False, True])
def test_zip_alternative_optimizer_matches_mgcv(optimizer, estimated):
    case = _optimizer_case(optimizer, estimated)
    ref = _reference(case)
    model = GAM(
        formula=case["formula"],
        family={"name": "zip", "theta": case["theta"], "b": case["b"]},
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
        optimize_smoothing=True,
        control=case.get("control"),
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")
    nd = pd.DataFrame({"x": np.linspace(-0.95, 0.95, 17), "off": 0.0})
    prediction, se = model.predict(nd, return_se=True)
    for name, actual in {
        "theta": model.family.getTheta(),
        "sp": model.smoothing_params,
        "coef": _coef_full(model),
        "prediction": prediction,
        "se": se,
        "Vp": model.vcov(),
        "Ve": model.vcov(freq=True),
        "edf": _edf_total(model),
        "score": model.smoothing_score_,
        "loglik": model.loglik(),
        "aic": model.aic(),
        "bic": model.bic(),
    }.items():
        expected = np.sum(ref[name]) if name == "edf" else ref[name]
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5, err_msg=name)
    if ref["Vc"] is None:
        assert _edf2(model) is None
        np.testing.assert_array_equal(model.vcov(unconditional=True), model.vcov())
    else:
        np.testing.assert_allclose(
            model.vcov(unconditional=True), ref["Vc"], atol=5e-5, rtol=5e-5
        )
        np.testing.assert_allclose(_edf2(model), ref["edf2"], atol=5e-5, rtol=5e-5)
    if ref["Vc"] is None:
        with pytest.warns(UserWarning, match="corrected covariance not available"):
            _, use = model.predict(nd, return_se=True, unconditional=True)
    else:
        _, use = model.predict(nd, return_se=True, unconditional=True)
    np.testing.assert_allclose(use, ref["unconditional_se"], atol=5e-5, rtol=5e-5)
    if ref["sp_cov"] is None:
        assert model.sp_vcov() is None
    else:
        np.testing.assert_allclose(model.sp_vcov(), ref["sp_cov"], atol=5e-5, rtol=5e-5)


def test_zip_vector_bfgs_receives_requested_newton_controls(monkeypatch):
    from nampy.gam.fit.selection.optimize import driver

    captured = {}

    class ReachedOptimizer(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise ReachedOptimizer

    monkeypatch.setattr(driver, "_optimize_outer_bfgs_strict", capture)
    case = _case(method="REML", estimated=True)
    model = GAM(
        formula=case["formula"],
        family={"name": "zip", "b": case["b"]},
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
        control={"newton": {"conv.tol": 2e-8, "maxNstep": 2.5, "maxSstep": 0.75}},
    )
    with pytest.raises(ReachedOptimizer):
        model.fit(data=pd.DataFrame(case["data"]))
    assert captured["conv_tol"] == 2e-8
    assert captured["max_nstep"] == 2.5
    assert captured["max_sstep"] == 0.75


def _initial_bfgs_reference():
    case = _optimizer_case("bfgs", True)
    source = SCRIPT.replace(
        "b<-do.call(gam,args)",
        "initial_calls<-list();"
        "trace('gam.fit3',where=asNamespace('mgcv'),print=FALSE,"
        "exit=quote({v<-returnValue();if(length(initial_calls)<4)"
        "initial_calls[[length(initial_calls)+1]]<<-list(coef=v$coefficients,"
        "score=v$REML,gradient=v$REML1)}));"
        "args$optimizer<-c('outer','bfgs');b<-do.call(gam,args)",
    ).replace("list(accepted=TRUE,", "list(initial_calls=initial_calls,accepted=TRUE,")
    key = reference_key(
        "zip-bfgs-initial-v1",
        {"case": case, "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
    )
    result = load_reference("mgcv_zip_optimizers", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            script, inputs, output = [
                Path(directory) / n for n in ("ref.R", "in.json", "out.json")
            ]
            script.write_text(source)
            inputs.write_text(json.dumps(case))
            run = subprocess.run(
                _build_r_command(script, str(inputs), str(output)),
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_zip_optimizers", key, result)
    assert result["accepted"], result
    return result["initial_calls"]


def test_zip_bfgs_initial_fit_and_finite_difference_hessian_match_mgcv(monkeypatch):
    """gam.fit4 keeps the converged beta when final gdi2 refreshes its QR."""
    from nampy.gam.fit.selection.optimize import bfgs_strict

    reference = _initial_bfgs_reference()
    calls = []
    inverses = []
    original_eval = bfgs_strict._eval_objective_at
    original_hessian = bfgs_strict._finite_difference_initial_inverse_hessian

    def evaluate(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        if len(calls) < 4:
            calls.append(result)
        return result

    def initial_hessian(*args, **kwargs):
        result = original_hessian(*args, **kwargs)
        inverses.append(result)
        return result

    monkeypatch.setattr(bfgs_strict, "_eval_objective_at", evaluate)
    monkeypatch.setattr(
        bfgs_strict, "_finite_difference_initial_inverse_hessian", initial_hessian
    )
    case = _optimizer_case("bfgs", True)
    GAM(
        formula=case["formula"],
        family={"name": "zip", "b": case["b"]},
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
        optimize_smoothing=True,
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")
    for actual, expected in zip(calls, reference, strict=True):
        np.testing.assert_allclose(actual[4], expected["coef"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            actual[1], np.ravel(expected["gradient"]), rtol=1e-11, atol=1e-11
        )
    initial_gradient = np.ravel(reference[0]["gradient"])
    hessian = np.array(
        [(np.ravel(row["gradient"]) - initial_gradient) / 1e-4 for row in reference[1:]]
    )
    hessian = (hessian + hessian.T) / 2
    values, vectors = np.linalg.eigh(hessian)
    values = np.maximum(np.abs(values), np.max(np.abs(values)) * 1e-4)
    inverse = (vectors / values) @ vectors.T
    np.testing.assert_allclose(inverses[0], inverse, rtol=1e-8, atol=1e-8)
