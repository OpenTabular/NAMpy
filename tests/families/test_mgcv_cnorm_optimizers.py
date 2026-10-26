"""Actual-R alternative optimizer acceptance for censored normal models."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.families.cnorm import CensoredNormalFamily
from nampy.gam.model_state import _edf2
from tests.families.test_mgcv_cnorm_workflows import (
    SCRIPT,
    _assert_fit,
    _case,
    _data,
    _newdata,
)
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

SOURCE = SCRIPT.replace(
    "sp=p$sp,method='REML'",
    "method='REML',optimizer=c('outer',p$optimizer),control=if(is.null(p$control))gam.control() else do.call(gam.control,p$control)",
).replace(
    "list(accepted=TRUE,",
    "list(sp_cov=sp.vcov(b),unconditional_se=suppressWarnings(predict(b,nd,se.fit=TRUE,unconditional=TRUE)$se.fit),outer=b$outer.info,accepted=TRUE,",
)


def _optimizer_case(optimizer, estimated, *, tight=False):
    case = _case()
    case.update(theta=None if estimated else 1.0, optimizer=optimizer)
    # Never let jsonlite coerce mixed numeric/infinity strings at low precision.
    case["data"]["bound"] = [str(v) for v in case["data"]["bound"]]
    if tight:
        case["control"] = {
            "epsilon": 1e-10,
            "nlm": {"gradtol": 1e-10, "steptol": 1e-10},
        }
    return case


def _reference(case):
    source = SOURCE
    if case["optimizer"] == "efs":
        source = source.replace("optimizer=c('outer',p$optimizer)", "optimizer='efs'")
    key = reference_key(
        "cnorm-optimizers-v1",
        {"case": case, "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
    )
    result = load_reference("mgcv_cnorm_optimizers", key)
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
        save_reference("mgcv_cnorm_optimizers", key, result)
    return result


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("estimated", [False, True])
def test_cnorm_upstream_optimizer_inventory(optimizer, estimated):
    reference = _reference(_optimizer_case(optimizer, estimated))
    assert reference["accepted"], reference
    assert reference["converged"]
    assert (reference["Vc"] is not None) == (optimizer == "bfgs")
    assert (reference["edf2"] is not None) == (optimizer == "bfgs")


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("estimated", [False, True])
def test_cnorm_optimizer_matches_mgcv(optimizer, estimated):
    case = _optimizer_case(optimizer, estimated, tight=optimizer == "nlm")
    reference = _reference(case)
    model = GAM(
        formula=case["formula"],
        family=CensoredNormalFamily(theta=case["theta"]),
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
        optimize_smoothing=True,
        control=case.get("control"),
    ).fit(data=_data(case), sample_weight="w")
    _assert_fit(case, reference, model)
    for name, actual in {
        "theta": model.family.getTheta(),
        "sp": model.smoothing_params,
        "score": model.smoothing_score_,
    }.items():
        np.testing.assert_allclose(
            actual, np.atleast_1d(reference[name]), atol=2e-7, rtol=2e-7, err_msg=name
        )
    if reference["Vc"] is None:
        assert _edf2(model) is None
        np.testing.assert_array_equal(model.vcov(unconditional=True), model.vcov())
        with pytest.warns(UserWarning, match="corrected covariance not available"):
            _, se = model.predict(_newdata(), return_se=True, unconditional=True)
    else:
        np.testing.assert_allclose(
            model.vcov(unconditional=True), reference["Vc"], atol=2e-7, rtol=2e-7
        )
        np.testing.assert_allclose(
            _edf2(model), reference["edf2"], atol=2e-7, rtol=2e-7
        )
        _, se = model.predict(_newdata(), return_se=True, unconditional=True)
    np.testing.assert_allclose(se, reference["unconditional_se"], atol=2e-7, rtol=2e-7)
    if reference["sp_cov"] is None:
        assert model.sp_vcov() is None
    else:
        np.testing.assert_allclose(
            model.sp_vcov(), reference["sp_cov"], atol=2e-7, rtol=2e-7
        )


def test_cnorm_fixed_theta_efs_upstream_inventory():
    reference = _reference(_optimizer_case("efs", False))
    assert reference["accepted"], reference
    assert reference["converged"]


def test_cnorm_fixed_theta_efs_matches_mgcv():
    test_cnorm_optimizer_matches_mgcv("efs", False)


@pytest.mark.parametrize("estimated", [False, True])
def test_cnorm_controlled_nlm_upstream_inventory(estimated):
    reference = _reference(_optimizer_case("nlm", estimated, tight=True))
    assert reference["accepted"], reference
    assert reference["converged"]


def test_cnorm_nonunit_scale_source_returns_stale_failed_outer_fit():
    path = (
        Path(__file__).parents[1]
        / "reference_fixtures/mgcv_cnorm_optimizers/scale2_outer_trace.json"
    )
    source = json.loads(path.read_text())
    assert source["outer"]["conv"] == "step failed"
    final_parameters = np.r_[np.atleast_1d(source["theta"]), np.log(source["sp"])]
    matching = [
        call for call in source["calls"] if np.array_equal(call["coef"], source["coef"])
    ]
    assert matching
    assert all(
        np.max(np.abs(np.asarray(call["params"]).ravel() - final_parameters)) > 0.1
        for call in matching
    )
    final = source["calls"][-1]
    # Failed step halving leaves the family theta at the last trial, while
    # reported sp remains at the accepted score's state. Neither owns coef.
    assert np.asarray(final["params"]).ravel()[0] == final_parameters[0]
    accepted = [call for call in source["calls"] if call["score"] == source["score"]]
    assert accepted
    np.testing.assert_allclose(
        np.asarray(accepted[-1]["params"]).ravel()[1],
        final_parameters[1],
        atol=1e-12,
        rtol=0,
    )
    assert np.max(np.abs(np.asarray(final["coef"]) - source["coef"])) > 0.04
