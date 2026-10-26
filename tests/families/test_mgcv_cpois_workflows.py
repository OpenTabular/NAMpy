"""Fitting, inference and response lifecycle for mgcv's censored Poisson."""

import hashlib
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from tests.families.test_mgcv_cnorm_workflows import SCRIPT, _assert_fit, _data
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_cpois_workflows"
SOURCE = SCRIPT.replace('do.call(cnorm,list(theta=p$theta,link=p$link))', 'do.call(cpois,list(link=p$link))').replace("sp=p$sp,method='REML'", "sp=p$sp,method=p$method,scale=p$scale,optimizer=if(p$optimizer=='efs')'efs' else c('outer',p$optimizer)")


def reference(case):
    source = SOURCE.replace("scale=p$scale,", "scale=p$scale,control=do.call(gam.control,p$control),") if case.get("control") else SOURCE
    key = reference_key("cpois-workflows-v1", {"case": case, "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    ref = load_reference("mgcv_cpois_workflows", key)
    if ref is None:
        with tempfile.TemporaryDirectory() as directory:
            script, inputs, output = [Path(directory) / name for name in ("run.R", "in.json", "out.json")]
            script.write_text(source)
            inputs.write_text(json.dumps(case))
            run = subprocess.run(["Rscript", str(script), str(inputs), str(output)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            ref = json.loads(output.read_text())
            save_reference("mgcv_cpois_workflows", key, ref)
    return ref


def case(response="mixed", link="log", method="fixed", optimizer="newton", weighted=False, scale=1):
    data = json.loads((ROOT / "inputs.json").read_text())
    if not weighted:
        data["w"] = [1.] * len(data["y"])
    return {"data": data, "response": response, "offset": True, "link": link,
            "formula": ("cbind(y,bound)" if response == "mixed" else "y") + "~s(x,bs='cr',k=7)+offset(off)",
            "method": method, "sp": [.7] if method == "fixed" else None,
            "optimizer": optimizer, "scale": scale}


def fit(payload):
    return GAM(family={"name": "cpois", "link": payload["link"]}, formula=payload["formula"],
               optimize_smoothing=payload["method"] != "fixed", smoothing_method=payload["method"],
               smoothing_params=payload["sp"], smoothing_optimizer=payload["optimizer"],
               scale=payload["scale"], control=payload.get("control")).fit(_data(payload), sample_weight="w")


@pytest.mark.parametrize("response", ["mixed", "vector"])
@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("method", ["fixed", "ML", "REML"])
def test_cpois_fit_and_inference_match_source(response, link, method):
    payload = case(response, link, method)
    ref = reference(payload)
    assert ref["accepted"] and ref["converged"], ref
    model = fit(payload)
    _assert_fit(payload, ref, model)
    if method != "fixed":
        np.testing.assert_allclose(model.smoothing_params, np.atleast_1d(ref["sp"]), rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(model.smoothing_score_, ref["score"], rtol=2e-7, atol=2e-7)
        if ref["Vc"] is not None:
            from nampy.gam.model_state import _edf2

            np.testing.assert_allclose(model.vcov(unconditional=True), ref["Vc"], rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(_edf2(model), ref["edf2"], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm", "efs"])
def test_cpois_alternative_optimizers(optimizer):
    payload = case(method="REML", optimizer=optimizer)
    if optimizer == "nlm":
        payload["control"] = {"epsilon": 1e-10, "nlm": {"gradtol": 1e-10, "steptol": 1e-10}}
    ref = reference(payload)
    assert ref["accepted"], ref
    _assert_fit(payload, ref, fit(payload))


@pytest.mark.parametrize("scale", [1, 2])
def test_cpois_source_weight_semantics_and_external_dispersion(scale):
    payload = case(weighted=True, scale=scale)
    ref = reference(payload)
    assert ref["accepted"], ref
    _assert_fit(payload, ref, fit(payload))


def test_cpois_zero_prior_weights_match_source_ignored_weight_semantics():
    payload = case(weighted=True)
    payload["data"]["w"][::7] = [0.] * len(payload["data"]["w"][::7])
    ref = reference(payload)
    assert ref["accepted"], ref
    _assert_fit(payload, ref, fit(payload))


@pytest.mark.parametrize("spec", ["cpois", {"name": "cpois", "link": "log"}, ("cpois", {"link": "log"})])
def test_cpois_public_family_specifications(spec):
    from nampy.gam.families import CensoredPoissonFamily, make_gam_family

    family = make_gam_family(spec)
    assert isinstance(family, CensoredPoissonFamily)
    assert make_gam_family(family) is not family


def test_cpois_public_components_and_censor_persistence(tmp_path):
    from nampy.models.gam import GAMRegressor

    payload = case()
    model = GAMRegressor(family="cpois", formula=payload["formula"],
                         optimize_smoothing=False, smoothing_params=[.7]).fit(_data(payload))
    model.predict_components(_data(payload)).validate_additive_reconstruction()
    path = tmp_path / "cpois.pkl"
    model.save_model(path)
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(_data(payload)), model.predict(_data(payload)))
    response = pickle.loads(pickle.dumps(model.gam_.response_))
    order = np.r_[np.arange(79, -1, -1), [0, 0]]
    np.testing.assert_array_equal(response.subset(order).bounds, response.bounds[order])
    assert not response.bounds.flags.writeable


def test_cpois_explicit_matrix_response_matches_formula_response():
    payload = case()
    data = _data(payload)
    expected = fit(payload)
    explicit = GAM(family="cpois", formula="y~s(x,bs='cr',k=7)+offset(off)",
                   optimize_smoothing=False, smoothing_params=[.7]).fit(
                       data=data, y=data[["y", "bound"]].to_numpy())
    np.testing.assert_array_equal(explicit.predict(), expected.predict())


def test_cpois_invalid_bound_does_not_relax_predictor_validation():
    payload = case()
    data = _data(payload)
    data.loc[0, "x"] = np.inf
    with pytest.raises(ValueError):
        GAM(family="cpois", formula=payload["formula"], optimize_smoothing=False).fit(data)
