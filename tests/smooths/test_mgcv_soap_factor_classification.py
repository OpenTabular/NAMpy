"""Bounded ordinary soap factor classification against unmodified mgcv."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_factor_classification.R"
TERM_SP_PROBE = PROBE.with_name("mgcv_soap_factor_delegated_sp.R")
ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_soap_factor_classification"
KEY = reference_key("ordinary-soap-factor-classification-v1", {
    "source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
})
DOMAINS = ("free", "mixed", "multiple_free", "known", "all_known", "outer_known")
CASES = [(mode, basis, domain) for domain in DOMAINS for basis in ("so", "sf", "sw") for mode in ("fs", "sz")]


@pytest.fixture(scope="module")
def reference():
    result = load_reference("mgcv_soap_factor_classification", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            run = subprocess.run(["Rscript", str(PROBE), str(output)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_soap_factor_classification", KEY, result)
    assert result["version"] == "1.9.4"
    return result


def _expected_rejection(mode, basis, domain):
    if basis == "sf" and domain in {"known", "all_known"}:
        return "no free coefs in sf smooth"
    if (basis == "so" and domain not in {"known", "all_known"}) or (basis == "sf" and domain == "multiple_free"):
        return f'"{mode}" smooth cannot use a multiply penalized basis (wrong basis in xt)'
    if mode == "fs" and (basis == "sw" or basis == "so"):
        return "subscript out of bounds"
    return None


@pytest.mark.parametrize("mode,basis,domain", CASES)
def test_ordinary_soap_factor_upstream_classification(reference, mode, basis, domain):
    case = reference["cases"][f"{mode}/{basis}/{domain}"]
    error = _expected_rejection(mode, basis, domain)
    for stage in ("constructor", "constrained", "fit"):
        result = case[stage]
        assert result["ok"] == (error is None), (stage, result)
        if error is not None:
            assert result["error"] == error
    if error == "subscript out of bounds":
        base = case["base"]["value"]
        assert base["npen"] == 1
        assert base["rank"] == base["ncol"]
        assert base["null_space_dim"] == 0
        assert "object$rank + i" in case["constructor"]["call"]
    if error is not None and "multiply penalized" in error:
        assert case["base"]["value"]["npen"] > 1


def _frame(data, levels):
    frame = pd.DataFrame(data)
    frame["f"] = pd.Categorical(frame.f, categories=np.atleast_1d(levels))
    return frame


def _model(reference, mode, basis, domain, *, plain=False):
    key = f"{mode}/{basis}/{domain}"
    case = reference["cases"][f"plain/{key}" if plain else key]
    xt = {"bs": basis, "bnd": case["boundary"], "nmax": 30}
    fit = case["fit"].get("value")
    sp = None if fit is None else fit["full_sp"]
    features = "x,y" if plain else "x,y,f"
    return GAM(formula=f"response~s({features},bs='{mode}',k=8,xt={xt!r})",
               knots=reference["knots"], smoothing_method="fixed",
               optimize_smoothing=False, smoothing_params=sp)


REJECTED = [case for case in CASES if _expected_rejection(*case) is not None]


@pytest.mark.parametrize("mode,basis,domain", REJECTED)
def test_ordinary_soap_factor_rejections_are_source_backed(reference, mode, basis, domain):
    expected = _expected_rejection(mode, basis, domain)
    with pytest.raises((NotImplementedError, ValueError)) as caught:
        _model(reference, mode, basis, domain).fit(_frame(reference["data"], reference["levels"]))
    if basis == "sw" and mode == "fs":
        assert "null" in str(caught.value)
    elif expected == "no free coefs in sf smooth":
        assert "free" in str(caught.value)


# Source-accepted so/sz has one full-rank interior penalty when every boundary
# is known, alongside single-penalty sf and sz/sw compositions.
SUPPORTED = [case for case in CASES if _expected_rejection(*case) is None]


@pytest.mark.parametrize("mode,basis,domain", SUPPORTED)
def test_ordinary_single_penalty_soap_factor_invariants(reference, mode, basis, domain):
    result = reference["cases"][f"{mode}/{basis}/{domain}"]["fit"]["value"]
    model = _model(reference, mode, basis, domain).fit(_frame(reference["data"], reference["levels"]))
    _assert_fit_invariants(reference, model, result)


def _assert_fit_invariants(reference, model, result):
    newdata = _frame(reference["newdata"], reference["levels"])
    predicted, se = model.predict(newdata, return_se=True)
    for actual, expected in ((predicted, result["prediction"]), (se, result["se"]),
                             (model.predict(), result["fitted"]), (_edf_total(model), result["edf"])):
        np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=4e-5)
    P = model.predict(newdata, type="lpmatrix")
    np.testing.assert_allclose(P @ model.vcov() @ P.T, result["covariance"], atol=4e-5, rtol=4e-5)
    X = model.predict(type="lpmatrix")[:, 1:]
    source = np.asarray(result["X"])[:, 1:]
    transport = np.linalg.lstsq(X, source, rcond=None)[0]
    np.testing.assert_allclose(X @ transport, source, atol=2e-7)
    np.testing.assert_allclose(P[:, 1:] @ transport, np.asarray(result["Xp"])[:, 1:], atol=2e-7)
    for penalty, expected in zip(_penalty_blocks_seq(model), result["S"], strict=True):
        np.testing.assert_allclose(transport.T @ penalty.matrix @ transport, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode,basis,domain", CASES)
def test_no_factor_soap_delegates_before_factor_restrictions(reference, mode, basis, domain):
    case = reference["cases"][f"plain/{mode}/{basis}/{domain}"]
    model = _model(reference, mode, basis, domain, plain=True)
    if basis == "sf" and domain in {"known", "all_known"}:
        assert case["base"]["error"] == "no free coefs in sf smooth"
        assert case["constructor"]["error"] == case["fit"]["error"] == case["base"]["error"]
        with pytest.raises(ValueError, match="free"):
            model.fit(_frame(reference["data"], reference["levels"]))
        return
    assert case["constructor"]["ok"] and case["constrained"]["ok"] and case["fit"]["ok"]
    for key in ("ncol", "npen", "rank", "null_space_dim"):
        assert case["constructor"]["value"][key] == case["base"]["value"][key]
    model.fit(_frame(reference["data"], reference["levels"]))
    _assert_fit_invariants(reference, model, case["fit"]["value"])


@pytest.mark.parametrize("mode,basis,domain", [
    ("sz", "so", "known"), ("sz", "so", "all_known"),
    ("fs", "sf", "mixed"), ("sz", "sf", "outer_known"),
])
def test_source_factor_reconstruction_drops_known_boundary_offset(reference, mode, basis, domain):
    case = reference["cases"][f"{mode}/{basis}/{domain}"]
    offset = case["base"]["value"]["offset"]
    assert offset is not None and np.max(np.abs(offset)) > .01
    assert case["constructor"]["value"]["offset"] is None
    assert case["constrained"]["value"]["offset"] is None
    delegated = reference["cases"][f"plain/{mode}/{basis}/{domain}"]
    np.testing.assert_allclose(delegated["constructor"]["value"]["offset"], offset, atol=2e-12, rtol=2e-12)


@pytest.fixture(scope="module")
def term_sp_reference():
    key = reference_key("ordinary-soap-factor-delegated-sp-v1", {
        "source_sha256": hashlib.sha256(TERM_SP_PROBE.read_bytes()).hexdigest(),
    })
    result = load_reference("mgcv_soap_factor_classification", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            run = subprocess.run(["Rscript", str(TERM_SP_PROBE), str(output)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_soap_factor_classification", key, result)
    return result


@pytest.mark.parametrize("mode,basis", [("fs", "so"), ("fs", "sw"), ("sz", "so"), ("sz", "sw")])
def test_no_factor_delegation_preserves_explicit_term_sp(term_sp_reference, mode, basis):
    reference = term_sp_reference
    result = reference["cases"][f"{mode}/{basis}"]
    sp = np.atleast_1d(result["term_sp"]).tolist()
    xt = {"bs": basis, "bnd": reference["boundary"], "nmax": 30}
    model = GAM(formula=f"response~s(x,y,bs='{mode}',k=8,xt={xt!r},sp={sp!r})",
                knots=reference["knots"], smoothing_method="REML", optimize_smoothing=True)
    model.fit(_frame(reference["data"], reference["levels"]))
    assert not result["free_sp"]
    np.testing.assert_array_equal(model.smoothing_fixed_mask_, np.ones(len(sp), dtype=bool))
    np.testing.assert_allclose(model.smoothing_params, result["full_sp"], atol=2e-12, rtol=2e-12)
    _assert_fit_invariants(reference, model, result)


def test_ordinary_soap_factor_capture_provenance():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert hashlib.sha256(PROBE.read_bytes()).hexdigest() == manifest["probe_sha256"]
    assert hashlib.sha256(TERM_SP_PROBE.read_bytes()).hexdigest() == manifest["term_sp_probe_sha256"]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest["generator_sha256"]
    for name, expected in manifest["fixture_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
