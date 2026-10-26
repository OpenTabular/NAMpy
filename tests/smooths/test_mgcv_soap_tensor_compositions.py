"""Source inventory for soap tensor single/multiple-penalty compositions."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import criterion_value
from nampy.gam.model_state import _design_matrix, _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.smooths import test_mgcv_soap_tensor as existing

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_tensor_compositions.R"
KEY = reference_key(
    "soap-tensor-compositions", hashlib.sha256(PROBE.read_bytes()).hexdigest()
)


@pytest.fixture(scope="module")
def source_compositions():
    expected = load_reference("mgcv_soap_tensor_compositions", KEY)
    if expected is None:
        ref = existing.references.__wrapped__()["te/sf/fixed"]
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = [
                Path(directory) / name for name in ("inputs.json", "results.json")
            ]
            inputs.write_text(
                json.dumps(
                    {key: ref[key] for key in ("data", "newdata", "bnd", "knots")}
                )
            )
            run = subprocess.run(
                ["Rscript", str(PROBE), str(inputs), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            expected = json.loads(output.read_text())
        save_reference("mgcv_soap_tensor_compositions", KEY, expected)
    return expected["cases"]


def test_source_soap_tensor_composition_inventory(source_compositions):
    assert len(source_compositions) == 30
    for name, result in source_compositions.items():
        _, basis, geometry = name.split("/")
        accepted = (basis == "so" and geometry in {"known", "known_hole"}) or (
            basis == "sf" and geometry in {"free", "mixed"}
        )
        assert result["accepted"] is accepted, name
        if accepted:
            assert result["marginal_npen"] == 1, name
            assert all(fit["accepted"] for fit in result["fits"].values()), name
        elif basis == "sf" and geometry in {"known", "known_hole"}:
            assert "no free coefs in sf smooth" in result["error"], name
        if (result["marginal_npen"] or 0) > 1:
            assert not result["accepted"], name
            assert "multiple penalties are not supported" in result["error"], name


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize(
    "basis,geometry", [("so", "free"), ("so", "hole"), ("so", "mixed"), ("sf", "hole")]
)
def test_upstream_multiple_penalty_soap_tensors_are_precisely_rejected(
    source_compositions, fun, basis, geometry
):
    ref = source_compositions[f"{fun}/{basis}/{geometry}"]
    assert ref["marginal_npen"] > 1
    assert "multiple penalties are not supported" in ref["error"]
    formula = _formula(fun, basis, ref)
    with pytest.raises(
        NotImplementedError, match="multiple penalties are not supported"
    ):
        GAM(formula=formula, knots=ref["knots"]).fit(pd.DataFrame(ref["data"]))


def _formula(fun, basis, ref):
    return f"response~{fun}(x,y,z,bs=['{basis}','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{ref['bnd']!r},'nmax':30}},None])+offset(o)"


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize(
    "basis,geometry", [("so", "known"), ("so", "known_hole"), ("sf", "mixed")]
)
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_single_penalty_known_boundary_soap_tensors_match_source(
    source_compositions, fun, basis, geometry, method
):
    ref = source_compositions[f"{fun}/{basis}/{geometry}"]
    assert ref["accepted"]
    assert ref["marginal_npen"] == 1
    assert ref["marginal_offset"] > 0
    assert ref["tensor_offset"] == 0  # Upstream discards the marginal offset attribute.
    expected = ref["fits"][method]
    assert expected["accepted"]
    data, newdata = pd.DataFrame(ref["data"]), pd.DataFrame(ref["newdata"])
    model = GAM(
        formula=_formula(fun, basis, ref),
        knots=ref["knots"],
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=np.atleast_1d(expected["sp"]) if method == "fixed" else None,
    ).fit(data)
    fit, se = model.predict(newdata, return_se=True)
    np.testing.assert_allclose(fit, expected["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, expected["se"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        model.predict(), expected["train_fit"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(_edf_total(model), expected["edf"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        model.smoothing_params, expected["sp"], atol=2e-3, rtol=2e-3
    )
    P = model.lpmatrix(newdata)
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, expected["covariance"], atol=3e-5, rtol=3e-5
    )
    X = model.lpmatrix(data)
    A = np.linalg.lstsq(X, np.asarray(expected["X"]), rcond=None)[0]
    np.testing.assert_allclose(X @ A, expected["X"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(P @ A, expected["P"], atol=2e-7, rtol=0)
    B = np.asarray(_design_matrix(model))
    T = np.linalg.lstsq(B, np.asarray(expected["fit_X"]), rcond=None)[0]
    np.testing.assert_allclose(B @ T, expected["fit_X"], atol=2e-7, rtol=0)
    for block, S in zip(_penalty_blocks_seq(model), expected["S"], strict=True):
        np.testing.assert_allclose(T.T @ block.matrix @ T, S, atol=2e-6, rtol=2e-6)
    score = criterion_value(
        model,
        data["response"].to_numpy(),
        np.log(model.smoothing_params),
        method="reml",
    )
    np.testing.assert_allclose(score, expected["score"], atol=3e-5, rtol=3e-5)
