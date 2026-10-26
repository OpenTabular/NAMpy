"""Live mgcv-generated soap tensor fit and prediction reference contracts."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _design_matrix, _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_tensor.R"
KEY = reference_key(
    "soap-tensor-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_soap_tensor", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(
                ["Rscript", str(PROBE), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
        save_reference("mgcv_soap_tensor", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize("basis", ["sf", "sw"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_soap_tensor_fits_predictions_and_uncertainty_match_mgcv(
    references, fun, basis, method
):
    ref = references[f"{fun}/{basis}/{method}"]
    formula = f"response~{fun}(x,y,z,bs=['{basis}','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{ref['bnd']!r},'nmax':30}},None])+offset(o)"
    model = GAM(
        formula=formula,
        knots=ref["knots"],
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=np.atleast_1d(ref["sp"]) if method == "fixed" else None,
    ).fit(data=pd.DataFrame(ref["data"]))
    frame = pd.DataFrame(ref["newdata"])
    fit, se = model.predict(frame, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=3e-5, rtol=3e-5)
    P = model.predict(frame, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=2e-3, rtol=2e-3)

    # Align full coefficient spaces: t2's fitting C and prediction Cp differ.
    X = model.predict(type="lpmatrix")
    R = np.asarray(ref["X"])
    alignment = np.linalg.lstsq(X, R, rcond=None)[0]
    np.testing.assert_allclose(X @ alignment, R, atol=2e-7)
    np.testing.assert_allclose(P @ alignment, ref["Xp"], atol=2e-7)
    # Penalties live in the fitting parameterization, not t2's Cp coordinates.
    fit_basis = np.asarray(_design_matrix(model))
    target_basis = np.asarray(ref["raw_X"])
    penalty_alignment = np.linalg.lstsq(fit_basis, target_basis, rcond=None)[0]
    np.testing.assert_allclose(fit_basis @ penalty_alignment, target_basis, atol=2e-7)
    for penalty, target in zip(_penalty_blocks_seq(model), ref["raw_S"], strict=True):
        np.testing.assert_allclose(
            penalty_alignment.T @ penalty.matrix @ penalty_alignment,
            target,
            atol=2e-6,
            rtol=2e-6,
        )


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize("basis", ["so", "ad", "fs", "sz"])
def test_upstream_unsupported_marginals_remain_rejected(fun, basis):
    frame = pd.DataFrame(
        {
            "response": np.arange(20),
            "x": np.linspace(-1, 1, 20),
            "y": np.linspace(1, -1, 20),
            "z": np.linspace(0, 1, 20),
        }
    )
    with pytest.raises(NotImplementedError, match="Tensor marginals"):
        GAM(formula=f"response~{fun}(x,y,z,bs=['{basis}','cr'],d=[2,1],k=[8,4])").fit(
            data=frame
        )


@pytest.mark.parametrize("basis", ["sf", "sw"])
def test_soap_tensor_outside_domain_predictions_and_persistence(
    references, basis, tmp_path
):
    ref = references[f"t2/{basis}/fixed"]
    formula = f"response~t2(x,y,z,bs=['{basis}','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{ref['bnd']!r},'nmax':30}},None])+offset(o)"
    model = GAM(
        formula=formula,
        knots=ref["knots"],
        smoothing_method="fixed",
        smoothing_params=np.atleast_1d(ref["sp"]),
    ).fit(data=pd.DataFrame(ref["data"]))
    frame = pd.DataFrame(ref["newdata"])
    frame.loc[0, ["x", "y"]] = [-1.2, 0]
    expected = model.predict(frame, return_se=True)
    path = tmp_path / "soap_tensor.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    for actual, wanted in zip(
        restored.predict(frame, return_se=True), expected, strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-13)
    assert np.isnan(restored.predict(frame, type="lpmatrix")[0, 1:]).all()


def test_soap_boundary_margin_with_multiple_free_loops_is_rejected(references):
    ref = references["te/sf/fixed"]
    boundary = [
        *ref["bnd"],
        {"x": [-0.2, 0.2, 0.2, -0.2, -0.2], "y": [-0.2, -0.2, 0.2, 0.2, -0.2]},
    ]
    frame = pd.DataFrame(ref["data"])
    frame = frame[(abs(frame.x) > 0.3) | (abs(frame.y) > 0.3)]
    formula = f"response~te(x,y,z,bs=['sf','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{boundary!r},'nmax':30}},None])"
    with pytest.raises(NotImplementedError, match="multiple penalties"):
        GAM(
            formula=formula,
            knots=ref["knots"],
            smoothing_method="fixed",
            smoothing_params=[0.3, 0.3],
        ).fit(data=frame)
