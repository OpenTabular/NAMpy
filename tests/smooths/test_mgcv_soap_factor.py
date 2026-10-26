"""Actual mgcv fits for ordinary soap factor-smooth nesting."""

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

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_factor.R"
KEY = reference_key(
    "soap-factor-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_soap_factor", KEY)
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
        save_reference("mgcv_soap_factor", KEY, result)
    assert result["version"] == "1.9.4"
    return result


def _frame(data, levels):
    frame = pd.DataFrame(data)
    frame["f"] = pd.Categorical(frame.f, categories=levels)
    return frame


@pytest.mark.parametrize("mode,basis", [("fs", "sf"), ("sz", "sf"), ("sz", "sw")])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("by", [False, True])
@pytest.mark.parametrize("domain", ["free", "mixed"])
def test_soap_factor_fitted_parity(references, mode, basis, method, by, domain):
    ref = references["cases"][f"{mode}/{basis}/{method}/{str(by).upper()}/{domain}"]
    xt = {"bs": basis, "bnd": ref["boundary"], "nmax": 30}
    model = GAM(
        formula=f"response~s(x,y,f,bs='{mode}',k=8,xt={xt!r}{',by=w' if by else ''})",
        knots=references["knots"],
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=ref["sp"] if method == "fixed" else None,
    )
    model.fit(data=_frame(ref["data"], references["levels"]))
    frame = _frame(ref["newdata"], references["levels"])
    fit, se = model.predict(frame, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(se, ref["se"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=4e-5, rtol=4e-5)
    P = model.predict(frame, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance"], atol=4e-5, rtol=4e-5
    )
    X = model.predict(type="lpmatrix")[:, 1:]
    R = np.asarray(ref["X"])[:, 1:]
    T = np.linalg.lstsq(X, R, rcond=None)[0]
    np.testing.assert_allclose(X @ T, R, atol=2e-7)
    np.testing.assert_allclose(P[:, 1:] @ T, np.asarray(ref["Xp"])[:, 1:], atol=2e-7)
    for penalty, reference in zip(_penalty_blocks_seq(model), ref["S"], strict=True):
        np.testing.assert_allclose(
            T.T @ penalty.matrix @ T, reference, atol=3e-6, rtol=3e-6
        )


@pytest.mark.parametrize("mode,basis", [("fs", "sf"), ("sz", "sf"), ("sz", "sw")])
def test_soap_factor_persistence_and_matrix_scope(references, mode, basis, tmp_path):
    ref = references["cases"][f"{mode}/{basis}/fixed/FALSE/mixed"]
    xt = {"bs": basis, "bnd": ref["boundary"], "nmax": 30}
    formula = f"response~s(x,y,f,bs='{mode}',k=8,xt={xt!r})"
    model = GAM(
        formula=formula,
        knots=references["knots"],
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=ref["sp"],
    )
    frame = _frame(ref["data"], references["levels"])
    model.fit(data=frame)
    new = _frame(ref["newdata"], references["levels"])
    restored = GAM.load_model(model.save_model(tmp_path / "soap-factor.pkl"))
    np.testing.assert_allclose(
        restored.predict(new, return_se=True), model.predict(new, return_se=True)
    )
    matrix = frame.copy()
    for column in ["x", "y"]:
        matrix[column] = [[v, v] for v in frame[column]]
    with pytest.raises(NotImplementedError, match="Matrix"):
        model.fit(data=matrix)


@pytest.mark.parametrize(
    "mode,basis", [("fs", "sw"), ("fs", "so"), ("sz", "so"), ("fs", "ad"), ("sz", "ad")]
)
def test_soap_factor_rejected_compositions(references, mode, basis):
    ref = references["cases"]["fs/sf/fixed/FALSE/free"]
    xt = {"bs": basis, "bnd": ref["boundary"], "nmax": 30}
    model = GAM(
        formula=f"response~s(x,y,f,bs='{mode}',k=8,xt={xt!r})",
        knots=references["knots"],
    )
    with pytest.raises(NotImplementedError):
        model.fit(data=_frame(ref["data"], references["levels"]))
