"""FS paired quadrature locations with vector factors, actual mgcv references."""

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

PROBE = Path(__file__).with_name("probes") / "mgcv_fs_functional.R"
KEY = reference_key(
    "fs-functional-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_fs_functional", KEY)
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
        save_reference("mgcv_fs_functional", KEY, result)
    assert result["version"] == "1.9.4"
    return result


def _frame(data, levels):
    frame = pd.DataFrame({key: list(value) for key, value in data.items()})
    frame["f"] = pd.Categorical(frame.f, categories=levels)
    return frame


CASES = [
    (b, k, m, d, repeated)
    for b in ["tp", "cr", "ps"]
    for k in ["unit", "varying", "constant"]
    for m in ["fixed", "REML"]
    for d in [1, 2]
    for repeated in [False, True]
    if (d == 1 or b == "tp") and (not repeated or b != "tp")
]


@pytest.mark.parametrize("basis,kind,method,dims,repeated", CASES)
def test_fs_functional_fit_predict_penalties_and_uncertainty(
    references, basis, kind, method, dims, repeated
):
    ref = references["cases"][
        f"{basis}/{kind}/{method}/{dims}" + ("/repeated" if repeated else "")
    ]
    formula = f"response~s({'X,Z' if dims == 2 else 'X'},f,bs='fs',k=6,xt={{'bs':'{basis}'}}{',by=L' if kind != 'unit' else ''})"
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=ref["sp"] if method == "fixed" else None,
    )
    if basis == "tp":
        with pytest.raises(NotImplementedError, match="natural-null-space"):
            model.fit(data=_frame(ref["data"], ref["levels"]))
        return
    model.fit(data=_frame(ref["data"], ref["levels"]))
    frame = _frame(ref["newdata"], ref["levels"])
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
    remaining = list(range(len(ref["S"])))
    for local_index, penalty in enumerate(_penalty_blocks_seq(model)):
        actual = T.T @ penalty.matrix @ T
        index = min(
            remaining, key=lambda j: np.linalg.norm(actual - np.asarray(ref["S"][j]))
        )
        np.testing.assert_allclose(actual, ref["S"][index], atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(
            model.smoothing_params[local_index], ref["sp"][index], atol=3e-4, rtol=3e-3
        )
        remaining.remove(index)
    if kind == "varying":
        np.testing.assert_allclose(P[1, 1:], 0, atol=1e-12)


@pytest.mark.parametrize("mode", ["fs", "sz"])
@pytest.mark.parametrize("basis", ["so", "sf", "sw", "ad"])
def test_upstream_soap_factor_nesting_inventory(references, mode, basis):
    ref = references["nesting"][f"{mode}/{basis}"]
    assert ref["accepted"] == (basis == "sf" or mode == "sz" and basis == "sw")
    if basis in {"so", "ad"}:
        assert "multiply penalized" in ref["error"]


@pytest.mark.parametrize("basis", ["cr", "ps"])
@pytest.mark.parametrize("kind", ["unit", "varying"])
def test_fs_functional_prediction_contracts(references, basis, kind, tmp_path):
    ref = references["cases"][f"{basis}/{kind}/fixed/1"]
    model = GAM(
        formula=f"response~s(X,f,bs='fs',k=6,xt={{'bs':'{basis}'}}{',by=L' if kind != 'unit' else ''})",
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=ref["sp"],
    ).fit(data=_frame(ref["data"], ref["levels"]))
    frame = _frame(ref["newdata"], ref["levels"])
    restored = GAM.load_model(model.save_model(tmp_path / "fs.pkl"))
    for output in ["link", "lpmatrix"]:
        np.testing.assert_allclose(
            restored.predict(frame, type=output), model.predict(frame, type=output)
        )
    np.testing.assert_allclose(
        restored.predict(frame, return_se=True), model.predict(frame, return_se=True)
    )
    # Width belongs to each prediction batch, not to fitted basis state.
    narrow = frame.copy()
    narrow["X"] = [np.asarray(row)[:2] for row in narrow.X]
    if kind != "unit":
        narrow["L"] = [np.asarray(row)[:2] for row in narrow.L]
    matrix = model.predict(narrow, type="lpmatrix")
    pieces = []
    for column in range(2):
        point = narrow.copy()
        point["X"] = [row[column] for row in narrow.X]
        if kind != "unit":
            point["L"] = [row[column] for row in narrow.L]
        pieces.append(model.predict(point, type="lpmatrix")[:, 1:])
    np.testing.assert_allclose(matrix[:, 1:], sum(pieces), atol=1e-12)
    reordered = narrow.copy()
    reordered["f"] = reordered.f.cat.reorder_categories(list(reversed(ref["levels"])))
    np.testing.assert_allclose(model.predict(reordered), model.predict(narrow))
    invalid = narrow.copy()
    invalid["f"] = invalid.f.astype(object)
    invalid.loc[0, "f"] = "unseen"
    invalid.loc[1, "f"] = None
    prediction = model.predict(invalid)
    assert np.isnan(prediction[:2]).all()
    np.testing.assert_allclose(prediction[2:], model.predict(narrow)[2:])
    if kind != "unit":
        malformed = narrow.copy()
        malformed["L"] = frame.L
        with pytest.raises(ValueError, match="equal shape"):
            model.predict(malformed)


def test_fs_functional_packing_normalizes_signed_zero():
    from nampy.gam.smooths.categorical.fs import (
        _functional_factor_point_data,
        _pack_functional_factor_data,
    )
    from nampy.gam.smooths.smooth_base import resolve_by_state

    # Preserve the upstream ExtractData call, rather than assuming how R
    # stringifies IEEE signed zeros in uniquecombs' mixed-coordinate keys.
    source = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
X<-matrix(c(0,-0,0,1,-0,0,-0,1),4,2)
f<-factor(c('a','a','b','b'))
z<-mgcv:::ExtractData(s(X,f,bs='fs',xt=list(bs='cr')),list(X=X,f=f),NULL)$data
write_json(list(x=z$X,f=as.character(z$f),index=attr(z,'index')),commandArgs(TRUE)[1],digits=17,auto_unbox=TRUE)
"""
    key = reference_key("fs-functional-signed-zero-packing", source)
    ref = load_reference("mgcv_fs_functional", key)
    if ref is None:
        with tempfile.TemporaryDirectory() as directory:
            script, output = [
                Path(directory) / name for name in ["probe.R", "out.json"]
            ]
            script.write_text(source)
            subprocess.run(
                ["Rscript", str(script), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            ref = json.loads(output.read_text())
        save_reference("mgcv_fs_functional", key, ref)
    coordinates = np.array([[0.0, -0.0], [-0.0, 0.0], [0.0, -0.0], [1.0, 1.0]])
    data = np.empty((4, 2), dtype=object)
    data[:, 0] = list(coordinates)
    data[:, 1] = ["a", "a", "b", "b"]
    points = _functional_factor_point_data(
        data, [0], coordinates[:, :, None], resolve_by_state(None, data, ["X", "f"])
    )
    packed, index = _pack_functional_factor_data(points, [0, 1], [0])
    np.testing.assert_array_equal(index + 1, ref["index"])
    np.testing.assert_array_equal(packed[:, 0].astype(float), ref["x"])
    np.testing.assert_array_equal(packed[:, 1], ref["f"])
