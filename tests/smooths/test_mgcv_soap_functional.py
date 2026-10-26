"""Matrix soap smooth references, including upstream offset rejection evidence."""

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

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_functional.R"
KEY = reference_key(
    "soap-functional-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_soap_functional", KEY)
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
        save_reference("mgcv_soap_functional", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


def _frame(data):
    return pd.DataFrame({key: list(value) for key, value in data.items()})


def _model(ref, basis, kind, method):
    formula = f"response~s(X,Y,bs='{basis}',k=8,xt={{'bnd':{ref['bnd']!r},'nmax':30}}{',by=L' if kind != 'unit' else ''})+offset(o)"
    return GAM(
        formula=formula,
        knots=ref["knots"],
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=np.atleast_1d(ref.get("sp", 0.3))
        if method == "fixed"
        else None,
    )


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
@pytest.mark.parametrize("kind", ["unit", "varying", "constant"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("known", ["FALSE", "TRUE"])
def test_soap_functional_fit_prediction_and_upstream_offset_boundary(
    references, basis, kind, method, known
):
    ref = references[f"{basis}/{kind}/{method}/{known}"]
    model = _model(ref, basis, kind, method)
    if not ref["accepted"]:
        assert basis in {"so", "sf"} and known == "TRUE"
        assert ref["error"] == "incorrect number of dimensions"
        with pytest.raises(NotImplementedError, match="known-boundary offsets"):
            model.fit(data=_frame(ref["data"]))
        return
    model.fit(data=_frame(ref["data"]))
    frame = _frame(ref["newdata"])
    fit, se = model.predict(frame, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(se, ref["se"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=4e-5, rtol=4e-5)
    P = model.predict(frame, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance"], atol=4e-5, rtol=4e-5
    )
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-3, rtol=3e-3)
    X = model.predict(type="lpmatrix")[:, 1:]
    T = np.linalg.lstsq(X, np.asarray(ref["X"])[:, 1:], rcond=None)[0]
    np.testing.assert_allclose(X @ T, np.asarray(ref["X"])[:, 1:], atol=1e-8)
    for penalty, target in zip(_penalty_blocks_seq(model), ref["S"], strict=True):
        np.testing.assert_allclose(
            T.T @ penalty.matrix @ T, target, atol=2e-6, rtol=2e-6
        )
    if kind == "varying":
        np.testing.assert_allclose(P[1, 1:], 0, atol=1e-12)


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
def test_soap_functional_persistence_and_changed_quadrature_width(
    references, basis, tmp_path
):
    ref = references[f"{basis}/varying/fixed/FALSE"]
    model = _model(ref, basis, "varying", "fixed").fit(data=_frame(ref["data"]))
    new = {
        name: np.asarray(value)[:, :2] if name in {"X", "Y", "L"} else value
        for name, value in ref["newdata"].items()
    }
    frame = _frame(new)
    expected = model.predict(frame, return_se=True)
    path = tmp_path / "soap_functional.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    for actual, wanted in zip(
        restored.predict(frame, return_se=True), expected, strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-13)
    P = model.predict(frame, type="lpmatrix")
    points = {name: np.asarray(new[name]).reshape(-1) for name in ["X", "Y"]}
    points["L"] = np.ones(len(points["X"]))
    points["o"] = np.zeros(len(points["X"]))
    point_basis = model.predict(_frame(points), type="lpmatrix")[:, 1:]
    W = np.asarray(new["L"])
    np.testing.assert_allclose(
        P[:, 1:],
        np.einsum("nq,nqk->nk", W, point_basis.reshape(*W.shape, -1)),
        atol=2e-12,
    )


@pytest.mark.parametrize("bad", ["coordinate_shape", "weight_shape"])
def test_soap_functional_shape_guard(references, bad):
    ref = references["so/varying/fixed/FALSE"]
    frame = _frame(ref["data"])
    key = "Y" if bad == "coordinate_shape" else "L"
    frame[key] = [row[:2] for row in frame[key]]
    with pytest.raises(ValueError, match="equal shape"):
        _model(ref, "so", "varying", "fixed").fit(data=frame)
