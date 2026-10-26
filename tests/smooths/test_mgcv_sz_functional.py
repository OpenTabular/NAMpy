"""SZ paired quadrature locations with vector factors, actual mgcv references."""

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

PROBE = Path(__file__).with_name("probes") / "mgcv_sz_functional.R"
KEY = reference_key(
    "sz-functional-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_sz_functional", KEY)
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
        save_reference("mgcv_sz_functional", KEY, result)
    assert result["version"] == "1.9.4"
    return result


def _frame(data, levels):
    frame = pd.DataFrame({key: list(value) for key, value in data.items()})
    frame["f"] = pd.Categorical(frame.f, categories=levels)
    if "g" in frame:
        frame["g"] = pd.Categorical(frame.g, categories=["v", "u"])
    return frame


CASES = [
    (b, k, m, d, repeated, factors)
    for b in ["tp", "cr", "ps"]
    for k in ["unit", "varying", "constant"]
    for m in ["fixed", "REML"]
    for d in [1, 2]
    for repeated in [False, True]
    for factors in [1, 2]
    if (d == 1 or b == "tp")
    and (not repeated or b != "tp")
    and (factors == 1 or b == "cr" and k != "constant" and not repeated)
]


@pytest.mark.parametrize("basis,kind,method,dims,repeated,factors", CASES)
def test_sz_functional_fit_predict_penalties_and_uncertainty(
    references, basis, kind, method, dims, repeated, factors
):
    ref = references["cases"][
        f"{basis}/{kind}/{method}/{dims}"
        + ("/repeated" if repeated else "")
        + ("/two" if factors == 2 else "")
    ]
    formula = f"response~s({'X,Z' if dims == 2 else 'X'}{',g' if factors == 2 else ''},f,bs='sz',k=6,xt={{'bs':'{basis}'}}{',by=L' if kind != 'unit' else ''})"
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=ref["sp"] if method == "fixed" else None,
    )
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


@pytest.mark.parametrize("basis", ["cr", "ps"])
@pytest.mark.parametrize("kind", ["unit", "varying"])
def test_sz_functional_prediction_contracts(references, basis, kind, tmp_path):
    ref = references["cases"][f"{basis}/{kind}/fixed/1"]
    model = GAM(
        formula=f"response~s(X,f,bs='sz',k=6,xt={{'bs':'{basis}'}}{',by=L' if kind != 'unit' else ''})",
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
