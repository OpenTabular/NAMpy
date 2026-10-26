"""Matrix spline sums against fresh mgcv smoothCon/gam/predict.gam fits."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _edf_total
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_spline_linear_functionals.R"
KEY = reference_key("spline-matrix-functions", hashlib.sha256(PROBE.read_bytes()).hexdigest())


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_spline_linear_functionals", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(["Rscript", str(PROBE), str(output)], check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
        save_reference("mgcv_spline_linear_functionals", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


def _frame(values):
    return pd.DataFrame({name: list(value) for name, value in values.items()})


@pytest.mark.parametrize("basis", ["cr", "cs", "cc", "ps", "cp", "bs"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("kind", ["unweighted", "weighted", "constant"])
def test_spline_matrix_functions_match_mgcv(references, basis, method, kind):
    ref = references[f"{basis}/{method}/{kind}/1"]
    formula = f"y ~ s(X,bs='{basis}',k=10{',by=L' if kind != 'unweighted' else ''}) + offset(o)"
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=[0.4] if method == "fixed" else None,
    ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))
    prediction, se = model.predict(_frame(ref["newdata"]), return_se=True)
    np.testing.assert_allclose(prediction, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    P = model.predict(_frame(ref["newdata"]), type="lpmatrix")
    np.testing.assert_allclose(P @ model.vcov() @ P.T, ref["covariance_kernel"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.predict(), ref["train_fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(_edf_total(model), np.sum(ref["edf"]), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)
    coordinates = np.asarray(ref["newdata"]["X"])
    point_data = pd.DataFrame({"X": coordinates.ravel(), "L": 1.0, "o": 0.0})
    point_basis = model.predict(point_data, type="lpmatrix")
    weights = np.ones_like(coordinates) if kind == "unweighted" else np.asarray(ref["newdata"]["L"])
    aggregate = np.einsum("nq,nqk->nk", weights, point_basis.reshape(*coordinates.shape, -1))
    aggregate[:, 0] = 1.0
    np.testing.assert_allclose(P, aggregate, atol=2e-12, rtol=2e-12)
