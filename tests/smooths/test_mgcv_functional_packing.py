"""Repeated quadrature-location packing follows mgcv ExtractData."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_functional_packing.R"
KEY = reference_key(
    "matrix-functional-packed-setup",
    {"sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)
BASES = ["tp", "ts", "ds", "gp", "sos", "cr", "cs", "cc", "ps", "cp", "bs", "ad", "ad2"]
CASES = [
    (bs, kind, method)
    for bs in BASES
    for kind in ["weighted", "unweighted", "constant"]
    for method in ["fixed", "REML"]
    if not (method == "REML" and kind != "weighted" and bs != "ad2")
    and not (bs == "ad2" and kind != "unweighted")
]


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_functional_packing", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.json"
            run = subprocess.run(
                ["Rscript", str(PROBE), str(out)], capture_output=True, text=True
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_functional_packing", KEY, result)
    return result


@pytest.mark.parametrize("bs,kind,method", CASES)
def test_upstream_packed_fit_identity(references, bs, kind, method):
    r = references["cases"][f"{bs}/{kind}/{method}"]
    np.testing.assert_allclose(np.asarray(r["X"]) @ r["coef"], r["fitted"], atol=2e-12)
    np.testing.assert_allclose(np.asarray(r["Xp"]) @ r["coef"], r["fit"], atol=2e-12)


def _frame(data):
    return pd.DataFrame({k: list(v) for k, v in data.items()})


@pytest.mark.parametrize("bs,kind,method", CASES)
def test_runtime_packed_fit_and_penalties(references, bs, kind, method):
    from nampy.gam import GAM
    from nampy.gam.model_state import _edf_total, _penalty_blocks_seq

    r = references["cases"][f"{bs}/{kind}/{method}"]
    dim = 2 if bs in ["tp", "ds", "gp", "sos", "ad2"] else 1
    extra = ",k=[5,5],m=3" if bs == "ad2" else ",k=10,m=3" if bs == "ad" else ",k=10"
    formula = f"y~s({'X,Z' if dim == 2 else 'X'},bs={'ad' if bs == 'ad2' else bs!r}{extra}{',by=L' if kind != 'unweighted' else ''})"
    ns = 9 if bs == "ad2" else 3 if bs == "ad" else 1
    if bs == "cs":
        with pytest.raises(
            NotImplementedError, match="packed repeated locations.*unsupported"
        ):
            GAM(
                formula=formula,
                smoothing_method=method,
                optimize_smoothing=method != "fixed",
                smoothing_params=[0.4] if method == "fixed" else None,
            ).fit(data=_frame(r["data"]))
        return
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=[0.4] * ns if method == "fixed" else None,
    ).fit(data=_frame(r["data"]), sample_weight=np.asarray(r["data"]["w"]))
    fit, se = model.predict(_frame(r["newdata"]), return_se=True)
    np.testing.assert_allclose(fit, r["fit"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(se, r["se"], atol=4e-5, rtol=4e-5)
    P = model.predict(_frame(r["newdata"]), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, r["kernel"], atol=4e-5, rtol=4e-5
    )
    np.testing.assert_allclose(_edf_total(model), r["edf"], atol=4e-5, rtol=4e-5)
    np.testing.assert_allclose(model.smoothing_params, r["sp"], atol=4e-4, rtol=4e-4)
    local = model.predict(type="lpmatrix")[:, 1:]
    expected = np.asarray(r["X"])[:, 1:]
    align = np.linalg.lstsq(local, expected, rcond=None)[0]
    np.testing.assert_allclose(local @ align, expected, atol=2e-7)
    penalties = list(r["S"].values()) if isinstance(r["S"], dict) else r["S"]
    for block, wanted in zip(_penalty_blocks_seq(model), penalties, strict=True):
        np.testing.assert_allclose(
            align.T @ block.matrix @ align, wanted, atol=3e-6, rtol=3e-6
        )


def test_helper_r_order_and_unpack(references):
    from nampy.gam.smooths.smooth_base import linear_functional_setup_points

    ref = references["helper"]
    locations = np.stack([ref["A"], ref["B"]], axis=-1)
    points, index = linear_functional_setup_points(locations)
    np.testing.assert_array_equal(points, ref["points"])
    np.testing.assert_array_equal(points[index], locations.reshape(-1, 2))
    f_index = (
        np.asarray(ref["index"]).reshape(locations.shape[:2], order="F").reshape(-1) - 1
    )
    np.testing.assert_array_equal(index, f_index)


@pytest.mark.parametrize("unique_count", [8, 9, 10])
def test_helper_exact_packing_threshold_and_unpacked_order(unique_count):
    from nampy.gam.smooths.smooth_base import linear_functional_setup_points

    flat = np.arange(10, dtype=float)
    flat[unique_count:] = 0
    locations = flat.reshape(5, 2, 1)
    points, index = linear_functional_setup_points(locations)
    assert len(points) == (unique_count if unique_count < 9 else 10)
    np.testing.assert_array_equal(points[index], locations.reshape(-1, 1))
    if unique_count >= 9:
        np.testing.assert_array_equal(
            points[:, 0], locations[:, :, 0].reshape(-1, order="F")
        )
