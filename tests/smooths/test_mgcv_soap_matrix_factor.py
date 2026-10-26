"""Source-defined matrix soap nesting with vector factors and paired points."""

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

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_soap_matrix_factor"
PROBE = Path(__file__).with_name("probes") / "mgcv_soap_matrix_factor.R"


def case(mode="fs", basis="sf", method="fixed", kind="varying", repeated=False, boundary="free", factor_shape="vector"):
    frozen = json.loads((ROOT / "inputs.json").read_text())
    data, new = frozen["data"], frozen["newdata"]
    x, y = np.array(data["X"]), np.array(data["Y"])
    if repeated:
        grid = np.array([[-.8, -.6], [-.6, .5], [-.4, -.3], [-.3, .7], [.3, -.7], [.4, .3], [.6, -.4], [.8, .6], [-.7, .2], [.7, .2], [.2, .7]])
        locations = np.resize(grid, (*x.shape, 2))
        x, y = locations[:, :, 0], locations[:, :, 1]
    for target in (data, new):
        weights = np.asarray(target["L"])
        if kind == "unit":
            weights[:] = 1
        if kind == "constant":
            weights[np.sum(weights, axis=1) == 0] = 1
            weights /= weights.sum(axis=1)[:, None]
        target["L"] = weights.tolist()
    data["X"], data["Y"] = x.tolist(), y.tolist()
    levels = frozen["levels"]
    data.pop("noise")
    data["response"] = frozen["responses"][f"{kind}/{repeated}"]
    bnd = [frozen["outer"]]
    if boundary in {"mixed", "two_free"}:
        bnd += [frozen["inner"]]
        if boundary == "mixed":
            bnd[1]["f"] = [.4] * 5
    if boundary == "known":
        bnd[0]["f"] = [-1., 1., 1., -1., -1.]
    return {"data": data, "newdata": new, "levels": levels, "knots": frozen["knots"],
            "boundary": bnd, "mode": mode, "basis": basis, "method": method,
            "kind": kind, "repeated": repeated, "boundary_kind": boundary,
            "factor_shape": factor_shape}


def reference(p, probe=PROBE):
    key = reference_key("soap-matrix-factor-v1", {"case": p, "source_sha256": hashlib.sha256(probe.read_bytes()).hexdigest()})
    result = load_reference("mgcv_soap_matrix_factor", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = [Path(tmp) / name for name in ("in.json", "out.json")]
            inp.write_text(json.dumps(p))
            run = subprocess.run(["Rscript", str(probe), str(inp), str(out)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_soap_matrix_factor", key, result)
    return result


def frame(data, levels):
    result = pd.DataFrame({key: list(value) for key, value in data.items()})
    result["f"] = pd.Categorical(result.f, categories=levels)
    return result


def model_for(p, ref):
    xt = {"bs": p["basis"], "bnd": p["boundary"], "nmax": 30}
    return GAM(formula=f"response~s(X,Y,f,bs='{p['mode']}',k=8,xt={xt!r}{',by=L' if p['kind'] != 'unit' else ''})+offset(o)", knots=p["knots"], smoothing_method=p["method"], optimize_smoothing=p["method"] != "fixed", smoothing_params=np.atleast_1d(ref["sp"]) if p["method"] == "fixed" else None)


@pytest.mark.parametrize("mode,basis,boundary", [("fs", "sf", "free"), ("sz", "sf", "free"), ("sz", "sw", "free"), ("sz", "so", "known")])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("kind", ["unit", "varying", "constant"])
@pytest.mark.parametrize("repeated", [False, True])
def test_matrix_soap_factor_fits_match_source(mode, basis, boundary, method, kind, repeated):
    p = case(mode, basis, method, kind, repeated, boundary)
    ref = reference(p)
    assert ref["accepted"], ref
    model = model_for(p, ref).fit(data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"])
    new = frame(p["newdata"], p["levels"])
    predicted, se = model.predict(new, return_se=True)
    np.testing.assert_allclose(predicted, ref["fit"], rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(se, ref["se"], rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(model.predict(), ref["fitted"], rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], rtol=4e-5, atol=4e-5)
    X, P = model.predict(type="lpmatrix"), model.lpmatrix(new)
    np.testing.assert_allclose(P @ model.vcov() @ P.T, ref["covariance"], rtol=4e-5, atol=4e-5)
    transform = np.linalg.lstsq(X[:, 1:], np.asarray(ref["X"])[:, 1:], rcond=None)[0]
    np.testing.assert_allclose(X[:, 1:] @ transform, np.asarray(ref["X"])[:, 1:], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(P[:, 1:] @ transform, np.asarray(ref["Xp"])[:, 1:], atol=2e-7, rtol=2e-7)
    for penalty, target in zip(_penalty_blocks_seq(model), ref["S"], strict=True):
        np.testing.assert_allclose(transform.T @ penalty.matrix @ transform, target, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("mode,basis,boundary", [("fs", "sf", "mixed"), ("sz", "sf", "mixed"), ("sz", "sw", "known"), ("sz", "sw", "two_free")])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_matrix_soap_factor_known_boundary_behavior(mode, basis, boundary, method):
    # Source fs/sz replaces X, discarding a single-penalty sf base's offset.
    test_matrix_soap_factor_fits_match_source(mode, basis, boundary, method, "varying", True)
    ref = reference(case(mode, basis, method, "varying", True, boundary))
    assert ref["factor_offset"] is None
    assert ref["term_prediction_offset"] is None
    if basis == "sf":
        assert np.max(np.abs(ref["base_offset"])) > .1


@pytest.mark.parametrize("mode,basis,boundary,source_error", [
    ("fs", "so", "free", "multiply penalized"),
    ("sz", "so", "mixed", "multiply penalized"),
    ("fs", "sf", "two_free", "multiply penalized"),
    ("sz", "sf", "two_free", "multiply penalized"),
    ("fs", "sw", "free", "subscript out of bounds"),
    ("fs", "so", "known", "subscript out of bounds"),
    ("fs", "sf", "known", "no free coefs"),
    ("sz", "sf", "known", "no free coefs"),
])
def test_matrix_soap_factor_source_rejections(mode, basis, boundary, source_error):
    p = case(mode, basis, boundary=boundary)
    ref = reference(p)
    assert not ref["accepted"]
    assert source_error in ref["error"]
    # These are source-rejected compositions, not numerical parity fallbacks.
    model = model_for(p, {"sp": [.3]})
    with pytest.raises((ValueError, NotImplementedError)):
        model.fit(data=frame(p["data"], p["levels"]))


@pytest.mark.parametrize("mode,basis,boundary", [("fs", "sf", "mixed"), ("sz", "sf", "mixed"), ("sz", "sw", "free"), ("sz", "so", "known")])
def test_matrix_soap_factor_new_width_point_sum_and_persistence(mode, basis, boundary, tmp_path):
    p = case(mode, basis, "REML", "varying", True, boundary)
    ref = reference(p)
    model = model_for(p, ref).fit(data=frame(p["data"], p["levels"]), sample_weight=p["data"]["w"])
    # Predictions can use a different number of paired locations per observation.
    new = frame(p["newdata"], p["levels"])
    for name in ("X", "Y", "L"):
        new[name] = [np.asarray(v)[:2] for v in new[name]]
    matrix = model.lpmatrix(new)
    summed = np.zeros_like(matrix)
    for column in range(2):
        point = new.copy()
        for name in ("X", "Y"):
            point[name] = [np.asarray(v)[column] for v in new[name]]
        point["L"] = 1.
        weight = np.array([np.asarray(v)[column] for v in new.L])
        summed += weight[:, None] * model.lpmatrix(point)
    # The model intercept is not a linear functional term.
    summed[:, 0] = 1.
    np.testing.assert_allclose(matrix, summed, rtol=2e-7, atol=2e-7)
    restored = GAM.load_model(model.save_model(tmp_path / "matrix-soap-factor.pkl"))
    np.testing.assert_allclose(restored.predict(new, return_se=True), model.predict(new, return_se=True), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(restored.lpmatrix(new), matrix, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape,accepted", [("factor_repeated", True), ("factor_varying", True), ("character", False), ("numeric", False)])
def test_source_matrix_factor_class_is_distinct_from_classless_matrices(shape, accepted):
    # Explicit categorical rows now represent the accepted factor+dim object.
    # Classless arrays must retain the separate source classification here;
    # they are not silently reinterpreted as factors.
    p = case(factor_shape=shape)
    ref = reference(p, PROBE.with_name("mgcv_soap_matrix_factor_shapes.R"))
    assert ref["accepted"] is accepted
    if not accepted:
        assert "names" in ref["error"] if shape == "character" else "bivariate" in ref["error"]
