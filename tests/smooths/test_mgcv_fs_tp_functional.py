"""Source-backed matrix FS/TP invariants and one-null-direction acceptance."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _edf_total, _penalty_blocks_seq, _term_blocks_seq
from nampy.gam.smooths.categorical.fs import FSmoothInteractionTerm
from nampy.gam.splines.basis.natparam import nat_param_type1
from tests.mgcv_invariant_policy import stable_column_space_projector
from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.smooths.test_mgcv_fs_functional import KEY as INPUT_KEY
from tests.smooths.test_mgcv_fs_functional import _frame

PROBE = Path(__file__).with_name("probes") / "mgcv_fs_tp_functional.R"


def reference(kind="unit", method="fixed", repeated=False, m=1):
    key = reference_key("fs-tp-functional-operators-v1", {
        "input_fixture": str(INPUT_KEY), "kind": kind, "method": method,
        "repeated": repeated, "m": m, "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
    })
    result = load_reference("mgcv_fs_tp_functional", key)
    if result is None:
        inputs = load_reference("mgcv_fs_functional", INPUT_KEY)["cases"]
        ref = inputs[f"{'ps' if repeated else 'tp'}/{kind}/{method}/1" + ("/repeated" if repeated else "")]
        payload = {key: ref[key] for key in ("data", "newdata", "levels")}
        payload.update(kind=kind, method=method, m=m)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text(json.dumps(payload))
            run = subprocess.run(["Rscript", str(PROBE), str(source), str(output)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_fs_tp_functional", key, result)
    assert result["version"] == "1.9.4"
    return result


def fit(ref, kind="unit", method="fixed", m=1):
    return GAM(formula=f"response~s(X,f,bs='fs',xt={{'bs':'tp'}},k=6,m={m}{',by=L' if kind != 'unit' else ''})",
               smoothing_method=method, optimize_smoothing=method != "fixed",
               smoothing_params=ref["sp"] if method == "fixed" else None,
               control={"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}}).fit(data=_frame(ref["data"], ref["levels"]))


@pytest.mark.parametrize("kind", ["unit", "varying", "constant"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("repeated", [False, True])
def test_fs_tp_single_null_functional_fit_penalties_and_uncertainty(kind, method, repeated):
    ref = reference(kind, method, repeated)
    assert ref["null_dim"] == 1
    model = fit(ref, kind, method)
    new = _frame(ref["newdata"], ref["levels"])
    predicted, se = model.predict(new, return_se=True)
    for actual, expected in ((predicted, ref["fit"]), (se, ref["se"]),
                             (model.predict(), ref["fitted"]), (_edf_total(model), ref["edf"])):
        np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=4e-5)
    P = model.predict(new, type="lpmatrix")
    np.testing.assert_allclose(P @ model.vcov() @ P.T, ref["covariance"], atol=4e-5, rtol=4e-5)
    X, R = model.predict(type="lpmatrix")[:, 1:], np.asarray(ref["X"])[:, 1:]
    transport = np.linalg.lstsq(X, R, rcond=None)[0]
    np.testing.assert_allclose(stable_column_space_projector(X), stable_column_space_projector(R), atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(X @ transport, R, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(P[:, 1:] @ transport, np.asarray(ref["Xp"])[:, 1:], atol=2e-7, rtol=2e-7)
    for penalty, expected in zip(_penalty_blocks_seq(model), ref["S"], strict=True):
        np.testing.assert_allclose(transport.T @ penalty.matrix @ transport, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-3)
    runtime = _term_blocks_seq(model)[0]
    assert runtime.basis_name == "fs"
    if kind == "varying":
        np.testing.assert_allclose(P[1, 1:], 0, atol=1e-12)


@pytest.mark.parametrize("kind", ["unit", "varying", "constant"])
def test_fs_tp_single_null_prediction_width_and_persistence(kind, tmp_path):
    ref = reference(kind)
    model = fit(ref, kind)
    new = _frame(ref["newdata"], ref["levels"])
    new["X"] = [np.asarray(row)[:2] for row in new.X]
    if kind != "unit":
        new["L"] = [np.asarray(row)[:2] for row in new.L]
    matrix = model.predict(new, type="lpmatrix")
    point_parts = []
    for column in range(2):
        point = new.copy()
        point["X"] = [row[column] for row in new.X]
        if kind != "unit":
            point["L"] = [row[column] for row in new.L]
        point_parts.append(model.predict(point, type="lpmatrix")[:, 1:])
    np.testing.assert_allclose(matrix[:, 1:], sum(point_parts), atol=1e-12, rtol=1e-12)
    restored = GAM.load_model(model.save_model(tmp_path / "fs-tp-m1.pkl"))
    np.testing.assert_array_equal(restored.predict(new), model.predict(new))
    np.testing.assert_array_equal(restored.predict(new, return_se=True), model.predict(new, return_se=True))


@pytest.mark.parametrize("m", [1, 2])
def test_fs_tp_source_input_injection_separates_range_and_null_operators(m):
    ref = reference(m=m)
    X, S = np.asarray(ref["nat_X"]), np.asarray(ref["nat_S"])
    rank = int(ref["nat_rank"])
    actual = nat_param_type1(X, S, rank=rank)
    expected = ref["nat"]
    A, B = np.asarray(actual["X"]), np.asarray(expected["X"])
    # Hold source constructor inputs fixed: repeated eigenspaces can rotate,
    # but these function spaces and original-penalty quadratic forms cannot.
    for columns in (slice(None, rank), slice(rank, None)):
        np.testing.assert_allclose(stable_column_space_projector(A[:, columns]),
                                   stable_column_space_projector(B[:, columns]), atol=2e-8, rtol=2e-8)
    transport = np.linalg.lstsq(A, B, rcond=None)[0]
    np.testing.assert_allclose(A @ transport, B, atol=2e-8, rtol=2e-8)
    expected_D = np.diag(np.r_[expected["D"], np.zeros(B.shape[1] - rank)])
    actual_D = actual["P"].T @ S @ actual["P"]
    np.testing.assert_allclose(transport.T @ actual_D @ transport, expected_D, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(actual["D"], expected["D"], atol=2e-8, rtol=2e-8)
    for columns in (slice(None, rank), slice(rank, None)):
        np.testing.assert_allclose(np.mean(A[:, columns]**2), np.mean(B[:, columns]**2), atol=2e-12, rtol=2e-12)


def _penalized_function_operator(design, penalties, sp):
    precision = design.T @ design
    for matrix, weight in zip(penalties, sp, strict=True):
        precision[1:, 1:] += weight * matrix
    return design @ np.linalg.solve(precision, design.T)


def test_fs_tp_multinull_guard_preserves_source_orientation_boundary(monkeypatch):
    ref = reference(m=2)
    assert ref["null_dim"] == 2
    with pytest.raises(NotImplementedError, match="natural-null-space"):
        fit(ref, m=2)

    # Diagnostic bypass only. The test establishes why this unsupported route
    # must not be declared portable by comparing basis spans alone.
    monkeypatch.setattr(FSmoothInteractionTerm, "_validate_matrix_tp_null_space", lambda self, null_dim: None)
    model = fit(ref, m=2)
    local = model.predict(type="lpmatrix")[:, 1:]
    source = np.asarray(ref["X"])[:, 1:]
    transport = np.linalg.lstsq(local, source, rcond=None)[0]
    np.testing.assert_allclose(local @ transport, source, atol=2e-7, rtol=2e-7)
    local_penalties = _penalty_blocks_seq(model)
    local_scales = np.array([p.metadata["penalty_rescale_factor"] for p in local_penalties])
    source_scales = np.asarray(ref["S_scale"])
    actual = [transport.T @ p.matrix @ transport for p in local_penalties]
    expected = [np.asarray(S) for S in ref["S"]]
    # Unscaled source range and pooled null penalties agree. The common
    # scaled-SP ratio is owned by smoothCon's absolute row-sum normalization,
    # not nat.param's rotation-invariant Frobenius normalization.
    np.testing.assert_allclose(actual[0] * local_scales[0], expected[0] * source_scales[0], atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(sum(A * s for A, s in zip(actual[1:], local_scales[1:], strict=True)),
                               sum(B * s for B, s in zip(expected[1:], source_scales[1:], strict=True)), atol=3e-6, rtol=3e-6)
    ratio = float(np.vdot(actual[0], expected[0]) / np.vdot(expected[0], expected[0]))
    np.testing.assert_allclose(source_scales / local_scales, ratio, atol=2e-8, rtol=2e-8)

    # Inject a deterministic allowed null-eigenspace rotation into source
    # coordinates, then repeat the literal FS/smoothCon penalty construction.
    # A valid coordinate transport transforms each penalty; reassigning the
    # same diagonal rank-one penalties after rotation changes the model when
    # their SPs differ. Equal-null SPs avoid that anisotropy only before the
    # subsequent orientation-dependent smoothCon rescaling.
    block = np.eye(6)
    angle = .37
    block[-2:, -2:] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotation = np.kron(np.eye(len(ref["levels"])), block)
    full_rotation = np.eye(rotation.shape[0] + 1)
    full_rotation[1:, 1:] = rotation
    design = np.asarray(ref["X"])
    rotated_design = design @ full_rotation
    raw_penalties = [np.asarray(S) for S in ref["raw_S"]]
    transported_penalties = [rotation.T @ S @ rotation for S in raw_penalties]
    anisotropic_sp = [.3, .1, .8]
    original = _penalized_function_operator(design, raw_penalties, anisotropic_sp)
    valid = _penalized_function_operator(rotated_design, transported_penalties, anisotropic_sp)
    reset = _penalized_function_operator(rotated_design, raw_penalties, anisotropic_sp)
    np.testing.assert_allclose(valid, original, atol=2e-10, rtol=2e-10)
    assert np.max(np.abs(reset - original)) > 1e-4
    equal_sp = [.3, .3, .3]
    np.testing.assert_allclose(_penalized_function_operator(rotated_design, raw_penalties, equal_sp),
                               _penalized_function_operator(design, raw_penalties, equal_sp), atol=2e-10, rtol=2e-10)
    raw_X = np.asarray(ref["raw_X"])
    maXX = np.max(np.sum(np.abs(raw_X), axis=1))**2
    rotated_maXX = np.max(np.sum(np.abs(raw_X @ rotation), axis=1))**2
    np.testing.assert_allclose([np.linalg.norm(S, ord=1) / maXX for S in raw_penalties], source_scales, atol=2e-12, rtol=2e-12)
    rescales = [S / (np.linalg.norm(S, ord=1) / rotated_maXX) for S in raw_penalties]
    scaled_original = _penalized_function_operator(design, expected, equal_sp)
    scaled_rotated = _penalized_function_operator(rotated_design, rescales, equal_sp)
    assert np.max(np.abs(scaled_rotated - scaled_original)) > 1e-4


@pytest.mark.parametrize("mode,base,m", [("fs", "tp", 1), ("sz", "tp", 3), ("sz", "ts", 1)])
def test_scalar_factor_smooth_forwards_nondefault_tp_derivative_order(mode, base, m):
    probe = PROBE.with_name("mgcv_factor_tp_orders.R")
    key = reference_key("factor-thinplate-derivative-order-v1", {
        "input_fixture": str(INPUT_KEY), "mode": mode, "base": base, "m": m,
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    })
    ref = load_reference("mgcv_fs_tp_functional", key)
    if ref is None:
        data = load_reference("mgcv_fs_functional", INPUT_KEY)["cases"]["tp/unit/fixed/1"]
        payload = {"data": data["data"], "levels": data["levels"], "mode": mode, "base": base, "m": m}
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = Path(directory) / "input.json", Path(directory) / "output.json"
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(["Rscript", str(probe), str(inputs), str(output)], capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            ref = json.loads(output.read_text())
        save_reference("mgcv_fs_tp_functional", key, ref)
    formula = f"response~s(X,f,bs='{mode}',xt={{'bs':'{base}'}},k=6,m={m})"
    model = GAM(formula=formula, smoothing_method="fixed", optimize_smoothing=False,
                smoothing_params=[.3] * len(ref["S"])).fit(data=_frame(ref["data"], ref["levels"]))
    X = _term_blocks_seq(model)[0].basis_train
    source = np.asarray(ref["X"])
    transport = np.linalg.lstsq(X, source, rcond=None)[0]
    np.testing.assert_allclose(X @ transport, source, atol=2e-7, rtol=2e-7)
    penalties = _penalty_blocks_seq(model)
    for actual, expected in zip(penalties, ref["S"], strict=True):
        np.testing.assert_allclose(transport.T @ actual.matrix @ transport, expected, atol=3e-6, rtol=3e-6)
    null_dim = X.shape[1] - np.linalg.matrix_rank(sum(p.matrix for p in penalties))
    assert null_dim == ref["null_dim"]
