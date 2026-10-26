"""Audit t2 B-spline null metrics with complete coefficient transport.

The guarded constructor is bypassed only inside these audit tests. Train and
newdata bases, every penalty, fitted coefficients and covariance are all
transported together; a raw eigenvector difference alone is not evidence.
"""

import importlib

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import criterion_value
from nampy.gam.model_state import _penalty_blocks_seq
from nampy.gam.splines.basis import natparam
from tests.smooths import test_mgcv_tensor_linear_functionals as matrix_references
from tests.smooths.test_mgcv_tensor_linear_functionals import _frame

t2 = importlib.import_module("nampy.gam.smooths.tensor.t2")


@pytest.fixture(scope="module")
def references():
    return matrix_references.references.__wrapped__()


def _bypass_matrix_guard(monkeypatch):
    prepare = t2.prepare_tensor_functional_fit
    finish = t2.finish_tensor_functional_fit

    def prepare_for_audit(term, X, names):
        X = prepare(term, X, names)
        term._audit_functional_shape = term._tensor_functional_shape
        term._tensor_functional_shape = None
        return X

    def restore_for_audit(term):
        term._tensor_functional_shape = term._audit_functional_shape
        finish(term)

    monkeypatch.setattr(t2, "prepare_tensor_functional_fit", prepare_for_audit)
    monkeypatch.setattr(t2, "finish_tensor_functional_fit", restore_for_audit)


def _rotate_null_eigenvectors(monkeypatch):
    original = natparam._r_symmetric_eigh_descending

    def rotated(matrix):
        values, vectors = original(matrix)
        if matrix.shape == (5, 5):
            if abs(values[-2]) < 1e-12 * values[0]:
                angle = 0.4
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                vectors[:, -2:] = vectors[:, -2:] @ rotation
            else:
                vectors[:, -1] *= -1
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(5), atol=2e-14)
            np.testing.assert_allclose(
                (vectors * values) @ vectors.T, matrix, atol=2e-14
            )
        return values, vectors

    monkeypatch.setattr(natparam, "_r_symmetric_eigh_descending", rotated)


def _fit(reference, *, full=False, method="fixed", ord=None, m=None, kind="weighted"):
    options = f",full={full}"
    if ord is not None:
        options += f",ord={ord}"
    if m is not None:
        options += f",m={m}"
    nsp = 5 if full else 3
    if ord == [2]:
        nsp = 1
    elif m == [[3, 1], [3, 1]]:
        nsp = 3
    elif m == [[3, 0], [3, 0]]:
        nsp = 1
    by = "" if kind == "unweighted" else ",by=L"
    return GAM(
        formula=f'y~t2(X,Z,bs=["bs","bs"],k=[5,5]{by}{options})+offset(o)',
        smoothing_params=np.linspace(0.4, 0.8, nsp) if method == "fixed" else None,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
    ).fit(
        data=_frame(reference["data"]), sample_weight=np.asarray(reference["data"]["w"])
    )


def _full_penalties(model):
    p = len(model.fit_result().coef_full)
    index = model.gam_result_.compiled_model.coef_reduced_to_full_idx
    result = []
    for block in _penalty_blocks_seq(model):
        S = np.zeros((p, p))
        S[np.ix_(index, index)] = block.matrix
        result.append(S)
    return result


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_t2_bs_null_metric_audit_transports_prediction_and_all_penalties(
    references, monkeypatch, full, method
):
    reference = references["t2/bs/fixed/weighted"]
    _bypass_matrix_guard(monkeypatch)
    baseline = _fit(reference, full=full, method=method)
    _rotate_null_eigenvectors(monkeypatch)
    alternative = _fit(reference, full=full, method=method)
    data, newdata = _frame(reference["data"]), _frame(reference["newdata"])
    X0, X1 = baseline.lpmatrix(data), alternative.lpmatrix(data)
    A = np.linalg.lstsq(X0, X1, rcond=None)[0]
    np.testing.assert_allclose(X0 @ A, X1, atol=2e-11, rtol=0)
    P0, P1 = baseline.lpmatrix(newdata), alternative.lpmatrix(newdata)
    np.testing.assert_allclose(P0 @ A, P1, atol=2e-11, rtol=0)
    S0, S1 = _full_penalties(baseline), _full_penalties(alternative)
    transported = [A.T @ S @ A for S in S0]
    errors = [
        np.max(np.abs(target - current))
        for target, current in zip(transported, S1, strict=True)
    ]
    w, y, offset = [np.asarray(reference["data"][key]) for key in ("w", "y", "o")]
    H = X1.T @ (w[:, None] * X1) + sum(
        lam * S for lam, S in zip(baseline.smoothing_params, transported, strict=True)
    )
    beta = np.linalg.solve(H, X1.T @ (w * (y - offset)))
    np.testing.assert_allclose(
        P1 @ beta + reference["newdata"]["o"],
        baseline.predict(newdata),
        atol=2e-10,
        rtol=0,
    )
    V1 = baseline.fit_result().scale * np.linalg.solve(H, np.eye(len(H)))
    np.testing.assert_allclose(
        P1 @ V1 @ P1.T, P0 @ baseline.vcov() @ P0.T, atol=2e-10, rtol=0
    )
    score0 = criterion_value(
        baseline, y, np.log(baseline.smoothing_params), method="reml"
    )
    score1 = criterion_value(
        alternative, y, np.log(alternative.smoothing_params), method="reml"
    )
    prediction_error = np.max(
        np.abs(baseline.predict(newdata) - alternative.predict(newdata))
    )
    covariance_error = np.max(
        np.abs(P0 @ baseline.vcov() @ P0.T - P1 @ alternative.vcov() @ P1.T)
    )
    assert max(errors) > 1e-4
    # The complete congruence transport above reproduces the old operator.
    # Reconstructing source-style identity blocks changes that operator.
    if method == "fixed" or not full:
        assert prediction_error > 1e-3
        assert covariance_error > 1e-5
    assert abs(score0 - score1) > 1e-2


@pytest.mark.parametrize("route", ["range_only", "one_null", "no_null"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_t2_bs_supported_routes_preserve_all_fitted_invariants(
    references, monkeypatch, route, full, method
):
    reference = references["t2/bs/fixed/weighted"]
    options = (
        {"ord": [2]}
        if route == "range_only"
        else {"m": [[3, 1], [3, 1]]}
        if route == "one_null"
        else {"m": [[3, 0], [3, 0]]}
    )
    baseline = _fit(reference, full=full, method=method, **options)
    _rotate_null_eigenvectors(monkeypatch)
    alternative = _fit(reference, full=full, method=method, **options)
    data, newdata = _frame(reference["data"]), _frame(reference["newdata"])
    X0, X1 = baseline.lpmatrix(data), alternative.lpmatrix(data)
    A = np.linalg.lstsq(X0, X1, rcond=None)[0]
    np.testing.assert_allclose(X0 @ A, X1, atol=2e-11, rtol=0)
    P0, P1 = baseline.lpmatrix(newdata), alternative.lpmatrix(newdata)
    np.testing.assert_allclose(P0 @ A, P1, atol=2e-11, rtol=0)
    for S0, S1 in zip(
        _full_penalties(baseline), _full_penalties(alternative), strict=True
    ):
        np.testing.assert_allclose(A.T @ S0 @ A, S1, atol=2e-9, rtol=2e-12)
    np.testing.assert_allclose(
        baseline.predict(newdata), alternative.predict(newdata), atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        P0 @ baseline.vcov() @ P0.T, P1 @ alternative.vcov() @ P1.T, atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        baseline.fit_result().edf_total,
        alternative.fit_result().edf_total,
        atol=2e-10,
        rtol=0,
    )
    y = np.asarray(reference["data"]["y"])
    score0 = criterion_value(
        baseline, y, np.log(baseline.smoothing_params), method="reml"
    )
    score1 = criterion_value(
        alternative, y, np.log(alternative.smoothing_params), method="reml"
    )
    np.testing.assert_allclose(score0, score1, atol=2e-10, rtol=0)


@pytest.mark.parametrize("route", ["range_only", "one_null", "no_null"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("kind", ["unweighted", "weighted", "constant", "scalar"])
def test_t2_bs_supported_matrix_routes_match_actual_mgcv(
    references, route, full, method, kind
):
    import hashlib
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    from tests.reference_fixtures import load_reference, reference_key, save_reference

    ref = references[f"t2/bs/fixed/{kind if kind != 'scalar' else 'unweighted'}"]
    if kind == "scalar":
        ref = {
            section: {
                name: (
                    np.asarray(value)[:, 0].tolist() if name in ("X", "Z") else value
                )
                for name, value in ref[section].items()
            }
            for section in ("data", "newdata")
        }
        kind = "unweighted"
    probe = Path(__file__).with_name("probes") / "mgcv_t2_bs_matrix_invariants.R"
    payload = {
        "route": route,
        "full": full,
        "method": method,
        "kind": kind,
        "data": ref["data"],
        "newdata": ref["newdata"],
    }
    key = reference_key(
        "t2-bs-matrix-invariants",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        },
    )
    expected = load_reference("mgcv_t2_bs_matrix_invariants", key)
    if expected is None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = [
                Path(directory) / name for name in ("inputs.json", "result.json")
            ]
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(
                ["Rscript", str(probe), str(inputs), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            expected = json.loads(output.read_text())
        save_reference("mgcv_t2_bs_matrix_invariants", key, expected)
    options = (
        {"ord": [2]}
        if route == "range_only"
        else {"m": [[3, 1], [3, 1]]}
        if route == "one_null"
        else {"m": [[3, 0], [3, 0]]}
    )
    model = _fit(ref, full=full, method=method, kind=kind, **options)
    data, newdata = _frame(ref["data"]), _frame(ref["newdata"])
    prediction, se = model.predict(newdata, return_se=True)
    P = model.lpmatrix(newdata)
    target_penalties = expected["penalties"]
    if isinstance(target_penalties, dict):
        term = next(
            term
            for term in model.gam_result_.compiled_model.compiled_terms
            if term.term_type == "tensor_t2"
        )
        target_penalties = [
            target_penalties[label]
            for label in term.predict_fn.__self__._penalty_labels
        ]
    np.testing.assert_allclose(prediction, expected["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, expected["se"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, expected["covariance"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        model.predict(), expected["train_fit"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        model.fit_result().edf_total, expected["edf"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(
        model.smoothing_params, np.atleast_1d(expected["sp"]), atol=3e-4, rtol=3e-4
    )
    X = model.lpmatrix(data)
    A = np.linalg.lstsq(X, np.asarray(expected["train_X"]), rcond=None)[0]
    np.testing.assert_allclose(X @ A, expected["train_X"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(P @ A, expected["new_X"], atol=2e-7, rtol=0)
    # smoothCon's fitted C and PredictMat's Cp need not use the same
    # coefficient coordinates. Align each S in the FIT basis; aligning it
    # against model.matrix/prediction columns mixes these parameterizations.
    B = np.asarray(model.gam_result_.compiled_model.design_matrix)
    target_B = np.asarray(expected["fit_X"])
    Afit = np.linalg.lstsq(B, target_B, rcond=None)[0]
    np.testing.assert_allclose(B @ Afit, target_B, atol=2e-7, rtol=0)
    for S, target in zip(_penalty_blocks_seq(model), target_penalties, strict=True):
        np.testing.assert_allclose(
            Afit.T @ S.matrix @ Afit, target, atol=2e-6, rtol=2e-6
        )
    score = criterion_value(
        model,
        np.asarray(ref["data"]["y"]),
        np.log(model.smoothing_params),
        method="reml",
    )
    np.testing.assert_allclose(score, expected["score"], atol=3e-5, rtol=3e-5)
