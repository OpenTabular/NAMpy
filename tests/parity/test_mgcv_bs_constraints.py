"""B-spline explicit constraints: source operators and fitted invariants."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.compiler.construct import smooth_con
from nampy.gam.fit.selection.criteria import criterion_value
from nampy.gam.model_state import _design_matrix, _edf_total, _penalty_blocks_seq
from nampy.gam.specs import TermSpec, build_smooth_spec
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_bs_constraints"
PROBE = Path(__file__).with_name("probes") / "mgcv_bs_constraints.R"
INPUTS = json.loads((ROOT / "inputs.json").read_text())
KEY = reference_key(
    "bs-explicit-constraints",
    {"probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(), "inputs": INPUTS},
)


@pytest.fixture(scope="module")
def references():
    expected = load_reference("mgcv_bs_constraints", KEY)
    if expected is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            run = subprocess.run(
                ["Rscript", str(PROBE), str(ROOT / "inputs.json"), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            expected = json.loads(output.read_text())
        save_reference("mgcv_bs_constraints", KEY, expected)
    assert expected["version"] == "1.9.4"
    return expected["cases"]


def _m(name):
    return [3, 2] if name == "single" else [3, 2, 1]


def _constraint(name):
    return (
        None
        if name == "ordinary"
        else np.ones((1, 8))
        if name == "one"
        else np.vstack((np.ones(8), np.arange(8)))
        if name == "two"
        else np.empty((0, 8))
    )


def _spec(mname, select=False):
    return TermSpec(
        kind="smooth",
        features=("x",),
        label="s(x)",
        smooth_spec=build_smooth_spec(
            special="s", bs="bs", k=8, m=_m(mname), select=select
        ),
    )


@pytest.mark.parametrize("mname", ["single", "multiple"])
@pytest.mark.parametrize("cname", ["ordinary", "one", "two", "empty"])
@pytest.mark.parametrize("select", [False, True])
def test_bs_raw_absorbed_and_analytic_derivative_source_operators(
    references, mname, cname, select
):
    ref = references[f"{mname}/{cname}/{str(select).upper()}"]
    X = np.asarray(INPUTS["data"]["x"])[:, None]
    C = _constraint(cname)
    raw = smooth_con(_spec(mname, select), X, ["x"], absorb_cons=False, constraints=C)
    con = smooth_con(_spec(mname, select), X, ["x"], constraints=C)
    np.testing.assert_allclose(
        raw.compiled_term.basis_train, ref["raw_X"], atol=2e-9, rtol=0
    )
    actual_C = raw.fit_constraint_operator
    if cname == "empty":
        assert np.asarray(actual_C).shape == (0, 8)
    else:
        np.testing.assert_allclose(actual_C, ref["C"], atol=2e-9, rtol=0)
    A = np.linalg.lstsq(
        con.compiled_term.basis_train, np.asarray(ref["X"]), rcond=None
    )[0]
    np.testing.assert_allclose(
        con.compiled_term.basis_train @ A, ref["X"], atol=2e-9, rtol=0
    )
    for block, S in zip(raw.penalty_specs, ref["raw_S"], strict=True):
        np.testing.assert_allclose(block.matrix, S, atol=2e-9, rtol=0)
    for block, S in zip(con.penalty_specs, ref["S"], strict=True):
        np.testing.assert_allclose(A.T @ block.matrix @ A, S, atol=2e-9, rtol=0)
    np.testing.assert_allclose(
        con.compiled_term.predict_matrix(pd.DataFrame(INPUTS["newdata"]).to_numpy())
        @ A,
        ref["P"],
        atol=2e-9,
        rtol=0,
    )
    for order, D in enumerate(ref["derivatives"], 1):
        np.testing.assert_allclose(
            con.compiled_term.derivative_matrix(
                pd.DataFrame(INPUTS["derivative_data"]).to_numpy(), order=order
            )
            @ A,
            D,
            atol=2e-8,
            rtol=0,
        )
    T = con.compiled_term.predict_coefficient_map
    if C is not None and T is not None:
        np.testing.assert_allclose(C @ T, 0, atol=2e-12, rtol=0)


@pytest.mark.parametrize("mname", ["single", "multiple"])
@pytest.mark.parametrize("cname", ["ordinary", "one", "two", "empty"])
@pytest.mark.parametrize("select", [False, True])
@pytest.mark.parametrize("method", ["fixed", "REML", "GCV"])
def test_bs_constraint_fits_prediction_covariance_and_penalty_transport(
    references, mname, cname, select, method, tmp_path
):
    ref = references[f"{mname}/{cname}/{str(select).upper()}"]
    expected = ref["fits"][method]
    control = {}
    if (mname, cname, select, method) == ("multiple", "empty", True, "GCV"):
        # Default magic stopping differs in this flat, redundant selection
        # direction. Keep that source endpoint below; automatic acceptance uses
        # explicitly matched tighter source/Python controls, not looser checks.
        expected = ref["fits"]["GCV_matched_controls"]
        control = {"mgcv.tol": 1e-12}
    _check_fit_reference(ref, mname, cname, select, method, expected, tmp_path, control)


def _check_fit_reference(
    ref, mname, cname, select, method, expected, tmp_path, control
):
    data, newdata = pd.DataFrame(INPUTS["data"]), pd.DataFrame(INPUTS["newdata"])
    C = _constraint(cname)
    model = GAM(
        formula=f"y~s(x,bs='bs',k=8,m={_m(mname)!r})",
        constraints=None if C is None else {1: C},
        select=select,
        control=control,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=np.atleast_1d(expected["sp"]) if method == "fixed" else None,
    ).fit(data=data)
    pred, se = model.predict(newdata, return_se=True)
    np.testing.assert_allclose(pred, expected["fit"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(se, expected["se"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        model.predict(), expected["train_fit"], atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(_edf_total(model), expected["edf"], atol=2e-5, rtol=2e-5)
    P = model.lpmatrix(newdata)
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, expected["covariance"], atol=2e-5, rtol=2e-5
    )
    X = model.lpmatrix(data)
    A = np.linalg.lstsq(X, np.asarray(expected["X"]), rcond=None)[0]
    np.testing.assert_allclose(X @ A, expected["X"], atol=2e-9, rtol=0)
    np.testing.assert_allclose(P @ A, expected["P"], atol=2e-9, rtol=0)
    B = np.asarray(_design_matrix(model))
    T = np.linalg.lstsq(B, np.asarray(ref["X"]), rcond=None)[0]
    np.testing.assert_allclose(B @ T, ref["X"], atol=2e-9, rtol=0)
    for block, S in zip(_penalty_blocks_seq(model), ref["S"], strict=True):
        np.testing.assert_allclose(T.T @ block.matrix @ T, S, atol=2e-9, rtol=0)
    restored = GAM.load_model(model.save_model(tmp_path / "bs.nampy"))
    np.testing.assert_array_equal(restored.predict(newdata), pred)
    return model


def test_bs_empty_selected_multi_gcv_default_source_endpoint(references, tmp_path):
    """Keep default source endpoint acceptance separate from automatic selection."""
    ref = references["multiple/empty/TRUE"]
    model = _check_fit_reference(
        ref, "multiple", "empty", True, "fixed", ref["fits"]["GCV"], tmp_path, {}
    )
    score = criterion_value(
        model,
        np.asarray(INPUTS["data"]["y"]),
        np.log(model.smoothing_params),
        method="gcv",
    )
    np.testing.assert_allclose(score, ref["fits"]["GCV"]["score"], atol=2e-12, rtol=0)


@pytest.mark.parametrize(
    "C", [np.ones((2, 8)), np.ones((1, 7)), np.full((1, 8), np.nan), np.eye(8)]
)
def test_bs_invalid_constraints(C):
    with pytest.raises(ValueError, match="Constraint"):
        smooth_con(
            _spec("single"),
            np.asarray(INPUTS["data"]["x"])[:, None],
            ["x"],
            constraints=C,
        )


@pytest.mark.parametrize("option", ["by=z", "pc=0.3", "id=1"])
def test_bs_constraint_compositions_remain_guarded(option):
    data = pd.DataFrame(INPUTS["data"]).assign(z=0.5)
    with pytest.raises(NotImplementedError, match="without by, pc, or linked id"):
        GAM(formula=f"y~s(x,bs='bs',k=8,{option})", constraints={1: np.ones(8)}).fit(
            data=data
        )


@pytest.mark.parametrize("mname", ["single", "multiple"])
@pytest.mark.parametrize("select", [False, True])
def test_bs_independent_constraint_rows_preserve_all_transported_operators(
    mname, select
):
    X = np.asarray(INPUTS["data"]["x"])[:, None]
    C = _constraint("two")
    first = smooth_con(_spec(mname, select), X, ["x"], constraints=C)
    second = smooth_con(
        _spec(mname, select),
        X,
        ["x"],
        constraints=np.array([[2.0, 1.0], [1.0, 3.0]]) @ C,
    )
    B0, B1 = first.compiled_term.basis_train, second.compiled_term.basis_train
    transport = np.linalg.lstsq(B0, B1, rcond=None)[0]
    np.testing.assert_allclose(B0 @ transport, B1, atol=2e-12, rtol=0)
    newdata = pd.DataFrame(INPUTS["newdata"]).to_numpy()
    np.testing.assert_allclose(
        first.compiled_term.predict_matrix(newdata) @ transport,
        second.compiled_term.predict_matrix(newdata),
        atol=2e-12,
        rtol=0,
    )
    for before, after in zip(first.penalty_specs, second.penalty_specs, strict=True):
        np.testing.assert_allclose(
            transport.T @ before.matrix @ transport, after.matrix, atol=2e-9, rtol=0
        )
    points = pd.DataFrame(INPUTS["derivative_data"]).to_numpy()
    for order in (1, 2, 3):
        np.testing.assert_allclose(
            first.compiled_term.derivative_matrix(points, order=order) @ transport,
            second.compiled_term.derivative_matrix(points, order=order),
            atol=2e-8,
            rtol=0,
        )
