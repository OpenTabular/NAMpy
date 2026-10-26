"""Static mgcv matrix-functional random-effect and MRF parity."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_categorical_functionals.R"
KEY = reference_key(
    "categorical-matrix-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)
KINDS = [
    "re_numeric",
    "re_numeric2",
    "re_factor",
    "re_mixed",
    "re_factor2",
    "mrf_full",
    "mrf_low",
]
NB = {
    "a": ["b"],
    "b": ["a", "c"],
    "c": ["b", "d"],
    "d": ["c", "e"],
    "e": ["d", "f"],
    "f": ["e"],
}


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_categorical_functionals", KEY)
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
        save_reference("mgcv_categorical_functionals", KEY, result)
    return result["cases"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("weighting", ["weighted", "unweighted", "constant"])
def test_upstream_categorical_functional_invariants(
    references, kind, method, weighting
):
    ref = references[f"{kind}/{method}/{weighting}"]
    P = np.asarray(ref["Xp"])
    np.testing.assert_allclose(P, ref["point_aggregate"], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ref["X"]) @ ref["coef"] + ref["data"]["o"], ref["fitted"], atol=1e-12
    )
    np.testing.assert_allclose(
        P @ ref["coef"] + ref["newdata"]["o"], ref["fit"], atol=1e-12
    )
    np.testing.assert_allclose(
        P @ ref["Vp"] @ P.T, ref["covariance_kernel"], atol=1e-12
    )
    assert ref["summation_error"] < 1e-12
    assert ref["converged"]


def _frame(data):
    return pd.DataFrame({key: list(value) for key, value in data.items()})


def _formula(kind, weighting):
    terms = {
        "re_numeric": "X",
        "re_numeric2": "X,Z",
        "re_factor": "F",
        "re_mixed": "X,F",
        "re_factor2": "F,G",
    }.get(kind, "F")
    options = (
        f"bs='mrf',k={4 if kind == 'mrf_low' else -1},xt={{'nb':{NB!r}}}"
        if kind.startswith("mrf")
        else "bs='re'"
    )
    return f"y~s({terms},{options}{',by=L' if weighting != 'unweighted' else ''})+offset(o)"


def _fit(ref, kind, method, weighting):
    from nampy.gam import GAM

    return GAM(
        formula=_formula(kind, weighting),
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=[0.4] if method == "fixed" else None,
    ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("weighting", ["weighted", "unweighted", "constant"])
def test_runtime_categorical_functionals_match_mgcv(
    references, kind, method, weighting
):
    from nampy.gam.model_state import _edf_total, _penalty_blocks_seq

    ref = references[f"{kind}/{method}/{weighting}"]
    model = _fit(ref, kind, method, weighting)
    data = _frame(ref["newdata"])
    fit, se = model.predict(data, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    P = model.predict(data, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance_kernel"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)
    local = model.predict(type="lpmatrix")[:, 1:]
    align = np.linalg.lstsq(local, np.asarray(ref["X"])[:, 1:], rcond=None)[0]
    np.testing.assert_allclose(local @ align, np.asarray(ref["X"])[:, 1:], atol=2e-8)
    penalties = list(ref["S"].values()) if isinstance(ref["S"], dict) else ref["S"]
    np.testing.assert_allclose(
        align.T @ _penalty_blocks_seq(model)[0].matrix @ align,
        penalties[0],
        atol=3e-6,
        rtol=3e-6,
    )
    nd = ref["newdata"]
    point = {
        key: np.asarray(value).reshape(-1) for key, value in nd.items() if key != "o"
    }
    point["L"] = np.ones(len(point["X"]))
    point["o"] = np.zeros(len(point["X"]))
    Pb = model.predict(_frame(point), type="lpmatrix")[:, 1:]
    weights = (
        np.asarray(nd["L"]) if weighting != "unweighted" else np.ones_like(nd["L"])
    )
    np.testing.assert_allclose(
        P[:, 1:],
        np.einsum("nq,nqk->nk", weights, Pb.reshape(*weights.shape, -1)),
        atol=2e-12,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_categorical_functional_persistence_and_width(references, kind, tmp_path):
    from nampy.gam import GAM

    ref = references[f"{kind}/fixed/weighted"]
    model = _fit(ref, kind, "fixed", "weighted")
    nd = {
        key: np.asarray(value)[:, :2] if key != "o" else value
        for key, value in ref["newdata"].items()
    }
    frame = _frame(nd)
    expected = model.predict(frame, return_se=True)
    path = tmp_path / "model.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    for actual, wanted in zip(
        restored.predict(frame, return_se=True), expected, strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-13)
    np.testing.assert_allclose(
        restored.predict(frame, type="lpmatrix")[2, 1:], 0, atol=1e-14
    )


@pytest.mark.parametrize("kind", ["re_mixed", "mrf_full", "mrf_low"])
@pytest.mark.parametrize("bad", ["by_shape", "ragged", "coordinate_shape"])
def test_categorical_matrix_shape_errors(references, kind, bad):
    from nampy.gam import GAM

    data = _frame(references[f"{kind}/fixed/weighted"]["data"])
    if bad == "by_shape":
        data["L"] = np.ones(len(data))
    elif bad == "ragged":
        data.at[0, "F"] = data.at[0, "F"][:2]
    else:
        data["F"] = [row[:2] for row in data.F]
    with pytest.raises(ValueError, match="shape|number"):
        GAM(
            formula=_formula(kind, "weighted"),
            smoothing_method="fixed",
            smoothing_params=[0.4],
        ).fit(data=data)


@pytest.mark.parametrize("kind", ["re_factor", "mrf_full", "mrf_low"])
def test_categorical_matrix_new_level_behavior(references, kind):
    ref = references[f"{kind}/fixed/weighted"]
    model = _fit(ref, kind, "fixed", "weighted")
    data = _frame(ref["newdata"])
    data.at[0, "F"] = ["unknown"] * 3
    if kind == "re_factor":
        np.testing.assert_allclose(
            model.predict(data, type="lpmatrix")[0, 1:], 0, atol=0
        )
    else:
        with pytest.raises(ValueError, match="unknown|unseen|levels"):
            model.predict(data)
