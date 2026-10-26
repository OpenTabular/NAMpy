"""Static mgcv adaptive matrix-functional fit and prediction parity."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_ad_functionals.R"
KEY = reference_key(
    "adaptive-matrix-fit-predict",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_ad_functionals", KEY)
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
        save_reference("mgcv_ad_functionals", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


@pytest.mark.parametrize("basis,dimensions", [("ad", 1), ("ad", 2)])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("weighted", ["weighted", "unweighted", "constant"])
def test_upstream_functional_fit_prediction_invariants(
    references, basis, method, weighted, dimensions
):
    ref = references[f"{basis}/{method}/{weighted}/{dimensions}"]
    X, P = np.asarray(ref["X"]), np.asarray(ref["Xp"])
    beta, covariance = np.asarray(ref["coef"]), np.asarray(ref["Vp"])
    np.testing.assert_allclose(
        X @ beta + ref["data"]["o"], ref["train_fit"], atol=3e-13
    )
    np.testing.assert_allclose(P @ beta + ref["newdata"]["o"], ref["fit"], atol=3e-13)
    kernel = P @ covariance @ P.T
    np.testing.assert_allclose(kernel, ref["covariance_kernel"], atol=3e-13)
    np.testing.assert_allclose(
        np.sqrt(np.maximum(np.diag(kernel), 0)), ref["se"], atol=3e-13
    )
    if dimensions == 2 and weighted != "unweighted":
        # Upstream ad constructs a tensor with by="NA": prediction loses L.
        assert ref["summation_error"] > 0.1
        return
    np.testing.assert_allclose(P, ref["point_aggregate"], atol=3e-13)
    assert ref["summation_error"] < 3e-13
    assert ref["converged"]
    assert np.count_nonzero(np.asarray(ref["data"]["w"]) == 0) == 2
    assert np.all(np.asarray(ref["sp"]) > 0)
    if weighted == "weighted":
        # A zero quadrature row contributes no smooth; the intercept survives.
        np.testing.assert_allclose(P[2, 1:], 0, atol=1e-14)
        assert P[2, 0] == 1
    elif weighted == "unweighted":
        # With constant row sums, the summed smooth is centered over rows.
        np.testing.assert_allclose(X[:, 1:].sum(axis=0), 0, atol=3e-11)


def _frame(values):
    import pandas as pd

    return pd.DataFrame({name: list(value) for name, value in values.items()})


@pytest.mark.parametrize("basis,dimensions", [("ad", 1), ("ad", 2)])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("weighted", ["weighted", "unweighted", "constant"])
def test_runtime_functional_fit_and_prediction_match_mgcv(
    references, basis, method, weighted, dimensions
):
    from nampy.gam import GAM

    ref = references[f"{basis}/{method}/{weighted}/{dimensions}"]
    formula = f"y ~ s({'X,Z' if dimensions == 2 else 'X'},bs='{basis}',k={10 if dimensions == 1 else [5, 5]},m=3{',by=L' if weighted != 'unweighted' else ''}) + offset(o)"
    if dimensions == 2 and weighted != "unweighted":
        with pytest.raises(NotImplementedError, match="fit/predict mismatch"):
            GAM(
                formula=formula, smoothing_method="fixed", smoothing_params=[0.4] * 9
            ).fit(data=_frame(ref["data"]))
        return
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=[0.4] * (3 if dimensions == 1 else 9)
        if method == "fixed"
        else None,
    ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))
    actual, se = model.predict(_frame(ref["newdata"]), return_se=True)
    np.testing.assert_allclose(actual, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    P = model.predict(_frame(ref["newdata"]), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance_kernel"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(model.predict(), ref["train_fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)

    from nampy.gam.model_state import _edf_total, _penalty_blocks_seq

    np.testing.assert_allclose(
        _edf_total(model), np.sum(ref["edf"]), atol=3e-5, rtol=3e-5
    )
    local = model.predict(type="lpmatrix")[:, 1:]
    align = np.linalg.lstsq(local, np.asarray(ref["X"])[:, 1:], rcond=None)[0]
    np.testing.assert_allclose(local @ align, np.asarray(ref["X"])[:, 1:], atol=2e-7)
    expected_penalties = (
        list(ref["S"].values()) if isinstance(ref["S"], dict) else ref["S"]
    )
    assert len(_penalty_blocks_seq(model)) == (3 if dimensions == 1 else 9)
    for block, penalty in zip(
        _penalty_blocks_seq(model), expected_penalties, strict=True
    ):
        np.testing.assert_allclose(
            align.T @ block.matrix @ align, penalty, atol=3e-6, rtol=3e-6
        )
    # Prediction at individual locations reconstructs the matrix functional.
    nd = ref["newdata"]
    point = {name: np.asarray(nd[name]).reshape(-1) for name in ["X", "Z"]}
    point["L"] = np.ones(len(point["X"]))
    point["o"] = np.zeros(len(point["X"]))
    point_basis = model.predict(_frame(point), type="lpmatrix")[:, 1:]
    weights = np.asarray(nd["L"]) if weighted != "unweighted" else np.ones_like(nd["L"])
    summed = np.einsum("nq,nqk->nk", weights, point_basis.reshape(*weights.shape, -1))
    np.testing.assert_allclose(P[:, 1:], summed, atol=2e-12)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_functional_model_persistence_changed_quadrature_width_and_zero_rows(
    references, dimensions, tmp_path
):
    from nampy.gam import GAM

    ref = references[
        f"ad/fixed/{'weighted' if dimensions == 1 else 'unweighted'}/{dimensions}"
    ]
    model = GAM(
        formula="y~s(X,bs='ad',by=L,k=10,m=3)+offset(o)"
        if dimensions == 1
        else "y~s(X,Z,bs='ad',k=[5,5],m=3)+offset(o)",
        smoothing_method="fixed",
        smoothing_params=[0.4] * (3 if dimensions == 1 else 9),
    ).fit(data=_frame(ref["data"]))
    prediction_data = {
        name: np.asarray(values)[:, :2] if name in {"X", "Z", "L"} else values
        for name, values in ref["newdata"].items()
    }
    frame = _frame(prediction_data)
    expected = model.predict(frame, return_se=True)
    path = tmp_path / "functional.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    for actual, wanted in zip(
        restored.predict(frame, return_se=True), expected, strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-13)
    P = restored.predict(frame, type="lpmatrix")
    if dimensions == 1:
        np.testing.assert_array_equal(P[2, 1:], 0)


@pytest.mark.parametrize("bad", ["coordinate_shape", "by_shape", "ragged"])
def test_functional_shape_errors_are_explicit(references, bad):
    from nampy.gam import GAM

    frame = _frame(references["ad/fixed/weighted/2"]["data"])
    if bad == "coordinate_shape":
        frame["Z"] = [row[:2] for row in frame.Z]
    elif bad == "by_shape":
        frame["L"] = np.ones(len(frame))
    else:
        frame.at[0, "X"] = frame.at[0, "X"][:2]
    if bad == "ragged":
        from nampy.gam.smooths.univariate.ad import AdaptiveSmoothTerm

        with pytest.raises(ValueError, match="same number"):
            AdaptiveSmoothTerm(["X", "Z"], by="L", k=[5, 5], m=3).fit(
                frame.to_numpy(), list(frame.columns)
            )
        return
    with pytest.raises(ValueError, match="equal shape|same number"):
        GAM(
            formula="y~s(X,bs='ad',by=L,k=10,m=3)"
            if bad == "by_shape"
            else "y~s(X,Z,bs='ad',k=[5,5],m=3)",
            smoothing_method="fixed",
            smoothing_params=[0.4] * (3 if bad == "by_shape" else 9),
        ).fit(data=frame)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_adaptive_tensor_marginals_remain_rejected(references, tensor):
    from nampy.gam import GAM

    data = _frame(references["ad/fixed/unweighted/2"]["data"])
    with pytest.raises((ValueError, NotImplementedError), match="tensor|marginal"):
        GAM(
            formula=f"y~{tensor}(X,Z,bs=['ad','ps'],k=[5,5])", smoothing_method="fixed"
        ).fit(data=data)
