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
from nampy.gam.model_state import _design_matrix, _edf_total, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_tensor_linear_functionals.R"
KEY = reference_key(
    "tensor-matrix-functions", hashlib.sha256(PROBE.read_bytes()).hexdigest()
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_tensor_linear_functionals", KEY)
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
        save_reference("mgcv_tensor_linear_functionals", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


def _frame(values):
    return pd.DataFrame({name: list(value) for name, value in values.items()})


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("basis", ["cr", "ps", "bs", "tp", "ds", "gp"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("kind", ["unweighted", "weighted", "constant"])
def test_tensor_matrix_functions_match_mgcv(references, tensor, basis, method, kind):
    ref = references[f"{tensor}/{basis}/{method}/{kind}"]
    formula = f"y ~ {tensor}(X,Z,bs=['{basis}','{basis}'],k=[5,5]{',by=L' if kind != 'unweighted' else ''}) + offset(o)"
    if tensor == "t2" and basis == "bs":
        # Retain actual-R fixtures as evidence for the portability boundary.
        with pytest.raises(NotImplementedError, match="natural null-space"):
            GAM(
                formula=formula,
                smoothing_method=method,
                optimize_smoothing=method != "fixed",
                smoothing_params=ref["sp"] if method == "fixed" else None,
            ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))
        return
    model = GAM(
        formula=formula,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=ref["sp"] if method == "fixed" else None,
    ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))
    prediction, se = model.predict(_frame(ref["newdata"]), return_se=True)
    np.testing.assert_allclose(prediction, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    P = model.predict(_frame(ref["newdata"]), type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance_kernel"], atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(model.predict(), ref["train_fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(
        _edf_total(model), np.sum(ref["edf"]), atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)
    # Compare penalty forms after aligning possibly rotated coefficient bases.
    fitted_basis = np.asarray(_design_matrix(model))
    target_basis = np.asarray(ref["fit_X"])
    alignment = np.linalg.lstsq(fitted_basis, target_basis, rcond=None)[0]
    np.testing.assert_allclose(fitted_basis @ alignment, target_basis, atol=2e-7)
    target_penalties = (
        [ref["fit_S"][name] for name in ["rr", "nr", "rn"]]
        if isinstance(ref["fit_S"], dict)
        else ref["fit_S"]
    )
    for penalty, target in zip(
        _penalty_blocks_seq(model), target_penalties, strict=True
    ):
        np.testing.assert_allclose(
            alignment.T @ penalty.matrix @ alignment, target, atol=2e-6, rtol=2e-6
        )
    coordinates = np.asarray(ref["newdata"]["X"])
    point_data = pd.DataFrame(
        {
            "X": coordinates.ravel(),
            "Z": np.asarray(ref["newdata"]["Z"]).ravel(),
            "L": 1.0,
            "o": 0.0,
        }
    )
    point_basis = model.predict(point_data, type="lpmatrix")
    weights = (
        np.ones_like(coordinates)
        if kind == "unweighted"
        else np.asarray(ref["newdata"]["L"])
    )
    aggregate = np.einsum(
        "nq,nqk->nk", weights, point_basis.reshape(*coordinates.shape, -1)
    )
    aggregate[:, 0] = 1.0
    np.testing.assert_allclose(P, aggregate, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize(
    "bad", ["coordinate_width", "mixed_scalar", "by_vector", "by_width"]
)
def test_tensor_matrix_arguments_reject_incompatible_shapes(references, tensor, bad):
    ref = references[f"{tensor}/cr/fixed/weighted"]
    data = _frame(ref["data"])
    if bad == "coordinate_width":
        data["Z"] = [row[:2] for row in data["Z"]]
    elif bad == "mixed_scalar":
        data["Z"] = [row[0] for row in data["Z"]]
    elif bad == "by_vector":
        data["L"] = 1.0
    else:
        data["L"] = [row[:2] for row in data["L"]]
    with pytest.raises(ValueError, match="equal shape"):
        GAM(
            formula=f"y~{tensor}(X,Z,bs=['cr','cr'],k=[5,5],by=L)",
            smoothing_params=ref["sp"],
            optimize_smoothing=False,
        ).fit(data=data)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_tensor_matrix_serialization_and_new_quadrature_width(
    references, tensor, tmp_path
):
    ref = references[f"{tensor}/cr/fixed/weighted"]
    model = GAM(
        formula=f"y~{tensor}(X,Z,bs=['cr','cr'],k=[5,5],by=L)+offset(o)",
        smoothing_params=ref["sp"],
        optimize_smoothing=False,
    ).fit(data=_frame(ref["data"]), sample_weight=np.asarray(ref["data"]["w"]))
    nd = _frame(ref["newdata"])
    for name in ["X", "Z", "L"]:
        nd[name] = [row[:2] for row in nd[name]]
    expected = model.predict(nd, return_se=True)
    path = tmp_path / "tensor.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(nd, return_se=True), expected, atol=1e-13
    )
    point = pd.DataFrame(
        {name: np.asarray(nd[name].to_list()).ravel() for name in ["X", "Z"]}
    )
    point["L"] = 1.0
    point["o"] = 0.0
    P = restored.predict(point, type="lpmatrix").reshape(len(nd), 2, -1)
    summed = np.einsum("nq,nqk->nk", np.asarray(nd["L"].to_list()), P)
    summed[:, 0] = 1.0
    np.testing.assert_allclose(
        restored.predict(nd, type="lpmatrix"), summed, atol=2e-12
    )


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
@pytest.mark.parametrize("kind", ["unweighted", "weighted", "constant"])
def test_tensor_matrix_repeated_paired_coordinates(references, tensor, method, kind):
    ref = references[f"{tensor}/cr/{method}/{kind}/repeated"]
    remapped = dict(references)
    remapped[f"{tensor}/cr/{method}/{kind}"] = ref
    test_tensor_matrix_functions_match_mgcv(remapped, tensor, "cr", method, kind)


@pytest.fixture(scope="module")
def soap_references():
    probe = PROBE.with_name("mgcv_tensor_soap_functional.R")
    key = reference_key(
        "tensor-soap-matrix-functions", hashlib.sha256(probe.read_bytes()).hexdigest()
    )
    result = load_reference("mgcv_tensor_linear_functionals", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(
                ["Rscript", str(probe), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
        save_reference("mgcv_tensor_linear_functionals", key, result)
    return result["cases"]


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("basis", ["sf", "sw"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_soap_tensor_matrix_functionals(soap_references, tensor, basis, method):
    ref = soap_references[f"{tensor}/{basis}/{method}"]
    formula = f"response~{tensor}(x,y,z,by=L,bs=['{basis}','cr'],d=[2,1],k=[8,4],xt=[{{'bnd':{ref['bnd']!r},'nmax':30}},None])+offset(o)"
    model = GAM(
        formula=formula,
        knots=ref["knots"],
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=np.atleast_1d(ref["sp"]) if method == "fixed" else None,
    ).fit(data=_frame(ref["data"]))
    frame = _frame(ref["newdata"])
    fit, se = model.predict(frame, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)
    P = model.predict(frame, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance"], atol=3e-5, rtol=3e-5
    )
    B = np.asarray(_design_matrix(model))
    target_B = np.asarray(ref["raw_X"])
    alignment = np.linalg.lstsq(B, target_B, rcond=None)[0]
    np.testing.assert_allclose(B @ alignment, target_B, atol=2e-7)
    for penalty, target in zip(_penalty_blocks_seq(model), ref["raw_S"], strict=True):
        np.testing.assert_allclose(
            alignment.T @ penalty.matrix @ alignment, target, atol=2e-6, rtol=2e-6
        )


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_tensor_matrix_linked_setup_is_explicitly_unsupported(references, tensor):
    ref = references[f"{tensor}/cr/fixed/weighted"]
    with pytest.raises(NotImplementedError, match="[Ll]inked"):
        GAM(
            formula=f"y~{tensor}(X,Z,bs=['cr','cr'],k=[5,5],by=L,id=1)",
            optimize_smoothing=False,
            smoothing_params=ref["sp"],
        ).fit(data=_frame(ref["data"]))


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_tensor_matrix_factor_by_is_rejected(references, tensor):
    ref = references[f"{tensor}/cr/fixed/weighted"]
    data = _frame(ref["data"])
    data["group"] = pd.Categorical(np.where(np.arange(len(data)) % 2, "a", "b"))
    with pytest.raises(
        (ValueError, NotImplementedError), match="factor|equal shape|matrix"
    ):
        GAM(
            formula=f"y~{tensor}(X,Z,bs=['cr','cr'],k=[5,5],by=group)",
            optimize_smoothing=False,
            smoothing_params=ref["sp"],
        ).fit(data=data)


def test_type3_null_normalization_depends_on_valid_eigenvector_orientation(monkeypatch):
    """Upstream's per-column null scaling is not rotation invariant."""
    from nampy.gam.splines.basis import natparam

    x = np.linspace(-0.8, 1.3, 31)
    X = np.column_stack([x**2, x**3, np.ones_like(x), x])
    S = np.diag([3.0, 1.0, 0.0, 0.0])
    original = natparam._r_symmetric_eigh_descending
    baseline = natparam.nat_param_type3(X, S)

    def rotated(matrix):
        values, vectors = original(matrix)
        if matrix.shape == S.shape:
            angle = 0.4
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            vectors[:, -2:] = vectors[:, -2:] @ rotation
            np.testing.assert_allclose((vectors * values) @ vectors.T, S, atol=1e-14)
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-14)
        return values, vectors

    monkeypatch.setattr(natparam, "_r_symmetric_eigh_descending", rotated)
    alternative = natparam.nat_param_type3(X, S)
    original_null = baseline["X"][:, 2:]
    rotated_null = alternative["X"][:, 2:]
    # Equal spaces, unequal ridge metrics: t2's range/null blocks penalize
    # these coordinates, changing fitted functions rather than merely signs.
    np.testing.assert_allclose(
        original_null @ np.linalg.pinv(original_null),
        rotated_null @ np.linalg.pinv(rotated_null),
        atol=1e-13,
    )
    assert not np.allclose(
        original_null @ original_null.T, rotated_null @ rotated_null.T
    )
