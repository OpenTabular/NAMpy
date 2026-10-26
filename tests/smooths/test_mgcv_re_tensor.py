"""Actual mgcv random-effect tensor marginal parity."""

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

PROBE = Path(__file__).with_name("probes") / "mgcv_re_tensor.R"
KEY = reference_key("re-tensor", hashlib.sha256(PROBE.read_bytes()).hexdigest())


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_re_tensor", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            subprocess.run(
                ["Rscript", str(PROBE), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
        save_reference("mgcv_re_tensor", KEY, result)
    return result["cases"]


@pytest.mark.parametrize("matrix_arg", [False, True])
@pytest.mark.parametrize("kind", ["numeric", "factor", "mixed"])
@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_re_tensor_reference(references, matrix_arg, kind, tensor, method):
    ref = references[f"{str(matrix_arg).upper()}/{kind}/{tensor}/{method}"]
    assert "error" not in ref, ref
    data = pd.DataFrame({k: list(v) for k, v in ref["data"].items()})
    newdata = pd.DataFrame({k: list(v) for k, v in ref["newdata"].items()})
    if not matrix_arg:
        data["f"] = pd.Categorical(data["f"], categories=list("abcd"))
        newdata["f"] = pd.Categorical(newdata["f"], categories=list("abcd"))
    variables = "x,z" if kind == "numeric" else "f,z" if kind == "factor" else "x,f,z"
    form = f"response~{tensor}({variables},bs=['re','cr'],d=[{2 if kind == 'mixed' else 1},1],k=[4,5]{',by=L' if matrix_arg else ''})+offset(o)"
    if ref.get("precision", False):
        precision = [np.diag([1, 2, 3, 4]).tolist()]
        form = form.replace(
            ")+offset(o)", f",xt=[{{'S':{precision!r},'rank':[4]}},None])+offset(o)"
        )
    model = GAM(
        formula=form,
        smoothing_method=method,
        optimize_smoothing=method != "fixed",
        smoothing_params=ref["sp"] if method == "fixed" else None,
    ).fit(data=data)
    fit, se = model.predict(newdata, return_se=True)
    np.testing.assert_allclose(fit, ref["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, ref["se"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=3e-4, rtol=3e-4)
    P = model.predict(newdata, type="lpmatrix")
    np.testing.assert_allclose(
        P @ model.vcov() @ P.T, ref["covariance"], atol=3e-5, rtol=3e-5
    )
    B = np.asarray(_design_matrix(model))
    target = np.asarray(ref["raw_X"])
    A = np.linalg.lstsq(B, target, rcond=None)[0]
    np.testing.assert_allclose(B @ A, target, atol=2e-7)
    for penalty, S in zip(_penalty_blocks_seq(model), ref["raw_S"], strict=True):
        np.testing.assert_allclose(A.T @ penalty.matrix @ A, S, atol=2e-6, rtol=2e-6)
    if kind != "numeric":
        unknown = newdata.copy(deep=True)
        if matrix_arg:
            unknown.at[0, "f"] = ["new"] * len(unknown.at[0, "f"])
        else:
            unknown["f"] = unknown["f"].cat.add_categories(["new"])
            unknown.at[0, "f"] = "new"
        fit, se = model.predict(unknown, return_se=True)
        np.testing.assert_allclose(fit, ref["unknown"]["fit"], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(se, ref["unknown"]["se"], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_re_tensor_rejects_multiple_precision_penalties(references, tensor):
    assert "multiple penalties" in references[f"unsupported/{tensor}"]["error"]
    data = pd.DataFrame(
        {
            "response": np.sin(np.arange(60)),
            "f": pd.Categorical(list("abcd") * 15),
            "z": np.linspace(-1, 1, 60),
        }
    )
    precision = [np.eye(4).tolist(), (2 * np.eye(4)).tolist()]
    formula = f"response~{tensor}(f,z,bs=['re','cr'],k=[4,5],xt=[{{'S':{precision!r},'rank':[4,4]}},None])"
    with pytest.raises(NotImplementedError, match="multiple penalties"):
        GAM(formula=formula).fit(data=data)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_re_tensor_single_supplied_precision(references, tensor, method):
    remapped = dict(references)
    remapped[f"FALSE/factor/{tensor}/{method}"] = references[
        f"FALSE/factor/{tensor}/{method}/precision"
    ]
    test_re_tensor_reference(remapped, False, "factor", tensor, method)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_re_tensor_repeated_mixed_coordinates(references, tensor, method):
    remapped = dict(references)
    remapped[f"TRUE/mixed/{tensor}/{method}"] = references[
        f"TRUE/mixed/{tensor}/{method}/repeated"
    ]
    test_re_tensor_reference(remapped, True, "mixed", tensor, method)


@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
def test_re_tensor_matrix_point_sum_and_persistence(references, tensor, tmp_path):
    ref = references[f"TRUE/factor/{tensor}/fixed"]
    data = pd.DataFrame({name: list(values) for name, values in ref["data"].items()})
    nd = pd.DataFrame({name: list(values) for name, values in ref["newdata"].items()})
    model = GAM(
        formula=f"response~{tensor}(f,z,bs=['re','cr'],k=[4,5],by=L)+offset(o)",
        smoothing_params=ref["sp"],
        optimize_smoothing=False,
    ).fit(data=data)
    target = model.predict(nd, return_se=True)
    path = tmp_path / "re_tensor.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(restored.predict(nd, return_se=True), target, atol=1e-13)
    point = pd.DataFrame(
        {name: np.asarray(nd[name].to_list()).ravel() for name in ["f", "z"]}
    )
    point["L"] = 1.0
    point["o"] = 0.0
    P = restored.predict(point, type="lpmatrix").reshape(len(nd), 3, -1)
    aggregate = np.einsum("nq,nqk->nk", np.asarray(nd["L"].to_list()), P)
    aggregate[:, 0] = 1.0
    np.testing.assert_allclose(
        restored.predict(nd, type="lpmatrix"), aggregate, atol=2e-12
    )


def test_shared_factor_feature_keeps_stricter_non_re_margin_guard():
    from nampy.gam.smooths.tensor.te import TensorProductSplineTerm

    # Formula construction rejects repeated smooth variables; the runtime
    # composition contract must nevertheless retain the stricter owner.
    nb = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    term = TensorProductSplineTerm(
        feature=[["f"], ["f"]],
        basis=["re", "mrf"],
        k=[4, 4],
        xt=[None, {"nb": nb}],
        metadata={"factor_levels_by_feature": {"f": {"levels": list("abcd")}}},
    )
    term.fit(np.asarray(list("abcd") * 15, dtype=object)[:, None], ["f"])
    assert term.metadata["zero_unseen_factor_features"] == []
    with pytest.raises(ValueError, match="unseen|unknown|levels"):
        term.transform_new(np.asarray([["new"]], dtype=object))
