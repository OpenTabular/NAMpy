"""Matrix cs: distinguish basis transport from null-penalty ambiguity.

The public matrix-cs guard remains intentional. These tests compare source
function spaces and coefficient-transport invariants, not raw eigenvectors.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve

from nampy.gam.splines.univariate.cr import add_full_rank_shrinkage
from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.smooths.test_mgcv_functional_packing import KEY as PACKED_KEY
from tests.smooths.test_mgcv_spline_linear_functionals import KEY as DISTINCT_KEY

SCRIPT = r'''
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=TRUE)
d<-p$data;nd<-p$newdata
f<-as.formula(p$formula)
b<-gam(f,data=d,weights=d$w,sp=.4,method="REML")
v<-as.vector(d$X);nv<-as.vector(nd$X)
rawcr<-smooth.construct(s(v,bs="cr",k=10),data=list(v=v),knots=NULL)
rawcs<-smooth.construct(s(v,bs="cs",k=10),data=list(v=v),knots=NULL)
out<-list(version=as.character(packageVersion("mgcv")),knots=rawcs$xp,raw_cr=rawcr$S[[1]],raw_cs=rawcs$S[[1]],point_basis=rawcs$X,point_new=Predict.matrix(rawcs,list(v=nv)),S=b$smooth[[1]]$S[[1]],S_scale=b$smooth[[1]]$S.scale,X=predict(b,type="lpmatrix"),Xp=predict(b,nd,type="lpmatrix"),coef=coef(b),Vp=b$Vp,pred=predict(b,nd),fitted=fitted(b),scale=b$sig2,edf=b$edf)
write_json(out,a[2],digits=17,auto_unbox=TRUE)
'''


def input_case(kind="weighted", repeated=False):
    if repeated:
        ref = load_reference("mgcv_functional_packing", PACKED_KEY)["cases"][f"cs/{kind}/fixed"]
        formula = f"y~s(X,bs='cs',k=10{',by=L' if kind != 'unweighted' else ''})"
    else:
        ref = load_reference("mgcv_spline_linear_functionals", DISTINCT_KEY)["cases"][f"cs/fixed/{kind}/1"]
        formula = f"y~s(X,bs='cs',k=10{',by=L' if kind != 'unweighted' else ''})+offset(o)"
    return {"kind": kind, "repeated": repeated, "data": ref["data"], "newdata": ref["newdata"], "formula": formula}


def reference(case):
    key = reference_key("matrix-cs-functional-invariants-v1", case)
    result = load_reference("mgcv_matrix_cs_invariants", key)
    if result is None:
        with tempfile.TemporaryDirectory() as tmp:
            source, inp, out = [Path(tmp) / name for name in ("source.R", "in.json", "out.json")]
            source.write_text(SCRIPT)
            inp.write_text(json.dumps(case))
            run = subprocess.run(["Rscript", str(source), str(inp), str(out)], capture_output=True, text=True)
            assert run.returncode == 0, run.stderr
            result = json.loads(out.read_text())
        save_reference("mgcv_matrix_cs_invariants", key, result)
    return result


def sum_basis(point_basis, coordinates, weights):
    coordinates = np.asarray(coordinates)
    # R point_basis uses column-major matrix flattening.
    point_basis = np.asarray(point_basis).reshape(coordinates.shape[1], coordinates.shape[0], -1).transpose(1, 0, 2)
    return np.einsum("nq,nqk->nk", weights, point_basis)


def design_state(case, source):
    x = np.asarray(case["data"]["X"])
    newx = np.asarray(case["newdata"]["X"])
    weight = np.ones_like(x) if case["kind"] == "unweighted" else np.asarray(case["data"]["L"])
    newweight = np.ones_like(newx) if case["kind"] == "unweighted" else np.asarray(case["newdata"]["L"])
    basis = sum_basis(source["point_basis"], x, weight)
    pred_basis = sum_basis(source["point_new"], newx, newweight)
    fitted = np.asarray(source["X"])
    predicted = np.asarray(source["Xp"])
    transform = np.linalg.lstsq(basis, fitted[:, 1:], rcond=None)[0]
    return basis, pred_basis, fitted, predicted, transform


@pytest.mark.parametrize("kind,repeated", [("weighted", False), ("constant", False), ("unweighted", False), ("weighted", True), ("constant", True), ("unweighted", True)])
def test_cs_source_basis_penalty_transport_and_spectrum(kind, repeated):
    case = input_case(kind, repeated)
    source = reference(case)
    basis, pred_basis, fitted, predicted, transform = design_state(case, source)
    np.testing.assert_allclose(basis @ transform, fitted[:, 1:], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(pred_basis @ transform, predicted[:, 1:], atol=2e-12, rtol=2e-12)
    raw = np.asarray(source["raw_cs"])
    np.testing.assert_allclose(transform.T @ raw @ transform / source["S_scale"], source["S"], atol=2e-10, rtol=2e-10)
    cr = np.asarray(source["raw_cr"])
    native = add_full_rank_shrinkage(cr)
    scale = np.linalg.norm(cr, ord=2)
    null = np.linalg.qr(np.column_stack([np.ones(10), source["knots"]]))[0]
    range_projector = np.eye(10) - null @ null.T
    np.testing.assert_allclose(cr @ null / scale, 0., atol=2e-14)
    for penalty in (raw, native):
        np.testing.assert_allclose(range_projector @ (penalty - cr) @ range_projector / scale, 0., atol=2e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(native) / scale, np.linalg.eigvalsh(raw) / scale, atol=2e-13, rtol=2e-13)


def test_unequal_null_penalties_change_fitted_function_and_covariance():
    case = input_case()
    source = reference(case)
    _, _, fitted, predicted, transform = design_state(case, source)
    raw = np.asarray(source["raw_cs"])
    null = np.linalg.qr(np.column_stack([np.ones(10), source["knots"]]))[0]
    rotation = np.array([[0., -1.], [1., 0.]])
    rotate = np.eye(10) + null @ (rotation - np.eye(2)) @ null.T
    rotated = rotate.T @ raw @ rotate
    # Same raw CR operator, same full shrunk spectrum, same fitted function
    # space: only the allocation of unequal null shrinkage directions changes.
    cr = np.asarray(source["raw_cr"])
    scale = np.linalg.norm(cr, ord=2)
    np.testing.assert_allclose(rotate.T @ cr @ rotate / scale, cr / scale, atol=2e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(rotated) / scale, np.linalg.eigvalsh(raw) / scale, atol=2e-13)
    w = np.asarray(case["data"]["w"])
    response = np.asarray(case["data"]["y"]) - np.asarray(case["data"]["o"])
    penalty = np.zeros((fitted.shape[1], fitted.shape[1]))
    penalty[1:, 1:] = transform.T @ rotated @ transform / source["S_scale"]
    hessian = fitted.T @ (w[:, None] * fitted) + .4 * penalty
    coef = solve(hessian, fitted.T @ (w * response), assume_a="sym")
    covariance = solve(hessian, np.eye(len(coef)), assume_a="sym") * source["scale"]
    original_function = predicted @ np.asarray(source["coef"])
    original_kernel = predicted @ np.asarray(source["Vp"]) @ predicted.T
    assert np.max(np.abs(predicted @ coef - original_function)) > 1e-7
    assert np.max(np.abs(predicted @ covariance @ predicted.T - original_kernel)) > 1e-9
