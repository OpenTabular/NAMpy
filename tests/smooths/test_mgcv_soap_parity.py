"""Static mgcv soap-film constructor and end-to-end reference coverage.

Refresh explicitly with NAMPY_REFRESH_REFERENCE_FIXTURES=1; normal tests never run R.
Reference specification: mgcv R/soap.r smooth.construct.{so,sf,sw}.smooth.spec.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _coef_full, _edf_total
from tests.mgcv_parity_utils import _build_r_command, _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = [pytest.mark.surface_regression]


def _case(
    basis="so",
    geometry="square",
    *,
    by=False,
    fx=False,
    select=False,
    method="fixed",
    nmax=35,
    family="gaussian",
    offset=False,
    boundary_spec=None,
    variant=None,
):
    rng = np.random.default_rng(20261026)
    xy = rng.uniform(0.06, 0.94, (150, 2))
    boundary = [{"x": [0.0, 1.0, 1.0, 0.0, 0.0], "z": [0.0, 0.0, 1.0, 1.0, 0.0]}]
    k = 8
    if geometry == "hole":
        boundary.append(
            {"x": [0.4, 0.6, 0.6, 0.4, 0.4], "z": [0.4, 0.4, 0.6, 0.6, 0.4]}
        )
        xy = xy[np.any((xy < 0.35) | (xy > 0.65), axis=1)]
        k = [8, 6]
    if geometry == "known":
        boundary[0]["f"] = [0.0, 1.0, 1.5, 0.5, 0.0]
    knot_xy = np.array(
        [
            [0.2, 0.2],
            [0.5, 0.2],
            [0.8, 0.2],
            [0.2, 0.5],
            [0.8, 0.5],
            [0.2, 0.8],
            [0.5, 0.8],
            [0.8, 0.8],
        ]
    )
    data = pd.DataFrame(xy, columns=["x", "z"])
    data["v"] = 0.4 + data.x
    data["y"] = (
        np.sin(4 * data.x) * np.cos(3 * data.z)
        + 0.3 * data.x
        + rng.normal(0, 0.12, len(data))
    )
    newdata = pd.DataFrame(
        {"x": [0.12, 0.28, 0.75, 0.89], "z": [0.15, 0.82, 0.2, 0.78]}
    )
    newdata["v"] = 0.4 + newdata.x
    if family == "poisson":
        data["y"] = rng.poisson(np.exp(data["y"]))
    elif family == "binomial":
        data["y"] = rng.binomial(1, 1 / (1 + np.exp(-2 * data["y"])))
    result = {
        "basis": basis,
        "geometry": geometry,
        "by": by,
        "fx": fx,
        "select": select,
        "method": method,
        "k": k,
        "xt": {"bnd": boundary, "nmax": nmax},
        "knots": {"x": knot_xy[:, 0].tolist(), "z": knot_xy[:, 1].tolist()},
        "data": data.to_dict("list"),
        "newdata": newdata.to_dict("list"),
    }

    if family != "gaussian":
        result["family"] = family
    if offset:
        result["offset"] = True
    if boundary_spec is not None:
        result["xt"]["bndSpec"] = boundary_spec
    if variant is not None:
        result["variant"] = variant
        base = f"s(x,z,bs={basis!r},k={k!r},xt={result['xt']!r}"
        if variant == "pc":
            result["formula"] = "y ~ " + base + ",pc=[0.2,0.2])"
        elif variant == "linked":
            result["formula"] = (
                "y ~ " + base + ",by=v,id='soap') + " + base + ",by=w,id='soap')"
            )
            result["data"]["w"] = (0.7 + data.z**2).tolist()
            result["newdata"]["w"] = (0.7 + newdata.z**2).tolist()
        elif variant == "factor":
            result["formula"] = "y ~ f + " + base + ",by=f)"
            result["data"]["f"] = np.resize(["a", "b"], len(data)).tolist()
            result["newdata"]["f"] = ["a", "b", "a", "b"]
    return result


_R_REFERENCE = r"""
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
a <- commandArgs(TRUE)
p <- fromJSON(a[1], simplifyVector=FALSE)
d <- as.data.frame(lapply(p$data, unlist)); nd <- as.data.frame(lapply(p$newdata,unlist))
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
xt <- p$xt; xt$bnd <- lapply(xt$bnd,function(b) lapply(b,unlist))
if (!is.null(xt$bndSpec$m)) xt$bndSpec$m <- unlist(xt$bndSpec$m)
k <- unlist(p$k); kn <- lapply(p$knots,unlist); bs <- p$basis
expr <- if (p$by) s(x,z,bs=bs,k=k,xt=xt,by=v,fx=p$fx) else s(x,z,bs=bs,k=k,xt=xt,fx=p$fx)
raw <- smooth.construct(expr,d,kn)
sc <- smoothCon(expr,d,knots=kn,absorb.cons=TRUE)[[1]]
Xp <- PredictMat(sc,nd)
f <- if (p$by) y~s(x,z,bs=bs,k=k,xt=xt,by=v,fx=p$fx) else y~s(x,z,bs=bs,k=k,xt=xt,fx=p$fx)
if (!is.null(p$formula_r)) f <- as.formula(p$formula_r)
if (isTRUE(p$offset)) f <- update(f,.~.+offset(v))
args <- list(formula=f,data=d,knots=kn,method=if(p$method=='fixed') 'REML' else p$method,select=p$select)
if (!is.null(p$family)) args$family <- get(p$family,mode='function')()
if (p$method=='fixed' && !p$fx) args$sp <- rep(.37,(if(identical(p$variant,'factor')) 2 else 1)*length(sc$S)+as.integer(p$select && sc$null.space.dim>0))
b <- do.call(gam,args)
pr <- predict(b,nd,se.fit=TRUE,type="response")
pack <- function(x) if(is.null(x)) NULL else unname(x)
ans <- list(version=as.character(packageVersion('mgcv')), X=pack(raw$X),
  S=lapply(raw$S,pack),rank=pack(raw$rank),null_space_dim=raw$null.space.dim,
  offset=pack(attr(raw$X,'offset')), Xsc=pack(sc$X), Ssc=lapply(sc$S,pack), Xnew=pack(Xp),
  coef=pack(coef(b)), Vp=pack(b$Vp), Ve=pack(b$Ve), sp=pack(b$sp),
  edf=sum(b$edf), fitted=pack(fitted(b)), prediction=pack(pr$fit), se=pack(pr$se.fit),
  lpmatrix=pack(predict(b,nd,type='lpmatrix')))
write_json(ans,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key("soap-end-to-end-v1", case, normalize_floats=True)
    result = load_reference("mgcv_soap", key)
    if result is not None:
        return result
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, inputs, output = (
            root / name for name in ("ref.R", "input.json", "out.json")
        )
        source.write_text(_R_REFERENCE)
        r_case = dict(case)
        if "formula" in case:
            r_case["formula_r"] = _normalize_python_formula_text(case["formula"])
        inputs.write_text(json.dumps(r_case))
        proc = subprocess.run(
            _build_r_command(source, str(inputs), str(output)),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(output.read_text())
    save_reference("mgcv_soap", key, result)
    return result


RAW_CASES = [(b, g) for b in ("so", "sf", "sw") for g in ("square", "hole")]
RAW_CASES += [("so", "known"), ("sw", "known")]


@pytest.mark.parametrize("basis,geometry", RAW_CASES)
def test_soap_raw_basis_penalties_rank_and_prediction(basis, geometry):
    case = _case(basis, geometry)
    expected = _reference(case)
    from nampy.gam.smooths.univariate.soap import SoapFilmTerm

    data, newdata = pd.DataFrame(case["data"]), pd.DataFrame(case["newdata"])
    term = SoapFilmTerm(
        feature=["x", "z"],
        basis=basis,
        k=case["k"],
        xt=case["xt"],
        knots=case["knots"],
        constraint_mode="never",
    ).fit(data[["x", "z"]].to_numpy(copy=True), ["x", "z"])
    np.testing.assert_allclose(
        term._raw_basis_train, expected["X"], atol=2e-10, rtol=2e-10
    )
    assert len(term._raw_penalties) == len(expected["S"])
    for actual, reference in zip(term._raw_penalties, expected["S"], strict=True):
        np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=2e-10)
    assert (
        np.asarray(term.rank).ravel().tolist()
        == np.asarray(expected["rank"]).ravel().tolist()
    )
    assert term._setup.null_space_dim == expected["null_space_dim"]
    # Public constrained construction must reproduce mgcv's prediction map.
    fitted = SoapFilmTerm(
        feature=["x", "z"], basis=basis, k=case["k"], xt=case["xt"], knots=case["knots"]
    ).fit(data[["x", "z"]].to_numpy(copy=True), ["x", "z"])
    np.testing.assert_allclose(
        fitted.basis_train, expected["Xsc"], atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        fitted.transform_new(newdata[["x", "z"]].to_numpy(copy=True)),
        expected["Xnew"],
        atol=2e-10,
        rtol=2e-10,
    )
    for actual, reference in zip(fitted.penalties, expected["Ssc"], strict=True):
        np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=2e-10)


FIT_CASES = [{"basis": b} for b in ("so", "sf", "sw")]
FIT_CASES += [
    {"basis": "so", "geometry": "hole"},
    {"basis": "so", "geometry": "known"},
    {"basis": "so", "by": True},
    {"basis": "so", "fx": True},
    {"basis": "so", "select": True},
    {"basis": "so", "geometry": "hole", "select": True},
    {"basis": "so", "geometry": "known", "by": True},
]
FIT_CASES += [{"basis": b, "method": "REML"} for b in ("so", "sf", "sw")]
FIT_CASES += [
    {"basis": "so", "family": family, "method": method}
    for family in ("poisson", "binomial")
    for method in ("fixed", "REML")
]
FIT_CASES += [
    {"basis": "so", "geometry": "known", "offset": True},
    {"basis": "so", "boundary_spec": {"bs": "cp", "m": [2, 2]}},
]

FIT_CASES += [{"basis": "so", "variant": v} for v in ("pc", "linked", "factor")]


@pytest.mark.parametrize(
    "options",
    FIT_CASES,
    ids=lambda p: "-".join(f"{key}={value}" for key, value in p.items()),
)
def test_soap_fit_predictions_uncertainty_and_covariance(options):
    case = _case(**options)
    expected = _reference(case)
    formula = (
        f"y ~ s(x,z,bs={case['basis']!r},k={case['k']!r},"
        f"xt={case['xt']!r},fx={case['fx']!r}" + (",by=v)" if case["by"] else ")")
    )
    formula = case.get("formula", formula)
    if case.get("offset"):
        formula += " + offset(v)"
    fixed = case["method"] == "fixed"
    n_penalties = len(expected["Ssc"]) + int(
        case["select"] and expected["null_space_dim"] > (0 if case["by"] else 1)
    )
    if case.get("variant") == "factor":
        n_penalties *= 2
    model = GAM(
        formula=formula,
        family=case.get("family", "gaussian"),
        knots=case["knots"],
        select=case["select"],
        smoothing_method="fixed" if fixed else case["method"],
        optimize_smoothing=not fixed,
        **(
            {"smoothing_params": np.full(n_penalties, 0.37)}
            if fixed and not case["fx"]
            else {}
        ),
    )
    model.fit(data=pd.DataFrame(case["data"]))
    nd = pd.DataFrame(case["newdata"])
    tolerance = 3e-5 if not fixed else 2e-8
    prediction, se = model.predict(nd, return_se=True)
    np.testing.assert_allclose(
        prediction, expected["prediction"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(se, expected["se"], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        model.predict(), expected["fitted"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        model.predict(nd, type="lpmatrix"), expected["lpmatrix"], atol=2e-9, rtol=2e-9
    )
    np.testing.assert_allclose(
        model.vcov(), expected["Vp"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        model.vcov(freq=True), expected["Ve"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        _coef_full(model), expected["coef"], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        _edf_total(model), expected["edf"], atol=tolerance, rtol=tolerance
    )
    if not fixed:
        np.testing.assert_allclose(
            np.log(model.smoothing_params),
            np.log(np.atleast_1d(expected["sp"])),
            atol=2e-4,
            rtol=2e-4,
        )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(nd), prediction)


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
def test_soap_rejects_training_points_outside_domain(basis):
    from nampy.gam.smooths.univariate.soap import SoapFilmTerm

    case = _case(basis)
    data = pd.DataFrame(case["data"])[["x", "z"]].to_numpy(copy=True)
    data[0] = [-0.2, 0.5]
    with pytest.raises(ValueError, match="outside soap boundary"):
        SoapFilmTerm(
            feature=["x", "z"],
            basis=basis,
            k=case["k"],
            xt=case["xt"],
            knots=case["knots"],
        ).fit(data, ["x", "z"])


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
def test_soap_prediction_preserves_valid_rows_and_marks_outside_nan(basis):
    from nampy.gam.smooths.univariate.soap import SoapFilmTerm

    case = _case(basis, "hole")
    data = pd.DataFrame(case["data"])[["x", "z"]].to_numpy(copy=True)
    term = SoapFilmTerm(
        feature=["x", "z"], basis=basis, k=case["k"], xt=case["xt"], knots=case["knots"]
    ).fit(data, ["x", "z"])
    query = np.array([[0.2, 0.2], [-0.3, 0.5], [0.5, 0.5], [0.8, 0.8]])
    prediction = term.transform_new(query)
    assert np.isfinite(prediction[[0, 3]]).all()
    assert np.isnan(prediction[[1, 2]]).all()
    np.testing.assert_allclose(prediction[[0, 3]], term.transform_new(query[[0, 3]]))


def test_soap_full_basis_is_boundary_and_interior_component_concatenation():
    from nampy.gam.smooths.univariate.soap import SoapFilmTerm

    case = _case()
    data = pd.DataFrame(case["data"])[["x", "z"]].to_numpy(copy=True)
    terms = [
        SoapFilmTerm(
            feature=["x", "z"],
            basis=b,
            k=case["k"],
            xt=case["xt"],
            knots=case["knots"],
            constraint_mode="never",
        ).fit(data, ["x", "z"])
        for b in ("so", "sf", "sw")
    ]
    full, film, interior = terms
    np.testing.assert_allclose(
        full.basis_train,
        np.column_stack([film.basis_train, interior.basis_train]),
        atol=3e-12,
        rtol=3e-12,
    )
    nfilm = film.basis_train.shape[1]
    np.testing.assert_allclose(
        full._raw_penalties[0][:nfilm, :nfilm],
        film._raw_penalties[0],
        atol=3e-12,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        full._raw_penalties[1][nfilm:, nfilm:],
        interior._raw_penalties[0],
        atol=3e-12,
        rtol=3e-12,
    )
    assert np.count_nonzero(full._raw_penalties[0][nfilm:]) == 0
    assert np.count_nonzero(full._raw_penalties[1][:nfilm]) == 0
