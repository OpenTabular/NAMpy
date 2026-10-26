"""Static mgcv constructor/PDE fixtures for the soap finite-difference port.

Refresh explicitly with NAMPY_REFRESH_REFERENCE_FIXTURES=1 pytest ... .
Ordinary runs only read the versioned JSON and never launch R.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.splines.soap import (
    boundary_grid,
    build_soap_basis,
    inside_boundary,
    pde_matrix,
    predict_soap_basis,
    process_boundary,
)
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

NAMESPACE = "mgcv_soap_numerics"


def _case(geometry, basis, boundary_basis):
    if geometry == "concave":
        boundary = [{"x": [-1, 1, 1, 0, 0, -1], "y": [-1, -1, 0, 0, 1, 1]}]
        knots = [[-0.6, -0.5], [0.5, -0.6], [-0.5, 0.5]]
    else:
        boundary = [{"x": [-1, 1, 1, -1], "y": [-1, -1, 1, 1]}]
        knots = [[-0.65, -0.6], [0.65, -0.6], [-0.6, 0.6], [0.65, 0.6]]
    if geometry == "circle":
        theta = np.linspace(0, 2 * np.pi, 41)
        boundary = [{"x": np.cos(theta).tolist(), "y": np.sin(theta).tolist()}]
    if geometry == "rectangle":
        boundary = [{"x": [-1, 1, 1, -1], "y": [-0.75, -0.75, 0.75, 0.75]}]
    if geometry in {"hole", "fixed"}:
        boundary.append(
            {"x": [-0.22, 0.22, 0.22, -0.22], "y": [-0.22, -0.22, 0.22, 0.22]}
        )
    if geometry == "fixed":
        boundary[1]["f"] = [1.0, 1.5, 2.0, 0.5]
    if geometry == "all_fixed":
        boundary[0]["f"] = [1.0, 2.0, 3.0, 2.0]
    if geometry == "shifted":
        boundary = [
            {"x": (np.array(b["x"]) + 3).tolist(), "y": (np.array(b["y"]) - 2).tolist()}
            for b in boundary
        ]
        knots = (np.array(knots) + [3, -2]).tolist()
    xx, yy = np.meshgrid(np.linspace(-0.91, 0.91, 9), np.linspace(-0.89, 0.89, 8))
    x, y = xx.ravel(), yy.ravel()
    if geometry == "shifted":
        x, y = x + 3, y - 2
    keep = inside_boundary(process_boundary(boundary), x, y)
    x, y = x[keep], y[keep]
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "boundary": boundary,
        "knots": knots,
        "k": [7] * len(boundary),
        "nmax": 27,
        "basis": basis,
        "boundary_spec": {"bs": boundary_basis, "m": [2, 2]},
    }


CASES = [
    (geometry, basis, bb)
    for geometry in (
        "square",
        "concave",
        "hole",
        "fixed",
        "shifted",
        "all_fixed",
        "circle",
        "rectangle",
    )
    for basis in ("so", "sf", "sw")
    for bb in ("cc", "cp")
    if not (geometry == "all_fixed" and basis == "sf")
]


def _reference(case):
    script = r"""
args <- commandArgs(TRUE)
lib <- Sys.getenv("MGCV_LIB_PATH")
if (nzchar(lib)) .libPaths(c(lib,.libPaths()))
library(mgcv)
library(jsonlite)
a <- fromJSON(args[1], simplifyVector=FALSE)
v <- function(x) as.numeric(unlist(x))
bnd <- lapply(a$boundary,function(b) { z <- list(x=v(b$x),y=v(b$y)); if (!is.null(b$f)) z$f <- v(b$f); z })
knt <- do.call(rbind,lapply(a$knots,v)); knt <- list(x=knt[,1],y=knt[,2])
xt <- list(bnd=bnd,nmax=a$nmax,bndSpec=if (is.null(a$boundary_spec)) NULL else list(bs=a$boundary_spec$bs,m=v(a$boundary_spec$m),knot.space="even"))
dat <- data.frame(x=v(a$x),y=v(a$y))
obj <- smooth.construct(s(x,y,bs=a$basis,k=v(a$k),xt=xt),dat,knt)
pred <- PredictMat(obj,data.frame(x=dat$x[seq(1,nrow(dat),3)]+.003,y=dat$y[seq(1,nrow(dat),3)]-.002))
cat(toJSON(list(version=as.character(packageVersion("mgcv")), X=obj$X,S=obj$S,rank=obj$rank,ns=obj$null.space.dim,offset=attr(obj$X,"offset"),G=obj$sd$G,d=obj$sd$d,irng=obj$irng,pred=pred,pred_offset=attr(pred,"offset")),digits=17,na="null",auto_unbox=FALSE))
"""
    with tempfile.TemporaryDirectory() as td:
        arg = Path(td) / "input.json"
        src = Path(td) / "reference.R"
        arg.write_text(json.dumps(case))
        src.write_text(script)
        result = subprocess.run(
            ["Rscript", str(src), str(arg)], capture_output=True, text=True, check=False
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("geometry,basis,bb", CASES)
def test_soap_raw_constructor_mgcv(geometry, basis, bb):
    case = _case(geometry, basis, bb)
    _assert_reference(case)


def _assert_reference(case):
    key = reference_key("soap_raw_constructor", case)
    if generation_enabled():
        save_reference(NAMESPACE, key, _reference(case))
    ref = load_reference(NAMESPACE, key)
    setup = build_soap_basis(**case)
    np.testing.assert_allclose(setup.basis_train, ref["X"], atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(setup.irng, ref["irng"], atol=2e-10, rtol=2e-10)
    assert setup.ranks == ref["rank"]
    assert setup.null_space_dim == ref["ns"][0]
    np.testing.assert_array_equal(setup.G.T, ref["G"])
    for actual, expected in zip(setup.penalties, ref["S"], strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
        np.testing.assert_allclose(actual, actual.T, atol=1e-12)
        assert np.linalg.eigvalsh(actual).min() > -1e-9
    pred, off = predict_soap_basis(
        np.array(case["x"])[::3] + 0.003, np.array(case["y"])[::3] - 0.002, setup
    )
    np.testing.assert_allclose(
        pred, np.asarray(ref["pred"], dtype=float), atol=2e-10, rtol=2e-10
    )
    if setup.offset is not None:
        np.testing.assert_allclose(setup.offset, ref["offset"], atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(off, ref["pred_offset"], atol=2e-10, rtol=2e-10)
    else:
        assert ref["offset"] == {} or ref["offset"] is None
    grid, distance, *_ = boundary_grid(process_boundary(case["boundary"]), case["nmax"])
    np.testing.assert_allclose(distance, ref["d"], atol=2e-14)
    assert pde_matrix(grid, setup.h).shape[0] == setup.grid_basis.shape[0]


def test_soap_fit_predict_identity_and_exterior():
    case = _case("hole", "so", "cc")
    setup = build_soap_basis(**case)
    pred, _ = predict_soap_basis(case["x"], case["y"], setup)
    np.testing.assert_array_equal(pred, setup.basis_train)
    pred, _ = predict_soap_basis([0, 5, -0.5], [0, 5, -0.5], setup)
    assert np.isnan(pred[:2]).all()
    assert np.isfinite(pred[2]).all()


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
def test_soap_rejects_outside_training_data(basis):
    case = _case("square", basis, "cc")
    case["x"][0] = 5
    with pytest.raises(ValueError, match="outside soap boundary"):
        build_soap_basis(**case)


def test_soap_rejects_boundary_knot():
    case = _case("square", "sw", "cc")
    case["knots"][0] = [0.999, 0.6]
    with pytest.raises(ValueError, match="on or outside boundary"):
        build_soap_basis(**case)


@pytest.mark.parametrize(
    "variant",
    [
        "default_k",
        "small_k",
        "cp_linear_ridge",
        "cp_single_order",
        "cp_high_order",
        "different_loop_k",
        "coarse_grid",
        "fine_grid",
    ],
)
def test_soap_options_mgcv(variant):
    case = _case("hole", "so", "cc")
    if variant == "default_k":
        case["k"] = -1
        case["boundary_spec"] = None
    elif variant == "small_k":
        case["k"] = 3
    elif variant.startswith("cp_"):
        case["boundary_spec"]["bs"] = "cp"
        case["boundary_spec"]["m"] = {
            "cp_linear_ridge": [0, 0],
            "cp_single_order": [1],
            "cp_high_order": [3, 3],
        }[variant]
    elif variant == "different_loop_k":
        case["k"] = [5, 8]
    else:
        case["nmax"] = 18 if variant == "coarse_grid" else 48
    if variant == "small_k":
        with pytest.warns(UserWarning, match="increased to minimum"):
            _assert_reference(case)
    else:
        _assert_reference(case)


def test_soap_components_and_harmonic_pde():
    case = _case("circle", "so", "cc")
    so = build_soap_basis(**case)
    sf = build_soap_basis(**dict(case, basis="sf"))
    sw = build_soap_basis(**dict(case, basis="sw"))
    np.testing.assert_allclose(
        so.basis_train, np.column_stack([sf.basis_train, sw.basis_train]), atol=1e-13
    )
    # The free boundary films solve the homogeneous interior Laplace equation.
    rhs = pde_matrix(sf.G, sf.h) @ sf.grid_basis
    interior = sf.G[sf.G > 0]
    np.testing.assert_allclose(rhs[interior], 0, atol=1e-11)
    np.testing.assert_allclose(sf.grid_basis.sum(axis=1), 1, atol=2e-12)
    assert so.need_constraint and sf.need_constraint and not sw.need_constraint


def test_soap_fixed_boundary_removes_centering():
    for basis in ("so", "sf", "sw"):
        setup = build_soap_basis(**_case("fixed", basis, "cc"))
        assert not setup.need_constraint
        assert (setup.offset is None) == (basis == "sw")
