"""Static constructor parity for mgcv's sc/scad (distinct from SCAM)."""

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.smooths.univariate.ad import AdaptiveSmoothTerm
from nampy.gam.smooths.univariate.bs import DerivativeBSplineTerm1D
from nampy.gam.smooths.univariate.sc import (
    ShapeConstrainedAdaptiveTerm,
    ShapeConstrainedBSplineTerm,
)
from nampy.gam.splines.univariate.bs import build_derivative_bspline_setup
from nampy.gam.splines.univariate.sc import sc_inequalities, scad_inequalities
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression

COMBINATIONS = [
    [v for v in combo if v]
    for combo in itertools.product(("", "c+", "c-"), ("", "m+", "m-"), ("", "+"))
] + [["c+", "c-", "m+", "m-", "+"], ["unknown"]]


@pytest.fixture(scope="module")
def upstream_constraints():
    key = reference_key("scasm_constructor_constraints_v1", COMBINATIONS)
    result = load_reference("mgcv_sc", key)
    if result is not None:
        return result
    source = (
        Path(__file__).resolve().parents[2] / "nampy_notes/upstreams/mgcv/R/scasm.r"
    )
    # Source only in the explicit fixture refresh workflow. Installed older mgcv
    # versions supply the stable bs/ad constructors used by this new R source.
    script = r"""
    suppressPackageStartupMessages(library(mgcv))
    library(jsonlite)
    args <- commandArgs(TRUE)
    env <- new.env(parent=asNamespace("mgcv"))
    sys.source(args[1], envir=env)
    combos <- fromJSON(args[2], simplifyVector=FALSE)
    pack <- function(a) if (is.null(a)) list(dim=NULL,data=NULL) else list(dim=dim(a), data=as.numeric(t(a)))
    result <- list()
    for (degree in c(2,3,4)) for (i in seq_along(combos)) {
      con <- unlist(combos[[i]])
      spec <- s(x, bs="sc", k=9, m=c(degree,1), xt=con)
      sm <- env$smooth.construct.sc.smooth.spec(spec,list(x=seq(0,1,length=31)),NULL)
      result[[paste(degree,i,sep="_")]] <- list(Ain=pack(sm$Ain),bin=sm$bin,
        bin0=sm$bin0,skip=!is.null(sm$C)&&nrow(sm$C)==0)
    }
    for (i in seq_along(combos)) {
      spec <- s(x,bs="scad",k=9,m=3,xt=unlist(combos[[i]]))
      sm <- env$smooth.construct.scad.smooth.spec(spec,list(x=seq(0,1,length=31)),NULL)
      result[[paste("ad",i,sep="_")]] <- list(Ain=pack(sm$Ain),bin=sm$bin,
        bin0=sm$bin0,skip=!is.null(sm$C)&&nrow(sm$C)==0)
    }
    cat(toJSON(result,auto_unbox=TRUE,digits=17,null="null"))
    """
    output = subprocess.run(
        ["Rscript", "-e", script, str(source), json.dumps(COMBINATIONS)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert output.returncode == 0, output.stderr
    result = json.loads(output.stdout)
    save_reference("mgcv_sc", key, result)
    return result


@pytest.mark.parametrize("index", range(len(COMBINATIONS)))
@pytest.mark.parametrize("degree", [2, 3, 4, "ad"])
def test_all_sc_shape_constraint_rows_match_upstream(
    upstream_constraints, index, degree
):
    con = COMBINATIONS[index]
    if degree == "ad":
        actual = scad_inequalities(9, con)
    else:
        setup = build_derivative_bspline_setup(
            np.linspace(0, 1, 31),
            feature_index=0,
            feature_name="x",
            bs_dim=9,
            m=(degree, 1),
        )
        actual = sc_inequalities(setup, con)
    expected = upstream_constraints[f"{degree}_{index + 1}"]
    # Empty R vectors become NULL in the first, unconstrained combination.
    matrix = expected["Ain"]
    if matrix["dim"] is None:
        assert actual.Ain.shape == (0, 9)
    else:
        wanted = np.asarray(matrix["data"]).reshape(matrix["dim"])
        np.testing.assert_allclose(actual.Ain, wanted, atol=2e-14, rtol=2e-14)
    for field in ("bin", "bin0"):
        wanted = expected[field]
        np.testing.assert_allclose(
            getattr(actual, field), [] if wanted is None else wanted
        )
    assert actual.skip_centering == expected["skip"]


@pytest.mark.parametrize("xt", [None, [], ["unknown"], {"ignored": "unknown"}])
@pytest.mark.parametrize("constraint_mode", ["never", "auto"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_unconstrained_runtime_is_exact_underlying_basis(xt, constraint_mode, adaptive):
    cls, base = (
        (ShapeConstrainedAdaptiveTerm, AdaptiveSmoothTerm)
        if adaptive
        else (ShapeConstrainedBSplineTerm, DerivativeBSplineTerm1D)
    )
    data = np.linspace(0, 1, 61)[:, None]
    options = {
        "k": 9,
        "m": 3 if adaptive else (3, 2),
        "constraint_mode": constraint_mode,
    }
    actual = cls("x", xt=xt, **options).fit(data, ["x"])
    expected = base("x", **options).fit(data, ["x"])
    np.testing.assert_array_equal(actual.basis_train, expected.basis_train)
    for a, b in zip(actual.penalties, expected.penalties, strict=True):
        np.testing.assert_array_equal(a, b)
    new = np.array([[-0.2], [0.3], [0.8], [1.2]])
    np.testing.assert_array_equal(
        actual.transform_new(new), expected.transform_new(new)
    )


@pytest.mark.parametrize("con", [c for c in COMBINATIONS if c and c != ["unknown"]])
@pytest.mark.parametrize(
    "cls", [ShapeConstrainedBSplineTerm, ShapeConstrainedAdaptiveTerm]
)
def test_shape_requests_cannot_silently_fit_unconstrained(con, cls):
    with pytest.raises(NotImplementedError, match="scasm"):
        cls("x", xt=con)


@pytest.mark.parametrize(
    "basis,base,m,sp",
    [("sc", "bs", "[3,2]", [0.2]), ("scad", "ad", "3", [0.2, 0.4, 0.6])],
)
@pytest.mark.parametrize("by", [False, True])
def test_public_formula_fit_matches_base(basis, base, m, sp, by):
    import pandas as pd

    from tests.mgcv_parity_utils import _fit_nampy_model_fixed_sp

    x = np.linspace(0, 1, 70)
    data = pd.DataFrame({"x": x, "z": 1 + x, "y": np.sin(6 * x) + 0.1 * np.cos(19 * x)})
    by_option = ",by=z" if by else ""
    formula = f'y ~ s(x,bs="{basis}",k=9,m={m}{by_option})'
    actual = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
    expected = _fit_nampy_model_fixed_sp(
        data, formula.replace(f'"{basis}"', f'"{base}"'), "gaussian", sp
    )
    np.testing.assert_allclose(
        actual.predict(data), expected.predict(data), atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("basis", ["sc", "scad"])
def test_public_formula_rejects_shape_constraints(basis):
    import pandas as pd

    from tests.mgcv_parity_utils import _fit_nampy_model_fixed_sp

    data = pd.DataFrame({"x": np.linspace(0, 1, 30), "y": np.linspace(0, 1, 30)})
    with pytest.raises(NotImplementedError, match="scasm"):
        _fit_nampy_model_fixed_sp(
            data, f'y ~ s(x,bs="{basis}",k=9,xt=["m+"])', "gaussian", [0.2]
        )
