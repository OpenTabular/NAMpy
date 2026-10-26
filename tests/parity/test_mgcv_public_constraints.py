"""Public raw/absorbed construction and custom C against actual mgcv smoothCon/gam."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.compiler.construct import smooth_con
from nampy.gam.specs import TermSpec, build_smooth_spec
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)


def reference(bs, C, optimized, select=False):
    code = r"""
library(mgcv); library(jsonlite)
a <- fromJSON(commandArgs(TRUE)[1]); d <- as.data.frame(a$data)
C <- if (is.null(a$C)) NULL else as.matrix(a$C)
smooth.construct.custom.smooth.spec <- function(object,data,knots) {
  object$bs <- a$bs
  class(object) <- paste0(a$bs,".smooth.spec")
  z <- smooth.construct(object,data,knots)
  if(!is.null(C)) z$C <- C
  z
}
spec <- s(x,bs="custom",k=7)
r <- smoothCon(spec,d,absorb.cons=FALSE,null.space.penalty=a$select)[[1]]
c <- smoothCon(spec,d,absorb.cons=TRUE,null.space.penalty=a$select)[[1]]
f <- if(a$optimized) y~s(x,bs="custom",k=7) else y~s(x,bs="custom",k=7,sp=.7)
b <- gam(f,data=d,method="REML",select=a$select)
nd <- data.frame(x=seq(-.1,1.1,length.out=25)); p <- predict(b,nd,se.fit=TRUE)
ans <- list(raw=r$X,C=r$C,Sraw=r$S[[1]],X=c$X,S=c$S[[1]],Sall=c$S,rank=c$rank,null_dim=c$null.space.dim,Xp=PredictMat(c,nd),fit=p$fit,se=p$se.fit,Vp=b$Vp,coef=coef(b),sp=b$sp,edf=sum(b$edf))
write_json(ans,commandArgs(TRUE)[2],digits=17,auto_unbox=TRUE)
"""
    x = np.linspace(0, 1, 85)
    dat = pd.DataFrame(
        {"x": x, "y": np.sin(x * 5) + 0.15 * np.cos(np.arange(85) * 1.7)}
    )
    width = 6 if bs == "cc" else 7
    payload = {
        "bs": bs,
        "C": None if C is None else C.tolist(),
        "width": width,
        "optimized": optimized,
        "select": select,
        "data": dat.to_dict("list"),
    }
    key = reference_key("public_constraints", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as folder:
            src, inp, out = [Path(folder) / n for n in ("ref.R", "in.json", "out.json")]
            src.write_text(code)
            inp.write_text(json.dumps(payload))
            run = subprocess.run(
                ["Rscript", str(src), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            save_reference("mgcv_public_constraints", key, json.loads(out.read_text()))
    return dat, load_reference("mgcv_public_constraints", key)


def _assert_cs_penalty_invariants(actual, expected):
    # smooth.r assigns unequal .1/.01 shrinkage floors to the two numerically
    # zero eigenvectors. Their orientation is not unique across LAPACK builds;
    # smoothCon's norm-based scaling consequently need not be identical either.
    # Check the complete normalized spectrum and the identifiable range operator.
    operators = []
    spectra = []
    for penalty in (actual, expected):
        values, vectors = np.linalg.eigh(np.asarray(penalty, dtype=float))
        values = values / values[-1]
        spectra.append(values)
        operators.append((vectors[:, 2:] * values[2:]) @ vectors[:, 2:].T)
    np.testing.assert_allclose(spectra[0], spectra[1], atol=2e-9)
    np.testing.assert_allclose(operators[0], operators[1], atol=2e-9)


def _assert_absorbed_penalty(bs, con, raw, reference, atol):
    actual = con.penalty_specs[0].matrix
    if bs != "cs":
        np.testing.assert_allclose(actual, reference["S"], atol=atol)
        return
    _assert_cs_penalty_invariants(raw.penalty_specs[0].matrix, reference["Sraw"])
    transform = con.compiled_term.predict_coefficient_map
    np.testing.assert_allclose(
        actual, transform.T @ raw.penalty_specs[0].matrix @ transform, atol=atol
    )
    reference_transform = np.linalg.lstsq(reference["raw"], reference["X"], rcond=None)[
        0
    ]
    np.testing.assert_allclose(
        reference["S"],
        reference_transform.T @ np.asarray(reference["Sraw"]) @ reference_transform,
        atol=atol,
    )


def spec(bs):
    return TermSpec(
        kind="smooth",
        features=("x",),
        label="s(x)",
        smooth_spec=build_smooth_spec(special="s", bs=bs, k=7),
    )


@pytest.mark.parametrize("bs", ["cr", "cs", "cc", "ps", "cp"])
@pytest.mark.parametrize("custom", [False, True])
def test_raw_and_absorbed_reference(bs, custom):
    width = 6 if bs == "cc" else 7
    C = np.vstack([np.ones(width), np.arange(width)]) if custom else None
    dat, r = reference(bs, C, False)
    raw = smooth_con(
        spec(bs), dat[["x"]].to_numpy(), ["x"], absorb_cons=False, constraints=C
    )
    con = smooth_con(spec(bs), dat[["x"]].to_numpy(), ["x"], constraints=C)
    np.testing.assert_allclose(raw.compiled_term.basis_train, r["raw"], atol=2e-9)
    if bs != "cs":
        np.testing.assert_allclose(raw.penalty_specs[0].matrix, r["Sraw"], atol=2e-9)
    np.testing.assert_allclose(raw.fit_constraint_operator, r["C"], atol=2e-9)
    np.testing.assert_allclose(con.compiled_term.basis_train, r["X"], atol=2e-9)
    _assert_absorbed_penalty(bs, con, raw, r, atol=2e-9)
    assert con.penalty_specs[0].rank == r["rank"]
    assert con.penalty_specs[0].null_space_dim == r["null_dim"]
    xp = np.linspace(-0.1, 1.1, 25)[:, None]
    np.testing.assert_allclose(con.compiled_term.predict_matrix(xp), r["Xp"], atol=2e-9)
    T = con.compiled_term.predict_coefficient_map
    np.testing.assert_allclose(raw.fit_constraint_operator @ T, 0, atol=2e-12)


@pytest.mark.parametrize("bs", ["cr", "cs", "ps", "cp", "cc"])
@pytest.mark.parametrize("optimized", [False, True])
def test_custom_constraint_fit_prediction_covariance(bs, optimized, tmp_path):
    width = 6 if bs == "cc" else 7
    C = np.vstack([np.ones(width), np.arange(width)])
    dat, r = reference(bs, C, optimized)
    model = GAM(
        formula=f"y ~ s(x,bs='{bs}',k=7)",
        constraints={1: C},
        optimize_smoothing=optimized,
        smoothing_method="REML",
        smoothing_params=None if optimized else [0.7],
    ).fit(data=dat)
    xp = pd.DataFrame({"x": np.linspace(-0.1, 1.1, 25)})
    pred = model.predict(xp, return_se=True)
    np.testing.assert_allclose(pred[0], r["fit"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(pred[1], r["se"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(model.vcov(), r["Vp"], atol=2e-5, rtol=2e-5)
    file = tmp_path / "constraint.pkl"
    model.save_model(file)
    loaded = GAM.load_model(file)
    np.testing.assert_allclose(loaded.predict(xp), pred[0])


@pytest.mark.parametrize(
    "C", [np.ones((2, 7)), np.ones((1, 6)), np.full((1, 7), np.nan), np.eye(7)]
)
def test_invalid_constraints(C):
    with pytest.raises(ValueError, match="Constraint"):
        smooth_con(spec("cr"), np.linspace(0, 1, 30)[:, None], ["x"], constraints=C)


def test_empty_constraints_and_selector_validation():
    X = np.linspace(0, 1, 30)[:, None]
    con = smooth_con(spec("cr"), X, ["x"], constraints=np.empty((0, 7)))
    assert con.compiled_term.basis_train.shape == (30, 7)
    with pytest.raises(ValueError, match="selector"):
        GAM(constraints={2: np.ones(7)}, optimize_smoothing=False).fit(
            X, np.sin(X[:, 0])
        )
    with pytest.raises(NotImplementedError, match="Explicit/raw"):
        smooth_con(spec("gp"), X, ["x"])


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e200, 1e300])
def test_constraint_scale_invariance(scale):
    X = np.linspace(0, 1, 45)[:, None]
    baseline = smooth_con(spec("cr"), X, ["x"], constraints=np.ones(7))
    scaled = smooth_con(spec("cr"), X, ["x"], constraints=np.ones(7) * scale)
    np.testing.assert_allclose(
        scaled.compiled_term.basis_train, baseline.compiled_term.basis_train, atol=1e-14
    )
    np.testing.assert_allclose(
        np.ones(7) @ scaled.compiled_term.predict_coefficient_map, 0, atol=1e-14
    )
    np.testing.assert_array_equal(
        scaled.fit_constraint_operator, np.ones((1, 7)) * scale
    )


@pytest.mark.parametrize("bs", ["ps", "cp"])
def test_constraint_prediction_derivatives_and_equivalent_rows(bs):
    X = np.linspace(0, 1, 70)[:, None]
    C = np.vstack([np.ones(7), np.arange(7)])
    con = smooth_con(spec(bs), X, ["x"], constraints=C)
    other = smooth_con(
        spec(bs), X, ["x"], constraints=np.array([[2.0, 1.0], [1.0, 3.0]]) @ C
    )
    A = con.compiled_term
    B = other.compiled_term
    # Rotation of independent constraint rows preserves the function space.
    np.testing.assert_allclose(
        A.basis_train @ np.linalg.pinv(A.basis_train),
        B.basis_train @ np.linalg.pinv(B.basis_train),
        atol=1e-12,
    )
    xp = np.linspace(0.1, 0.9, 15)[:, None]
    h = 1e-5
    finite = (A.predict_matrix(xp + h) - A.predict_matrix(xp - h)) / (2 * h)
    np.testing.assert_allclose(A.derivative_matrix(xp, order=1), finite, atol=1e-7)


def test_constraint_selectors_transaction_and_adapter():
    from sklearn.base import clone

    from nampy.models import GAMRegressor

    X = pd.DataFrame({"x": np.linspace(0, 1, 45)})
    y = np.sin(X.x * 5)
    estimator = GAMRegressor(
        formula="y ~ s(x,bs='cr',k=7)",
        constraints={1: np.ones(7)},
        optimize_smoothing=False,
    )
    model = clone(estimator).fit(X, y)
    pred = model.predict(X)
    model.gam_.hparams["constraints"] = {99: np.ones(7)}
    with pytest.raises(ValueError, match="selector"):
        model.gam_.fit(data=X.assign(y=y))
    np.testing.assert_allclose(model.predict(X), pred)
    label = next(
        t.label
        for p in model.gam_.predictor_specs
        for t in p.terms
        if t.kind == "smooth"
    )
    model.gam_.hparams["constraints"] = {"s(x)": np.ones(7)}
    model.gam_.fit(data=X.assign(y=y))
    np.testing.assert_allclose(model.predict(X), pred)
    model.gam_.hparams["constraints"] = {1: np.ones(7), label: np.ones(7)}
    with pytest.raises(ValueError, match="Multiple constraints"):
        model.gam_.fit(data=X.assign(y=y))


def test_constraint_unsupported_compositions_and_cancellation():
    from dataclasses import replace

    X = np.linspace(0, 1, 30)[:, None]
    with pytest.raises(NotImplementedError, match="linked id"):
        smooth_con(replace(spec("cr"), smoothing_id="shared"), X, ["x"])
    C = np.ones((2, 7))
    C[1, 0] = -1
    C[1, 1] = 2
    with pytest.raises(ValueError, match="cancelling"):
        smooth_con(spec("cr"), X, ["x"], constraints=C)


@pytest.mark.parametrize("bs", ["cr", "ps", "cs"])
@pytest.mark.parametrize("rows", [1, 2])
def test_selection_penalty_after_constraint_absorption(bs, rows):
    from dataclasses import replace

    C = np.vstack([np.ones(7), np.arange(7)])[:rows]
    dat, r = reference(bs, C, True, select=True)
    term = spec(bs)
    term = replace(
        term, smooth_spec=build_smooth_spec(special="s", bs=bs, k=7, select=True)
    )
    con = smooth_con(term, dat[["x"]].to_numpy(), ["x"], constraints=C)
    assert len(con.penalty_specs) == len(r["Sall"])
    if bs == "cs":
        raw = smooth_con(
            spec(bs), dat[["x"]].to_numpy(), ["x"], absorb_cons=False, constraints=C
        )
        _assert_absorbed_penalty(bs, con, raw, r, atol=1e-10)
    else:
        for p, S in zip(con.penalty_specs, r["Sall"], strict=True):
            np.testing.assert_allclose(p.matrix, S, atol=1e-10)
    model = GAM(
        formula=f"y ~ s(x,bs='{bs}',k=7)",
        constraints={1: C},
        select=True,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=dat)
    fit, se = model.predict(
        pd.DataFrame({"x": np.linspace(-0.1, 1.1, 25)}), return_se=True
    )
    np.testing.assert_allclose(fit, r["fit"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(se, r["se"], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("bs", ["cr", "ps"])
def test_localized_sparse_constraints_reference(bs):
    C = np.zeros((1, 7))
    C[0, 1:4] = [1, 2, 3]
    dat, r = reference(bs, C, False)
    con = smooth_con(spec(bs), dat[["x"]].to_numpy(), ["x"], constraints=C)
    np.testing.assert_allclose(con.compiled_term.basis_train, r["X"], atol=1e-10)
    np.testing.assert_allclose(con.penalty_specs[0].matrix, r["S"], atol=1e-10)
    np.testing.assert_allclose(
        C @ con.compiled_term.predict_coefficient_map, 0, atol=1e-12
    )


@pytest.mark.parametrize("options", ["by=z", "pc=0.3", "id=1"])
def test_constraint_unsupported_compositions_in_fitting(options):
    x = np.linspace(0, 1, 35)
    data = pd.DataFrame({"x": x, "z": x + 0.2, "y": np.sin(x * 4)})
    with pytest.raises(NotImplementedError, match="without by, pc, or linked id"):
        GAM(formula=f"y ~ s(x,bs='cr',k=7,{options})", constraints={1: np.ones(7)}).fit(
            data=data
        )
