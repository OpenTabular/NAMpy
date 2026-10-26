"""Static mgcv parametricPenalty setup, fitting and inference parity."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.summary import summary_gam
from nampy.gam.model_state import _coef_full, _penalty_blocks_seq
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)


def _reference(payload, extra_term=False):
    code = r"""
d <- as.data.frame(a$data[c("x","u","y","f")]); d$f <- factor(d$f)
if(a$term=="B") d$B <- I(do.call(cbind,a$data[c("x","u","v")]))
P <- lapply(a$penalties, function(S) do.call(rbind,lapply(S,unlist)))
P$sp <- unlist(a$sp)
if(!is.null(a$L)) P$L <- do.call(rbind,lapply(a$L,unlist))
pp <- setNames(list(P),a$term)
b <- gam(as.formula(a$formula),data=d,family=if(a$family=="gaussian") gaussian() else poisson(),paraPen=pp,method="REML")
G <- gam(as.formula(a$formula),data=d,family=if(a$family=="gaussian") gaussian() else poisson(),paraPen=pp,method="REML",fit=FALSE)
p <- predict(b,se.fit=TRUE)
ans <- list(coef=unname(coef(b)),fit=as.numeric(p$fit),se=as.numeric(p$se.fit),Vp=unname(b$Vp),edf=sum(b$edf),scale=b$sig2,sp=unname(b$sp),aic=AIC(b),p_table=unname(summary(b)$p.table),pterms=unname(summary(b)$pTerms.table),X=unname(G$X),rank=G$rank,off=G$off,S=G$S)
"""
    if extra_term:
        code = code.replace(
            "pp <- setNames(list(P),a$term)",
            "pp <- setNames(list(P),a$term); pp$x <- list(matrix(2,1,1),sp=.3)",
        )
    key = reference_key("parametric_penalties", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            src, inp, out = [
                Path(directory) / n for n in ("ref.R", "in.json", "out.json")
            ]
            src.write_text(
                "args <- commandArgs(TRUE)\nlibrary(mgcv)\nlibrary(jsonlite)\na <- fromJSON(args[1],simplifyVector=FALSE)\na$data <- lapply(a$data,unlist)\na$sp <- unlist(a$sp)\n"
                + code
                + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="null")\n'
            )
            inp.write_text(json.dumps(payload))
            proc = subprocess.run(
                ["Rscript", str(src), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            save_reference(
                "mgcv_parametric_penalties", key, json.loads(out.read_text())
            )
    return load_reference("mgcv_parametric_penalties", key)


def _data(family):
    rng = np.random.default_rng(347)
    x, u, v = rng.normal(size=(3, 160))
    f = np.array(["a", "b", "c", "d"] * 40)
    eta = 0.4 + 0.5 * x - 0.3 * u + 0.2 * v + np.tile([0, 0.1, -0.25, 0.3], 40)
    y = (
        eta + rng.normal(size=160) * 0.6
        if family == "gaussian"
        else rng.poisson(np.exp(eta))
    )
    return pd.DataFrame({"x": x, "u": u, "v": v, "f": f, "y": y})


def _case(
    term,
    family,
    optimized,
    mixed=False,
    shared=False,
    override_sp=None,
    no_intercept=False,
    extra_term=False,
):
    dat = _data(family)
    width = 1 if term == "x" else 3
    penalties = (
        [np.eye(width)]
        if width == 1
        else [np.diag([1.0, 0.0, 1.0]), np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0])]
    )
    L = np.ones((2, 1)) if shared else None
    sp = (
        [-1.0] * (1 if shared else len(penalties))
        if optimized
        else ([0.7] if shared else [0.7, 1.1][: len(penalties)])
    )
    if override_sp is not None:
        sp = override_sp
    formula = "y ~ " + term + (' + s(u,bs="cr",k=7,sp=.4)' if mixed else "")
    if extra_term:
        formula = formula.replace("y ~ ", "y ~ x + ")
    if no_intercept:
        formula += " - 1"
    payload = {
        "data": dat.to_dict(orient="list"),
        "term": term,
        "family": family,
        "formula": formula,
        "penalties": [s.tolist() for s in penalties],
        "sp": sp,
        "L": None if L is None else L.tolist(),
    }
    ref = _reference(payload, extra_term=extra_term)
    if term == "B":
        dat["B"] = list(dat[["x", "u", "v"]].to_numpy())
    pp = {term: {"penalties": penalties, "sp": sp}}
    if extra_term:
        pp["x"] = {"penalties": [np.array([[2.0]])], "sp": [0.3]}
    if L is not None:
        pp[term]["L"] = L
    model = GAM(
        formula=formula,
        family=family,
        optimize_smoothing=optimized,
        smoothing_method="REML",
        para_pen=pp,
    ).fit(data=dat)
    return model, ref


@pytest.mark.parametrize("term", ["x", "f", "B"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("optimized", [False, True])
def test_parametric_penalty_fit(term, family, optimized):
    model, ref = _case(term, family, optimized)
    _assert_fit(model, ref)


def _assert_fit(model, ref, *, covariance_invariant=False):
    fit, se = model.predict(type="link", return_se=True)
    actual = {
        "coef": _coef_full(model),
        "fit": fit,
        "se": se,
        "Vp": model.vcov(),
        "X": model.predict(type="lpmatrix"),
        "edf": model.fit_result().edf_total,
        "scale": model.fit_result().scale,
        "aic": model.aic(),
        "p_table": summary_gam(model).p_table.to_numpy(),
        "pterms": summary_gam(model)
        .pterms_table[["df", "wald_stat", "p_value"]]
        .to_numpy(),
    }
    if covariance_invariant:
        # B contains u, also in the smooth null space. The final smoothing
        # parameter is unidentified: compare covariance on the observable
        # predictor space, not covariance along aliased coefficient directions.
        X = actual["X"]
        actual["Vp"] = X @ actual["Vp"] @ X.T
        ref = dict(ref, Vp=X @ np.asarray(ref["Vp"]) @ X.T)
        actual.pop("p_table")
        actual.pop("pterms")
        actual.pop("aic")
    for name, value in actual.items():
        np.testing.assert_allclose(
            value, np.asarray(ref[name]), rtol=3e-5, atol=3e-6, err_msg=name
        )
    if not covariance_invariant:
        np.testing.assert_allclose(
            model.smoothing_params[~model.smoothing_fixed_mask_],
            np.atleast_1d(ref["sp"]),
            rtol=3e-4,
            atol=3e-6,
            err_msg="sp",
        )
    penalties = list(_penalty_blocks_seq(model))
    np.testing.assert_array_equal(
        [p.rank for p in penalties], np.atleast_1d(ref["rank"])
    )
    np.testing.assert_array_equal(
        [p.coef_slice.start + 1 + int(model.fit_intercept) for p in penalties],
        np.atleast_1d(ref["off"]),
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("optimized", [False, True])
def test_parametric_penalty_smooth_mix(family, optimized):
    model, ref = _case("f", family, optimized, mixed=True)
    _assert_fit(model, ref)


@pytest.mark.parametrize("optimized", [False, True])
def test_parametric_penalty_shared_L(optimized):
    model, ref = _case("B", "gaussian", optimized, shared=True)
    _assert_fit(model, ref)


@pytest.mark.parametrize(
    "spec,error,match",
    [
        ({"penalties": [np.eye(2)]}, ValueError, "wrong dimension"),
        ({"penalties": [np.eye(3)], "rank": [1, 2]}, ValueError, "rank.*length"),
        ({"penalties": [np.eye(3)], "rank": [1.5]}, ValueError, "finite integers"),
        ({"penalties": [np.eye(3)], "rank": [4]}, ValueError, "rank must"),
        ({"penalties": [np.eye(3)], "sp": [1, 2]}, ValueError, "sp.*dimension"),
        ({"penalties": [np.eye(3)], "L": [[1], [1]]}, ValueError, "L has wrong"),
        ({"penalties": [np.eye(3)], "L": [[0.5]]}, NotImplementedError, "one-hot"),
        ({"penalties": [np.full((3, 3), np.nan)]}, ValueError, "nonfinite"),
    ],
)
def test_invalid_parametric_penalty(spec, error, match):
    with pytest.raises(error, match=match):
        GAM(formula="y ~ f", para_pen={"f": spec}).fit(data=_data("gaussian"))


def test_parametric_penalty_supplied_rank_and_persistence(tmp_path):
    model, ref = _case("B", "gaussian", False)
    model.hparams["para_pen"]["B"]["rank"] = [2, 1]
    dat = _data("gaussian")
    dat["B"] = list(dat[["x", "u", "v"]].to_numpy())
    model.fit(data=dat)
    _assert_fit(model, ref)
    model.save_model(tmp_path / "penalized.pkl")
    restored = GAM.load_model(tmp_path / "penalized.pkl")
    _assert_fit(restored, ref)
    np.testing.assert_allclose(
        restored.predict(dat.iloc[:9]), model.predict(dat.iloc[:9])
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_parametric_penalty_partial_fixed(family):
    model, ref = _case("B", family, True, mixed=True, override_sp=[0.7, -1.0])
    _assert_fit(model, ref, covariance_invariant=True)
    np.testing.assert_array_equal(model.smoothing_fixed_mask_, [True, False, True])


@pytest.mark.parametrize("value", [0.7])
def test_parametric_penalty_without_intercept(value):
    model, ref = _case("B", "gaussian", False, override_sp=[value, 1.1], no_intercept=True)
    _assert_fit(model, ref)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_parametric_penalty_partial_fixed_identified(family):
    model, ref = _case("f", family, True, mixed=True, override_sp=[0.7, -1.0])
    _assert_fit(model, ref)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("optimized", [False, True])
def test_parametric_penalty_multiple_original_terms(family, optimized):
    model, ref = _case("f", family, optimized, mixed=True, extra_term=True)
    _assert_fit(model, ref)
    assert [p.label for p in _penalty_blocks_seq(model)][:3] == ["x", "f", "f"]


def test_parametric_penalty_zero_fixed():
    model, ref = _case("f", "gaussian", False, override_sp=[0.0, 1.1])
    _assert_fit(model, ref)


def test_parametric_penalty_unmatched_labels_follow_upstream():
    dat = _data("gaussian")
    plain = GAM(formula="y ~ x").fit(data=dat)
    penalized = GAM(
        formula="y ~ x",
        para_pen={"not_in_formula": [np.eye(3)], "(Intercept)": [np.eye(1)]},
    ).fit(data=dat)
    np.testing.assert_array_equal(plain.predict(), penalized.predict())
    assert len(_penalty_blocks_seq(penalized)) == 0


def test_parametric_penalty_multiformula_rejected():
    with pytest.raises(ValueError, match="multi-formula"):
        GAM(formula=["y ~ x", "~ 1"], family="gaulss", para_pen={"x": [np.eye(1)]}).fit(
            data=_data("gaussian")
        )
