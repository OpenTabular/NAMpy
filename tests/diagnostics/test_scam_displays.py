"""Actual plot.scam/vis.scam/scam.check numeric reference displays."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.scam_plots import (
    prepare_plot_scam_data,
    prepare_scam_check_data,
    prepare_vis_scam_data,
)
from nampy.gam.splines.shape import (
    build_bivariate_shape_setup,
    build_scop_univariate_setup,
)
from tests.mgcv_parity_utils import _build_r_command, _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.scam.scam_reference_utils import _require_local_scam

pytestmark = [pytest.mark.surface_regression]


def _case(
    code="mpi",
    family="gaussian",
    weighted=False,
    se=True,
    se_with_mean=False,
    custom=False,
    mixed=False,
):
    rng = np.random.default_rng(924)
    data = pd.DataFrame(
        {"x": np.sort(rng.uniform(-1.7, 2.1, 96)), "z": rng.uniform(-2, 1.5, 96)}
    )
    by = code.endswith("by")
    bivariate = code.startswith(("te", "ti"))
    data["v"] = 0.5 + 0.2 * (data.x + 1.7)
    if bivariate:
        setup = build_bivariate_shape_setup(
            data.x.to_numpy(),
            data.z.to_numpy(),
            basis_code=code,
            bs_dim=(5, 6),
            spline_order=(2, 1),
        )
        formula = f"y ~ s(x,z,bs='{code}',k=[5,6],m=[2,1])"
        sp = [0.45, 0.75]
        term_start = np.linspace(-2.2, -1.15, setup.n_coef)
        if code in {"tecvcv", "tecxcx", "tecxcv"}:
            sp = [1e-4, 1e-4]
            term_start = np.linspace(-1.25, -0.45, setup.n_coef)
    else:
        setup = build_scop_univariate_setup(
            data.x.to_numpy(), basis_code=code, bs_dim=8, spline_order=2
        )
        formula = f"y ~ s(x,bs='{code}',k=8,m=2" + (",by=v)" if by else ")")
        sp = [0.42]
        term_start = np.linspace(-1.35, -0.5, setup.n_coef)
    term_start[~setup.positive_mask] = np.linspace(
        -0.12, 0.12, np.count_nonzero(~setup.positive_mask)
    )
    start = np.r_[0.31, term_start]
    basis = setup.basis_train * (data.v.to_numpy()[:, None] if by else 1)
    eta = 0.31 + basis @ np.where(setup.positive_mask, np.exp(term_start), term_start)
    data["y"] = (
        eta + rng.normal(0, 0.025, len(data))
        if family == "gaussian"
        else rng.poisson(np.exp(eta))
    )
    # Keep response inputs identical to those used for the committed R fits.
    # BLAS reductions in basis @ coefficients can cross hash rounding bounds.
    input_path = (
        Path(__file__).parents[1]
        / "reference_fixtures/scam_displays/input_responses.json"
    )
    data["y"] = json.loads(input_path.read_text())[f"{code}/{family}"]
    data["weight"] = rng.uniform(0.5, 2, len(data)) if weighted else 1.0
    if mixed:
        formula += " + s(z,bs='cr',k=6)"
        start = np.r_[start, np.zeros(5)]
        sp = [*sp, 0.6]
    result = {
        "code": code,
        "family": family,
        "data": data.to_dict("list"),
        "formula": formula,
        "start": start.tolist(),
        "sp": sp,
        "se": se,
        "se_with_mean": se_with_mean,
        "custom": custom,
        "weighted": weighted,
        "bivariate": bivariate,
    }

    if mixed:
        result["mixed"] = True
    return result


_R = r"""
suppressPackageStartupMessages(library(scam));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
b<-scam(as.formula(p$formula),data=d,family=get(p$family)(),sp=p$sp,start=p$start,weights=d$weight,
optimizer=c('bfgs','newton'),control=list(maxit=200,devtol.fit=1e-7,steptol.fit=1e-7))
pdf(file=NULL)
r<-if(p$custom) seq(-.1,.1,length=nrow(d)) else TRUE
out<-plot(b,n=41,n2=12,se=p$se,seWithMean=p$se_with_mean,residuals=r,too.far=.1)
pack<-function(P) list(x=unname(P$x),y=unname(P$y),fit=as.numeric(P$fit),se=unname(P$se),resid=unname(P$p.resid),exclude=unname(P$exclude))
resid<-residuals(b,type='deviance');histogram<-hist(resid,plot=FALSE)
quantile_residuals<-residuals(b,type='rquantile',setseed=42)
qqold<-qqnorm(resid,plot.it=FALSE)
ans<-list(plots=lapply(out,pack),weights=b$weights,
check=list(resid=unname(resid),eta=unname(b$linear.predictors),fitted=unname(fitted(b)),response=unname(b$y),
breaks=histogram$breaks,counts=histogram$counts,qq=unname(qqold$x),rquantile=unname(quantile_residuals)),version=as.character(packageVersion('scam')))
if(p$bivariate) {
  # vis.scam explicitly delegates its grid predictions to predict.scam.
  grid<-expand.grid(x=seq(min(d$x),max(d$x),length=12),z=seq(min(d$z),max(d$z),length=12))
  pr<-predict(b,grid,se.fit=TRUE)
  ans$vis<-list(fit=unname(pr$fit),se=unname(pr$se.fit))
  invisible(vis.scam(b,view=c('x','z'),n.grid=12,too.far=0))
}
if(isTRUE(p$qq)) {
  counter<-0L
  qfun<-scam:::qq.scam
  env<-new.env(parent=environment(qfun))
  env$sample<-function(x,size,...) {counter<<-counter+1L;(p$perms[counter,]+.5)/nrow(d)}
  environment(qfun)<-env
  ans$qq<-as.numeric(qfun(b,s.rep=nrow(p$perms)))
}
dev.off();write_json(ans,a[2],digits=17,auto_unbox=FALSE,na='null',null='null')
"""


def _reference(case):
    key = reference_key("scam-displays-v1", case, normalize_floats=True)
    result = load_reference("scam_displays", key)
    if result is not None:
        return result
    library = _require_local_scam()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "ref.R"
        inp = root / "in.json"
        out = root / "out.json"
        script.write_text(_R)
        formula = (
            _normalize_python_formula_text(case["formula"])
            .replace("'mpiby'", "'mpiBy'")
            .replace('"mpiby"', '"mpiBy"')
            .replace("'mpdby'", "'mpdBy'")
            .replace('"mpdby"', '"mpdBy"')
        )
        inp.write_text(json.dumps({**case, "formula": formula}))
        proc = subprocess.run(
            _build_r_command(script, str(inp), str(out)),
            capture_output=True,
            text=True,
            env={**os.environ, "R_LIBS_USER": library},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(out.read_text())
    save_reference("scam_displays", key, result)
    return result


def _fit(case):
    frame = pd.DataFrame(case["data"])
    return GAM(
        formula=case["formula"],
        family=case["family"],
        smoothing_method="fixed",
        optimize_smoothing=False,
        smoothing_params=case["sp"],
        start=case["start"],
        max_irls_iter=200,
        irls_tol=1e-7,
    ).fit(data=frame, sample_weight=frame.weight)


CASES = [
    {"code": c}
    for c in (
        "mpi",
        "mpd",
        "po",
        "ipo",
        "dpo",
        "mpiby",
        "mpdby",
        "miso",
        "mifo",
        "cv",
        "cx",
        "mdcv",
        "mdcx",
        "micv",
        "micx",
        "cpop",
        "tedmi",
        "tesmi1",
        "tedmd",
        "temicx",
        "temicv",
        "tedecv",
        "tedecx",
        "tecvcv",
        "tecxcx",
        "tecxcv",
        "tescv",
        "tescx",
        "tesmd1",
        "tesmi2",
        "tesmd2",
        "tismi",
        "tismd",
    )
]
CASES += [
    {"weighted": True},
    {"family": "poisson"},
    {"se": 3.0},
    {"se_with_mean": True},
    {"custom": True},
    {"mixed": True, "se_with_mean": True},
    {"se": False},
]


@pytest.mark.parametrize(
    "options", CASES, ids=lambda p: "-".join(f"{k}={v}" for k, v in p.items())
)
def test_scam_plot_fit_se_and_partial_residuals_match_upstream(options):
    case = _case(**options)
    expected = _reference(case)
    model = _fit(case)
    residuals = np.linspace(-0.1, 0.1, len(model.y_)) if case["custom"] else True
    actual = prepare_plot_scam_data(
        model,
        n=41,
        n2=12,
        se=case["se"],
        se_with_mean=case["se_with_mean"],
        residuals=residuals,
        too_far=0.1,
    )
    for plot, ref in zip(actual["pd"], expected["plots"], strict=True):
        np.testing.assert_allclose(
            plot["fit"], np.asarray(ref["fit"], dtype=float), atol=3e-6, rtol=3e-6
        )
        if case["se"]:
            np.testing.assert_allclose(
                plot["se"],
                np.asarray(ref["se"], dtype=float).reshape(-1),
                atol=4e-6,
                rtol=4e-6,
            )
        else:
            # R retains P$se as a capability flag when no widths are computed.
            assert plot["se"] is False
            assert not plot["plot_ci"]
        np.testing.assert_allclose(plot["p_resid"], ref["resid"], atol=3e-6, rtol=3e-6)
    if case["bivariate"]:
        vis = prepare_vis_scam_data(model, view=["x", "z"], n_grid=12)
        np.testing.assert_allclose(
            vis["fit"], expected["vis"]["fit"], atol=3e-6, rtol=3e-6
        )
        np.testing.assert_allclose(
            vis["se_fit"], expected["vis"]["se"], atol=4e-6, rtol=4e-6
        )


def test_scam_check_old_style_and_continuous_quantile_residuals():
    case = _case(weighted=True)
    expected = _reference(case)["check"]
    model = _fit(case)
    actual = prepare_scam_check_data(model, old_style=True)
    for field, key in [("x", "eta"), ("y", "resid")]:
        np.testing.assert_allclose(
            actual["residual_vs_linear"][field], expected[key], atol=3e-7
        )
    np.testing.assert_allclose(
        actual["histogram"]["breaks"], expected["breaks"], atol=1e-12
    )
    np.testing.assert_array_equal(actual["histogram"]["counts"], expected["counts"])
    quantile = prepare_scam_check_data(model, type="rquantile", seed=42)
    np.testing.assert_allclose(
        quantile["residual_vs_linear"]["y"], expected["rquantile"], atol=4e-6, rtol=4e-6
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_qq_scam_quantile_reference_matches_with_identical_uniform_permutations(family):
    from nampy.gam.diagnostics.scam_plots import prepare_qq_scam_data

    case = _case(family=family)
    rng = np.random.default_rng(42)
    permutations = np.stack([rng.permutation(len(case["data"]["y"])) for _ in range(3)])
    case.update(qq=True, perms=permutations.tolist())
    expected = _reference(case)
    model = _fit(case)
    actual = prepare_qq_scam_data(model, s_rep=3, quantile_permutations=permutations)
    np.testing.assert_allclose(
        actual["theoretical"], expected["qq"], atol=4e-6, rtol=4e-6
    )


def test_wrong_length_partial_residuals_warn_and_reuse_weighted_working_residuals():
    model = _fit(_case(weighted=True))
    expected = prepare_plot_scam_data(model, residuals=True, n=41)
    with pytest.warns(UserWarning, match="wrong length"):
        actual = prepare_plot_scam_data(model, residuals=[1.0, 2.0], n=41)
    np.testing.assert_allclose(actual["pd"][0]["p_resid"], expected["pd"][0]["p_resid"])


def test_scam_preparation_preserves_fitted_state_and_negative_se_clamps_to_zero():
    model = _fit(_case())
    from nampy.gam.model_state import _coef_full

    coefficients = _coef_full(model).copy()
    covariance = model.vcov().copy()
    predictions = model.predict().copy()
    prepared = prepare_plot_scam_data(model, residuals=True, se=-2, n=41)
    np.testing.assert_array_equal(prepared["pd"][0]["se"], 0.0)
    np.testing.assert_array_equal(_coef_full(model), coefficients)
    np.testing.assert_array_equal(model.vcov(), covariance)
    np.testing.assert_array_equal(model.predict(), predictions)


@pytest.fixture
def pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


@pytest.mark.parametrize("code", ["mpi", "dpo", "tedmi"])
def test_public_plot_dispatches_scam_data_with_render_options(code, pyplot):
    model = _fit(_case(code=code))
    options = {
        "n": 41,
        "n2": 12,
        "residuals": True,
        "se": 3.0,
        "shift": 0.2,
        "too_far": 0.1,
        "main": "SCAM partial effect",
        "rug": False,
    }
    expected = prepare_plot_scam_data(model, **options)
    actual = model.plot(**options, figsize=(6, 4), shade_col="silver", pages=1)
    assert actual["figures"]
    for field in ["fit", "se", "se_fit", "p_resid"]:
        np.testing.assert_allclose(actual["pd"][0][field], expected["pd"][0][field])
    for figure in actual["figures"]:
        figure.canvas.draw()


def test_public_scam_qq_dispatch_with_quantile_permutations(pyplot):
    from nampy.gam.diagnostics.scam_plots import prepare_qq_scam_data

    model = _fit(_case(family="poisson"))
    rng = np.random.default_rng(14)
    permutations = np.stack([rng.permutation(len(model.y_)) for _ in range(3)])
    options = {"s_rep": 3, "quantile_permutations": permutations}
    expected = prepare_qq_scam_data(model, **options)
    actual = model.qq(**options, figsize=(5, 4))
    for field in ["theoretical", "residuals"]:
        np.testing.assert_allclose(actual[field], expected[field])
    assert len(actual["figures"]) == 1
    actual["figures"][0].canvas.draw()


@pytest.mark.parametrize("type", ["deviance", "rquantile"])
def test_public_scam_check_dispatch_and_quantile_residuals(type, pyplot):
    model = _fit(_case(weighted=True))
    options = {"type": type, "seed": 42, "old_style": True}
    expected = prepare_scam_check_data(model, **options)
    actual = model.plot_check(**options, figsize=(7, 5))
    assert actual["backend"] == "scam"
    for panel in ["residual_vs_linear", "response_vs_fitted"]:
        for field in ["x", "y"]:
            np.testing.assert_allclose(actual[panel][field], expected[panel][field])
    np.testing.assert_array_equal(
        actual["histogram"]["counts"], expected["histogram"]["counts"]
    )
    assert len(actual["figures"]) == 1
    actual["figures"][0].canvas.draw()


def test_public_scam_vis_returns_full_model_surface(pyplot):
    model = _fit(_case(code="tedmi"))
    options = {"view": ["x", "z"], "n_grid": 12, "se": 2.0}
    expected = prepare_vis_scam_data(model, **options)
    with pytest.warns(UserWarning, match="No contour display with errors"):
        actual = model.vis(**options, plot_type="contour", figsize=(6, 4))
    for field in ["fit", "se_fit"]:
        np.testing.assert_allclose(actual[field], expected[field])
    partial = prepare_plot_scam_data(model, n2=12, too_far=0)["pd"][0]["fit"]
    assert not np.allclose(partial, actual["fit"])
    assert actual["figures"]
    for figure in actual["figures"]:
        figure.canvas.draw()
