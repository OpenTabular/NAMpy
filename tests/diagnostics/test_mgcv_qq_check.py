"""Static mgcv QQ/check diagnostics with shared permutations/response draws."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.diagnostics.check_plots import _r_histogram, prepare_gam_check_data
from nampy.gam.diagnostics.qq import _family_draw, _normal_qq, prepare_qq_gam_data
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression

_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=FALSE)
if(!is.null(p$hist)) {
 h<-hist(unlist(p$hist),plot=FALSE)
 write_json(unclass(h),a[2],digits=17,auto_unbox=TRUE);quit()
}
d<-as.data.frame(lapply(p$data,unlist))
fam<-switch(p$family,negbin=negbin(theta=3),betar=betar(theta=5),ocat=ocat(theta=c(1,1)),tw=tw(theta=1.5),get(p$family)())
f<-if(p$family %in% c('gaulss','gammals')) list(y~s(x,bs='cr',k=5),~1) else y~s(x,bs='cr',k=5)
b<-gam(f,data=d,weights=d$w,sp=.4,method='REML',family=fam)
perms<-do.call(rbind,lapply(p$permutations,unlist));draws<-do.call(rbind,lapply(p$draws,unlist))
if(p$rep==0) {
 b$family<-mgcv:::fix.family.qf(b$family);original<-b$family$qf;i<-0L
 b$family$qf<-function(p,mu,wt,scale) {i<<-i+1L;original((perms[i,]+.5)/length(p),mu,wt,scale)}
} else {
 i<-0L;b$family$rd<-function(mu,wt,scale) {i<<-i+1L;draws[i,]}
}
trace('qq.gam',where=asNamespace('mgcv'),print=FALSE,exit=quote({
 assign('.qq_snapshot',list(theoretical=Dq,residuals=sort(D),simulated_residuals=dm,lim=lim),envir=.GlobalEnv)
}))
pdf(a[3]);do.call(get('qq.gam',asNamespace('mgcv')),list(object=b,rep=p$rep,s.rep=3,level=p$level,type=p$type));dev.off()
r<-residuals(b,type=p$type)
h<-hist(r,plot=FALSE)
.qq_snapshot$histogram<-unclass(h)
.qq_snapshot$eta<-b$linear.predictors
.qq_snapshot$fitted<-if(inherits(b$family,"extended.family")) predict(b,type="response") else b$fitted.values
.qq_snapshot$y<-b$y
.qq_snapshot$raw_residuals<-r
write_json(.qq_snapshot,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key("qq-check-v1", case, normalize_floats=True)
    value = load_reference("mgcv_qq_check", key)
    if value is not None:
        return value
    with tempfile.TemporaryDirectory() as tmp:
        script, inp, out, pdf = [
            Path(tmp) / name for name in ["ref.R", "in.json", "out.json", "discard.pdf"]
        ]
        script.write_text(_SCRIPT)
        inp.write_text(json.dumps(case))
        result = subprocess.run(
            _build_r_command(script, str(inp), str(out), str(pdf)),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        value = json.loads(out.read_text())
    save_reference("mgcv_qq_check", key, value)
    return value


def _case(family="gaussian", type="deviance", rep=0, level=0.9):
    rng = np.random.default_rng(987)
    x = np.linspace(0.1, 1, 40)
    mu = np.exp(0.4 * np.sin(4 * x))
    w = np.tile([1.0, 2.0, 3.0, 1.0], 10)
    if family in {"gaussian", "gaulss"}:
        y = rng.normal(mu, 0.3 / np.sqrt(w))
        draws = rng.normal(mu, 0.35 / np.sqrt(w), (7, len(x)))
    elif family == "poisson":
        y = rng.poisson(mu)
        draws = rng.poisson(mu, (7, len(x)))
    elif family == "binomial":
        mu = mu / (1 + mu)
        y = rng.binomial(w.astype(int), mu) / w
        draws = rng.binomial(w.astype(int), mu, (7, len(x))) / w
    elif family == "ocat":
        y = np.searchsorted([-1, 0, 1], mu + rng.logistic(size=len(x))) + 1
        draws = np.searchsorted([-1, 0, 1], mu + rng.logistic(size=(7, len(x)))) + 1
    elif family == "betar":
        mu = mu / (1 + mu)
        y = rng.beta(5 * mu, 5 * (1 - mu))
        draws = rng.beta(5 * mu, 5 * (1 - mu), (7, len(x)))
    elif family == "negbin":
        y = rng.negative_binomial(3, 3 / (3 + mu))
        draws = rng.negative_binomial(3, 3 / (3 + mu), (7, len(x)))
    elif family == "inverse.gaussian":
        y = rng.wald(mu, 4)
        draws = rng.wald(mu, 4, (7, len(x)))
    else:
        y = rng.gamma(4, mu / 4)
        draws = rng.gamma(4, mu / 4, (7, len(x)))
    return {
        "data": {"x": x.tolist(), "y": y.tolist(), "w": w.tolist()},
        "family": family,
        "type": type,
        "rep": rep,
        "level": level,
        "permutations": [rng.permutation(len(x)).tolist() for _ in range(3)],
        "draws": draws.tolist(),
    }


def _fit(case):
    d = pd.DataFrame(case["data"])
    from nampy.gam.families import (
        BetaRegressionFamily,
        NegativeBinomialLogFamily,
        OrderedCategoricalFamily,
    )
    from nampy.gam.families.tweedie import TweedieTwFamily

    family = {
        "negbin": NegativeBinomialLogFamily(theta=3),
        "betar": BetaRegressionFamily(theta=5),
        "ocat": OrderedCategoricalFamily(theta=[1, 1]),
        "tw": TweedieTwFamily(theta=1.5),
    }.get(case["family"], case["family"].lower())
    formula = "y~s(x,bs='cr',k=5)"
    if case["family"] in {"gaulss", "gammals"}:
        formula = [formula, "~1"]
    return GAM(
        formula=formula,
        family=family,
        optimize_smoothing=False,
        smoothing_params=[0.4],
        smoothing_method="fixed",
    ).fit(data=d, sample_weight=d.w)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial", "Gamma"])
@pytest.mark.parametrize("type", ["deviance", "pearson", "response"])
@pytest.mark.parametrize("rep", [0, 7])
def test_qq_and_all_check_panels_match_actual_mgcv(family, type, rep):
    case = _case(family, type, rep)
    expected = _reference(case)
    model = _fit(case)
    original = model.y_.copy()
    options = {
        "type": type,
        "rep": rep,
        "s_rep": 3,
        "quantile_permutations": case["permutations"] if not rep else None,
        "response_draws": case["draws"] if rep else None,
    }
    actual = prepare_gam_check_data(model, **options)
    qq = actual["qq"]
    for key in ["theoretical", "residuals", "simulated_residuals"]:
        np.testing.assert_allclose(qq[key], expected[key], rtol=2e-6, atol=2e-7)
    if rep:
        np.testing.assert_allclose(
            np.stack([qq["lower"], qq["upper"]]), expected["lim"], rtol=2e-6, atol=2e-7
        )
    for key in ["breaks", "counts", "density", "mids"]:
        np.testing.assert_allclose(
            actual["histogram"][key], expected["histogram"][key], atol=2e-12
        )
    for panel, x, y in [
        ("residual_vs_linear", "eta", "raw_residuals"),
        ("response_vs_fitted", "fitted", "y"),
    ]:
        np.testing.assert_allclose(
            actual[panel]["x"], expected[x], rtol=2e-6, atol=2e-7
        )
        np.testing.assert_allclose(
            actual[panel]["y"], expected[y], rtol=2e-6, atol=2e-7
        )
    np.testing.assert_array_equal(model.y_, original)


@pytest.mark.parametrize(
    "values",
    [
        [0] * 20,
        [1] * 20,
        [-1] * 20,
        [0.1] * 20,
        [-0.1] * 20,
        list(range(-5, 6)),
        [0, 0, 1e-11],
        list(np.linspace(-0.001, 0.003, 37)),
        [1e10, 1e10 + 1, 1e10 + 2],
    ],
)
def test_r_histogram_pretty_breaks_fuzz_and_constant_ranges(values):
    expected = _reference({"hist": values})
    actual = _r_histogram(values)
    for key in ["breaks", "counts", "density", "mids"]:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("level", [-0.1, 0, 1, 2])
def test_envelope_level_boundaries(level):
    case = _case(rep=7, level=level)
    expected = _reference(case)
    actual = prepare_qq_gam_data(
        _fit(case), rep=7, level=level, response_draws=case["draws"]
    )
    assert actual["lower"] is actual["upper"] is None
    limits = np.asarray(expected["residuals"])
    if level >= 1:
        limits = np.append(limits, level)
    np.testing.assert_allclose(actual["ylim"], [limits.min(), limits.max()], atol=2e-7)
    np.testing.assert_allclose(
        actual["theoretical"], expected["theoretical"], atol=2e-7
    )
    np.testing.assert_allclose(
        actual["simulated_residuals"], expected["simulated_residuals"], atol=2e-7
    )


def test_numpy_seed_reproducibility_and_normal_old_style():
    model = _fit(_case())
    a = prepare_qq_gam_data(model, seed=42, rep=7)
    b = prepare_qq_gam_data(model, seed=42, rep=7)
    np.testing.assert_array_equal(a["simulated_residuals"], b["simulated_residuals"])
    c = prepare_qq_gam_data(model, seed=43, rep=7)
    assert not np.array_equal(a["simulated_residuals"], c["simulated_residuals"])
    check = prepare_gam_check_data(model, old_style=True)
    assert check["qq"]["reference"] == "normal"
    np.testing.assert_array_equal(
        check["qq"]["theoretical"], _normal_qq(model.y_, "deviance")["theoretical"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rep": -1},
        {"rep": 1.5},
        {"s_rep": 0},
        {"level": np.nan},
        {"type": "working"},
        {"quantile_permutations": [[0]]},
        {"rep": 2, "response_draws": [[0]]},
    ],
)
def test_qq_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        prepare_qq_gam_data(_fit(_case()), **kwargs)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial", "Gamma"])
def test_simulator_moments_and_weight_semantics(family):
    model = _fit(_case(family))
    rng = np.random.default_rng(486)
    draws = np.asarray([_family_draw(model, rng) for _ in range(10000)])
    from nampy.gam.model_state import _fit_scale, _fitted_mu

    mean = _fitted_mu(model)
    scale = _fit_scale(model)
    wt = model.prior_weights_
    variance = {
        "gaussian": scale / wt,
        "poisson": mean,
        "binomial": mean * (1 - mean) / wt,
        "Gamma": scale * mean**2,
    }[family]
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.04, rtol=0.04)
    np.testing.assert_allclose(draws.var(axis=0), variance, atol=0.04, rtol=0.08)


@pytest.mark.parametrize(
    "family", ["negbin", "betar", "gaulss", "gammals", "inverse.gaussian", "ocat", "tw"]
)
@pytest.mark.parametrize("type", ["deviance", "response"])
def test_extended_and_general_family_qq_check(family, type):
    rep = 7 if family in {"gaulss", "gammals", "inverse.gaussian", "ocat", "tw"} else 0
    case = _case(family, type, rep)
    expected = _reference(case)
    actual = prepare_gam_check_data(
        _fit(case),
        type=type,
        rep=rep,
        s_rep=3,
        response_draws=case["draws"] if rep else None,
        quantile_permutations=case["permutations"] if not rep else None,
    )
    for key in ["theoretical", "residuals", "simulated_residuals"]:
        np.testing.assert_allclose(
            actual["qq"][key], expected[key], rtol=2e-5, atol=2e-6
        )
    expected_eta = np.asarray(expected["eta"])
    expected_fit = np.asarray(expected["fitted"])
    np.testing.assert_allclose(
        actual["residual_vs_linear"]["x"],
        expected_eta[:, 0] if expected_eta.ndim == 2 else expected_eta,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual["response_vs_fitted"]["x"],
        expected_fit[:, 0] if expected_fit.ndim == 2 else expected_fit,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("weight_kind", ["zero", "fractional"])
def test_binomial_quantile_denominators_and_zero_weights(weight_kind):
    case = _case("binomial", "response")
    case["data"]["w"][::4] = [0.0 if weight_kind == "zero" else 1.5] * 10
    expected = _reference(case)
    model = _fit(case)
    if weight_kind == "fractional":
        with pytest.warns(RuntimeWarning, match="non-integer binomial denominator"):
            actual = prepare_qq_gam_data(
                model,
                type="response",
                s_rep=3,
                quantile_permutations=case["permutations"],
            )
    else:
        actual = prepare_qq_gam_data(
            model, type="response", s_rep=3, quantile_permutations=case["permutations"]
        )
    np.testing.assert_allclose(
        actual["theoretical"], expected["theoretical"], atol=2e-7
    )


@pytest.mark.parametrize("family", ["inverse.gaussian", "gaulss", "gammals"])
def test_missing_qf_uses_fifty_simulations_without_envelope(family):
    model = _fit(_case(family))
    result = prepare_qq_gam_data(model, seed=32)
    assert result["rep"] == 50
    assert result["level"] == 0
    assert result["lower"] is result["upper"] is None
    assert result["simulated_residuals"].shape == (len(model.y_), 50)
    assert np.all(np.isfinite(result["theoretical"]))


def test_normal_fallback_and_small_sample_positions():
    from scipy.stats import norm

    np.testing.assert_allclose(
        _normal_qq([3, 1, 2], "response")["theoretical"],
        norm.ppf((np.arange(1, 4) - 0.375) / 3.25),
    )
    model = _fit(_case())
    model.family.name = "custom_without_sampler"
    result = prepare_qq_gam_data(model, seed=42)
    assert result["reference"] == "normal"
    assert result["simulated_residuals"] is None


@pytest.mark.parametrize(
    "family", ["negbin", "betar", "inverse.gaussian", "gaulss", "gammals", "tw", "ocat"]
)
def test_extended_simulator_moments(family):
    from nampy.gam.model_state import _fit_scale, _fitted_mu

    model = _fit(_case(family))
    rng = np.random.default_rng(271)
    draws = np.asarray([_family_draw(model, rng) for _ in range(6000)])
    mu = _fitted_mu(model)
    scale = _fit_scale(model)
    if family == "ocat":
        probabilities = model.family.response_from_eta(mu)
        for j in range(probabilities.shape[1]):
            np.testing.assert_allclose(
                (draws == j + 1).mean(axis=0), probabilities[:, j], atol=0.025, rtol=0
            )
        return
    mean = mu
    if family == "negbin":
        variance = mu + mu**2 / model.family.theta
    elif family == "betar":
        variance = mu * (1 - mu) / (1 + model.family.getTheta(True))
    elif family == "inverse.gaussian":
        variance = scale * mu**3
    elif family == "gaulss":
        mean = mu[:, 0]
        variance = scale / model.prior_weights_ / mu[:, 1] ** 2
    elif family == "gammals":
        mean = mu[:, 0]
        variance = mean**2 * np.exp(mu[:, 1])
    else:
        variance = scale * mu**model.family.p
    assert np.all(
        np.abs(draws.mean(axis=0) - mean) < 6 * np.sqrt(variance / len(draws))
    )
    np.testing.assert_allclose(draws.var(axis=0), variance, rtol=0.15, atol=0.005)
