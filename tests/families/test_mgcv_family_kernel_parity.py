"""Direct family-kernel parity against R/mgcv family objects.

Upstream references: stats family objects (dev.resids, variance, mu.eta,
linkfun/linkinv, validmu/valideta, initialize/mustart, aic),
mgcv/R/gam.fit3.r::fix.family.ls (saturated log-likelihood kernels), and
mgcv/R/efam.r::nb (Dd, ls, dev.resids, aic, theta transforms).

nampy method mapping: deviance_obs ~ dev.resids, initialize_mu ~ mustart,
saturated_loglik ~ ls[1], ls ~ fix.family.ls / nb()$ls, Dd ~ nb()$Dd,
aic(edf=0) ~ family$aic.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from nampy.gam.families.registry import make_gam_family
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _build_r_command,
    _make_binomial_data,
    _mgcv_fixture_key,
    _mgcv_fixture_load,
    _mgcv_fixture_save,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_derivatives]

_FAMILY_KERNEL_FIXTURE_VERSION = 1


def _run_mgcv_family_kernels(
    family_key: str,
    link: str,
    *,
    y,
    mu,
    eta,
    wt,
    scale: float,
    ltheta: float | None = None,
):
    payload = {
        "y": np.asarray(y, dtype=np.float64).tolist(),
        "mu": np.asarray(mu, dtype=np.float64).tolist(),
        "eta": np.asarray(eta, dtype=np.float64).tolist(),
        "wt": np.asarray(wt, dtype=np.float64).tolist(),
        "scale": float(scale),
        "ltheta": None if ltheta is None else float(ltheta),
    }
    cache_key = _mgcv_fixture_key(
        "family_kernels",
        {
            "version": _FAMILY_KERNEL_FIXTURE_VERSION,
            "family_key": family_key,
            "link": link,
            "payload": payload,
        },
    )
    cached = _mgcv_fixture_load(cache_key)
    if cached is not None:
        return cached

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
family_key <- args[[1]]
link_name <- args[[2]]
payload <- fromJSON(args[[3]])
out <- args[[4]]
y <- as.numeric(payload$y)
mu <- as.numeric(payload$mu)
eta <- as.numeric(payload$eta)
wt <- as.numeric(payload$wt)
scale <- as.numeric(payload$scale)
ltheta <- if (is.null(payload$ltheta)) NULL else as.numeric(payload$ltheta)

fam <- switch(
  family_key,
  gaussian = gaussian(link = link_name),
  poisson = poisson(link = link_name),
  binomial = binomial(link = link_name),
  gamma = Gamma(link = link_name),
  negbin = do.call(mgcv::nb, list(theta = exp(ltheta), link = link_name)),
  stop(sprintf("Unsupported family key: %s", family_key))
)

res <- list(
  dev_resids = unname(as.numeric(fam$dev.resids(y, mu, wt))),
  variance = unname(as.numeric(fam$variance(mu))),
  mu_eta = unname(as.numeric(fam$mu.eta(eta))),
  linkinv = unname(as.numeric(fam$linkinv(eta))),
  linkfun = unname(as.numeric(fam$linkfun(mu))),
  validmu = isTRUE(fam$validmu(mu)),
  valideta = isTRUE(fam$valideta(eta))
)

ev <- new.env()
assign("y", y, envir = ev)
assign("nobs", length(y), envir = ev)
assign("weights", wt, envir = ev)
assign("etastart", NULL, envir = ev)
assign("start", NULL, envir = ev)
assign("mustart", NULL, envir = ev)
assign("family", fam, envir = ev)
mustart <- tryCatch({
  eval(fam$initialize, envir = ev)
  get("mustart", envir = ev)
}, error = function(e) NULL)
res$mustart <- if (is.null(mustart)) NULL else unname(as.numeric(mustart))

if (family_key %in% c("gaussian", "poisson", "binomial", "gamma")) {
  lsfam <- mgcv:::fix.family.ls(fam)
  res$ls <- unname(as.numeric(lsfam$ls(y, wt, rep(1, length(y)), scale)))
}
if (family_key %in% c("poisson", "binomial")) {
  res$aic <- unname(as.numeric(fam$aic(y, rep(1, length(y)), mu, wt, 0)))
}
if (family_key == "negbin") {
  dd <- fam$Dd(y, mu, ltheta, wt, level = 2)
  res$Dd <- lapply(dd, function(v) unname(as.numeric(v)))
  lsres <- fam$ls(y, wt, ltheta, scale)
  res$nb_ls <- list(
    ls = unname(as.numeric(lsres$ls)),
    lsth1 = unname(as.numeric(lsres$lsth1)),
    LSTH1 = unname(as.numeric(lsres$LSTH1)),
    lsth2 = unname(as.numeric(lsres$lsth2))
  )
  res$aic <- unname(as.numeric(fam$aic(y, mu, ltheta, wt, 0)))
}

write_json(res, out, auto_unbox = TRUE, digits = 17, null = "null")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        json_path = tmpdir_path / "kernels.json"
        script_path = tmpdir_path / "kernels.R"
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                family_key,
                link,
                json.dumps(payload),
                str(json_path),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(cache_key, result)
    return result


_WT_MIXED = np.asarray([0.0, 1e6, 1.0, 2.5, 0.4, 1.0], dtype=np.float64)

_KERNEL_CASES = {
    "gaussian": {
        # Positive responses so R's gaussian(link="log"/"inverse") initialize
        # expression accepts them for the mustart comparison.
        "links": ["identity", "log", "inverse"],
        "y": np.asarray([0.2, 0.5, 0.9, 1.7, 2.2, 3.0]),
        "mu": np.asarray([0.5, 0.2, 0.9, 1.4, 2.0, 2.8]),
        "eta": {
            "identity": np.asarray([-1.0, -0.2, 0.3, 0.9, 1.5, 2.1]),
            "log": np.asarray([-1.0, -0.2, 0.3, 0.9, 1.5, 2.1]),
            "inverse": np.asarray([0.4, 0.9, 1.3, 2.0, 2.6, 3.1]),
        },
        "scale": 0.37,
    },
    "poisson": {
        "links": ["log", "identity", "sqrt"],
        "y": np.asarray([0.0, 0.0, 1.0, 2.0, 5.0, 12.0]),
        "mu": np.asarray([0.3, 1e-6, 0.7, 2.5, 4.0, 9.0]),
        "eta": {
            "log": np.asarray([-2.0, -0.5, 0.2, 0.9, 1.4, 2.2]),
            "identity": np.asarray([0.3, 0.6, 1.0, 2.5, 4.0, 9.0]),
            "sqrt": np.asarray([0.2, 0.7, 1.0, 1.6, 2.0, 3.0]),
        },
        "scale": 1.0,
    },
    "binomial": {
        "links": ["logit", "probit", "cloglog", "cauchit", "log"],
        "y": np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        "mu": np.asarray([1e-7, 1.0 - 1e-7, 0.4, 0.6, 0.985, 0.015]),
        "eta": {
            "logit": np.asarray([-6.0, -1.2, -0.1, 0.4, 1.7, 6.0]),
            "probit": np.asarray([-3.5, -1.0, -0.1, 0.4, 1.2, 3.5]),
            "cloglog": np.asarray([-4.0, -1.0, -0.1, 0.4, 1.0, 2.0]),
            "cauchit": np.asarray([-25.0, -1.0, -0.1, 0.4, 1.0, 25.0]),
            # binomial(link="log") requires eta < 0; the final value probes
            # the upper probability boundary used by stats::make.link("log").
            "log": np.asarray([-16.0, -6.0, -2.0, -0.7, -0.1, -1e-7]),
        },
        "scale": 1.0,
    },
    "gamma": {
        "links": ["inverse", "log", "identity"],
        "y": np.asarray([1e-4, 0.5, 1.1, 2.0, 4.5, 7.0]),
        "mu": np.asarray([0.2, 0.8, 1.5, 2.5, 3.5, 6.0]),
        "eta": {
            "inverse": np.asarray([0.2, 0.6, 1.0, 1.8, 2.4, 3.0]),
            "log": np.asarray([-1.5, -0.4, 0.2, 0.9, 1.3, 1.9]),
            "identity": np.asarray([0.2, 0.8, 1.5, 2.5, 3.5, 6.0]),
        },
        "scale": 0.55,
    },
}

_FAMILY_LINK_PARAMS = [
    (family_key, link)
    for family_key, spec in _KERNEL_CASES.items()
    for link in spec["links"]
]


def _nampy_family(family_key: str, link: str, theta: float | None = None):
    spec: dict = {"name": family_key, "link": link}
    if theta is not None:
        spec["theta"] = theta
    return make_gam_family(spec)


@pytest.mark.parametrize(
    ("family_key", "link"),
    _FAMILY_LINK_PARAMS,
    ids=[f"{family_key}_{link}" for family_key, link in _FAMILY_LINK_PARAMS],
)
def test_ordinary_family_kernels_match_r(family_key, link):
    """dev.resids/variance/links/mustart/validity match R at boundaries."""
    spec = _KERNEL_CASES[family_key]
    y = spec["y"]
    mu = spec["mu"]
    eta = spec["eta"][link]
    wt = np.ones_like(y)
    expected = _run_mgcv_family_kernels(
        family_key, link, y=y, mu=mu, eta=eta, wt=wt, scale=spec["scale"]
    )
    family = _nampy_family(family_key, link)

    np.testing.assert_allclose(
        family.deviance_obs(y, mu, weights=wt),
        np.asarray(expected["dev_resids"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        family.variance(mu),
        np.asarray(expected["variance"], dtype=np.float64),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        family.mu_eta(eta),
        np.asarray(expected["mu_eta"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        family.inverse_link(eta),
        np.asarray(expected["linkinv"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        family.link(mu),
        np.asarray(expected["linkfun"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-14,
    )
    # Gaussian families impose no mu/eta domain in R (validmu is TRUE) and
    # nampy correspondingly defines no valid_mu/valid_eta there.
    if hasattr(family, "valid_mu"):
        assert family.valid_mu(mu) == bool(expected["validmu"])
    else:
        assert bool(expected["validmu"])
    if hasattr(family, "valid_eta"):
        assert family.valid_eta(eta) == bool(expected["valideta"])
    else:
        assert bool(expected["valideta"])

    expected_mustart = expected.get("mustart", None)
    assert expected_mustart is not None
    try:
        actual_mustart = family.initialize_mu(y, weights=wt)
    except TypeError:
        actual_mustart = family.initialize_mu(y)
    np.testing.assert_allclose(
        actual_mustart,
        np.asarray(expected_mustart, dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("family_key", ["gaussian", "poisson", "binomial", "gamma"])
def test_saturated_loglik_matches_fix_family_ls_reference(family_key):
    """saturated_loglik equals fix.family.ls()[1], zero/extreme weights incl."""
    spec = _KERNEL_CASES[family_key]
    y = spec["y"]
    link = spec["links"][0]
    for wt in (np.ones_like(y), _WT_MIXED):
        expected = _run_mgcv_family_kernels(
            family_key,
            link,
            y=y,
            mu=spec["mu"],
            eta=spec["eta"][link],
            wt=wt,
            scale=spec["scale"],
        )
        family = _nampy_family(family_key, link)
        expected_ls = np.asarray(expected["ls"], dtype=np.float64)
        actual_saturated = family.saturated_loglik(
            y, weights=wt, n=np.ones_like(y), scale=spec["scale"]
        )
        np.testing.assert_allclose(
            actual_saturated, expected_ls[0], rtol=1e-10, atol=1e-10
        )
        if family_key in {"gaussian", "gamma"}:
            actual_ls = np.asarray(
                family.ls(y, wt, n=np.ones_like(y), scale=spec["scale"]),
                dtype=np.float64,
            )
            # scipy polygamma vs R trigamma agree to ~1e-9 relative.
            np.testing.assert_allclose(actual_ls, expected_ls, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("family_key", ["poisson", "binomial"])
def test_known_scale_family_aic_kernel_matches_r(family_key):
    """family$aic (edf=0 contribution) matches R for known-scale families.

    Binomial includes R's ``m <- if (any(n > 1)) n else wt`` quirk: non-unit
    prior weights on 0/1 responses are reinterpreted as binomial denominators
    (zero-weight rows contribute nothing, fractional weights round through
    ``dbinom(round(m*y), round(m), mu)``).
    """
    spec = _KERNEL_CASES[family_key]
    y = spec["y"]
    mu = spec["mu"]
    link = spec["links"][0]
    for wt in (np.ones_like(y), _WT_MIXED):
        expected = _run_mgcv_family_kernels(
            family_key,
            link,
            y=y,
            mu=mu,
            eta=spec["eta"][link],
            wt=wt,
            scale=1.0,
        )
        family = _nampy_family(family_key, link)
        np.testing.assert_allclose(
            family.aic(y, mu, edf=0.0, weights=wt),
            float(np.ravel(expected["aic"])[0]),
            rtol=1e-10,
            atol=1e-10,
        )


_NB_Y = np.asarray([0.0, 1.0, 3.0, 9.0, 0.0, 25.0], dtype=np.float64)
_NB_MU = np.asarray([0.4, 1.2, 2.5, 6.0, 3.5, 14.0], dtype=np.float64)
_NB_DD_KEYS_L0 = ("Dmu", "Dmu2", "EDmu2")
_NB_DD_KEYS_L1 = ("Dth", "Dmuth", "Dmu3", "Dmu2th", "EDmu2th")
_NB_DD_KEYS_L2 = ("Dmu4", "Dth2", "Dmuth2", "Dmu2th2", "Dmu3th")


@pytest.mark.parametrize("theta", [0.4, 1.6, 45.0])
@pytest.mark.parametrize(
    "wt_id", ["unit", "mixed"], ids=["unit_weights", "mixed_weights"]
)
def test_negbin_dd_kernel_matches_mgcv(theta, wt_id):
    """nb()$Dd derivatives match at y=0 boundaries for all levels."""
    wt = np.ones_like(_NB_Y) if wt_id == "unit" else _WT_MIXED
    ltheta = float(np.log(theta))
    expected = _run_mgcv_family_kernels(
        "negbin",
        "log",
        y=_NB_Y,
        mu=_NB_MU,
        eta=np.log(_NB_MU),
        wt=wt,
        scale=1.0,
        ltheta=ltheta,
    )
    family = _nampy_family("negbin", "log", theta=theta)

    np.testing.assert_allclose(
        family.deviance_obs(_NB_Y, _NB_MU, weights=wt),
        np.asarray(expected["dev_resids"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    # nampy's negbin exposes the aic kernel through loglik_obs
    # (mgcv/R/efam.r:239-246: aic = -2 * sum(loglik_obs * wt)).
    actual_aic = -2.0 * float(
        np.sum(wt * np.asarray(family.loglik_obs(_NB_Y, _NB_MU), dtype=np.float64))
    )
    np.testing.assert_allclose(
        actual_aic,
        float(np.ravel(expected["aic"])[0]),
        rtol=1e-10,
        atol=1e-10,
    )

    actual = family.Dd(_NB_Y, _NB_MU, theta=ltheta, wt=wt, level=2)
    for key in (*_NB_DD_KEYS_L0, *_NB_DD_KEYS_L1, *_NB_DD_KEYS_L2):
        assert key in actual, f"missing Dd component {key}"
        np.testing.assert_allclose(
            np.asarray(actual[key], dtype=np.float64),
            np.asarray(expected["Dd"][key], dtype=np.float64),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"Dd[{key}] mismatch",
        )

    level0 = family.Dd(_NB_Y, _NB_MU, theta=ltheta, wt=wt, level=0)
    assert set(level0.keys()) == set(_NB_DD_KEYS_L0)
    level1 = family.Dd(_NB_Y, _NB_MU, theta=ltheta, wt=wt, level=1)
    assert set(level1.keys()) == set(_NB_DD_KEYS_L0) | set(_NB_DD_KEYS_L1)


@pytest.mark.parametrize("theta", [0.4, 1.6, 45.0])
def test_negbin_ls_kernel_matches_mgcv(theta):
    """nb()$ls saturated log-likelihood and theta derivatives match."""
    ltheta = float(np.log(theta))
    for wt in (np.ones_like(_NB_Y), _WT_MIXED):
        expected = _run_mgcv_family_kernels(
            "negbin",
            "log",
            y=_NB_Y,
            mu=_NB_MU,
            eta=np.log(_NB_MU),
            wt=wt,
            scale=1.0,
            ltheta=ltheta,
        )["nb_ls"]
        family = _nampy_family("negbin", "log", theta=theta)
        actual = family.ls(_NB_Y, wt, theta=ltheta)
        np.testing.assert_allclose(
            actual["ls"], float(np.ravel(expected["ls"])[0]), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            actual["lsth1"],
            float(np.ravel(expected["lsth1"])[0]),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["LSTH1"], dtype=np.float64).ravel(),
            np.asarray(expected["LSTH1"], dtype=np.float64).ravel(),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual["lsth2"],
            float(np.ravel(expected["lsth2"])[0]),
            rtol=1e-10,
            atol=1e-10,
        )


def test_negbin_theta_transforms_follow_mgcv_convention():
    """putTheta stores log(theta); getTheta(trans=True) returns theta."""
    family = _nampy_family("negbin", "log", theta=2.5)
    np.testing.assert_allclose(family.getTheta(False), np.log(2.5), rtol=1e-12)
    np.testing.assert_allclose(family.getTheta(True), 2.5, rtol=1e-12)
    family.putTheta(np.log(7.0))
    np.testing.assert_allclose(family.getTheta(False), np.log(7.0), rtol=1e-12)
    np.testing.assert_allclose(family.getTheta(True), 7.0, rtol=1e-12)


def test_weighted_binomial_fitted_loglik_aic_bic_match_mgcv():
    """Fitted weighted-binomial logLik/AIC/BIC inherit the m <- wt quirk.

    stats::binomial()$aic treats the non-unit prior weights as binomial
    denominators; the model-level logLik.gam/AIC/BIC chain must therefore
    match mgcv on a genuinely weighted 0/1 fit, not only at unit weights.
    """
    data = _make_binomial_data().copy()
    row = np.arange(len(data))
    data["w"] = np.asarray([1.0, 2.0, 3.0, 4.0])[row % 4]
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    gam = GAM(
        family="binomial",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data, sample_weight=data["w"].to_numpy(dtype=np.float64))
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "binomial",
        "REML",
        weights_column="w",
        optimizer="newton",
        allow_live_run=True,
    )
    expected_loglik = float(np.ravel(expected["fit"]["loglik"])[0])
    expected_aic = float(np.ravel(expected["fit"]["aic"])[0])
    expected_bic = float(np.ravel(expected["parity"]["diagnostics"]["bic"])[0])
    np.testing.assert_allclose(gam.loglik(), expected_loglik, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gam.aic(), expected_aic, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gam.bic(), expected_bic, rtol=1e-6, atol=1e-6)
