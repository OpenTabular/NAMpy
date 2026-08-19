"""Behavioral parity for min_sp smoothing-parameter floors against mgcv.

Upstream reference: mgcv/R/mgcv.r::gam.setup (min.sp builds the fixed offset
penalty H, mgcv.r:1465-1508), so mgcv's reported ``fit$sp`` is the *free*
multiplier and the identified quantity is the total penalty multiplier
``fit$sp + min.sp``.  nampy parameterizes the same total directly with a lower
bound (nampy/gam/smoothing_selection/optimize/driver.py), so parity is asserted
on total smoothing parameters, criterion, EDF, fit and covariance.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _build_r_command,
    _df_cache_repr,
    _family_specs,
    _mgcv_cache_key,
    _mgcv_cache_load,
    _mgcv_cache_save,
)

pytestmark = [pytest.mark.surface_regression]

_MIN_SP_CACHE_VERSION = 2


def _wiggly_gaussian_data(seed=311, n=200) -> pd.DataFrame:
    """Wiggly low-noise x0 signal (small optimal sp) and a curved x1 effect.

    Both unconstrained optima must be interior: a linear-truth smooth would
    ride its sp to the upper boundary, where the endpoint is not identified.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(3.1 * x0) + 0.4 * x1**2 + rng.normal(scale=0.08, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _wiggly_poisson_data(seed=313, n=260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    eta = 0.4 + 0.9 * np.sin(3.0 * x0) + 0.25 * x1**2
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _run_mgcv_min_sp_snapshot(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    min_sp,
    optimizer: str = "newton",
):
    """Reference mgcv fit with min.sp= (not supported by mgcv_snapshot.R)."""
    _family_nampy, family_token = _family_specs(family)
    min_sp_list = np.asarray(min_sp, dtype=np.float64).tolist()
    cache_key = _mgcv_cache_key(
        "min_sp_snapshot",
        {
            "version": _MIN_SP_CACHE_VERSION,
            "data": _df_cache_repr(data),
            "formula": str(formula),
            "family_token": family_token,
            "method": method,
            "min_sp": min_sp_list,
            "optimizer": str(optimizer),
        },
    )
    cached = _mgcv_cache_load(cache_key)
    if cached is not None:
        return cached

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
formula_obj <- as.formula(args[[3]])
family_name <- tolower(args[[4]])
method_name <- args[[5]]
min_sp <- as.numeric(fromJSON(args[[6]]))
optimizer_name <- args[[7]]
family_obj <- switch(
  family_name,
  gaussian = gaussian(),
  poisson = poisson(),
  binomial = binomial(),
  gamma = Gamma(link = "log"),
  stop(sprintf("Unsupported family for min.sp parity: %s", family_name))
)
fit <- gam(
  formula_obj,
  data = d,
  family = family_obj,
  method = method_name,
  min.sp = min_sp,
  optimizer = c("outer", optimizer_name)
)
pred_link <- predict(fit, type = "link", se.fit = TRUE)
write_json(
  list(
    sp_free = unname(as.numeric(fit$sp)),
    min_sp = unname(as.numeric(min_sp)),
    sp_total = unname(as.numeric(fit$sp) + as.numeric(min_sp)),
    criterion_value = unname(as.numeric(fit$gcv.ubre)),
    scale = unname(as.numeric(fit$sig2)),
    deviance = unname(as.numeric(fit$deviance)),
    edf_total = unname(as.numeric(sum(fit$edf))),
    edf_by_term = unname(as.numeric(summary(fit)$edf)),
    cov_bayes = unname(fit$Vp),
    response = unname(as.numeric(predict(fit, type = "response"))),
    link = unname(as.numeric(pred_link$fit)),
    se_link = unname(as.numeric(pred_link$se.fit))
  ),
  out,
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "min_sp.json"
        script_path = tmpdir_path / "min_sp.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                str(formula),
                family_token,
                method,
                json.dumps(min_sp_list),
                str(optimizer),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(cache_key, result)
    return result


def _fitted_sp(gam) -> np.ndarray:
    return np.asarray(
        gam.fit_result(include_covariances=False).smoothing_params,
        dtype=np.float64,
    )


def _fit_min_sp_model(data, formula, family, method, min_sp) -> GAM:
    return GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
        smoothing_optimizer="outer_newton",
        min_sp=np.asarray(min_sp, dtype=np.float64),
    ).fit(data=data)


def _assert_min_sp_parity(gam, expected, *, log_sp_atol, pred_atol, cov_atol):
    actual_sp = _fitted_sp(gam)
    expected_total = np.asarray(expected["sp_total"], dtype=np.float64)
    assert actual_sp.shape == expected_total.shape
    np.testing.assert_allclose(
        np.log(actual_sp), np.log(expected_total), atol=log_sp_atol, rtol=0.0
    )

    result = gam.fit_result(include_covariances=True)
    np.testing.assert_allclose(
        result.criterion_value,
        float(expected["criterion_value"]),
        atol=1e-6,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        result.edf_total, float(expected["edf_total"]), atol=5e-5, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result.edf_by_term, dtype=np.float64),
        np.asarray(expected["edf_by_term"], dtype=np.float64),
        atol=5e-5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.scale, float(expected["scale"]), atol=1e-8, rtol=1e-6
    )

    actual_link, actual_se = gam.predict(type="link", return_se=True)
    np.testing.assert_allclose(
        actual_link,
        np.asarray(expected["link"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_atol,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected["se_link"], dtype=np.float64),
        atol=10.0 * pred_atol,
        rtol=10.0 * pred_atol,
    )

    # cr bases identify the coefficient basis uniquely, so Vp is comparable
    # elementwise (same convention as _assert_exact_mgcv_snapshot_parity users).
    actual_cov = np.asarray(gam.vcov(), dtype=np.float64)
    expected_cov = np.asarray(expected["cov_bayes"], dtype=np.float64)
    assert actual_cov.shape == expected_cov.shape
    scale_ref = max(1.0, float(np.max(np.abs(expected_cov))))
    np.testing.assert_allclose(
        actual_cov, expected_cov, atol=cov_atol * scale_ref, rtol=cov_atol
    )


_MIN_SP_FORMULA = 'y ~ s(x0, bs="cr", k=10) + s(x1, bs="cr", k=8)'


def test_gaussian_reml_binding_min_sp_matches_mgcv():
    """A binding min.sp floor moves sp, criterion and covariance like mgcv."""
    data = _wiggly_gaussian_data()
    min_sp = [5.0, 0.0]

    unconstrained = GAM(
        family="gaussian",
        formula=_MIN_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    unconstrained_sp = _fitted_sp(unconstrained)
    assert unconstrained_sp[0] < min_sp[0], (
        "test setup requires the floor to bind on the first smooth"
    )

    gam = _fit_min_sp_model(data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp)
    expected = _run_mgcv_min_sp_snapshot(
        data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp
    )
    _assert_min_sp_parity(
        gam, expected, log_sp_atol=5e-3, pred_atol=1e-6, cov_atol=1e-5
    )

    # The binding floor must be active and must worsen the (minimized) REML
    # criterion relative to the unconstrained optimum, in both implementations.
    assert _fitted_sp(gam)[0] >= min_sp[0] * (1.0 - 1e-9)
    unconstrained_criterion = unconstrained.fit_result(
        include_covariances=False
    ).criterion_value
    assert float(expected["criterion_value"]) > unconstrained_criterion
    assert (
        gam.fit_result(include_covariances=False).criterion_value
        > unconstrained_criterion
    )


def test_gaussian_reml_non_binding_min_sp_matches_unconstrained_and_mgcv():
    """A slack min.sp floor leaves the optimum untouched, as in mgcv."""
    data = _wiggly_gaussian_data()
    min_sp = [1e-8, 1e-8]

    gam = _fit_min_sp_model(data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp)
    expected = _run_mgcv_min_sp_snapshot(
        data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp
    )
    _assert_min_sp_parity(
        gam, expected, log_sp_atol=2e-3, pred_atol=1e-6, cov_atol=1e-5
    )

    unconstrained = GAM(
        family="gaussian",
        formula=_MIN_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    np.testing.assert_allclose(
        np.log(_fitted_sp(gam)),
        np.log(_fitted_sp(unconstrained)),
        atol=1e-6,
        rtol=0.0,
    )


def test_poisson_reml_binding_min_sp_matches_mgcv():
    """min.sp floors carry over to non-Gaussian PIRLS fits."""
    data = _wiggly_poisson_data()
    min_sp = [400.0, 0.0]

    unconstrained = GAM(
        family="poisson",
        formula=_MIN_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    assert _fitted_sp(unconstrained)[0] < min_sp[0]

    gam = _fit_min_sp_model(data, _MIN_SP_FORMULA, "poisson", "REML", min_sp)
    expected = _run_mgcv_min_sp_snapshot(
        data, _MIN_SP_FORMULA, "poisson", "REML", min_sp
    )
    _assert_min_sp_parity(
        gam, expected, log_sp_atol=5e-3, pred_atol=5e-6, cov_atol=5e-5
    )
    assert _fitted_sp(gam)[0] >= min_sp[0] * (1.0 - 1e-9)


def test_min_sp_via_fit_argument_matches_constructor_route():
    """fit(min_sp=...) and GAM(min_sp=...) resolve to the same clamped fit."""
    data = _wiggly_gaussian_data()
    min_sp = [5.0, 0.0]
    via_constructor = _fit_min_sp_model(
        data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp
    )
    via_fit = GAM(
        family="gaussian",
        formula=_MIN_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data, min_sp=np.asarray(min_sp, dtype=np.float64))
    np.testing.assert_allclose(
        _fitted_sp(via_fit),
        _fitted_sp(via_constructor),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        via_fit.predict(type="link"),
        via_constructor.predict(type="link"),
        rtol=0.0,
        atol=0.0,
    )


def test_gaussian_ml_binding_min_sp_matches_mgcv():
    """The min.sp floor must bind identically under ML (previously REML-only)."""
    data = _wiggly_gaussian_data()
    min_sp = [5.0, 0.0]

    gam = _fit_min_sp_model(data, _MIN_SP_FORMULA, "gaussian", "ML", min_sp)
    expected = _run_mgcv_min_sp_snapshot(
        data, _MIN_SP_FORMULA, "gaussian", "ML", min_sp
    )
    _assert_min_sp_parity(
        gam, expected, log_sp_atol=5e-3, pred_atol=1e-6, cov_atol=1e-5
    )
    # Under mgcv's total = min.sp + exp(rho) parameterization the optimizer
    # approaches a binding floor asymptotically, so the endpoint sits just
    # above it rather than exactly on it.
    sp0 = float(_fitted_sp(gam)[0])
    assert min_sp[0] <= sp0 <= min_sp[0] * (1.0 + 1e-4)


def test_gaussian_reml_bfgs_binding_min_sp_matches_mgcv():
    """min.sp with the BFGS outer optimizer on both sides (previously newton-only)."""
    data = _wiggly_gaussian_data()
    min_sp = [5.0, 0.0]

    gam = GAM(
        family="gaussian",
        formula=_MIN_SP_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="bfgs",
        min_sp=np.asarray(min_sp, dtype=np.float64),
    ).fit(data=data)
    expected = _run_mgcv_min_sp_snapshot(
        data, _MIN_SP_FORMULA, "gaussian", "REML", min_sp, optimizer="bfgs"
    )
    _assert_min_sp_parity(
        gam, expected, log_sp_atol=5e-3, pred_atol=1e-6, cov_atol=1e-4
    )
    sp0 = float(_fitted_sp(gam)[0])
    assert min_sp[0] <= sp0 <= min_sp[0] * (1.0 + 1e-4)
