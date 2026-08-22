"""vcov.gam option parity for ordinary families: sandwich, freq, dispersion.

Upstream reference: mgcv/R/mgcv.r::vcov.gam — freq=TRUE returns Ve,
unconditional=TRUE returns Vc when available (silently falling back to Vp),
sandwich=TRUE builds the sandwich covariance, and dispersion= rescales by
dispersion/sig2.  Compared through tests/parity/mgcv_snapshot.R plus a local
runner for the dispersion argument.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam import GAM
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _build_r_command,
    _df_cache_repr,
    _family_specs,
    _make_gaussian_data,
    _make_poisson_data,
    _mgcv_cache_key,
    _mgcv_cache_load,
    _mgcv_cache_save,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_output]

_TWO_CR_FORMULA = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
_VCOV_OPTIONS_CACHE_VERSION = 1


def _reml_fit(data, family):
    return GAM(
        family=family,
        formula=_TWO_CR_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)


def _assert_cov_close(actual, expected, *, tol):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    scale_ref = max(float(np.max(np.abs(expected))), 1e-12)
    np.testing.assert_allclose(actual, expected, atol=tol * scale_ref, rtol=tol)


@pytest.mark.parametrize(
    ("family", "data_factory", "tol"),
    [
        pytest.param("gaussian", _make_gaussian_data, 1e-6, id="gaussian"),
        pytest.param("poisson", _make_poisson_data, 1e-6, id="poisson"),
    ],
)
def test_ordinary_family_sandwich_and_freq_vcov_match_mgcv(family, data_factory, tol):
    """vcov(sandwich=) and vcov(freq=True) match mgcv for ordinary families."""
    data = data_factory()
    gam = _reml_fit(data, family)
    expected = _run_mgcv_snapshot(
        data,
        _TWO_CR_FORMULA,
        family,
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    e_fit = expected["fit"]

    _assert_cov_close(gam.vcov(freq=True), e_fit["cov_freq"], tol=tol)
    _assert_cov_close(gam.vcov(), e_fit["cov_bayes"], tol=tol)

    expected_sandwich_bayes = e_fit.get("cov_sandwich_bayes", None)
    assert expected_sandwich_bayes is not None
    _assert_cov_close(gam.vcov(sandwich=True), expected_sandwich_bayes, tol=10.0 * tol)
    expected_sandwich_freq = e_fit.get("cov_sandwich_freq", None)
    assert expected_sandwich_freq is not None
    _assert_cov_close(
        gam.vcov(sandwich=True, freq=True), expected_sandwich_freq, tol=10.0 * tol
    )


def _run_mgcv_vcov_dispersion(data, formula, family, method, dispersion):
    """vcov.gam(dispersion=) reference (not exposed by mgcv_snapshot.R)."""
    _family_nampy, family_token = _family_specs(family)
    cache_key = _mgcv_cache_key(
        "vcov_dispersion",
        {
            "version": _VCOV_OPTIONS_CACHE_VERSION,
            "data": _df_cache_repr(data),
            "formula": str(formula),
            "family_token": family_token,
            "method": method,
            "dispersion": float(dispersion),
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
out <- args[[2]]
formula_obj <- as.formula(args[[3]])
family_name <- tolower(args[[4]])
method_name <- args[[5]]
dispersion <- as.numeric(args[[6]])
family_obj <- switch(
  family_name,
  gaussian = gaussian(),
  poisson = poisson(),
  stop(sprintf("Unsupported family: %s", family_name))
)
fit <- gam(formula_obj, data = d, family = family_obj, method = method_name)
write_json(
  list(
    vcov_dispersion = unname(vcov(fit, dispersion = dispersion)),
    vcov_freq_dispersion = unname(vcov(fit, freq = TRUE, dispersion = dispersion))
  ),
  out,
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "vcov_dispersion.json"
        script_path = tmpdir_path / "vcov_dispersion.R"
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
                repr(float(dispersion)),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(cache_key, result)
    return result


def test_vcov_dispersion_rescale_matches_mgcv():
    """vcov(dispersion=) rescales the covariance exactly like mgcv."""
    data = _make_gaussian_data()
    gam = _reml_fit(data, "gaussian")
    dispersion = 0.05
    expected = _run_mgcv_vcov_dispersion(
        data, _TWO_CR_FORMULA, "gaussian", "REML", dispersion
    )
    _assert_cov_close(
        gam.vcov(dispersion=dispersion), expected["vcov_dispersion"], tol=1e-6
    )
    _assert_cov_close(
        gam.vcov(freq=True, dispersion=dispersion),
        expected["vcov_freq_dispersion"],
        tol=1e-6,
    )


def test_unconditional_vcov_falls_back_to_vp_for_fixed_sp_like_mgcv():
    """Without sp uncertainty (fixed sp), unconditional falls back to Vp."""
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    expected = _run_mgcv_snapshot(
        data, formula, "gaussian", "fixed", allow_live_run=True
    )
    # mgcv vcov.gam silently returns Vp when Vc is unavailable.
    np.testing.assert_allclose(
        gam.vcov(unconditional=True), gam.vcov(), rtol=0.0, atol=0.0
    )
    _assert_cov_close(
        gam.vcov(unconditional=True), expected["fit"]["cov_bayes"], tol=1e-8
    )


def test_unconditional_vcov_matches_mgcv_vc_for_reml():
    """vcov(unconditional=True) equals mgcv's Vc for an optimized fit."""
    data = _make_gaussian_data()
    gam = _reml_fit(data, "gaussian")
    expected = _run_mgcv_snapshot(
        data,
        _TWO_CR_FORMULA,
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    expected_vc = expected["fit"].get("vcov_unconditional", None)
    if expected_vc is None:
        expected_vc = expected["fit"].get("cov_unconditional", None)
    assert expected_vc is not None
    _assert_cov_close(gam.vcov(unconditional=True), expected_vc, tol=1e-5)
