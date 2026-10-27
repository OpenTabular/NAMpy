"""Independent high-precision Gamma saturated-density scale derivatives."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.families.registry import make_gam_family
from tests.reference_fixtures import load_reference, reference_key, save_reference

Y = np.array([1e-30, 0.5, 1, 2, 1e30])
WEIGHTS = np.array([0.2, 0.5, 1, 2, 10])
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if (nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); family <- mgcv:::fix.family.ls(Gamma())
out <- list(values=family$ls(p$y,p$weights,length(p$y),p$phi),
            version=as.character(packageVersion('mgcv')),r_version=R.version.string)
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='string')
"""


def upstream_reference(phi, weights):
    payload = {"phi": phi, "y": Y.tolist(), "weights": weights.tolist()}
    key = reference_key(
        "gamma-scale-ls-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_gamma_scale_ls", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / name for name in ("source.R", "out.json")
            )
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_gamma_scale_ls", key, result)
    assert result["version"] == "1.9.4"
    return np.asarray(result["values"], float)


def high_precision_reference(phi, zero_weights):
    weights = WEIGHTS.copy()
    if zero_weights:
        weights[::2] = 0
    payload = {"phi": phi, "y": Y.tolist(), "weights": weights.tolist(), "dps": 100}
    key = reference_key("gamma-saturated-density-scale-diff-v1", payload)
    expected = load_reference("gamma_dispersion_high_precision", key)
    if expected is None:
        import mpmath as mp

        with mp.workdps(payload["dps"]):
            y = [mp.mpf(float(value)) for value in Y]
            w = [mp.mpf(float(value)) for value in weights]

            def log_density(scale):
                total = mp.mpf(0)
                for response, weight in zip(y, w, strict=True):
                    if not weight:
                        continue
                    shape = weight / scale
                    # Full Gamma density at the saturated mean response.
                    total += (
                        (shape - 1) * mp.log(response)
                        - shape
                        - mp.loggamma(shape)
                        - shape * mp.log(response * scale / weight)
                    )
                return total

            value = [
                float(mp.diff(log_density, mp.mpf(phi), order)) for order in range(3)
            ]
        expected = {"mpmath_version": mp.__version__, "values": value}
        save_reference("gamma_dispersion_high_precision", key, expected)
    return expected["values"], weights


@pytest.mark.parametrize("phi", [1e-15, 1e-12, 1e-9, 1e-5, 1e-3, 0.1, 1, 10])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_gamma_ls_small_dispersion_and_derivatives(phi, zero_weights):
    expected, weights = high_precision_reference(phi, zero_weights)
    family = make_gam_family("gamma")
    np.testing.assert_allclose(
        family.ls(Y, weights, scale=phi), expected, atol=2e-9, rtol=2e-10
    )
    np.testing.assert_allclose(
        family.saturated_loglik(Y, weights=weights, scale=phi),
        expected[0],
        atol=2e-9,
        rtol=2e-10,
    )
    if phi >= 1e-3:
        np.testing.assert_allclose(
            family.ls(Y, weights, scale=phi),
            upstream_reference(phi, weights),
            atol=2e-8,
            rtol=2e-8,
        )
    elif phi <= 1e-9:
        # Retain actual upstream observations, but avoid copying cancellation
        # that can even reverse its second derivative at phi=1e-15.
        raw = upstream_reference(phi, weights)
        denominator = 1 + np.abs(expected)
        own_error = np.linalg.norm(
            (family.ls(Y, weights, scale=phi) - expected) / denominator
        )
        source_error = np.linalg.norm((raw - expected) / denominator)
        assert own_error < source_error


def test_gamma_ls_excludes_all_zero_weight_rows():
    family = make_gam_family("gamma")
    np.testing.assert_array_equal(
        family.ls(Y, np.zeros(len(Y)), scale=1e-15), np.zeros(3)
    )


@pytest.mark.parametrize("phi", [1e-15, 1e-12, 1e-9, 1e-5])
@pytest.mark.parametrize("method", ["ML", "REML"])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_gamma_scale_profile_retains_small_stationary_solutions(
    phi, method, zero_weights
):
    from nampy.gam.fit.selection.criteria.pirls.family_gamma import (
        _solve_gamma_profile_scale,
    )

    # The independently differentiated density determines a positive Dp whose
    # profile has its stationary point at phi. This exercises the actual scale
    # solver with the same weight, gamma and null-rank corrections as fitting.
    expected, weights = high_precision_reference(phi, zero_weights)
    model = SimpleNamespace(
        family=make_gam_family("gamma"),
        prior_weights_=weights,
        score_gamma=1.3,
        n_true_=None,
    )
    mp = 1
    dp = -2 * phi**2 * expected[1]
    if method == "REML":
        dp -= model.score_gamma * mp * phi
    fitted = _solve_gamma_profile_scale(
        model,
        Y,
        dp,
        mp,
        method=method,
        init_scale=phi * 1.1,
    )
    np.testing.assert_allclose(fitted, phi, rtol=2e-10, atol=0)
