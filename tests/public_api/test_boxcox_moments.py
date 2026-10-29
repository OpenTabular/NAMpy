"""Moment existence and independent GAMLSS response-space quadrature."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy.distributions import BCPELaw, BCTLaw, DistributionPrediction


def prediction(kind, mu=1.0, sigma=0.4, nu=1.0, tau=5.0):
    law = BCTLaw() if kind == "bct" else BCPELaw()
    return DistributionPrediction(
        law,
        {
            k: np.atleast_1d(v)
            for k, v in {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau}.items()
        },
    )


def test_moments_match_independent_gamlss_density_integrals():
    source = pd.read_csv(
        Path(__file__).parents[1] / "reference_fixtures/boxcox_moments.csv"
    )
    for kind, rows in source.groupby("family"):
        law = prediction(
            kind, **{k: rows[k].to_numpy() for k in ("mu", "sigma", "nu", "tau")}
        )
        assert_allclose(law.mean(), rows["mean"], rtol=2e-8, atol=2e-9)
        assert_allclose(law.variance(), rows.variance, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("kind", ["bct", "bcpe"])
def test_endpoint_power_controls_negative_nu_existence(kind):
    assert np.isinf(prediction(kind, nu=-0.8).mean()[0])
    assert np.isnan(prediction(kind, nu=-0.8).variance()[0])
    law = prediction(kind, nu=-1.5)
    assert np.isfinite(law.mean()[0])
    assert np.isinf(law.variance()[0])


def test_latent_t_tail_and_log_t_moments_diverge():
    assert np.isinf(prediction("bct", nu=0).mean()[0])
    assert np.isinf(prediction("bct", nu=0.2, tau=5).mean()[0])
    law = prediction("bct", nu=0.3, tau=5)
    assert np.isfinite(law.mean()[0])
    assert np.isinf(law.variance()[0])


def test_log_pe_mgf_boundary_and_log_normal_identity():
    assert np.isinf(prediction("bcpe", nu=0, tau=0.8).mean()[0])
    assert np.isinf(prediction("bcpe", nu=0, tau=1, sigma=2).mean()[0])
    assert np.isfinite(prediction("bcpe", nu=0, tau=1, sigma=0.4).variance()[0])
    law = prediction("bcpe", mu=1.4, nu=0, tau=2, sigma=0.4)
    assert_allclose(law.mean(), [1.4 * np.exp(0.4**2 / 2)], rtol=2e-10)
    assert_allclose(
        law.variance(), [1.4**2 * np.exp(0.4**2) * np.expm1(0.4**2)], rtol=2e-10
    )
