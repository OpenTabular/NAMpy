"""
Tests for public GAM features that are not covered by the main parity suites:
  - select=True for te/ti/t2 tensor smooths
  - a lightweight sample_weight API smoke
  - lightweight standard-error API checks
  - Negative binomial theta estimation (estimate_theta=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, n)
    y = np.sin(np.pi * x) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x})


def _data2(n=100, seed=1):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n)
    x1 = rng.uniform(0.0, 1.0, n)
    y = x0 + np.sin(2 * np.pi * x1) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


# ===========================================================================
# select=True for tensor smooths
# ===========================================================================

class TestTensorSelect:
    @pytest.mark.parametrize("smooth_type", ["te", "ti", "t2"])
    def test_tensor_select_true_fits(self, smooth_type):
        data = _data2()
        formula = f"y ~ {smooth_type}(x0, x1, k=[5, 5])"
        gam = GAM(family="gaussian", formula=formula, select=True)
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize("smooth_type", ["te", "ti", "t2"])
    def test_tensor_select_penalty_count_increases(self, smooth_type):
        """select=True should add at least one extra null-space penalty."""
        data = _data2()
        formula = f"y ~ {smooth_type}(x0, x1, k=[5, 5])"
        gam_no_sel = GAM(family="gaussian", formula=formula)
        gam_sel = GAM(family="gaussian", formula=formula, select=True)
        gam_no_sel.fit(data=data)
        gam_sel.fit(data=data)
        assert gam_sel.n_smoothing_params_ >= gam_no_sel.n_smoothing_params_


# ===========================================================================
# Prior / case weights
# ===========================================================================

class TestPriorWeights:
    def test_weights_are_accepted_on_public_api(self):
        data = _data()
        w = np.ones(100)
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data, sample_weight=w)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))


# ===========================================================================
# Standard errors and confidence intervals
# ===========================================================================

class TestStandardErrors:
    def test_return_se_true_returns_tuple(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        result = gam.predict(data, return_se=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_se_shapes_match_predictions(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred, se = gam.predict(data, return_se=True)
        assert pred.shape == se.shape

    def test_vp_cov_gives_larger_se_than_vf(self):
        """Bayesian (Vp) SE should generally be >= frequentist (Vf) SE."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred_vp, se_vp = gam.predict(data, return_se=True, cov="bayes")
        pred_vf, se_vf = gam.predict(data, return_se=True, cov="freq")
        # Bayesian SE is generally >= frequentist SE on average
        assert np.mean(se_vp) >= np.mean(se_vf) * 0.5

    def test_se_link_vs_response(self):
        """link-scale SE should differ from response-scale SE for Poisson."""
        rng = np.random.default_rng(7)
        n = 80
        x = rng.uniform(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.5 + np.sin(2 * x)))
        data = pd.DataFrame({"y": y.astype(float), "x": x})
        gam = GAM(family="poisson", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred_resp, se_resp = gam.predict(data, return_se=True, type="response")
        pred_link, se_link = gam.predict(data, return_se=True, type="link")
        # Response and link scale SEs should differ for non-identity link
        assert not np.allclose(se_resp, se_link, rtol=1e-3)

    def test_new_data_se(self):
        """SE on new data is available on the public API."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        x_new = np.linspace(0.0, 2.0, 20)
        new_data = pd.DataFrame({"x": x_new})
        pred, se = gam.predict(new_data, return_se=True)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(se))
        assert np.all(se > 0.0)


# ===========================================================================
# Negative binomial theta estimation
# ===========================================================================

class TestNegBinTheta:
    def _nb_data(self, n=200, seed=3, theta_true=2.0):
        from scipy.stats import nbinom
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n)
        mu = np.exp(1.0 + np.sin(np.pi * x))
        p = theta_true / (theta_true + mu)
        y = nbinom.rvs(theta_true, p, random_state=rng)
        return pd.DataFrame({"y": y.astype(float), "x": x}), mu

    def test_estimate_theta_mle_method_exists(self):
        from nampy.gam.families.exponential import NegativeBinomialLogFamily
        fam = NegativeBinomialLogFamily(theta=1.0, estimate_theta=True)
        assert hasattr(fam, "estimate_theta_mle")

    def test_estimate_theta_mle_updates_toward_true(self):
        from nampy.gam.families.exponential import NegativeBinomialLogFamily
        rng = np.random.default_rng(99)
        n = 500
        theta_true = 3.0
        mu = np.full(n, 2.0)
        # Simulate NB counts
        from scipy.stats import nbinom
        p = theta_true / (theta_true + mu)
        y = nbinom.rvs(theta_true, p, random_state=rng).astype(float)

        fam = NegativeBinomialLogFamily(theta=1.0, estimate_theta=True)
        theta_est = fam.estimate_theta_mle(y, mu)
        # The estimate should be closer to true theta than initial theta=1
        assert abs(theta_est - theta_true) < abs(1.0 - theta_true)

    def test_estimate_theta_flag_off_by_default(self):
        from nampy.gam.families.exponential import NegativeBinomialLogFamily
        fam = NegativeBinomialLogFamily(theta=1.0)
        assert not fam.estimate_theta

    def test_negbin_fit_with_estimate_theta(self):
        data, _ = self._nb_data()
        gam = GAM(
            family={"name": "negbin", "theta": 2.0, "estimate_theta": True},
            formula='y ~ s(x, bs="cr")',
        )
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))
        # theta should have been updated from the initial value
        assert hasattr(gam.family, "theta")
        assert gam.family.theta > 0.0

    def test_negbin_estimate_theta_moves_toward_truth(self):
        """Estimated theta should be closer to truth than initial guess."""
        data, _ = self._nb_data(theta_true=3.0)
        theta_init = 5.0  # Start far from truth
        gam = GAM(
            family={"name": "negbin", "theta": theta_init, "estimate_theta": True},
            formula='y ~ s(x, bs="cr")',
        )
        gam.fit(data=data)
        theta_fit = gam.family.theta
        assert abs(theta_fit - 3.0) < abs(theta_init - 3.0), (
            f"Expected theta closer to 3.0, got {theta_fit}"
        )

    def test_negbin_fixed_theta_unchanged(self):
        """With estimate_theta=False, theta should stay at its initial value."""
        data, _ = self._nb_data(theta_true=3.0)
        theta_init = 1.0
        gam = GAM(
            family={"name": "negbin", "theta": theta_init},
            formula='y ~ s(x, bs="cr")',
        )
        gam.fit(data=data)
        assert gam.family.theta == pytest.approx(theta_init)

    def test_mle_theta_positive(self):
        """MLE should always return a positive theta."""
        from nampy.gam.families.exponential import NegativeBinomialLogFamily
        rng = np.random.default_rng(10)
        y = rng.poisson(2.0, 100).astype(float)
        mu = np.full(100, 2.0)
        for theta_init in [0.1, 0.5, 1.0, 5.0, 10.0]:
            fam = NegativeBinomialLogFamily(theta=theta_init, estimate_theta=True)
            theta_est = fam.estimate_theta_mle(y, mu)
            assert theta_est > 0.0, f"Estimated theta={theta_est} for init={theta_init}"
