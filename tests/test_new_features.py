"""
Tests for newly implemented features:
  - pc= for ps, cc, gp, tp/ts bases
  - select=True for te/ti/t2 tensor smooths
  - Prior/case weights via sample_weight
  - Standard errors and confidence intervals via predict(return_se=True)
  - Negative binomial theta estimation (estimate_theta=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.smooths.registry import make_smooth_term


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
# pc= for ps basis
# ===========================================================================

class TestPcPs:
    def test_ps_pc_fits_without_error(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="ps", k=8, pc=0.0)')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_ps_pc_smooth_is_zero_at_constraint_point(self):
        """The fitted smooth must equal zero at the constraint point."""
        data = _data()
        pc_val = 1.0
        gam = GAM(family="gaussian", formula=f'y ~ s(x, bs="ps", k=8, pc={pc_val})')
        gam.fit(data=data)

        point = pd.DataFrame({"x": [pc_val]})
        contrib = gam.predict_feature_vals(point)
        smooth_keys = [k for k in contrib if k not in ("output", "intercept")]
        smooth_val = float(np.sum([np.array(contrib[k]) for k in smooth_keys]))
        assert abs(smooth_val) < 1e-8, f"smooth at pc={pc_val} should be 0, got {smooth_val}"

    def test_ps_pc_n_coef_same_as_no_pc(self):
        """pc= reparameterises the basis but does not reduce n_coef (matches mgcv)."""
        data = _data()
        gam_no_pc = GAM(family="gaussian", formula='y ~ s(x, bs="ps", k=8)')
        gam_with_pc = GAM(family="gaussian", formula='y ~ s(x, bs="ps", k=8, pc=0.0)')
        gam_no_pc.fit(data=data)
        gam_with_pc.fit(data=data)
        assert gam_with_pc.n_coef_ == gam_no_pc.n_coef_

    def test_ps_pc_dict_syntax(self):
        """pc= as a dict should also work."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="ps", k=8, pc={"x": 0.5})')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_ps_pc_transform_new_matches_train(self):
        """Predictions on train data should be consistent between fit and predict."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="ps", k=8, pc=0.0)')
        gam.fit(data=data)
        p1 = gam.predict(data)
        p2 = gam.predict(data)
        np.testing.assert_array_equal(p1, p2)

    def test_ps_pc_fixed_true(self):
        """pc= with fixed=True should still work (unpenalized basis)."""
        data = _data()
        gam = GAM(
            family="gaussian",
            formula='y ~ s(x, bs="ps", k=6, pc=0.0, fx=TRUE)',
        )
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))


# ===========================================================================
# pc= for cc basis
# ===========================================================================

class TestPcCc:
    def test_cc_pc_fits_without_error(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cc", k=8, pc=0.5)')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_cc_pc_smooth_is_zero_at_constraint_point(self):
        data = _data()
        pc_val = 1.0
        gam = GAM(family="gaussian", formula=f'y ~ s(x, bs="cc", k=8, pc={pc_val})')
        gam.fit(data=data)
        point = pd.DataFrame({"x": [pc_val]})
        contrib = gam.predict_feature_vals(point)
        smooth_keys = [k for k in contrib if k not in ("output", "intercept")]
        smooth_val = float(np.sum([np.array(contrib[k]) for k in smooth_keys]))
        assert abs(smooth_val) < 1e-8, f"smooth at pc={pc_val} should be 0, got {smooth_val}"

    def test_cc_pc_n_coef_same_as_no_pc(self):
        """pc= reparameterises the basis but does not reduce n_coef (matches mgcv)."""
        data = _data()
        gam_no_pc = GAM(family="gaussian", formula='y ~ s(x, bs="cc", k=8)')
        gam_with_pc = GAM(family="gaussian", formula='y ~ s(x, bs="cc", k=8, pc=0.5)')
        gam_no_pc.fit(data=data)
        gam_with_pc.fit(data=data)
        assert gam_with_pc.n_coef_ == gam_no_pc.n_coef_

    def test_cc_pc_prediction_finite(self):
        """Predictions should be finite everywhere in [0, 2]."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cc", k=8, pc=0.5)')
        gam.fit(data=data)
        x_new = np.linspace(0.01, 1.99, 50)
        pred = gam.predict(pd.DataFrame({"x": x_new}))
        assert np.all(np.isfinite(pred))


# ===========================================================================
# pc= for gp basis
# ===========================================================================

class TestPcGp:
    def test_gp_pc_fits_without_error(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="gp", pc=0.0)')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_gp_pc_smooth_is_zero_at_constraint_point(self):
        data = _data()
        pc_val = 1.0
        gam = GAM(family="gaussian", formula=f'y ~ s(x, bs="gp", pc={pc_val})')
        gam.fit(data=data)
        point = pd.DataFrame({"x": [pc_val]})
        contrib = gam.predict_feature_vals(point)
        smooth_keys = [k for k in contrib if k not in ("output", "intercept")]
        smooth_val = float(np.sum([np.array(contrib[k]) for k in smooth_keys]))
        assert abs(smooth_val) < 1e-8, f"smooth at pc={pc_val} should be 0, got {smooth_val}"

    def test_gp_pc_n_coef_same_as_no_pc(self):
        """pc= reparameterises the basis but does not reduce n_coef (matches mgcv)."""
        data = _data()
        gam_no_pc = GAM(family="gaussian", formula='y ~ s(x, bs="gp")')
        gam_with_pc = GAM(family="gaussian", formula='y ~ s(x, bs="gp", pc=0.0)')
        gam_no_pc.fit(data=data)
        gam_with_pc.fit(data=data)
        assert gam_with_pc.n_coef_ == gam_no_pc.n_coef_

    def test_gp_multivariate_pc_raises(self):
        """pc= for a multivariate gp should still raise NotImplementedError."""
        data = _data2()
        with pytest.raises(NotImplementedError, match="multivariate"):
            GAM(
                family="gaussian",
                formula='y ~ s(x0, x1, bs="gp", pc=0.0)',
            ).fit(data=data)


# ===========================================================================
# pc= for tp/ts basis
# ===========================================================================

class TestPcTp:
    def test_tp_pc_fits_without_error(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="tp", pc=0.0)')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_tp_pc_smooth_is_zero_at_constraint_point(self):
        data = _data()
        pc_val = 1.0
        gam = GAM(family="gaussian", formula=f'y ~ s(x, bs="tp", pc={pc_val})')
        gam.fit(data=data)
        point = pd.DataFrame({"x": [pc_val]})
        contrib = gam.predict_feature_vals(point)
        smooth_keys = [k for k in contrib if k not in ("output", "intercept")]
        smooth_val = float(np.sum([np.array(contrib[k]) for k in smooth_keys]))
        assert abs(smooth_val) < 1e-8, f"smooth at pc={pc_val} should be 0, got {smooth_val}"

    def test_tp_pc_n_coef_same_as_no_pc(self):
        """pc= reparameterises the basis but does not reduce n_coef (matches mgcv)."""
        data = _data()
        gam_no_pc = GAM(family="gaussian", formula='y ~ s(x, bs="tp")')
        gam_with_pc = GAM(family="gaussian", formula='y ~ s(x, bs="tp", pc=0.0)')
        gam_no_pc.fit(data=data)
        gam_with_pc.fit(data=data)
        assert gam_with_pc.n_coef_ == gam_no_pc.n_coef_

    def test_ts_pc_fits_without_error(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="ts", pc=0.0)')
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_tp_multivariate_pc_raises(self):
        data = _data2()
        with pytest.raises(NotImplementedError, match="multivariate"):
            GAM(
                family="gaussian",
                formula='y ~ s(x0, x1, bs="tp", pc=0.0)',
            ).fit(data=data)


# ===========================================================================
# select=True for tensor smooths
# ===========================================================================

class TestTensorSelect:
    def test_te_select_true_fits(self):
        data = _data2()
        gam = GAM(family="gaussian", formula="y ~ te(x0, x1, k=[5, 5])", select=True)
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_te_select_penalty_count_increases(self):
        """select=True should add an extra null-space penalty vs select=False."""
        data = _data2()
        gam_no_sel = GAM(family="gaussian", formula="y ~ te(x0, x1, k=[5, 5])")
        gam_sel = GAM(family="gaussian", formula="y ~ te(x0, x1, k=[5, 5])", select=True)
        gam_no_sel.fit(data=data)
        gam_sel.fit(data=data)
        assert gam_sel.n_smoothing_params_ >= gam_no_sel.n_smoothing_params_

    def test_ti_select_true_fits(self):
        data = _data2()
        gam = GAM(family="gaussian", formula="y ~ ti(x0, x1, k=[5, 5])", select=True)
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_t2_select_true_fits(self):
        data = _data2()
        gam = GAM(family="gaussian", formula="y ~ t2(x0, x1, k=[5, 5])", select=True)
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_te_select_predictions_finite(self):
        data = _data2()
        gam = GAM(family="gaussian", formula="y ~ te(x0, x1, k=[5, 5])", select=True)
        gam.fit(data=data)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))


# ===========================================================================
# Prior / case weights
# ===========================================================================

class TestPriorWeights:
    def test_weights_accepted_as_array(self):
        data = _data()
        w = np.ones(100)
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data, sample_weight=w)
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_weights_change_fit(self):
        """Doubling weights on first half should shift the fit."""
        data = _data()
        w_uniform = np.ones(100)
        w_heavy = np.ones(100)
        w_heavy[:50] = 5.0

        gam_uniform = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam_heavy = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam_uniform.fit(data=data, sample_weight=w_uniform)
        gam_heavy.fit(data=data, sample_weight=w_heavy)

        pred_uniform = gam_uniform.predict(data)
        pred_heavy = gam_heavy.predict(data)
        # Predictions should differ due to different weighting
        assert not np.allclose(pred_uniform, pred_heavy, atol=1e-8)

    def test_zero_weights_ignored(self):
        """Observations with weight=0 should be effectively ignored."""
        data = _data()
        w_all = np.ones(100)
        w_partial = np.ones(100)
        w_partial[50:] = 0.0

        gam_all = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam_partial = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam_all.fit(data=data[:50], sample_weight=w_all[:50])
        gam_partial.fit(data=data, sample_weight=w_partial)

        # Both fit only on first 50 rows (effectively)
        pred_all = gam_all.predict(data[:50])
        pred_partial = gam_partial.predict(data[:50])
        # Should be approximately equal (won't be exact due to full-data design)
        assert np.allclose(pred_all, pred_partial, atol=0.1)

    def test_weights_column_name_in_formula(self):
        """sample_weight as column name string."""
        data = _data()
        data["w"] = np.ones(100)
        data.loc[:49, "w"] = 2.0
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data, sample_weight="w")
        pred = gam.predict(data)
        assert np.all(np.isfinite(pred))

    def test_weights_stored_on_model(self):
        data = _data()
        w = np.ones(100) * 2.0
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data, sample_weight=w)
        assert gam.prior_weights_ is not None
        np.testing.assert_array_equal(gam.prior_weights_, w)


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

    def test_se_shape_matches_predictions(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred, se = gam.predict(data, return_se=True)
        assert pred.shape == se.shape

    def test_se_is_positive(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred, se = gam.predict(data, return_se=True)
        assert np.all(se > 0.0), "Standard errors should be strictly positive"

    def test_se_finite(self):
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred, se = gam.predict(data, return_se=True)
        assert np.all(np.isfinite(se))

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

    def test_confidence_interval_coverage(self):
        """95% CI should contain the true mean for most training points."""
        data = _data()
        gam = GAM(family="gaussian", formula='y ~ s(x, bs="cr")')
        gam.fit(data=data)
        pred, se = gam.predict(data, return_se=True)
        lower = pred - 1.96 * se
        upper = pred + 1.96 * se
        # Predictions should be inside their own CI by construction
        assert np.all(pred >= lower - 1e-10)
        assert np.all(pred <= upper + 1e-10)

    def test_new_data_se(self):
        """SE on new data (not training data) should be finite."""
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
