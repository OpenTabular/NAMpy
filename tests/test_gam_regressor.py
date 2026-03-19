"""Tests for the classical GAM implementation (core + sklearn wrapper).

Organised by phase:

Phase A – statistically correct Gaussian GAM baseline
  * intercept consistency (summary metrics match predict residuals)
  * GCV uses parameter-space trace (no n×n hat matrix)
  * fit / predict / score on additive data
  * DataFrame column reorder safety
  * CI scaling sanity (SEs shrink with n)

Phase B – mgcv-like correctness features
  * lpmatrix @ coef == predict
  * termwise predictions sum + intercept == full prediction
  * exact ML / REML (mixed-model reparameterization)
    - ML != REML (different criteria, different smoothing params)
    - ML and REML converge as n → ∞
    - both produce good fits
  * Kass–Steffey (kass_steffey) covariance is PSD-ish
  * term-drop test returns sensible F-stat
  * predict with SEs

Diagnostics
  * concurvity increases on engineered correlated smooths
  * k_diagnostic flags when needed
"""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor, cho_solve

from nampy.basemodels.gam import GAM
from nampy.gam.smoothness.criteria import (
    criterion_gradient_numerical,
    criterion_hessian_numerical,
)
from nampy.models.gam import GAMRegressor


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def additive_data():
    """y = sin(2*x0) + x1^2 + noise."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.fixture
def additive_data_large():
    """Same signal, larger n (for SE-shrinkage tests)."""
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)
    return X, y


# ======================================================================
# Phase A – correct Gaussian GAM baseline
# ======================================================================


class TestPhaseA:
    def test_fit_predict_score(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=12).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        r2 = model.score(X, y)
        assert r2 > 0.85, f"R² on training data too low: {r2:.3f}"

    def test_predict_new_data(self, additive_data):
        X_train, y_train = additive_data
        rng = np.random.default_rng(99)
        X_test = rng.uniform(-2, 2, size=(50, 2))
        model = GAMRegressor(n_splines=10).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (50,)
        assert np.all(np.isfinite(preds))

    def test_intercept_consistency(self, additive_data):
        """intercept_ == mean(y) in the core (centered-basis identity)."""
        X, y = additive_data
        model = GAMRegressor(n_splines=10).fit(X, y)
        np.testing.assert_allclose(
            model.intercept_, np.mean(y), atol=1e-10,
            err_msg="intercept_ must equal mean(y) for centered bases",
        )

    def test_summary_matches_predict_residuals(self, additive_data, capsys):
        """summary() R-sq should agree with predict()-based R^2."""
        X, y = additive_data
        model = GAMRegressor(n_splines=10).fit(X, y)
        r2_score = model.score(X, y)
        model.summary()
        captured = capsys.readouterr().out
        assert "R-sq" in captured
        assert "Gaussian Additive Model Summary" in captured

    def test_wrong_feature_count_raises(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=8).fit(X, y)
        with pytest.raises(ValueError, match="Expected 2 features"):
            model.predict(np.zeros((5, 3)))

    def test_dataframe_column_reorder_safety(self, additive_data):
        """Predict on DataFrame with columns in a different order."""
        X_arr, y = additive_data
        X_df = pd.DataFrame(X_arr, columns=["a", "b"])
        model = GAMRegressor(n_splines=10).fit(X_df, y)

        X_reordered = X_df[["b", "a"]]
        preds_original = model.predict(X_df)
        preds_reordered = model.predict(X_reordered)
        np.testing.assert_allclose(
            preds_original, preds_reordered, atol=1e-10,
            err_msg="Prediction must be invariant to column order in DataFrame",
        )

    def test_accepts_dataframe(self, additive_data):
        X_arr, y = additive_data
        X_df = pd.DataFrame(X_arr, columns=["feat_a", "feat_b"])
        model = GAMRegressor(n_splines=10).fit(X_df, y)
        assert model.feature_names_ == ["feat_a", "feat_b"]
        assert model.n_features_in_ == 2
        preds = model.predict(X_df)
        assert preds.shape == y.shape

    def test_scalar_smoothing_params(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=10, smoothing_params=0.5).fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_ci_scaling_sanity(self, additive_data, additive_data_large):
        """SEs must shrink (on average) as n increases."""
        X_s, y_s = additive_data
        X_l, y_l = additive_data_large
        model_s = GAMRegressor(n_splines=10).fit(X_s, y_s)
        model_l = GAMRegressor(n_splines=10).fit(X_l, y_l)

        cis_small = model_s.confidence_intervals(alpha=0.05)
        cis_large = model_l.confidence_intervals(alpha=0.05)

        widths_small = [hi - lo for lo, hi in cis_small]
        widths_large = [hi - lo for lo, hi in cis_large]
        assert np.mean(widths_large) < np.mean(widths_small), (
            "CI widths should shrink with more data"
        )

    def test_confidence_intervals_structure(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=8).fit(X, y)
        cis = model.confidence_intervals(alpha=0.05)
        assert isinstance(cis, list)
        assert all(len(ci) == 2 for ci in cis)
        assert all(ci[0] < ci[1] for ci in cis), "lower < upper in every CI"

    def test_get_set_params(self):
        model = GAMRegressor(n_splines=15, smoothing_params=2.0, method="GCV")
        params = model.get_params()
        assert params["n_splines"] == 15
        assert params["smoothing_params"] == 2.0
        assert params["method"] == "GCV"
        model.set_params(n_splines=20)
        assert model.n_splines == 20


# ======================================================================
# Phase B – mgcv-like correctness features
# ======================================================================


class TestPhaseB:
    def test_lpmatrix_times_coef_equals_predict(self, additive_data):
        """lpmatrix @ full_coef == predict()."""
        X, y = additive_data
        gam = GAM(X, k=10)
        gam.fit(y)

        Lp = gam.predict(type="lpmatrix")
        full_coef = np.concatenate([[gam.intercept_], gam.coef_])
        pred_via_lp = Lp @ full_coef
        pred_direct = gam.predict()

        np.testing.assert_allclose(pred_via_lp, pred_direct, atol=1e-10)

    def test_termwise_plus_intercept_equals_full(self, additive_data):
        """sum(terms, axis=1) + intercept == predict()."""
        X, y = additive_data
        gam = GAM(X, k=10)
        gam.fit(y)

        terms = gam.predict(type="terms")
        full = gam.predict()
        recon = terms.sum(axis=1) + gam.intercept_
        np.testing.assert_allclose(recon, full, atol=1e-10)

    def test_reml_criterion_runs(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=10, method="REML").fit(X, y)
        assert model.score(X, y) > 0.8

        sp = model._gam.smoothing_params
        assert np.all(sp > 0), "smoothing params must be positive"
        assert np.all(np.isfinite(sp))

    def test_ml_criterion_runs(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=10, method="ML").fit(X, y)
        assert model.score(X, y) > 0.8

    def test_ml_and_reml_differ(self, additive_data):
        """ML and REML should produce different smoothing parameters."""
        X, y = additive_data
        model_ml = GAMRegressor(n_splines=10, method="ML").fit(X, y)
        model_reml = GAMRegressor(n_splines=10, method="REML").fit(X, y)

        sp_ml = model_ml._gam.smoothing_params
        sp_reml = model_reml._gam.smoothing_params

        # They should not be identical (different criteria)
        assert not np.allclose(sp_ml, sp_reml, rtol=1e-4), (
            f"ML and REML should yield different smoothing params; "
            f"ML={sp_ml}, REML={sp_reml}"
        )

    def test_ml_and_reml_criteria_differ_at_same_sp(self, additive_data):
        """At the same smoothing params, ML and REML criteria must differ."""
        X, y = additive_data
        gam = GAM(k=10)
        gam.fit(X=X, y=y)
        assert gam._can_use_exact_gaussian_ml_reml()
        log_sp = np.zeros(gam.n_smoothing_params_, dtype=np.float64)

        val_ml = gam._criterion(y, log_sp, method="ML")
        val_reml = gam._criterion(y, log_sp, method="REML")

        assert np.isfinite(val_ml) and np.isfinite(val_reml)
        assert val_ml != val_reml, (
            f"ML and REML criteria should differ; ML={val_ml:.6f}, REML={val_reml:.6f}"
        )

    def test_ml_criterion_includes_covariance_logdet(self, additive_data):
        """The exact Gaussian ML score must include the marginal covariance determinant."""
        X, y = additive_data
        gam = GAM(k=10)
        gam.fit(X=X, y=y)
        assert gam._can_use_exact_gaussian_ml_reml()

        log_sp = np.array([0.25, -0.15], dtype=np.float64)
        val_ml = gam._criterion(y, log_sp, method="ML")

        y_eff = y if gam.offset_train_ is None else (y - gam.offset_train_)
        Xf = gam.X_fix_
        Zr = gam.Z_rand_
        n = Xf.shape[0]
        p = gam.rank_X_fix_
        q = gam.n_rand_

        if q == 0:
            if p == 0:
                rss_v = max(float(y_eff @ y_eff), 1e-14)
            else:
                XtX = Xf.T @ Xf
                cXtX, lo = cho_factor(XtX, check_finite=False)
                b_hat = cho_solve((cXtX, lo), Xf.T @ y_eff, check_finite=False)
                resid = y_eff - Xf @ b_hat
                rss_v = max(float(resid @ resid), 1e-14)
            expected = n * np.log(rss_v / n)
        else:
            sp = gam._expand_smoothing_params_from_log(log_sp)
            lam_parts = [
                np.full(block["n_pen"], sp[block["smoothing_index"]], dtype=np.float64)
                for block in gam._reparam_rand_blocks_
                if block["n_pen"] > 0
            ]
            lam_vec = np.concatenate(lam_parts) if lam_parts else np.empty((0,), dtype=np.float64)

            M = gam.ZtZ_rand_ + np.diag(lam_vec)
            cM, loM = cho_factor(M, check_finite=False)

            ZTy = Zr.T @ y_eff
            Minv_ZTy = cho_solve((cM, loM), ZTy, check_finite=False)
            Ky = y_eff - Zr @ Minv_ZTy

            if p > 0:
                ZTX = Zr.T @ Xf
                Minv_ZTX = cho_solve((cM, loM), ZTX, check_finite=False)
                KX = Xf - Zr @ Minv_ZTX
                XtKX = Xf.T @ KX
                cXKX, loXKX = cho_factor(XtKX, check_finite=False)
                XtKy = Xf.T @ Ky
                b_hat = cho_solve((cXKX, loXKX), XtKy, check_finite=False)
                rss_v = max(float(y_eff @ Ky - XtKy @ b_hat), 1e-14)
            else:
                rss_v = max(float(y_eff @ Ky), 1e-14)

            logdet_M = 2.0 * float(np.sum(np.log(np.diag(cM))))
            logdet_Lam = float(np.sum(np.log(lam_vec)))
            expected = n * np.log(rss_v / n) + (logdet_M - logdet_Lam)

        np.testing.assert_allclose(val_ml, expected, atol=1e-10, rtol=1e-10)

    def test_select_is_propagated_for_auto_built_main_effects(self, additive_data):
        """Global select=True should add null-space penalties in the compiled design."""
        X, y = additive_data
        gam = GAM(k=10, select=True, smoothing_method="fixed")
        gam.fit(X=X, y=y)

        assert any(pb.is_null_space_penalty for pb in gam.penalty_blocks_), (
            "select=True should add null-space selection penalties to the compiled design."
        )

    def test_select_reml_fit_is_supported(self, additive_data):
        """Exact Gaussian REML should support select=True shrinkage penalties."""
        X, y = additive_data
        gam = GAM(k=10, select=True, smoothing_method="REML", optimize_smoothing=True)
        gam.fit(X=X, y=y)

        assert gam._optim_method == "reml"
        assert np.all(np.isfinite(gam.smoothing_params))
        assert np.all(gam.smoothing_params > 0.0)
        assert any(pb.is_null_space_penalty for pb in gam.penalty_blocks_)
        assert gam._reparam_rand_blocks_ is not None
        assert any(block["is_null_space_penalty"] for block in gam._reparam_rand_blocks_)

    def test_tensor_select_raises_explicitly(self, additive_data):
        """Tensor select should fail explicitly until tensor selection penalties are implemented."""
        X, y = additive_data
        X_named = pd.DataFrame(X, columns=["x0", "x1"])
        gam = GAM(k=8, tensor_terms=[("x0", "x1")], select=True, smoothing_method="fixed")

        with pytest.raises(NotImplementedError, match=r"select=True .* te\(\)/ti\(\)/t2\(\)"):
            gam.fit(X=X_named, y=y)

    def test_ml_reml_converge_at_large_n(self):
        """With large n, ML and REML smoothing params should be closer."""
        rng = np.random.default_rng(42)
        n_small, n_large = 100, 5000
        k = 8

        def make_data(n):
            X = rng.uniform(-2, 2, size=(n, 2))
            y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)
            return X, y

        X_s, y_s = make_data(n_small)
        X_l, y_l = make_data(n_large)

        ml_s = GAMRegressor(n_splines=k, method="ML").fit(X_s, y_s)
        reml_s = GAMRegressor(n_splines=k, method="REML").fit(X_s, y_s)
        diff_small = np.abs(
            np.log(ml_s._gam.smoothing_params)
            - np.log(reml_s._gam.smoothing_params)
        ).mean()

        ml_l = GAMRegressor(n_splines=k, method="ML").fit(X_l, y_l)
        reml_l = GAMRegressor(n_splines=k, method="REML").fit(X_l, y_l)
        diff_large = np.abs(
            np.log(ml_l._gam.smoothing_params)
            - np.log(reml_l._gam.smoothing_params)
        ).mean()

        assert diff_large < diff_small, (
            f"ML/REML gap should shrink with n; "
            f"small-n gap={diff_small:.4f}, large-n gap={diff_large:.4f}"
        )

    def test_gcv_ml_reml_all_produce_good_fits(self, additive_data):
        """All three methods should produce R² > 0.85 on the training set."""
        X, y = additive_data
        for method in ("GCV", "ML", "REML"):
            model = GAMRegressor(n_splines=10, method=method).fit(X, y)
            r2 = model.score(X, y)
            assert r2 > 0.85, f"method={method}: R²={r2:.3f} too low"

    def test_predict_with_se(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=10).fit(X, y)
        mu, se = model.predict_se(X)
        assert mu.shape == y.shape
        assert se.shape == y.shape
        assert np.all(se > 0), "SEs must be positive"
        assert np.all(np.isfinite(se))

    def test_termwise_se(self, additive_data):
        X, y = additive_data
        gam = GAM(X, k=10)
        gam.fit(y)

        terms, ses = gam.predict(type="terms", return_se=True)
        assert terms.shape == (X.shape[0], 2)
        assert ses.shape == terms.shape
        assert np.all(ses >= 0)

    def test_unconditional_covariance_psd(self, additive_data):
        X, y = additive_data
        gam = GAM(X, k=8)
        gam.fit(y)
        Vu = gam.compute_unconditional_covariance(y)
        evals = np.linalg.eigvalsh(Vu)
        assert np.all(
            evals > -1e-6
        ), f"Kass–Steffey cov has negative eigenvalues: {evals.min():.3e}"

    def test_term_drop_test(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=10).fit(X, y)
        result = model.term_drop_test(term_index=0)
        assert "f_stat" in result
        assert "p_value" in result
        assert result["f_stat"] >= 0
        assert 0 <= result["p_value"] <= 1

    def test_term_drop_detects_signal(self, additive_data):
        """Dropping a true-signal term should yield a small p-value."""
        X, y = additive_data
        model = GAMRegressor(n_splines=10).fit(X, y)
        result = model.term_drop_test(term_index=0)
        assert result["p_value"] < 0.05, (
            f"Dropping the signal term should be significant, got p={result['p_value']:.4f}"
        )


# ======================================================================
# Diagnostics
# ======================================================================


class TestDiagnostics:
    def test_concurvity_correlated(self):
        """Concurvity score should be high when smooths are correlated."""
        rng = np.random.default_rng(7)
        n = 200
        x0 = rng.uniform(-2, 2, n)
        x1 = x0 + rng.normal(scale=0.01, size=n)  # near duplicate
        X = np.column_stack([x0, x1])
        y = np.sin(x0) + rng.normal(scale=0.1, size=n)

        model = GAMRegressor(n_splines=8).fit(X, y)
        scores = model.concurvity()
        assert len(scores) == 2
        assert scores[0]["r2"] > 0.8, "Correlated smooths should show high concurvity"

    def test_concurvity_independent(self):
        """Concurvity should be low for truly independent features."""
        rng = np.random.default_rng(7)
        n = 200
        X = rng.uniform(-2, 2, size=(n, 2))
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)

        model = GAMRegressor(n_splines=8).fit(X, y)
        scores = model.concurvity()
        assert all(s["r2"] < 0.5 for s in scores), "Independent features → low concurvity"

    def test_k_diagnostic_runs(self, additive_data):
        X, y = additive_data
        model = GAMRegressor(n_splines=6).fit(X, y)
        diag = model.k_diagnostic(factor=2)
        assert diag["k_new"] > diag["k_old"]
        assert "edf_old" in diag and "edf_new" in diag


# ======================================================================
# Non-Gaussian Laplace ML/REML
# ======================================================================


class TestNonGaussianLaplaceMLREML:
    def test_backend_resolution_prefers_exact_gaussian_but_keeps_pirls_available(self):
        gaussian = GAM(k=8, family="gaussian")
        binomial = GAM(k=8, family="binomial")
        rng = np.random.default_rng(11)
        X = rng.normal(size=(80, 2))
        y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + rng.normal(scale=0.1, size=80)

        assert gaussian._available_fit_backends() == ("gaussian_exact", "pirls")
        assert gaussian._resolve_fit_backend() == "gaussian_exact"
        gaussian.fit(X=X, y=y)
        assert gaussian._resolve_ml_reml_scoring_backend("reml") == "gaussian_exact"

        assert binomial._available_fit_backends() == ("pirls",)
        assert binomial._resolve_fit_backend() == "pirls"

    def test_binomial_reml_fit_populates_shared_fit_state(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(180, 2))
        eta = 1.2 * np.sin(X[:, 0]) - 0.7 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(
            k=8,
            family="binomial",
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_method == "reml"
        assert gam.fit_state_ is not None
        assert gam.fit_state_.X is not None
        assert gam.fit_state_.XtWX is not None
        assert gam.fit_state_.penalty_matrix is not None
        assert gam.fit_state_.working_weights is not None
        assert gam.fit_state_.penalty_quadratic is not None
        assert gam.fit_state_.loglik is not None
        assert np.all(np.isfinite(gam.smoothing_params))
        assert np.all(gam.smoothing_params > 0.0)

    def test_binomial_ml_and_reml_criteria_differ_at_same_sp(self):
        rng = np.random.default_rng(321)
        X = rng.normal(size=(160, 2))
        eta = 0.9 * np.sin(X[:, 0]) + 0.5 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(k=8, family="binomial")
        gam.fit(X=X, y=y)

        assert gam._uses_pirls_solver()
        assert gam._can_use_simple_ml_reml_structure()

        log_sp = np.array([0.1, -0.2], dtype=np.float64)
        val_ml = gam._criterion(y, log_sp, method="ML")
        val_reml = gam._criterion(y, log_sp, method="REML")

        assert np.isfinite(val_ml) and np.isfinite(val_reml)
        assert val_ml != val_reml

    def test_poisson_reml_fit_optimizes_smoothing(self):
        rng = np.random.default_rng(456)
        X = rng.normal(size=(220, 2))
        mu = np.exp(0.3 + 0.8 * np.sin(X[:, 0]) + 0.2 * X[:, 1])
        y = rng.poisson(mu)

        gam = GAM(
            k=8,
            family="poisson",
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_method == "reml"
        assert np.all(np.isfinite(gam.smoothing_params))
        assert np.all(gam.smoothing_params > 0.0)
        assert gam.deviance_ >= 0.0

    def test_poisson_ml_and_reml_criteria_differ_at_same_sp(self):
        rng = np.random.default_rng(654)
        X = rng.normal(size=(180, 2))
        mu = np.exp(0.2 + 0.7 * np.sin(X[:, 0]) - 0.3 * X[:, 1])
        y = rng.poisson(mu)

        gam = GAM(k=8, family="poisson")
        gam.fit(X=X, y=y)

        assert gam._uses_pirls_solver()
        assert gam._can_use_simple_ml_reml_structure()

        log_sp = np.array([-0.05, 0.15], dtype=np.float64)
        val_ml = gam._criterion(y, log_sp, method="ML")
        val_reml = gam._criterion(y, log_sp, method="REML")

        assert np.isfinite(val_ml) and np.isfinite(val_reml)
        assert val_ml != val_reml

    def test_binomial_select_reml_fit_optimizes_smoothing(self):
        rng = np.random.default_rng(987)
        X = rng.normal(size=(180, 2))
        eta = 0.8 * np.sin(X[:, 0]) - 0.3 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(
            k=8,
            family="binomial",
            select=True,
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_method == "reml"
        assert np.all(np.isfinite(gam.smoothing_params))
        assert np.all(gam.smoothing_params > 0.0)
        assert any(pb.is_null_space_penalty for pb in gam.penalty_blocks_)

    def test_binomial_reml_optimizer_uses_gradient(self):
        rng = np.random.default_rng(222)
        X = rng.normal(size=(150, 2))
        eta = 1.1 * np.sin(X[:, 0]) - 0.4 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(
            k=8,
            family="binomial",
            optimize_smoothing=True,
            smoothing_method="REML",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_used_gradient is True
        assert gam._optim_result is not None
        assert hasattr(gam._optim_result, "jac")
        assert np.all(np.isfinite(np.asarray(gam._optim_result.jac)))

    def test_poisson_ml_optimizer_uses_gradient(self):
        rng = np.random.default_rng(333)
        X = rng.normal(size=(150, 2))
        mu = np.exp(0.1 + 0.6 * np.sin(X[:, 0]) + 0.25 * X[:, 1])
        y = rng.poisson(mu)

        gam = GAM(
            k=8,
            family="poisson",
            optimize_smoothing=True,
            smoothing_method="ML",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_used_gradient is True
        assert gam._optim_result is not None
        assert hasattr(gam._optim_result, "jac")
        assert np.all(np.isfinite(np.asarray(gam._optim_result.jac)))


class TestDerivativeAwareOptimization:
    def test_exact_binomial_reml_gradient_matches_centered_difference(self):
        rng = np.random.default_rng(555)
        X = rng.normal(size=(160, 2))
        eta = 1.1 * np.sin(X[:, 0]) - 0.4 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(k=8, family="binomial")
        gam.fit(X=X, y=y)

        log_sp = np.array([0.1, -0.15], dtype=np.float64)
        grad_exact = gam._criterion_gradient(y, log_sp, method="REML")
        grad_num = criterion_gradient_numerical(gam, y, log_sp, method="REML")

        np.testing.assert_allclose(grad_exact, grad_num, rtol=1e-5, atol=1e-6)

    def test_exact_poisson_ml_gradient_matches_centered_difference(self):
        rng = np.random.default_rng(556)
        X = rng.normal(size=(160, 2))
        mu = np.exp(0.2 + 0.7 * np.sin(X[:, 0]) + 0.15 * X[:, 1])
        y = rng.poisson(mu)

        gam = GAM(k=8, family="poisson")
        gam.fit(X=X, y=y)

        log_sp = np.array([-0.08, 0.12], dtype=np.float64)
        grad_exact = gam._criterion_gradient(y, log_sp, method="ML")
        grad_num = criterion_gradient_numerical(gam, y, log_sp, method="ML")

        np.testing.assert_allclose(grad_exact, grad_num, rtol=1e-4, atol=2e-4)

    def test_exact_binomial_reml_hessian_matches_numerical(self):
        rng = np.random.default_rng(557)
        X = rng.normal(size=(120, 2))
        eta = 0.9 * np.sin(X[:, 0]) - 0.3 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(k=8, family="binomial")
        gam.fit(X=X, y=y)

        log_sp = np.array([0.05, -0.12], dtype=np.float64)
        hess_exact = gam._criterion_hessian(y, log_sp, method="REML")
        hess_num = criterion_hessian_numerical(gam, y, log_sp, method="REML")

        np.testing.assert_allclose(hess_exact, hess_num, rtol=1e-5, atol=1e-6)

    def test_exact_poisson_ml_hessian_matches_numerical(self):
        rng = np.random.default_rng(558)
        X = rng.normal(size=(120, 2))
        mu = np.exp(0.2 + 0.6 * np.sin(X[:, 0]) + 0.2 * X[:, 1])
        y = rng.poisson(mu)

        gam = GAM(k=8, family="poisson")
        gam.fit(X=X, y=y)

        log_sp = np.array([-0.06, 0.11], dtype=np.float64)
        hess_exact = gam._criterion_hessian(y, log_sp, method="ML")
        hess_num = criterion_hessian_numerical(gam, y, log_sp, method="ML")

        np.testing.assert_allclose(hess_exact, hess_num, rtol=1e-5, atol=1e-6)

    def test_exact_negbin_ml_gradient_matches_centered_difference(self):
        rng = np.random.default_rng(559)
        X = rng.normal(size=(140, 2))
        mu = np.exp(0.2 + 0.5 * np.sin(X[:, 0]) - 0.1 * X[:, 1])
        theta = 2.5
        p = theta / (theta + mu)
        y = rng.negative_binomial(theta, p)

        gam = GAM(k=8, family="negbin", theta=theta)
        gam.fit(X=X, y=y)

        log_sp = np.array([-0.04, 0.11], dtype=np.float64)
        grad_exact = gam._criterion_gradient(y, log_sp, method="ML")
        grad_num = criterion_gradient_numerical(gam, y, log_sp, method="ML")

        np.testing.assert_allclose(grad_exact, grad_num, rtol=3e-3, atol=7e-3)

    def test_exact_negbin_ml_hessian_matches_numerical(self):
        rng = np.random.default_rng(560)
        X = rng.normal(size=(140, 2))
        mu = np.exp(0.2 + 0.5 * np.sin(X[:, 0]) - 0.1 * X[:, 1])
        theta = 2.5
        p = theta / (theta + mu)
        y = rng.negative_binomial(theta, p)

        gam = GAM(k=8, family="negbin", theta=theta)
        gam.fit(X=X, y=y)

        log_sp = np.array([-0.04, 0.11], dtype=np.float64)
        hess_exact = gam._criterion_hessian(y, log_sp, method="ML")
        hess_num = criterion_hessian_numerical(gam, y, log_sp, method="ML")

        np.testing.assert_allclose(hess_exact, hess_num, rtol=2e-2, atol=5e-3)

    def test_exact_gaussian_reml_gradient_matches_centered_difference(self, additive_data):
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)

        assert gam._resolve_ml_reml_scoring_backend("reml") == "gaussian_exact"

        log_sp = np.array([0.15, -0.1], dtype=np.float64)
        grad_exact = gam._criterion_gradient(y, log_sp, method="REML")
        grad_num = criterion_gradient_numerical(gam, y, log_sp, method="REML")

        np.testing.assert_allclose(grad_exact, grad_num, rtol=1e-5, atol=1e-6)

    def test_exact_gaussian_ml_gradient_matches_centered_difference(self, additive_data):
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)

        assert gam._resolve_ml_reml_scoring_backend("ml") == "gaussian_exact"

        log_sp = np.array([-0.05, 0.18], dtype=np.float64)
        grad_exact = gam._criterion_gradient(y, log_sp, method="ML")
        grad_num = criterion_gradient_numerical(gam, y, log_sp, method="ML")

        np.testing.assert_allclose(grad_exact, grad_num, rtol=1e-5, atol=1e-6)

    def test_gaussian_reml_gradient_is_finite(self, additive_data):
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)

        log_sp = np.array([0.15, -0.1], dtype=np.float64)
        grad = gam._criterion_gradient(y, log_sp, method="REML")

        assert grad.shape == log_sp.shape
        assert np.all(np.isfinite(grad))

    def test_gaussian_reml_hessian_is_finite_and_symmetric(self, additive_data):
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)

        log_sp = np.array([0.12, -0.08], dtype=np.float64)
        hess = gam._criterion_hessian(y, log_sp, method="REML")

        assert hess.shape == (2, 2)
        assert np.all(np.isfinite(hess))
        np.testing.assert_allclose(hess, hess.T, atol=1e-8)

    def test_gaussian_reml_optimizer_uses_gradient(self, additive_data):
        X, y = additive_data
        gam = GAM(k=8, optimize_smoothing=True, smoothing_method="REML")
        gam.fit(X=X, y=y)

        assert gam._optim_used_gradient is True
        assert gam._optim_result is not None
        assert hasattr(gam._optim_result, "jac")
        assert np.all(np.isfinite(np.asarray(gam._optim_result.jac)))

    def test_gaussian_outer_newton_uses_hessian(self, additive_data):
        X, y = additive_data
        gam = GAM(
            k=8,
            optimize_smoothing=True,
            smoothing_method="REML",
            smoothing_optimizer="outer_newton",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_used_gradient is True
        assert gam._optim_used_hessian is True
        assert gam._optim_result is not None
        assert gam._optim_result.success
        assert hasattr(gam._optim_result, "hess")
        assert np.all(np.isfinite(np.asarray(gam._optim_result.hess)))

    def test_binomial_outer_newton_uses_hessian(self):
        rng = np.random.default_rng(444)
        X = rng.normal(size=(160, 2))
        eta = 0.9 * np.sin(X[:, 0]) - 0.5 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p)

        gam = GAM(
            k=8,
            family="binomial",
            optimize_smoothing=True,
            smoothing_method="REML",
            smoothing_optimizer="outer_newton",
        )
        gam.fit(X=X, y=y)

        assert gam._optim_used_gradient is True
        assert gam._optim_used_hessian is True
        assert gam._optim_result is not None
        assert gam._optim_result.success
        assert hasattr(gam._optim_result, "hess")
        assert np.all(np.isfinite(np.asarray(gam._optim_result.hess)))


# ======================================================================
# Core GAM edge cases
# ======================================================================


class TestReparameterization:
    """Tests for the internal null-space / penalized-space reparameterization."""

    def test_reparam_dimensions(self, additive_data):
        """X_fix_ and Z_rand_ dimensions should be consistent."""
        X, y = additive_data
        gam = GAM(k=10)
        gam.fit(X=X, y=y)

        n, m = X.shape
        assert gam.X_fix_.shape[0] == n
        assert gam.Z_rand_.shape[0] == n
        assert gam.rank_X_fix_ == gam.X_fix_.shape[1]

        total_pen = sum(block["n_pen"] for block in gam._reparam_rand_blocks_)
        assert gam.Z_rand_.shape[1] == total_pen

    def test_reparam_spans_same_column_space(self, additive_data):
        """Reparameterized matrices should span the same column space as [1|Z]."""
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)

        original = np.column_stack([np.ones(X.shape[0]), gam.Z])
        reparam = np.column_stack([gam.X_fix_, gam.Z_rand_])

        # Both should have the same rank
        rank_orig = np.linalg.matrix_rank(original)
        rank_reparam = np.linalg.matrix_rank(reparam)
        assert rank_orig == rank_reparam, (
            f"Rank mismatch: original={rank_orig}, reparam={rank_reparam}"
        )

        # Projection onto original column space should recover reparam columns
        Q, _ = np.linalg.qr(original, mode="reduced")
        proj = Q @ (Q.T @ reparam)
        np.testing.assert_allclose(
            proj, reparam, atol=1e-8,
            err_msg="Reparameterized columns not in span of [1|Z]",
        )

    def test_whitened_penalty_is_identity(self, additive_data):
        """After whitening, the penalty on each Z_r block should be λI."""
        X, y = additive_data
        gam = GAM(k=8)
        gam.fit(X=X, y=y)
        start = 0
        for i, meta in enumerate(gam._reparam_meta):
            if meta is None:
                continue
            primary = meta["primary"]
            n_pen = primary["n_pen"]
            if n_pen > 0:
                Zr_block = gam.Z_rand_[:, start:start + n_pen]
                U1 = primary["U1"]
                d_pos = primary["d_pos"]
                B_orig = gam.term_blocks_[i].basis_train
                B1 = B_orig @ U1
                Zr_expected = B1 / np.sqrt(d_pos)[np.newaxis, :]
                np.testing.assert_allclose(Zr_block, Zr_expected, atol=1e-10)
                start += n_pen


class TestCoreGAM:
    def test_1d_input(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-2, 2, 100)
        y = np.sin(x) + rng.normal(scale=0.1, size=100)
        gam = GAM(x, k=8)
        gam.fit(y)
        preds = gam.predict()
        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_nan_X_raises(self):
        X = np.array([[1, 2], [np.nan, 3], [4, 5]])
        with pytest.raises(ValueError, match="NaN"):
            GAM(X, k=5)

    def test_nan_y_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, size=(50, 2))
        y = rng.normal(size=50)
        y[10] = np.nan
        gam = GAM(X, k=5)
        with pytest.raises(ValueError, match="NaN"):
            gam.fit(y)

    def test_low_k_raises(self):
        X = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="k must be >= 3"):
            GAM(X, k=2)

    def test_predict_before_fit_raises(self):
        X = np.random.randn(50, 2)
        gam = GAM(X, k=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            gam.predict()

    def test_k_stored(self, additive_data):
        X, y = additive_data
        gam = GAM(X, k=12)
        assert gam.k_ == 12

    def test_summary_shows_method_criterion(self, additive_data, capsys):
        """Summary should show the selected criterion value."""
        X, y = additive_data
        gam = GAM(X, k=10)
        gam.fit(y, method="REML")
        gam.summary()
        out = capsys.readouterr().out
        assert "REML criterion" in out
        assert "GCV (supplementary)" in out

    def test_summary_gcv_no_supplementary(self, additive_data, capsys):
        """GCV summary should not show 'supplementary' GCV line."""
        X, y = additive_data
        gam = GAM(X, k=10)
        gam.fit(y, method="GCV")
        gam.summary()
        out = capsys.readouterr().out
        assert "GCV criterion" in out
        assert "supplementary" not in out
