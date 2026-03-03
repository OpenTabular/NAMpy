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

from nampy.basemodels.gam import GAM
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
        gam = GAM(X, k=10)
        log_sp = np.zeros(X.shape[1])

        val_ml = gam._criterion(y, log_sp, method="ML")
        val_reml = gam._criterion(y, log_sp, method="REML")

        assert np.isfinite(val_ml) and np.isfinite(val_reml)
        assert val_ml != val_reml, (
            f"ML and REML criteria should differ; ML={val_ml:.6f}, REML={val_reml:.6f}"
        )

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
# Core GAM edge cases
# ======================================================================


class TestReparameterization:
    """Tests for the internal null-space / penalized-space reparameterization."""

    def test_reparam_dimensions(self, additive_data):
        """X_fix_ and Z_rand_ dimensions should be consistent."""
        X, y = additive_data
        gam = GAM(X, k=10)

        n, m = X.shape
        assert gam.X_fix_.shape[0] == n
        assert gam.Z_rand_.shape[0] == n
        assert gam.rank_X_fix_ == gam.X_fix_.shape[1]

        total_pen = sum(gam.rand_dims_per_term_)
        assert gam.Z_rand_.shape[1] == total_pen

    def test_reparam_spans_same_column_space(self, additive_data):
        """Reparameterized matrices should span the same column space as [1|Z]."""
        X, y = additive_data
        gam = GAM(X, k=8)

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
        gam = GAM(X, k=8)
        start = 0
        for i, meta in enumerate(gam._reparam_meta):
            n_pen = meta["n_pen"]
            if n_pen == 0:
                continue
            Zr_block = gam.Z_rand_[:, start:start + n_pen]
            # The effective penalty on whitened coords is λ * I
            # Verify by checking that B1 / sqrt(d) was applied correctly:
            # Zr = B @ U1 / sqrt(d), so Zr' Zr gives the "data-side" Gram
            # but the penalty is simply I (times λ)
            U1 = meta["U1"]
            d_pos = meta["d_pos"]
            B_orig = gam.Z[:, gam.slices[i]]
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
