"""Parity and finite-difference checks for Gaussian smoothness post-processing."""
from __future__ import annotations

import numpy as np
import pytest

from nampy.basemodels.gam import GAM
from nampy.gam.fit.postprocess.gaussian_smoothness_postprocess import (
    gaussian_smoothness_postprocess,
)
from nampy.gam.smoothness.criteria.gaussian import criterion_ml_reml_exact_dynamic
from nampy.gam.smoothness.criteria.gaussian_reml_algebra import (
    gaussian_reml_laplace_score,
    gaussian_weighted_residual_sum_squares,
    prior_weights_diagonal_from_fit,
    quadratic_form_penalty,
)
from nampy.gam.smoothness.criteria.penalty import (
    _static_penalty_null_dim,
    _stable_penalty_logdet,
)

from test_gam_mgcv_parity import (
    _fit_nampy_model_fixed_sp,
    _make_gaussian_data,
    _make_random_effect_data_noisy,
    _run_mgcv_snapshot,
)


def _fd_grad(f, x0, eps=1e-6):
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    g = np.zeros_like(x0)
    for i in range(x0.size):
        h = np.zeros_like(x0)
        h[i] = eps
        g[i] = (f(x0 + h) - f(x0 - h)) / (2.0 * eps)
    return g


def _fd_hess_diag(f, x0, eps=1e-5):
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    assert x0.size == 1
    e = eps
    return (f(x0 + e) - 2.0 * f(x0) + f(x0 - e)) / (e * e)


class TestGaussianSmoothnessPostprocess:
    def test_profiled_reml_matches_dynamic_criterion(self):
        df = _make_gaussian_data(seed=11, n=100)
        gam = GAM(formula='y ~ s(x0, bs="cr", k=8)', smoothing_method="REML")
        gam.fit(df, df["y"])
        sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        log_sp = np.log(np.maximum(sp, np.finfo(np.float64).tiny))
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))
        post = gaussian_smoothness_postprocess(gam, y, sp, score_type="REML", deriv=0)
        dyn = criterion_ml_reml_exact_dynamic(gam, y, log_sp, "REML")
        assert np.isfinite(post["reml_score_profiled_scale"])
        np.testing.assert_allclose(
            post["reml_score_profiled_scale"],
            dyn,
            rtol=0.0,
            atol=1e-12,
        )

    def test_profiled_reml_matches_mgcv_snapshot_noisy_re(self):
        data = _make_random_effect_data_noisy()
        formula = 'y ~ s(f, bs="re")'
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64).ravel()
        gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
        y = gam.family.validate_y(np.asarray(data["y"], dtype=np.float64))
        post = gaussian_smoothness_postprocess(gam, y, sp, score_type="REML", deriv=0)
        np.testing.assert_allclose(
            post["reml_score_profiled_scale"],
            float(expected["fit"]["criterion_value"]),
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            post["tr_a"],
            float(expected["fit"]["edf_total"]),
            rtol=0.0,
            atol=1e-10,
        )

    def test_gcv_derivatives_match_finite_differences(self):
        df = _make_gaussian_data(seed=21, n=95)
        gam = GAM(formula='y ~ s(x0, bs="cr", k=8)', smoothing_method="GCV")
        gam.fit(df, df["y"])
        sp0 = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))

        def gcv_score(sp):
            return gaussian_smoothness_postprocess(
                gam, y, sp, score_type="GCV", deriv=0
            )["gcv_score"]

        post = gaussian_smoothness_postprocess(gam, y, sp0, score_type="GCV", deriv=2)
        fdg = _fd_grad(lambda t: gcv_score(np.exp(t)), np.log(sp0), eps=1e-6)
        np.testing.assert_allclose(
            post["gcv_grad_log_sp"].ravel(),
            fdg,
            rtol=0.0,
            atol=5e-7,
        )
        fdh = _fd_hess_diag(lambda t: gcv_score(np.exp(t)), np.log(sp0), eps=1e-3)
        np.testing.assert_allclose(
            float(post["gcv_hess_log_sp"][0, 0]),
            float(fdh),
            rtol=0.0,
            atol=5e-7,
        )

    def test_reml_gradient_matches_fd_with_frozen_scale(self):
        df = _make_gaussian_data(seed=33, n=88)
        gam = GAM(formula='y ~ s(x0, bs="cr", k=8)', smoothing_method="REML")
        gam.fit(df, df["y"])
        sp0 = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))
        post = gaussian_smoothness_postprocess(gam, y, sp0, score_type="REML", deriv=1)
        sigma0 = float(post["scale_est"])
        Mp = float(_static_penalty_null_dim(gam) + int(gam.fit_intercept))
        sol0 = gam._solve_gaussian_given_smoothing(y, sp0)
        w = prior_weights_diagonal_from_fit(sol0, gam.n_samples_)

        def reml_frozen(sp):
            sol = gam._solve_gaussian_given_smoothing(y, sp)
            beta = np.asarray(sol["coef_full"], dtype=np.float64).ravel()
            mu = np.asarray(sol["mu"], dtype=np.float64).ravel()
            dev = gaussian_weighted_residual_sum_squares(y, mu, w)
            pen = quadratic_form_penalty(
                beta, np.asarray(sol["penalty_matrix"], dtype=np.float64)
            )
            A = np.asarray(sol["A"], dtype=np.float64)
            ldet = sol.get("log_det_XtWX_plus_penalty")
            if ldet is None or not np.isfinite(float(ldet)):
                from scipy.linalg import cho_factor

                cA, _ = cho_factor(A, check_finite=False)
                ldet = 2.0 * float(np.sum(np.log(np.abs(np.diag(cA)))))
            lds = _stable_penalty_logdet(gam, sp)
            return gaussian_reml_laplace_score(
                dev,
                pen,
                sigma0,
                float(ldet) - float(lds),
                Mp,
                w,
                gamma=float(gam.score_gamma),
                reml=True,
            )

        fdg = _fd_grad(lambda t: reml_frozen(np.exp(t)), np.log(sp0), eps=1e-6)
        np.testing.assert_allclose(
            post["reml_grad_log_sp"].ravel(),
            fdg,
            rtol=1e-6,
            atol=1e-8,
        )

    def test_fit_penalized_irls_from_model_attaches_smoothness_postprocess(self):
        df = _make_gaussian_data(seed=7, n=60)
        from nampy.gam.fit.solvers.penalized_irls import fit_penalized_irls_from_model

        gam = GAM(formula='y ~ s(x0, bs="cr", k=8)', smoothing_method="REML")
        gam.fit(df, df["y"])
        sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))
        out = fit_penalized_irls_from_model(
            gam, y, sp, attach_smoothness_postprocess=True
        )
        assert out.get("postproc") == "gaussian_smoothness_postprocess"
        assert np.isfinite(float(out["tr_a"]))
        assert np.isfinite(float(out["reml_score_profiled_scale"]))
