"""Parity and finite-difference checks for Gaussian smoothness post-processing."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from nampy.gam import GAM
from nampy.gam.fit.postprocess.gaussian_smoothness_postprocess import (
    gaussian_smoothness_postprocess,
)
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
)
from nampy.gam.fit.selection.criteria.gaussian import (
    criterion_ml_reml_exact_dynamic,
)
from tests._mgcv_snapshot_parity_shared import (
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


@lru_cache(maxsize=None)
def _fit_single_smooth_model(seed: int, n: int, method: str):
    df = _make_gaussian_data(seed=seed, n=n)
    gam = GAM(formula='y ~ s(x0, bs="cr", k=8)', smoothing_method=method)
    gam.fit(df, df["y"])
    return df, gam


class TestGaussianSmoothnessPostprocess:
    """
    Parity coverage for Gaussian smoothness post-processing, including profiled
    criteria, derivatives, and fit attachment.
    """
    def test_profiled_reml_matches_dynamic_criterion(self):
        """Verify that profiled REML matches dynamic criterion."""
        df, gam = _fit_single_smooth_model(seed=11, n=100, method="REML")
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
        """Verify that profiled REML matches mgcv snapshot noisy re."""
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
        """Verify that GCV derivatives match finite differences."""
        df, gam = _fit_single_smooth_model(seed=21, n=95, method="GCV")
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

    def test_reml_derivatives_match_dispatch(self):
        """Verify that REML derivatives match dispatch."""
        df, gam = _fit_single_smooth_model(seed=33, n=88, method="REML")
        sp0 = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))
        log_sp0 = np.log(sp0)
        post = gaussian_smoothness_postprocess(gam, y, sp0, score_type="REML", deriv=2)
        np.testing.assert_allclose(
            post["reml_grad_log_sp"].ravel(),
            criterion_gradient(gam, y, log_sp0, method="reml"),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            post["reml_hess_log_sp"],
            criterion_hessian(gam, y, log_sp0, method="reml"),
            rtol=0.0,
            atol=1e-12,
        )

    def test_fit_irls_from_model_attaches_smoothness_postprocess(self):
        """Verify that fit IRLS from model attaches smoothness postprocess."""
        df, gam = _fit_single_smooth_model(seed=7, n=60, method="REML")
        from nampy.gam.fit.solvers.irls_core import fit_irls_from_model

        sp = np.asarray(gam.smoothing_params, dtype=np.float64).ravel()
        y = gam.family.validate_y(np.asarray(df["y"], dtype=np.float64))
        out = fit_irls_from_model(gam, y, sp, attach_smoothness_postprocess=True)
        assert out.get("postproc") == "gaussian_smoothness_postprocess"
        assert np.isfinite(float(out["tr_a"]))
        assert np.isfinite(float(out["reml_score_profiled_scale"]))
