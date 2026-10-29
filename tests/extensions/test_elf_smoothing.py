"""ELF exact coefficient and outer derivatives before calibration integration."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy.gam import GAM
from nampy.gam.families.elf import ELFFamily
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)


def test_elf_derivatives_match_independent_torch_and_outer_finite_differences():
    import torch
    from torch.nn.functional import softplus

    rng = np.random.default_rng(81)
    X = np.column_stack([np.ones(40), rng.uniform(-1, 1, 40)])
    y = X @ np.array([0.4, 0.7]) + rng.normal(0, 0.4, 40)
    beta = np.array([0.2, 0.5])
    family = ELFFamily(level=0.3, bandwidth=0.2, loss_scale=0.15)
    value = family.Dd(y, X @ beta, family.getTheta(), np.ones(40), level=2)

    def ll(b):
        residual = torch.tensor(y) - torch.tensor(X) @ b
        return torch.sum((0.7 * residual - 0.2 * softplus(residual / 0.2)) / 0.15)

    assert_allclose(
        -0.5 * X.T @ value["Dmu"], torch.func.grad(ll)(torch.tensor(beta)), atol=2e-13
    )
    assert_allclose(
        -0.5 * X.T @ (value["Dmu2"][:, None] * X),
        torch.func.hessian(ll)(torch.tensor(beta)),
        atol=2e-13,
    )
    data = pd.DataFrame({"x": X[:, 1]})
    gam = GAM(
        formula="~s(x,k=4)",
        family=family,
        scale=1.0,
        optimize_smoothing=True,
        smoothing_method="reml",
        smoothing_optimizer="bfgs",
        irls_tol=1e-9,
    ).fit(data, y)
    rho = np.log(gam.smoothing_params)
    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient,
        approx_derivative(
            lambda r: np.array([criterion_value(gam, gam.y_, r, method="reml")]), rho
        ).ravel(),
        atol=2e-5,
        rtol=2e-5,
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda r: criterion_gradient(gam, gam.y_, r, method="reml"), rho
        ).reshape(hessian.shape),
        atol=2e-5,
        rtol=2e-5,
    )


def test_calibrated_quantile_reselects_smoothing_at_each_scale_and_persists(tmp_path):
    from nampy import CalibratedQuantileGAM

    root = __import__("pathlib").Path(__file__).parents[1] / "reference_fixtures"
    data = pd.read_csv(root / "qgam_calfast_fixed_input.csv")
    X = data[["x", "z"]]
    model = CalibratedQuantileGAM(
        formula="~s(x,k=4)+z",
        level=0.3,
        bandwidth=0.2,
        optimize_smoothing=True,
        log_scale_bounds=(-3, 2),
        calibration_tol=1e-5,
    ).fit(X, data.y)
    assert np.isfinite(model.smoothing_params_).all()
    assert model.smoothing_params_.shape == (1,)
    assert model.calibration_profile_.shape[1] == 2
    interval = model.predict_interval(X.iloc[:3])
    assert np.isfinite(interval.lower).all()
    restored = CalibratedQuantileGAM.load_model(
        model.save_model(tmp_path / "auto-qgam.nampy")
    )
    assert_allclose(
        restored.predict_interval(X.iloc[:3]).lower, interval.lower, atol=0, rtol=0
    )


@pytest.mark.parametrize("nonlinear", [False, True])
def test_automatic_smoothing_calibration_matches_qgam_profiles_and_endpoint(nonlinear):
    from pathlib import Path

    from nampy import CalibratedQuantileGAM
    from nampy.gam.design import CompiledDesign
    from nampy.gam.losses.objectives import LossFit
    from nampy.gam.losses.quantile_calibration import _calibration_covariances

    root = Path(__file__).parents[1] / "reference_fixtures"
    stem = "qgam_calfast_auto" + ("_nonlinear" if nonlinear else "")
    source = pd.read_csv(root / f"{stem}_input.csv")
    profile = pd.read_csv(root / f"{stem}_profile.csv")
    endpoint = pd.read_csv(root / f"{stem}.csv")
    X, y = source[["x", "z"]], source.y.to_numpy()
    for log_sigma in (-2.0, -1.0, 0.0, 1.0):
        sigma = np.exp(log_sigma)
        gam = GAM(
            formula="~s(x,k=4)+z",
            family=ELFFamily(level=0.3, bandwidth=0.2, loss_scale=sigma),
            scale=1.0,
            optimize_smoothing=True,
            smoothing_method="reml",
            smoothing_optimizer="bfgs",
            irls_tol=1e-9,
        ).fit(X, y)
        expected = profile[profile.log_sigma == log_sigma]
        # R's finite stopping endpoint on a flat oversmoothing limit need not
        # identify SP coordinates. Check the actual REML criterion first, then
        # compare coefficient/probability functionals at the same source SP.
        selected_score = criterion_value(
            gam, gam.y_, np.log(gam.smoothing_params), method="reml"
        )
        source_score = criterion_value(
            gam, gam.y_, np.log([expected.sp.iloc[0]]), method="reml"
        )
        assert selected_score <= source_score + 1e-6 * (1 + abs(source_score))
        fixed = GAM(
            formula="~s(x,k=4)+z",
            family=ELFFamily(level=0.3, bandwidth=0.2, loss_scale=sigma),
            scale=1.0,
            optimize_smoothing=False,
            smoothing_params=[expected.sp.iloc[0]],
            irls_tol=1e-10,
        ).fit(X, y)
        design = CompiledDesign.from_fitted_gam(fixed, X)
        fitted = LossFit(
            fixed.fit_result().coef_full,
            design.matrix @ fixed.fit_result().coef_full,
            0.0,
            0,
            True,
            0.0,
        )
        V, VS, loss = _calibration_covariances(
            design.matrix,
            y,
            design.penalty,
            fitted,
            level=0.3,
            bandwidth=0.2,
            sigma=sigma,
        )
        assert_allclose(fitted.fitted, expected.fitted, rtol=2e-6, atol=2e-6)
        assert_allclose(
            np.einsum("np,pq,nq->n", design.matrix, V, design.matrix),
            expected.posterior_variance,
            rtol=2e-5,
            atol=2e-6,
        )
        assert_allclose(
            np.einsum("np,pq,nq->n", design.matrix, VS, design.matrix),
            expected.sandwich_variance,
            rtol=2e-5,
            atol=2e-6,
        )
        assert_allclose(loss, expected.loss.iloc[0], rtol=2e-6, atol=2e-6)
    model = CalibratedQuantileGAM(
        formula="~s(x,k=4)+z",
        level=0.3,
        bandwidth=0.2,
        optimize_smoothing=True,
        log_scale_bounds=(-3, 2),
        calibration_tol=1e-7,
    ).fit(X, y)
    assert_allclose(
        np.log(model.loss_scale_), endpoint.log_sigma.iloc[0], rtol=2e-4, atol=2e-4
    )
    if nonlinear:
        assert_allclose(
            model.smoothing_params_, [endpoint.sp.iloc[0]], rtol=2e-4, atol=2e-5
        )
    assert_allclose(model.predict(X), endpoint.fitted, rtol=2e-5, atol=2e-5)
    assert_allclose(
        model.predict_interval(X).standard_error ** 2,
        endpoint.posterior_variance,
        rtol=2e-5,
        atol=2e-6,
    )
    assert (
        model.predict_interval(X).coverage
        == "approximate_pointwise_conditional_on_selected_smoothing"
    )


def test_selected_smoothing_pointwise_median_coverage_in_independent_simulation():
    from scipy.stats import binomtest

    from nampy import CalibratedQuantileGAM

    rng = np.random.default_rng(983)
    covered = 0
    repetitions = 30
    query = pd.DataFrame({"x": [0.0]})
    for _ in range(repetitions):
        X = pd.DataFrame({"x": rng.uniform(-1, 1, 160)})
        y = (
            0.4
            + 0.6 * X.x.to_numpy()
            + 1.5 * X.x.to_numpy() ** 2
            + rng.normal(0, 0.5, 160)
        )
        model = CalibratedQuantileGAM(
            formula="~s(x,k=4)",
            level=0.5,
            bandwidth=0.1,
            optimize_smoothing=True,
            log_scale_bounds=(-4, 1),
        ).fit(X, y)
        interval = model.predict_interval(query, level=0.9)
        covered += bool(interval.lower[0] <= 0.4 <= interval.upper[0])
    assert binomtest(covered, repetitions, p=0.9).pvalue > 0.001, covered
