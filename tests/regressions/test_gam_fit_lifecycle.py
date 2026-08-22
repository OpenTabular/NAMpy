"""Regression tests for isolated and transactional ``GAM.fit`` calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families import make_gam_family
from nampy.gam.model_state import _fit_workspace


def _gaussian_data(seed: int, n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = 0.4 + np.sin(np.pi * x) - 0.3 * z**2 + rng.normal(0.0, 0.12, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _poisson_data(seed: int, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    eta = 0.2 + 0.55 * np.sin(np.pi * x)
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x": x})


def _gaulss_data(seed: int, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    mu = 0.3 + np.sin(np.pi * x)
    sigma = np.exp(-0.35 + 0.25 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _ocat_data(seed: int, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.2, 1.2, n)
    latent = 0.2 + 0.9 * x - 0.25 * x**2
    uniform = rng.uniform(size=n)
    latent += np.log(uniform / (1.0 - uniform))
    cuts = np.asarray([-1.0, -0.15, 0.85])
    y = 1 + np.searchsorted(cuts, latent, side="left")
    return pd.DataFrame({"y": y, "x": x})


def _assert_same_fit(left: GAM, right: GAM, data: pd.DataFrame) -> None:
    left_result = left.fit_result(include_covariances=False)
    right_result = right.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        left_result.core.coef_full,
        right_result.core.coef_full,
    )
    np.testing.assert_allclose(
        left_result.smoothing_params,
        right_result.smoothing_params,
    )
    np.testing.assert_allclose(
        left.predict(data, type="link"),
        right.predict(data, type="link"),
    )


def test_refit_with_changed_formula_matches_a_fresh_model() -> None:
    first = _gaussian_data(1)
    second = _gaussian_data(2)
    first_formula = 'y ~ s(x, bs="cr", k=6)'
    second_formula = 'y ~ s(x, bs="cr", k=7) + s(z, bs="ps", k=6)'
    reused = GAM(
        family="gaussian",
        formula=first_formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=first)
    first_workspace = _fit_workspace(reused)

    reused.fit(data=second, formula=second_formula)
    fresh = GAM(
        family="gaussian",
        formula=second_formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=second)

    assert _fit_workspace(reused) is not first_workspace
    _assert_same_fit(reused, fresh, second)


def test_pirls_refit_on_different_data_matches_a_fresh_model() -> None:
    formula = 'y ~ s(x, bs="cr", k=6)'
    first = _poisson_data(3)
    second = _poisson_data(4)
    reused = GAM(
        family="poisson",
        formula=formula,
        smoothing_params=[0.7],
    ).fit(data=first)
    first_workspace = _fit_workspace(reused)

    reused.fit(data=second)
    fresh = GAM(
        family="poisson",
        formula=formula,
        smoothing_params=[0.7],
    ).fit(data=second)

    assert _fit_workspace(reused) is not first_workspace
    _assert_same_fit(reused, fresh, second)


def test_general_family_refit_does_not_reuse_outer_evaluations() -> None:
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]
    first = _gaulss_data(5)
    second = _gaulss_data(6)
    reused = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="ML",
    ).fit(data=first)

    reused.fit(data=second)
    fresh = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="ML",
    ).fit(data=second)

    _assert_same_fit(reused, fresh, second)


def test_failed_refit_preserves_the_last_successful_fit() -> None:
    data = _gaussian_data(7)
    model = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=6)',
        smoothing_params=[0.5],
    ).fit(data=data)
    result_before = model.gam_result_
    family_before = model.family
    workspace_before = _fit_workspace(model)
    prediction_before = model.predict(data)

    with pytest.raises((KeyError, ValueError)):
        model.fit(data=data, formula='y ~ s(missing, bs="cr", k=6)')

    assert model.gam_result_ is result_before
    assert model.family is family_before
    assert _fit_workspace(model) is workspace_before
    np.testing.assert_array_equal(model.predict(data), prediction_before)


def test_mutable_family_instances_are_never_shared_across_models() -> None:
    supplied = make_gam_family({"name": "ocat", "R": 4})
    first = GAM(family=supplied)
    second = GAM(family=supplied)

    assert first.family is not supplied
    assert second.family is not supplied
    assert first.family is not second.family
    first.family.putTheta(np.asarray([0.2, 0.4]))
    assert not np.array_equal(first.family.getTheta(), second.family.getTheta())
    assert not np.array_equal(first.family.getTheta(), supplied.getTheta())


def test_ocat_refit_reinitializes_estimated_cutpoints() -> None:
    formula = 'y ~ s(x, bs="cr", k=7)'
    first_data = _ocat_data(31)
    second_data = _ocat_data(32)
    reused = GAM(
        family={"name": "ocat", "R": 4},
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=first_data)

    reused.fit(data=second_data)
    fresh = GAM(
        family={"name": "ocat", "R": 4},
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=second_data)

    np.testing.assert_allclose(reused.family.getTheta(), fresh.family.getTheta())
    _assert_same_fit(reused, fresh, second_data)
