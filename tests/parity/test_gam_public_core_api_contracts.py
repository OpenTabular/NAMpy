"""Observable contracts for the public low-level GAM integration surface."""

from __future__ import annotations

import numpy as np

from nampy.gam import GAM, fit_model_core, solve_fit
from tests.mgcv_parity_utils import _make_gaussian_data


def test_public_fit_model_core_matches_the_array_facade() -> None:
    data = _make_gaussian_data(seed=455, n=110)
    features = ["x0", "x1"]
    X = data[features].to_numpy(dtype=np.float64)
    y = data["y"].to_numpy(dtype=np.float64)
    kwargs = {
        "family": "gaussian",
        "basis": "cr",
        "k": 7,
        "smoothing_params": [0.6, 1.1],
    }

    direct = GAM(**kwargs)
    returned = fit_model_core(direct, X, features, y)
    facade = GAM(**kwargs).fit(X=X, y=y)

    assert returned is direct
    np.testing.assert_allclose(
        direct.fit_result(include_covariances=True).core.coef_full,
        facade.fit_result(include_covariances=True).core.coef_full,
    )
    np.testing.assert_allclose(direct.predict(X), facade.predict(X))


def test_public_solve_fit_reproduces_the_published_fixed_sp_solution() -> None:
    data = _make_gaussian_data(seed=456, n=100)
    X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    y = data["y"].to_numpy(dtype=np.float64)
    model = GAM(
        family="gaussian",
        basis="cr",
        k=7,
        smoothing_params=[0.7, 1.2],
    )
    fit_model_core(model, X, ["x0", "x1"], y)

    solution = solve_fit(model, model.y_, model.smoothing_params)

    np.testing.assert_allclose(
        solution.fit_result.coef_full,
        model.gam_result_.fit_core_solution.fit_result.coef_full,
    )
    np.testing.assert_allclose(
        solution.fit_result.mu,
        model.gam_result_.fit_core_solution.fit_result.mu,
    )
