"""Native partial likelihood, mgcv risk-set ties, baseline and coefficient checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize._numdiff import approx_derivative
from sklearn.exceptions import NotFittedError

from nampy import CoxGAM
from nampy.gam.losses.cox import cox_partial_likelihood
from nampy.survival import SurvivalPrediction, SurvivalTarget


@pytest.mark.parametrize("route", ["linear", "smooth", "stratified", "categorical"])
def test_native_cox_endpoints_baseline_and_covariance_match_mgcv(route, tmp_path):
    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / "cox_fixed_input.csv")
    reference = pd.read_csv(root / "cox_fixed.csv")
    expected = reference[reference.route == route]
    X = source[["x", "z", "group"]].copy()
    X["group"] = pd.Categorical(X.group, categories=["a", "b"])
    target = SurvivalTarget(source.time, source.event)
    strata = source.stratum.to_numpy() if route == "stratified" else None
    formula = {
        "linear": "~x+z",
        "smooth": "~s(x,k=4)+z",
        "stratified": "~x+z",
        "categorical": "~x+factor(group)",
    }[route]
    model = CoxGAM(
        formula=formula, smoothing_params=[0.4] if route == "smooth" else None, tol=1e-8
    ).fit(X, target, strata=strata, offset=source.offset.to_numpy())
    query = pd.DataFrame(
        {
            "x": [-0.5, 0.2, 0.8],
            "z": [0.1, -0.3, 0.7],
            "group": pd.Categorical(["a", "b", "a"], categories=["a", "b"]),
        }
    )
    options = {"offset": np.zeros(3)}
    if strata is not None:
        options["strata"] = [1, 2, 1]
    prediction = model.predict_distribution(query, **options)
    assert_allclose(
        prediction.survival([0.5, 1, 2]).ravel(),
        expected.survival,
        atol=2e-7,
        rtol=2e-7,
    )
    assert_allclose(
        np.repeat(model.predict_link(query, offset=np.zeros(3)), 3),
        expected.link,
        atol=2e-7,
        rtol=2e-7,
    )
    matrix = model.design_.evaluate(query).matrix[:, model.free_indices_]
    se = np.sqrt(np.einsum("np,pq,nq->n", matrix, model.covariance_, matrix))
    assert_allclose(np.repeat(se, 3), expected.se, atol=2e-7, rtol=2e-7)
    components = model.predict_components(query, offset=np.zeros(3))
    assert_allclose(
        sum(components.terms.values()) + components.offset, components.link, atol=1e-12
    )
    restored = CoxGAM.load_model(model.save_model(tmp_path / f"{route}.nampy"))
    assert_array_equal(
        restored.predict_distribution(query, **options).survival([0.5, 1, 2]),
        prediction.survival([0.5, 1, 2]),
    )


def test_risk_set_derivatives_include_tied_censoring_and_match_independent_autodiff():
    import torch

    rng = np.random.default_rng(51)
    X = rng.normal(size=(12, 3))
    time = np.resize([1.0, 1.0, 2.0, 3.0], 12)
    event = np.resize([1, 0, 1, 1], 12).astype(bool)
    strata = np.repeat([0, 1], 6)
    offset = rng.normal(size=12) * 0.03
    beta = np.array([0.1, -0.2, 0.3])
    value, gradient, hessian = cox_partial_likelihood(
        beta, X, time, event, strata, offset
    )

    def independent(b):
        eta = torch.tensor(X) @ b + torch.tensor(offset)
        result = torch.zeros((), dtype=torch.float64)
        for label in (0, 1):
            for t in np.unique(time[(strata == label) & event]):
                deaths = (strata == label) & event & (time == t)
                risk = (strata == label) & (time >= t)
                result = (
                    result
                    + eta[deaths].sum()
                    - deaths.sum() * torch.logsumexp(eta[risk], dim=0)
                )
        return result

    b = torch.tensor(beta, requires_grad=True)
    assert_allclose(value, independent(b).detach(), atol=1e-13)
    assert_allclose(gradient, torch.func.grad(independent)(b).detach(), atol=1e-13)
    assert_allclose(hessian, torch.func.hessian(independent)(b).detach(), atol=1e-13)
    assert_allclose(
        hessian,
        approx_derivative(
            lambda b: cox_partial_likelihood(b, X, time, event, strata, offset)[1], beta
        ),
        atol=1e-9,
    )


def test_step_survival_quantiles_conditioning_rmst_and_density_rejection():
    prediction = SurvivalPrediction(
        cuts=[0, 1, 2, 3], cumulative_hazard_steps=[[0, 0.2, 0.7, 0.7]]
    )
    assert_allclose(
        prediction.survival([0, 0.5, 1, 2, 3]),
        [[1, 1, np.exp(-0.2), np.exp(-0.7), np.exp(-0.7)]],
    )
    assert_allclose(prediction.cumulative_hazard_at([2], conditional_on=[1]), [0.5])
    assert_allclose(prediction.rmst(3), [1 + np.exp(-0.2) + np.exp(-0.7)])
    assert_allclose(prediction.rmst(2, conditional_on=1), [1])
    assert_array_equal(prediction.quantile([0, 0.1, 0.3]), [[0, 1, 2]])
    assert np.isnan(prediction.quantile([0.9])).all()
    assert_array_equal(
        prediction.take([0, 0]).cumulative_hazard_steps,
        np.repeat([[0, 0.2, 0.7, 0.7]], 2, axis=0),
    )
    with pytest.raises(NotImplementedError, match="density"):
        prediction.hazard([1])
    tail = SurvivalPrediction(
        cuts=[0, 1, 2, 3], cumulative_hazard_steps=[[0, 1000, 1001, 1002]]
    )
    assert_allclose(tail.rmst(3, conditional_on=2), [1])


def test_cox_rejects_unsupported_routes_and_failed_refits_preserve_model():
    X = pd.DataFrame({"x": np.sin(np.arange(30))})
    target = SurvivalTarget(np.resize([1.0, 2.0, 3.0], 30), np.resize([1, 0, 1], 30))
    model = CoxGAM(formula="~x", tol=1e-8).fit(X, target)
    before = model.predict_link(X)
    with pytest.raises(NotImplementedError, match="Delayed-entry"):
        model.fit(X, SurvivalTarget(target.time, target.event, entry=np.full(30, 0.1)))
    assert_array_equal(model.predict_link(X), before)
    with pytest.raises(ValueError, match="align"):
        model.score(X.iloc[:-1], target)
    model.set_params(tol=1e-7)
    with pytest.raises(NotFittedError, match="Fit CoxGAM"):
        model.predict_link(X)
