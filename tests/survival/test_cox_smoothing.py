"""Exact coupled risk-set higher/outer derivatives and native smoothing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy import CoxGAM
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.losses.cox import cox_higher_derivatives
from nampy.survival import SurvivalTarget


def data():
    rng = np.random.default_rng(471)
    x = rng.uniform(-1, 1, 100)
    event_time = rng.exponential(np.exp(-1.2 * np.sin(3 * x)))
    censor = rng.uniform(0.4, 3, 100)
    return pd.DataFrame({"x": x}), SurvivalTarget(
        np.minimum(event_time, censor), event_time <= censor
    )


def test_coupled_cox_third_fourth_derivatives_match_independent_autodiff():
    import torch

    rng = np.random.default_rng(88)
    X = rng.normal(size=(14, 3))
    time = np.resize([1.0, 2.0, 2.0, 3.0], 14)
    event = np.resize([1, 0, 1], 14).astype(bool)
    strata = np.repeat([0, 1], 7)
    offset = np.linspace(-0.05, 0.05, 14)
    beta = np.array([0.1, -0.2, 0.15])

    def likelihood(b):
        eta = torch.tensor(X) @ b + torch.tensor(offset)
        value = torch.zeros((), dtype=torch.float64)
        for label in (0, 1):
            for t in np.unique(time[(strata == label) & event]):
                risk = (strata == label) & (time >= t)
                dead = (strata == label) & (time == t) & event
                value += eta[dead].sum() - dead.sum() * torch.logsumexp(eta[risk], 0)
        return value

    third, fourth = cox_higher_derivatives(beta, X, time, event, strata, offset)
    b = torch.tensor(beta)
    expected3 = torch.func.jacrev(torch.func.hessian(likelihood))(b)
    expected4 = torch.func.jacrev(torch.func.jacrev(torch.func.hessian(likelihood)))(b)
    assert_allclose(third, expected3, atol=2e-13)
    assert_allclose(fourth, expected4, atol=3e-13)


def test_native_cox_automatic_smoothing_and_exact_outer_derivatives():
    X, y = data()
    model = CoxGAM(formula="~s(x,k=4)", optimize_smoothing=True, tol=1e-9).fit(X, y)
    gam = model.gam_
    rho = np.log(model.smoothing_params_)
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
    assert_allclose(
        model.predict_link(X)[:, 0], gam.predict(X, type="link").ravel(), atol=1e-10
    )
    assert np.isfinite(
        model.predict_distribution(X.iloc[:2]).survival([0, 0.1, 0.3])
    ).all()


@pytest.mark.parametrize(
    "stratified,optimizer", [(False, "outer_newton"), (False, "bfgs"), (True, "bfgs")]
)
def test_cox_smoothing_endpoints_match_mgcv(stratified, optimizer, tmp_path):
    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / "cox_smoothing_input.csv")
    reference = pd.read_csv(root / "cox_smoothing.csv")
    expected = reference[reference.stratified == stratified]
    X = source[["x", "z"]]
    y = SurvivalTarget(source.time, source.event)
    model = CoxGAM(
        formula="~s(x,k=5)+z",
        optimize_smoothing=True,
        smoothing_optimizer=optimizer,
        tol=1e-10,
    ).fit(
        X,
        y,
        strata=source.stratum if stratified else None,
        offset=source.offset.to_numpy(),
    )
    query = pd.DataFrame({"x": [-0.7, 0.1, 0.8], "z": [0.1, -0.2, 0.4]})
    options = {"offset": np.zeros(3)}
    if stratified:
        options["strata"] = [1, 2, 1]
    assert_allclose(
        model.smoothing_params_, [expected.sp.iloc[0]], rtol=3e-5, atol=2e-6
    )
    assert_allclose(
        model.predict_distribution(query, **options).survival([0.1, 0.3, 0.5]).ravel(),
        expected.survival,
        rtol=2e-6,
        atol=2e-6,
    )
    assert_allclose(
        np.repeat(model.predict_link(query, offset=np.zeros(3)), 3),
        expected.link,
        rtol=2e-6,
        atol=2e-6,
    )
    assert_allclose(
        criterion_value(
            model.gam_, model.gam_.y_, np.log(model.smoothing_params_), method="reml"
        ),
        expected.score.iloc[0],
        rtol=2e-6,
        atol=2e-6,
    )
    restored = CoxGAM.load_model(model.save_model(tmp_path / "cox-auto.nampy"))
    assert_allclose(
        restored.predict_distribution(query, **options).survival([0.1, 0.3, 0.5]),
        model.predict_distribution(query, **options).survival([0.1, 0.3, 0.5]),
        atol=0,
        rtol=0,
    )


def test_unvalidated_cox_smoothing_optimizers_are_explicit():
    X, y = data()
    with pytest.raises(NotImplementedError, match="supported smoothing optimizers"):
        CoxGAM(
            formula="~s(x,k=4)", optimize_smoothing=True, smoothing_optimizer="efs"
        ).fit(X, y)


def test_stratified_newton_smoothing_is_explicitly_unsupported():
    X, y = data()
    with pytest.raises(NotImplementedError, match="requires bfgs"):
        CoxGAM(formula="~s(x,k=4)", optimize_smoothing=True).fit(
            X, y, strata=np.arange(len(y)) % 2
        )
