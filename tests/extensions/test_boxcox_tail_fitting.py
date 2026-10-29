"""Observed BCT/BCPE likelihood derivatives against independent probabilities."""

from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.boxcox_tail import BCPELikelihood, BCTLikelihood


@pytest.mark.parametrize("name", ["bct", "bcpe"])
def test_automatic_smoothing_is_explicitly_unsupported(name):
    from nampy import GAMLSS

    frame = pd.DataFrame({"x": np.linspace(0, 1, 12), "y": np.linspace(1, 3, 12)})
    model = GAMLSS(
        family=name,
        formula={"mu": "y~s(x,k=4)", "sigma": "~1", "nu": "~1", "tau": "~1"},
        optimize_smoothing=True,
    )
    with pytest.raises(NotImplementedError, match="(?i)reml|smoothing|derivative"):
        model.fit(frame)


@pytest.mark.parametrize("family", [BCTLikelihood, BCPELikelihood])
def test_saved_r_log_density(family):
    frame = pd.read_csv(
        Path(__file__).parents[1]
        / "reference_fixtures"
        / "boxcox_tail"
        / f"{family.kind}.csv"
    )
    p = frame[["mu", "sigma", "nu", "tau"]].to_numpy()
    eta = p.copy()
    eta[:, [0, 1, 3]] = np.log(eta[:, [0, 1, 3]])
    variables = [Taylor(eta[:, k], 4, 0, k) for k in range(4)]
    assert_allclose(
        family()._log_density(frame.y.to_numpy(), *variables).value,
        frame.log_prob,
        atol=5e-12,
        rtol=5e-12,
    )


def _mp_logdensity(kind, y, *eta):
    lm, ls, nu, lt = eta
    tau, sigma = mp.exp(lt), mp.exp(ls)
    r = mp.log(y) - lm
    z = mp.expm1(nu * r) / (nu * sigma)
    a = 1 / (sigma * abs(nu))
    if kind == "bct":
        logf = (
            mp.loggamma((tau + 1) / 2)
            - mp.loggamma(tau / 2)
            - mp.log(tau * mp.pi) / 2
            - (tau + 1) / 2 * mp.log1p(z * z / tau)
        )
        normalizer = (
            1
            - mp.betainc(
                tau / 2, mp.mpf("0.5"), 0, tau / (tau + a * a), regularized=True
            )
            / 2
        )
    else:
        alpha = mp.sqrt(mp.gamma(1 / tau) / mp.gamma(3 / tau))
        logf = (
            mp.log(tau)
            - mp.log(2)
            - mp.log(alpha)
            - mp.loggamma(1 / tau)
            - (abs(z) / alpha) ** tau
        )
        normalizer = (
            1 + mp.gammainc(1 / tau, 0, (a / alpha) ** tau, regularized=True)
        ) / 2
    return nu * r - ls - mp.log(y) + logf - mp.log(normalizer)


@pytest.mark.parametrize("family", [BCTLikelihood, BCPELikelihood])
@pytest.mark.parametrize("nu", [-1.2, 0.7])
def test_exact_observed_gradient_and_mixed_hessian(family, nu):
    eta = [0.2, -0.1, nu, 0.3]
    variables = [Taylor(np.array([eta[k]]), 4, 2, k) for k in range(4)]
    value = family()._log_density(np.array([2.3]), *variables)
    with mp.workdps(45):

        def f(*args):
            return _mp_logdensity(family.kind, mp.mpf("2.3"), *args)

        point = tuple(mp.mpf(str(x)) for x in eta)
        for index, coefficient in value.coefficients.items():
            expected = float(mp.diff(f, point, index))
            multiplier = np.prod([mp.factorial(k) for k in index])
            assert_allclose(
                float(coefficient[0]) * float(multiplier),
                expected,
                rtol=3e-9,
                atol=3e-10,
            )


@pytest.mark.parametrize("family", [BCTLikelihood, BCPELikelihood])
def test_public_fixed_smoothing_fit_matches_independent_optimum(family):
    from scipy.optimize import minimize

    from nampy import GAMLSS
    from nampy.distributions import DistributionPrediction
    from nampy.gam.model_state import _fit_result

    n = 160
    x = np.linspace(-1, 1, n)
    probability = (np.arange(n) + 0.5) / n
    np.random.default_rng(41).shuffle(probability)
    mu = np.exp(0.3 + 0.2 * x)
    truth = {
        "mu": mu,
        "sigma": np.full(n, 0.7),
        "nu": np.full(n, 1.2),
        "tau": np.full(n, 4 if family.kind == "bct" else 1.4),
    }
    dist = DistributionPrediction(family.law(), truth)
    # One quantile per row avoids sampling noise and weakly identified tails.
    y = dist.quantile(probability)[np.arange(n), np.arange(n)]
    frame = pd.DataFrame({"x": x, "y": y})
    model = GAMLSS(
        family=family.name,
        formula={"mu": "y~x", "sigma": "~1", "nu": "~1", "tau": "~1"},
        optimize_smoothing=False,
    ).fit(frame)
    assert _fit_result(model.gam_).converged
    eta = model.predict(frame, raw=True)
    initial = [eta[:, 0].mean(), (eta[-1, 0] - eta[0, 0]) / 2, *eta[0, 1:]]

    def objective(b):
        parameters = np.column_stack(
            [
                np.exp(b[0] + b[1] * x),
                np.full(n, np.exp(b[2])),
                np.full(n, b[3]),
                np.full(n, np.exp(b[4])),
            ]
        )
        return -np.sum(family().logpdf_from_parameters(y, parameters))

    reference = minimize(objective, initial, method="BFGS", options={"gtol": 1e-6})
    assert np.linalg.norm(reference.jac, ord=np.inf) < 2e-4
    assert_allclose(objective(initial), reference.fun, atol=2e-7, rtol=0)
    assert_allclose(
        model.predict_distribution(frame).log_prob(y),
        family().logpdf_from_parameters(y, model.predict(frame)),
        atol=2e-12,
    )
