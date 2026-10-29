"""Explicit trials, exact BB likelihood and public context boundary."""

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from sklearn.base import clone

from nampy import GAMLSS
from nampy.diagnostics import quantile_residuals
from nampy.distributions import BetaBinomialLaw, DistributionPrediction
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.betabinomial import BetaBinomialLocationPrecision
from nampy.gam.model_state import _fit_result


def _data():
    rng = np.random.default_rng(135)
    x = np.linspace(-1, 1, 150)
    trials = rng.integers(3, 16, len(x))
    mu = expit(-0.2 + 0.7 * x)
    y = rng.binomial(trials, rng.beta(mu * 3, (1 - mu) * 3))
    trials[:2], y[:2] = [0, 1], [0, 1]
    return pd.DataFrame({"x": x, "y": y}), trials


def _model(**kwargs):
    return GAMLSS(
        family="betabinomial_lss",
        formula={"mu": "y ~ x", "kappa": "~ 1"},
        optimize_smoothing=False,
        **kwargs,
    )


def _distribution(mu, kappa, trials):
    return DistributionPrediction(
        BetaBinomialLaw(),
        {"mu": np.asarray(mu), "kappa": np.asarray(kappa)},
        {"trials": np.asarray(trials)},
    )


def test_saved_r_bb_probabilities_and_unpenalized_fit():
    root = Path(__file__).parents[1] / "reference_fixtures" / "betabinomial_lss"
    reference = pd.read_csv(root / "probability.csv")
    distribution = _distribution(reference.mu, reference.kappa, reference.trials)
    assert_allclose(distribution.log_prob(reference.y), reference.log_prob, atol=2e-13)
    assert_allclose(distribution.cdf(reference.y), reference.cdf, atol=2e-13)
    for i, probability in enumerate(reference.probability):
        assert distribution.quantile(probability)[i, 0] == reference["quantile"][i]
    frame = pd.read_csv(root / "fit.csv")
    model = _model().fit(frame, context={"trials": frame.trials})
    assert _fit_result(model.gam_).converged
    assert_allclose(model.predict_parameters(frame), frame[["mu", "kappa"]], atol=3e-5, rtol=0)
    assert_allclose(
        model.predict_distribution(frame, context={"trials": frame.trials}).log_prob(
            frame.y
        ),
        frame.log_prob,
        atol=2e-5,
    )


def test_fixed_smooth_weights_match_repeated_observations():
    frame, trials = _data()
    weights = np.tile([1, 2, 3], 50)
    formula = {"mu": 'y ~ s(x,bs="cr",k=5)', "kappa": '~ s(x,bs="cr",k=4)'}
    kwargs = {
        "family": "betabinomial_lss",
        "formula": formula,
        "optimize_smoothing": False,
        "smoothing_params": [0.3, 0.7],
        "irls_tol": 1e-11,
    }
    model = GAMLSS(**kwargs).fit(
        frame, context={"trials": trials}, sample_weight=weights
    )
    repeated = np.repeat(np.arange(len(frame)), weights)
    expanded = GAMLSS(**kwargs).fit(
        frame.iloc[repeated].reset_index(drop=True),
        context={"trials": trials[repeated]},
    )
    assert _fit_result(model.gam_).converged
    assert_allclose(model.predict_parameters(frame), expanded.predict_parameters(frame), atol=3e-6)
    model.predict_components(frame).validate_additive_reconstruction()
    assert np.isfinite(model.standard_errors(frame)).all()


def test_probability_normalization_moments_boundaries_and_binomial_limit():
    trials = np.array([0, 1, 7, 12])
    mu, kappa = np.array([0.2, 0.7, 0.3, 0.8]), np.array([2, 0.4, 3, 5])
    dist = _distribution(mu, kappa, trials)
    support = np.arange(13)
    mass = np.array([np.exp(dist.log_prob(np.full(4, y))) for y in support])
    assert_allclose(mass.sum(axis=0), 1, atol=1e-14)
    assert_allclose(support @ mass, dist.mean(), atol=1e-13)
    assert_allclose(support**2 @ mass - dist.mean() ** 2, dist.variance(), atol=1e-12)
    assert_allclose(dist.quantile([0, 1]), np.column_stack([np.zeros(4), trials]))
    assert np.isneginf(dist.log_prob(np.full(4, -1))).all()
    assert np.isneginf(dist.log_prob(np.full(4, 0.5))).all()
    assert_allclose(dist.cdf(np.full(4, np.inf)), 1)
    assert_allclose(dist.cdf_left(np.zeros(4)), 0)
    assert_allclose(dist.cdf(np.zeros(4)), mass[0])
    assert_allclose(dist.cdf_left(np.full(4, 1.2)), dist.cdf(np.ones(4)))
    # Exact one-trial law is Bernoulli for every precision.
    assert_allclose(mass[:2, 1], [1 - mu[1], mu[1]], atol=1e-14)
    limit = _distribution([0.3, 0.3], [1e3, 1e5], [12, 12])
    errors = np.abs(np.exp(limit.log_prob([4, 4])) - stats.binom.pmf(4, 12, 0.3))
    assert errors[1] < errors[0] / 90
    assert errors[1] < 2e-5


def test_sampling_quantiles_and_shared_discrete_diagnostics():
    dist = _distribution([0.3, 0.7], [2.0, 4.0], [10, 15])
    draws = dist.sample(n_draws=20000, random_state=43)
    assert draws.shape == (20000, 2)
    assert_allclose(draws.mean(axis=0), dist.mean(), atol=0.06)
    assert_allclose(draws.var(axis=0), dist.variance(), atol=0.2)
    for p in [0.01, 0.2, 0.5, 0.99]:
        q = dist.quantile(p)[:, 0]
        assert np.all(dist.cdf_left(q) < p)
        assert np.all(dist.cdf(q) >= p)
    y = np.array([0, 15])
    residual = quantile_residuals(dist, y, uniforms=np.array([0.25, 0.75]))
    assert np.isfinite(residual.residuals).all()
    assert_allclose(
        residual.residuals,
        stats.norm.ppf(
            dist.cdf_left(y) + [0.25, 0.75] * (dist.cdf(y) - dist.cdf_left(y))
        ),
    )


@pytest.mark.parametrize("eta", [[-0.7, -0.4], [0.3, 1.2], [1.4, 0.5]])
def test_fourth_order_likelihood_against_independent_torch(eta):
    torch = pytest.importorskip("torch")
    y, trials = np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 8.0])
    family = BetaBinomialLocationPrecision()
    family.bind_fit_context({"trials": trials}, len(y))
    variables = [Taylor(np.full(len(y), eta[k]), 2, 4, k) for k in range(2)]
    result = family._log_density(y, *variables)

    def likelihood(z):
        mu, kappa = z[0].sigmoid(), z[1].exp()
        a, b = mu * kappa, (1 - mu) * kappa
        # Finite rising-factorial products avoid sharing the implementation's
        # lgamma/polygamma path (Torch polygamma also loses accuracy here).
        rows = []
        for successes, count in zip(y.astype(int), trials.astype(int), strict=True):
            value = z.sum() * 0
            for j in range(successes):
                value = value + torch.log(a + j) + np.log((count - j) / (j + 1))
            for j in range(count - successes):
                value = value + torch.log(b + j)
            for j in range(count):
                value = value - torch.log(kappa + j)
            rows.append(value)
        return torch.stack(rows)

    tensor = torch.tensor(eta, dtype=torch.float64)
    assert_allclose(result.value, likelihood(tensor).numpy(), atol=2e-12)
    function = likelihood
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        derivative = function(tensor).detach().numpy()
        packed = np.column_stack(
            [
                derivative[(slice(None), *idx)]
                for idx in combinations_with_replacement(range(2), degree)
            ]
        )
        assert_allclose(
            result.packed_derivatives(degree), packed, atol=3e-10, rtol=3e-9
        )
        assert np.all(result.packed_derivatives(degree)[0] == 0)


def test_public_fit_matches_independent_weighted_mle_offsets_and_persistence(tmp_path):
    frame, trials = _data()
    weight = np.tile([1.0, 2.0, 0.0], 50)
    offset = [0.12 * np.sin(frame.x.to_numpy()), np.full(len(frame), 0.08)]
    model = _model().fit(frame, sample_weight=weight, offset=np.column_stack(offset), context={'trials': trials})
    assert _fit_result(model.gam_).converged
    design = np.column_stack([np.ones(len(frame)), frame.x])

    def objective(beta):
        mu = expit(design @ beta[:2] + offset[0])
        kappa = np.exp(beta[2] + offset[1])
        return -np.sum(
            weight
            * stats.betabinom.logpmf(frame.y, trials, mu * kappa, (1 - mu) * kappa)
        )

    oracle = minimize(
        objective, [-0.2, 0.7, 1.0], method="BFGS", options={"gtol": 1e-6}
    )
    prediction = model.predict_parameters(frame, offset=np.column_stack(offset))
    assert_allclose(
        prediction[:, 0], expit(design @ oracle.x[:2] + offset[0]), atol=2e-6
    )
    assert_allclose(prediction[:, 1], np.exp(oracle.x[2] + offset[1]), atol=2e-5)
    distribution = model.predict_distribution(frame, offset=np.column_stack(offset), context={'trials': trials})
    assert_allclose(
        distribution.log_prob(frame.y),
        stats.betabinom.logpmf(
            frame.y,
            trials,
            prediction[:, 0] * prediction[:, 1],
            (1 - prediction[:, 0]) * prediction[:, 1],
        ),
        atol=1e-12,
    )
    assert_allclose(
        model.predict_point(frame, offset=np.column_stack(offset), context={'trials': trials}),
        trials * prediction[:, 0],
    )
    assert np.isfinite(model.standard_errors(frame, offset=np.column_stack(offset))).all()
    components = model.predict_components(frame, offset=np.column_stack(offset))
    assert_allclose(components.link, model.predict_link(frame, offset=np.column_stack(offset)))
    assert clone(model).family == "betabinomial_lss"
    path = tmp_path / "bb.pkl"
    model.save_model(path)
    loaded = GAMLSS.load_model(path)
    assert_allclose(
        loaded.predict_distribution(frame, offset=np.column_stack(offset), context={"trials": trials}).mean(),
        model.predict_distribution(frame, offset=np.column_stack(offset), context={"trials": trials}).mean()
    )
    assert not model.gam_.family.fit_context(len(frame))["trials"].flags.writeable


def test_weight_duplication_context_isolation_and_refit():
    frame, trials = _data()
    weights = np.tile([1, 2, 3], 50)
    fitted = _model().fit(frame, sample_weight=weights, context={"trials": trials})
    index = np.repeat(np.arange(len(frame)), weights)
    expanded = _model().fit(
        frame.iloc[index].reset_index(drop=True), context={"trials": trials[index]}
    )
    assert_allclose(fitted.predict_parameters(frame), expanded.predict_parameters(frame), atol=2e-7)
    expected = trials.copy()
    trials[:] += 1
    assert_allclose(fitted.gam_.family.fit_context(len(frame))["trials"], expected)
    # New fit owns fresh context, even when the number of rows is unchanged.
    fitted.fit(frame, context={"trials": trials})
    assert_allclose(fitted.gam_.family.fit_context(len(frame))["trials"], trials)
    assert_allclose(
        fitted.score(frame, frame.y, context={"trials": trials}),
        np.mean(fitted.predict_distribution(frame, context={"trials": trials}).log_prob(frame.y)),
    )
    assert_allclose(
        fitted.evaluate(frame, frame.y, context={"trials": trials})["Negative Log-Likelihood"],
        -fitted.score(frame, frame.y, context={"trials": trials}),
    )


def test_held_out_rows_require_new_context_even_with_same_length():
    frame, trials = _data()
    model = _model().fit(frame, context={"trials": trials})
    with pytest.raises(ValueError, match="context"):
        model.predict_distribution(frame)
    with pytest.raises(ValueError, match="context"):
        model.predict_point(frame)
    with pytest.raises(ValueError, match="context"):
        model.score(frame, frame.y)
    new_trials = trials[::-1].copy()
    assert_allclose(
        model.predict_point(frame, context={"trials": new_trials}),
        new_trials * model.predict_parameters(frame)[:, 0],
    )
    missing = frame.copy()
    missing.loc[3, "x"] = np.nan
    dist = model.predict_distribution(missing, context={"trials": new_trials})
    assert np.isnan(dist.mean()[3])
    assert_allclose(
        np.delete(dist.mean(), 3), np.delete(new_trials * model.predict_parameters(frame)[:, 0], 3)
    )
    with pytest.raises(NotImplementedError, match="row alignment"):
        model.predict_distribution(frame, context={"trials": trials}, na_action="omit")


def test_array_mode_and_permuted_prediction_context():
    frame, trials = _data()
    X = frame[["x"]].to_numpy()
    model = GAMLSS(
        family="betabinomial_lss",
        k=5,
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.7],
    ).fit(X, frame.y, context={"trials": trials})
    assert _fit_result(model.gam_).converged
    original = model.predict_distribution(X, context={"trials": trials})
    permuted = model.predict_distribution(X[::-1], context={"trials": trials[::-1]})
    assert_allclose(permuted.mean(), original.mean()[::-1])
    assert_allclose(
        permuted.log_prob(frame.y.to_numpy()[::-1]),
        original.log_prob(frame.y)[::-1],
    )


def test_context_is_fit_state_and_failed_native_refit_is_transactional():
    frame, trials = _data()
    family = BetaBinomialLocationPrecision()
    model = GAMLSS(
        family=family,
        formula={"mu": "y ~ x", "kappa": "~1"},
        optimize_smoothing=False,
    ).fit(frame, context={"trials": trials})
    assert not hasattr(family, "_fit_context")
    before = model.predict_parameters(frame)
    with pytest.raises(ValueError, match="unidentified"):
        model.gam_.fit(data=frame, context={"trials": np.ones(len(frame))})
    assert_allclose(model.predict_parameters(frame), before)
    assert_allclose(model.gam_.family.fit_context(len(frame))["trials"], trials)


@pytest.mark.parametrize(
    "trials", [[-1, 2], [1.5, 2], [np.nan, 2], [np.inf, 2], [1], 2]
)
def test_invalid_trial_context(trials):
    with pytest.raises(ValueError, match="trials|shape"):
        _distribution([0.2, 0.6], [2, 3], trials)


@pytest.mark.parametrize(
    "kind", ["missing", "one_trial", "zero_weight_information", "bad_count", "boundary"]
)
def test_explicit_fit_guards(kind):
    frame, trials = _data()
    weights = np.ones(len(frame))
    context = {"trials": trials}
    if kind == "missing":
        context = None
    elif kind in {"one_trial", "zero_weight_information"}:
        trials[:] = 1
        frame.y = frame.y % 2
        if kind == "zero_weight_information":
            trials[0], weights[0] = 5, 0
    elif kind == "bad_count":
        frame["y"] = frame.y.astype(float)
        frame.loc[2, "y"] = 0.5
    else:
        frame.y = 0
    with pytest.raises(
        ValueError,
        match="context|unidentified|integer successes|successes and failures",
    ):
        _model().fit(frame, context=context, sample_weight=weights)


def test_unsupported_optimizers_and_unrelated_context_are_explicitly_rejected():
    frame, trials = _data()
    for optimizer in ["magic", "nlm.fd", "optim"]:
        with pytest.raises(NotImplementedError):
            GAMLSS(
                family="betabinomial_lss",
                formula={"mu": "y ~ s(x,k=5)", "kappa": "~1"},
                smoothing_optimizer=optimizer,
            ).fit(frame, context={"trials": trials})
    with pytest.raises(NotImplementedError, match="context"):
        GAMLSS(
            family="normal",
            formula={"mu": "y ~ x", "sigma": "~1"},
            optimize_smoothing=False,
        ).fit(frame, context={"trials": trials})


@pytest.mark.parametrize("weighted", [False, True])
def test_outer_reml_derivatives_with_trial_context_and_offsets(weighted):
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    frame, trials = _data()
    weights = np.tile([0.0, 1.0, 2.0], 50) if weighted else None
    offsets = [0.1 * np.sin(frame.x.to_numpy()), np.linspace(-0.1, 0.1, len(frame))]
    model = GAMLSS(family='betabinomial_lss', formula={'mu': 'y ~ s(x,bs="cr",k=5)', 'kappa': '~ s(x,bs="cr",k=4)'}, optimize_smoothing=False, smoothing_params=[0.3, 0.7], irls_tol=1e-11).fit(frame, sample_weight=weights, offset=np.column_stack(offsets), context={'trials': trials})
    gam = model.gam_
    rho = np.log(gam.smoothing_params)

    def value(z):
        return np.array([criterion_value(gam, gam.y_, z, method="reml")])

    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient, approx_derivative(value, rho).ravel(), atol=2e-4, rtol=2e-4
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda z: criterion_gradient(gam, gam.y_, z, method="reml"), rho
        ),
        atol=2e-3,
        rtol=2e-3,
    )
    assert_allclose(gam.family.fit_context(len(frame))["trials"], trials)


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_automatic_reml_with_two_modeled_predictors(optimizer):
    frame, trials = _data()
    model = GAMLSS(
        family="betabinomial_lss",
        formula={"mu": 'y ~ s(x,bs="cr",k=5)', "kappa": '~ s(x,bs="cr",k=4)'},
        smoothing_optimizer=optimizer,
    ).fit(frame, context={"trials": trials})
    assert _fit_result(model.gam_).converged
    assert model.gam_._optim_method == "reml"
    assert np.all(np.isfinite(model.gam_.smoothing_params))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
    assert np.isfinite(model.score(frame, frame.y, context={"trials": trials}))
    assert np.isfinite(model.standard_errors(frame)).all()
