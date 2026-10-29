"""BCCG initialization and fixed-smoothing endpoint validation."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special, stats
from scipy.optimize import minimize

from nampy import GAMLSS
from nampy.diagnostics import quantile_residuals
from nampy.gam.families.registry import make_gam_family
from nampy.gam.model_state import _fit_result


def _data(n=180, *, sigma=0.28, nu=0.7):
    rng = np.random.default_rng(41)
    x = np.linspace(-1, 1, n)
    p = (np.arange(n) + 0.5) / n
    rng.shuffle(p)
    mu = np.exp(0.3 + 0.2 * x)
    c = stats.norm.cdf(1 / (sigma * abs(nu)))
    z = stats.norm.isf((1 - p) * c) if nu > 0 else stats.norm.ppf(p * c)
    return pd.DataFrame({"x": x, "y": mu * (1 + nu * sigma * z) ** (1 / nu)})


def _model(**kwargs):
    return GAMLSS(
        family="bccg_lss",
        formula={"mu": "y~x", "sigma": "~1", "nu": "~1"},
        optimize_smoothing=False,
        **kwargs,
    )


def _log_density(y, eta):
    mu, sigma, nu = np.exp(eta[:, 0]), np.exp(eta[:, 1]), eta[:, 2]
    lr = np.log(y) - np.log(mu)
    z = lr / sigma * special.exprel(nu * lr)
    with np.errstate(divide="ignore"):
        normalizer = special.log_ndtr(1 / (sigma * np.abs(nu)))
    return (
        nu * lr
        - np.log(sigma)
        - z * z / 2
        - np.log(y)
        - np.log(2 * np.pi) / 2
        - normalizer
    )


@pytest.mark.parametrize("sigma,nu", [(0.28, 0.7), (0.9, 1.0), (0.18, -3.0)])
def test_unpenalized_endpoint_matches_independent_normalized_likelihood_optimum(
    sigma, nu
):
    frame = _data(sigma=sigma, nu=nu)
    model = _model().fit(frame)
    assert _fit_result(model.gam_).converged
    x, y = frame.x.to_numpy(), frame.y.to_numpy()

    def objective(b):
        eta = np.column_stack(
            [b[0] + b[1] * x, np.full(len(x), b[2]), np.full(len(x), b[3])]
        )
        return -np.sum(_log_density(y, eta))

    reference = minimize(
        objective, [0.3, 0.2, np.log(sigma), nu], method="BFGS", options={"gtol": 1e-6}
    )
    assert np.linalg.norm(reference.jac, ord=np.inf) < 2e-4
    eta = np.column_stack(
        [
            reference.x[0] + reference.x[1] * x,
            np.full(len(x), reference.x[2]),
            np.full(len(x), reference.x[3]),
        ]
    )
    assert_allclose(model.predict_link(frame), eta, atol=2e-5, rtol=0)
    assert_allclose(model.score(frame, y), -reference.fun / len(y), atol=2e-10)
    assert_allclose(
        model.predict_distribution(frame).log_prob(y), _log_density(y, eta), atol=2e-5
    )


def test_initialization_projects_upstream_targets_and_offsets_without_a_new_solver():
    family = make_gam_family("bccg_lss")
    y = np.array([0.4, 0.8, 1.2, 2.1, 3.0])
    x = np.linspace(-1, 1, len(y))
    X = np.column_stack([np.ones(len(y)), x, np.ones(len(y)), np.ones(len(y))])
    jj = [np.array([0, 1]), np.array([2]), np.array([3])]
    offsets = [x * 0.1, np.full(len(y), 0.03), x * 0.2]
    start = family.initialize(y, X, jj, offset=offsets)
    targets = [
        np.log((y + y.mean()) / 2),
        np.full(len(y), np.log(0.1)),
        np.full(len(y), 0.5),
    ]
    for k, columns in enumerate(jj):
        expected = np.linalg.lstsq(X[:, columns], targets[k] - offsets[k], rcond=None)[
            0
        ]
        assert_allclose(start[columns], expected, atol=2e-15)


@pytest.mark.parametrize("case", ["positive", "negative", "weighted"])
def test_saved_r_unpenalized_endpoints(case):
    from pathlib import Path

    frame = pd.read_csv(
        Path(__file__).parents[1]
        / "reference_fixtures"
        / "bccg_fitting"
        / f"{case}.csv"
    )
    model = _model().fit(frame, sample_weight=frame.w.to_numpy())
    assert _fit_result(model.gam_).converged, (
        _fit_result(model.gam_).failure_reason,
        np.max(np.abs(model.predict_parameters(frame) - frame[["mu", "sigma", "nu"]].to_numpy())),
    )
    assert_allclose(
        model.predict_parameters(frame), frame[["mu", "sigma", "nu"]], atol=2e-5, rtol=0
    )
    assert_allclose(
        model.predict_distribution(frame).log_prob(frame.y), frame.log_prob, atol=2e-5
    )
    assert_allclose(
        model.score(frame, frame.y, sample_weight=frame.w),
        np.average(frame.log_prob, weights=frame.w),
        atol=2e-10,
    )


def _smooth_model(**kwargs):
    return GAMLSS(
        family="bccg_lss",
        formula={
            "mu": "y~s(x,bs='cr',k=5)",
            "sigma": "~s(x,bs='cr',k=4)",
            "nu": "~s(x,bs='cr',k=4)",
        },
        optimize_smoothing=False,
        smoothing_params=[0.3, 0.5, 0.7],
        **kwargs,
    )


def test_three_predictor_fixed_smoothing_endpoint_and_full_covariance(tmp_path):
    from scipy.optimize._numdiff import approx_derivative
    from sklearn.base import clone

    from nampy.gam.model_state import _fit_state, _predictor_full_indices
    from nampy.gam.predict.general import _general_family_covariance_for_prediction
    from nampy.gam.predict.predictions import select_prediction_covariance_matrix

    frame = _data()
    x, y = frame.x.to_numpy(), frame.y.to_numpy()
    weights = np.resize([1.0, 0.0, 2.0, 0.5, 3.0], len(frame))
    offsets = [0.05 * x, -0.03 * x, 0.1 * x]
    model = _smooth_model().fit(frame, sample_weight=weights, offset=np.column_stack(offsets))
    result = _fit_result(model.gam_)
    assert result.converged
    assert_allclose(model.gam_.smoothing_params, [0.3, 0.5, 0.7])
    X = model.lpmatrix(frame)
    jj = _predictor_full_indices(model.gam_)
    penalty = _fit_state(model.gam_).penalty_matrix
    beta = result.coef_full

    def predictors(b):
        return np.column_stack(
            [
                X[:, columns] @ b[columns] + o
                for columns, o in zip(jj, offsets, strict=True)
            ]
        )

    def objective(b):
        return -np.sum(weights * _log_density(y, predictors(b))) + 0.5 * b @ penalty @ b

    oracle = minimize(
        objective,
        beta + 0.01 * np.cos(np.arange(len(beta))),
        method="BFGS",
        options={"gtol": 1e-6},
    )
    assert np.linalg.norm(oracle.jac, ord=np.inf) < 2e-4
    assert_allclose(oracle.fun, objective(beta), atol=2e-9)
    assert_allclose(
        predictors(oracle.x),
        model.predict_link(frame, offset=np.column_stack(offsets)),
        atol=2e-5,
        rtol=0,
    )
    score = approx_derivative(lambda b: np.array([objective(b)]), beta).ravel()
    assert np.max(np.abs(score)) < 2e-4
    covariance = _general_family_covariance_for_prediction(
        model.gam_, select_prediction_covariance_matrix(model.gam_, cov="bayes")
    )
    observed_information = approx_derivative(
        lambda b: approx_derivative(lambda bb: np.array([objective(bb)]), b).ravel(),
        beta,
    )
    assert_allclose(covariance @ observed_information, np.eye(len(beta)), atol=2e-4)
    errors = []
    for row in range(3):
        derivative = approx_derivative(
            lambda b, row=row: model.gam_.family.distribution_parameters_from_eta(
                predictors(b)
            )[row],
            beta,
        )
        errors.append(np.sqrt(np.diag(derivative @ covariance @ derivative.T)))
    assert_allclose(model.standard_errors(frame, offset=np.column_stack(offsets))[:3], errors, atol=2e-8)
    components = model.predict_components(frame, offset=np.column_stack(offsets))
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(frame), 3)
    distribution = model.predict_distribution(frame, offset=np.column_stack(offsets))
    assert_allclose(model.predict_point(frame, offset=np.column_stack(offsets)), distribution.mean())
    assert distribution.sample(3, random_state=3).shape == (3, len(frame))
    assert np.isfinite(quantile_residuals(distribution, y).residuals).all()
    assert clone(model).get_params() == model.get_params()
    restored = GAMLSS.load_model(model.save_model(tmp_path / "bccg.pkl"))
    assert_allclose(
        restored.predict_distribution(frame, offset=np.column_stack(offsets)).quantile([0.1, 0.5, 0.9]),
        distribution.quantile([0.1, 0.5, 0.9]),
    )
    missing = frame.copy()
    missing.loc[4, "x"] = np.nan
    assert np.isnan(model.predict_distribution(missing, offset=np.column_stack(offsets)).cdf(1)[4])
    assert np.isnan(model.standard_errors(missing, offset=np.column_stack(offsets))[4]).all()


def test_frequency_weights_and_offsets_match_duplicated_fixed_smoothing_fit():
    frame = _data()
    weights = np.resize([1, 0, 2, 3], len(frame))
    offsets = [
        frame.x.to_numpy() * 0.02,
        frame.x.to_numpy() * -0.01,
        frame.x.to_numpy() * 0.03,
    ]

    # Fix the same physical knots: default quantile knots change when rows are
    # duplicated, which would compare different penalized function spaces.
    def make_model():
        return GAMLSS(
            family="bccg_lss",
            formula={
                "mu": "y~s(x,bs='cr',k=5)",
                "sigma": "~s(x,bs='cr',k=5)",
                "nu": "~s(x,bs='cr',k=5)",
            },
            knots={"x": np.linspace(-1, 1, 5)},
            optimize_smoothing=False,
            smoothing_params=[0.3, 0.5, 0.7],
        )

    model = make_model().fit(frame, sample_weight=weights, offset=np.column_stack(offsets))
    index = np.repeat(np.arange(len(frame)), weights)
    repeated = make_model().fit(
        frame.iloc[index].reset_index(drop=True), offset=np.column_stack(offsets)[index]
    )
    assert _fit_result(model.gam_).converged and _fit_result(repeated.gam_).converged
    assert_allclose(
        model.predict(frame, offset=np.column_stack(offsets)),
        repeated.predict(frame, offset=np.column_stack(offsets)),
        atol=3e-6,
        rtol=0,
    )


@pytest.mark.parametrize(
    "optimizer,message",
    [("optim", "null deviance"), ("magic", "Gaussian GCV/UBRE"), ("nlm.fd", "SCAM")],
)
def test_unsupported_automatic_optimizers_stay_explicit(optimizer, message):
    model = GAMLSS(
        family="bccg_lss",
        formula={"mu": "y~s(x,k=5)", "sigma": "~1", "nu": "~1"},
        smoothing_optimizer=optimizer,
    )
    with pytest.raises(NotImplementedError, match=message):
        model.fit(_data())


@pytest.mark.parametrize("method", ["ml", "gcv", "ubre", "ncv"])
def test_other_method_labels_retain_general_family_reml_coercion(method):
    from types import SimpleNamespace

    from nampy.gam.fit.capabilities import coerce_general_family_smoothing_method
    from nampy.gam.fit.selection import supports_smoothing_method

    model = SimpleNamespace(family=make_gam_family("bccg_lss"), n_samples_=0)
    assert coerce_general_family_smoothing_method(model, method) == "reml"
    if method in {"gcv", "ubre", "ncv"}:
        assert not supports_smoothing_method(model, method)


@pytest.mark.parametrize(
    "y,weights",
    [([1.0, 1.0, 1.0], None), ([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]), ([1.0, 2.0], None)],
)
def test_unidentified_small_or_degenerate_positive_weight_data_fail(y, weights):
    frame = pd.DataFrame({"y": y})
    model = GAMLSS(
        family="bccg_lss",
        formula={"mu": "y~1", "sigma": "~1", "nu": "~1"},
        optimize_smoothing=False,
    )
    with pytest.raises(ValueError, match="three distinct positive-weight"):
        model.fit(frame, sample_weight=weights)


@pytest.mark.parametrize("y", [[0, 1, 2], [-1, 1, 2], [1, np.inf, 2], [1, np.nan, 2]])
def test_response_support_is_not_clipped(y):
    with pytest.raises(ValueError, match="strictly positive"):
        make_gam_family("bccg_lss").validate_y(y)


def test_overly_tight_stopping_rule_remains_honest_about_nonconvergence():
    from pathlib import Path

    frame = pd.read_csv(
        Path(__file__).parents[1]
        / "reference_fixtures"
        / "bccg_fitting"
        / "weighted.csv"
    )
    model = _model(irls_tol=1e-11).fit(frame, sample_weight=frame.w)
    # A float64 plateau can occur even very close to the independent R endpoint.
    # Preserve the existing mgcv-style solver status, with no rescue/flag rewrite.
    status = model.distribution_fit_status()
    if not _fit_result(model.gam_).converged:
        assert status.state == "nonconverged" and not status.eligible
        assert _fit_result(model.gam_).failure_reason
    else:
        assert status.eligible
    assert_allclose(
        model.predict_parameters(frame), frame[["mu", "sigma", "nu"]], atol=2e-5, rtol=0
    )


def test_zero_power_endpoint_and_response_unit_equivariance():
    n = 180
    probabilities = (np.arange(n) + 0.5) / n
    frame = pd.DataFrame({"y": np.exp(0.3 + 0.4 * stats.norm.ppf(probabilities))})
    formula = {"mu": "y~1", "sigma": "~1", "nu": "~1"}
    models = []
    for factor in [1.0, 0.001, 1000.0]:
        scaled = frame * factor
        model = GAMLSS(
            family="bccg_lss", formula=formula, optimize_smoothing=False
        ).fit(scaled)
        assert _fit_result(model.gam_).converged
        models.append((factor, model, model.predict_parameters(scaled)))
    reference = models[0][2]
    assert abs(reference[0, 2]) < 2e-5
    for factor, model, parameters in models:
        assert_allclose(parameters / [factor, 1, 1], reference, atol=2e-5, rtol=0)
        assert_allclose(
            model.score(frame * factor, frame.y * factor),
            models[0][1].score(frame, frame.y) - np.log(factor),
            atol=2e-10,
        )
