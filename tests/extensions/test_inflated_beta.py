"""Four-predictor inflated beta, exact likelihood and coupled uncertainty."""

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, softmax
from sklearn.base import clone

from nampy import GAMLSS
from nampy.distributions.mixtures import inflated_beta_prediction
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.inflated_beta import (
    InflatedBeta,
)
from nampy.gam.model_state import _fit_result, _predictor_full_indices


def _data():
    rng = np.random.default_rng(612)
    x = np.linspace(-1, 1, 210)
    mu, phi = expit(-0.2 + 0.5 * x), 6.0
    y = rng.beta(mu * phi, (1 - mu) * phi)
    probabilities = softmax(
        np.column_stack([-0.9 + 0.4 * x, -1.1 - 0.3 * x, np.zeros(len(x))]), axis=1
    )
    u = rng.uniform(size=len(x))
    y[u < probabilities[:, 0]] = 0
    y[(u >= probabilities[:, 0]) & (u < probabilities[:, :2].sum(axis=1))] = 1
    return pd.DataFrame({"x": x, "y": y})


def _model(*, smooth=False, **kwargs):
    formula = {
        "mu": 'y ~ s(x,bs="cr",k=5)' if smooth else "y ~ x",
        "phi": "~1",
        "pi0": '~ s(x,bs="cr",k=4)' if smooth else "~x",
        "pi1": "~x",
    }
    return GAMLSS(family="beinf_lss", formula=formula, **kwargs)


def test_independent_weighted_mle_with_four_offsets_and_normalized_likelihood():
    frame = _data()
    weights = np.tile([0.0, 1.0, 2.0], 70)
    offsets = [
        0.1 * np.sin(frame.x.to_numpy()),
        np.full(len(frame), 0.05),
        np.linspace(-0.1, 0.1, len(frame)),
        np.linspace(0.1, -0.05, len(frame)),
    ]
    model = _model(optimize_smoothing=False).fit(
        frame, sample_weight=weights, offset=offsets
    )
    assert _fit_result(model.gam_).converged
    design = np.column_stack([np.ones(len(frame)), frame.x])
    y = frame.y.to_numpy()
    interior = (y > 0) & (y < 1)

    def objective(beta):
        mu, phi = expit(design @ beta[:2] + offsets[0]), np.exp(beta[2] + offsets[1])
        logits = np.column_stack(
            [
                design @ beta[3:5] + offsets[2],
                design @ beta[5:] + offsets[3],
                np.zeros(len(frame)),
            ]
        )
        p = softmax(logits, axis=1)
        logp = np.log(p[np.arange(len(y)), np.where(y == 0, 0, np.where(y == 1, 1, 2))])
        logp[interior] += stats.beta.logpdf(
            y[interior],
            mu[interior] * phi[interior],
            (1 - mu[interior]) * phi[interior],
        )
        return -np.sum(weights * logp)

    oracle = minimize(
        objective,
        [-0.2, 0.5, np.log(6), -0.9, 0.4, -1.1, -0.3],
        method="BFGS",
        options={"gtol": 1e-6},
    )
    expected = np.column_stack(
        [
            expit(design @ oracle.x[:2] + offsets[0]),
            np.exp(oracle.x[2] + offsets[1]),
            softmax(
                np.column_stack(
                    [
                        design @ oracle.x[3:5] + offsets[2],
                        design @ oracle.x[5:] + offsets[3],
                        np.zeros(len(frame)),
                    ]
                ),
                axis=1,
            ),
        ]
    )
    parameters = model.predict(frame, offset=offsets)
    assert_allclose(parameters, expected, atol=2e-5, rtol=0)
    dist = model.predict_distribution(frame, offset=offsets)
    assert_allclose(-np.sum(weights * dist.log_prob(frame.y)), oracle.fun, atol=2e-8)
    assert_allclose(dist.mean(), parameters[:, 3] + parameters[:, 4] * parameters[:, 0])
    assert_allclose(model.predict_point(frame, offset=offsets), dist.mean(), atol=1e-14)
    assert_allclose(
        model.gam_.family.variance_from_parameters(parameters),
        dist.variance(),
        atol=2e-15,
    )
    assert model.predict(raw=True).shape == (len(frame), 4)
    assert model.predictor_names_ == ("mu", "phi", "pi0", "pi1")
    assert model.parameter_names_ == ("mu", "phi", "pi0", "pi1", "pi_interior")


@pytest.mark.parametrize("eta", [[-0.4, 1.3, -0.8, -1.1], [0.5, 0.4, 0.3, -0.4]])
def test_exact_mixed_tensors_through_fourth_order_with_60_digit_oracle(eta):
    mp = pytest.importorskip("mpmath")
    y = np.array([0.0, 0.3, 1.0])
    family = InflatedBeta()
    value = family._log_density(
        y, *[Taylor(np.full(len(y), eta[k]), 4, 4, k) for k in range(4)]
    )
    with mp.workdps(60):
        point = tuple(mp.mpf(str(x)) for x in eta)
        for row, response in enumerate(y):

            def likelihood(em, lp, e0, e1, response=response):
                normalizer = mp.log(1 + mp.exp(e0) + mp.exp(e1))
                if response == 0:
                    return e0 - normalizer
                if response == 1:
                    return e1 - normalizer
                mu, phi = 1 / (1 + mp.exp(-em)), mp.exp(lp)
                a, b = mu * phi, (1 - mu) * phi
                yy = mp.mpf(str(response))
                return (
                    mp.loggamma(phi)
                    - mp.loggamma(a)
                    - mp.loggamma(b)
                    + (a - 1) * mp.log(yy)
                    + (b - 1) * mp.log1p(-yy)
                    - normalizer
                )

            assert_allclose(value.value[row], float(likelihood(*point)), atol=2e-12)
            for degree in range(1, 5):
                indices = list(combinations_with_replacement(range(4), degree))
                expected = [
                    float(
                        mp.diff(
                            likelihood, point, tuple(idx.count(k) for k in range(4))
                        )
                    )
                    for idx in indices
                ]
                assert_allclose(
                    value.packed_derivatives(degree)[row],
                    expected,
                    atol=3e-11,
                    rtol=3e-11,
                )
                if response in {0.0, 1.0}:
                    assert all(
                        value.packed_derivatives(degree)[row, j] == 0
                        for j, idx in enumerate(indices)
                        if 0 in idx or 1 in idx
                    )


def test_coupled_simplex_and_tiny_continuous_weight_are_not_lost():
    family = InflatedBeta()
    eta = np.array(
        [[-0.3, 1.2, 0.4, -0.7], [0.5, 0.6, -0.8, 0.1], [0.0, np.log(4), 50.0, 50.0]]
    )
    parameters = family.distribution_parameters_from_eta(eta)
    assert_allclose(parameters[:, 2:].sum(axis=1), 1, atol=2e-16)
    assert parameters[2, 4] > 0
    assert 1 - parameters[2, 2] - parameters[2, 3] == 0
    dist = inflated_beta_prediction(parameters)
    expected = np.log(parameters[2, 4]) + stats.beta.logpdf(0.5, 2, 2)
    assert_allclose(dist.log_prob([0.4, 0.6, 0.5])[2], expected, atol=2e-14)
    for row in range(2):
        jacobian = approx_derivative(
            lambda z: family.distribution_parameters_from_eta(z[None, :])[0], eta[row]
        )
        assert_allclose(
            family.distribution_parameter_jacobian(eta)[row], jacobian, atol=2e-10
        )
    # Endpoint/interior likelihoods never evaluate log(0) in the other branch.
    value = family._log_density(
        np.array([0.0, 1.0, 0.5]), *[Taylor(eta[:, k], 4, 4, k) for k in range(4)]
    )
    assert_allclose(value.value, dist.log_prob([0.0, 1.0, 0.5]), atol=2e-14)


@pytest.mark.parametrize("atom_logits", [[1000.0, 0.0], [-1000.0, 0.0]])
def test_unrepresentable_finite_logit_probabilities_are_explicit(atom_logits):
    with pytest.raises(NotImplementedError, match="float64 interior range"):
        InflatedBeta().distribution_parameters_from_eta(
            np.array([[0.0, 1.0, *atom_logits]])
        )


def test_boundary_weights_remain_supported_as_probability_queries():
    dist = inflated_beta_prediction(
        [[0.5, 4.0, 0.0, 0.0, 1.0], [0.5, 4.0, 1.0, 0.0, 0.0]]
    )
    assert_allclose(dist.mean(), [0.5, 0.0])
    assert_allclose(dist.variance(), [0.05, 0.0])
    assert_allclose(dist.log_prob([0.5, 0.0]), [stats.beta.logpdf(0.5, 2, 2), 0.0])


def test_full_coefficient_delta_covariance_with_shared_predictor_blocks():
    family = InflatedBeta()
    x = np.linspace(-0.5, 0.5, 4)
    X = np.column_stack([np.ones(4), x, np.ones(4), x, np.ones(4), x, np.ones(4)])
    jj = [np.array([0, 1]), np.array([2]), np.array([3, 4]), np.array([3, 5, 6])]
    coef = np.array([0.2, 0.3, 1.4, -0.3, 0.2, -0.1, 0.4])
    rng = np.random.default_rng(13)
    A = rng.normal(size=(7, 7))
    covariance = A @ A.T / 100
    parameters, se = family.predict_response(X, jj, coef, se=True, Vb=covariance)
    expected = []
    for row in range(len(X)):
        derivative = approx_derivative(
            lambda b, row=row: family.predict_fitted(X, jj, b)[row], coef
        )
        expected.append(np.sqrt(np.diag(derivative @ covariance @ derivative.T)))
    assert parameters.shape == se.shape == (4, 5)
    assert_allclose(se, expected, atol=1e-9)


def test_fixed_smoothing_duplicate_weights_components_persistence_and_missing_rows(
    tmp_path,
):
    frame = _data()
    weights = np.tile([1, 2, 3], 70)
    kwargs = {
        "smooth": True,
        "optimize_smoothing": False,
        "smoothing_params": [0.3, 0.7],
        "irls_tol": 1e-10,
    }
    model = _model(**kwargs).fit(frame, sample_weight=weights)
    index = np.repeat(np.arange(len(frame)), weights)
    repeated = _model(**kwargs).fit(frame.iloc[index].reset_index(drop=True))
    assert_allclose(model.predict(frame), repeated.predict(frame), atol=3e-6)
    components = model.predict_components(frame)
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(frame), 4)
    assert model.predict(frame).shape == (len(frame), 5)
    assert_allclose(components.link, model.predict_link(frame))
    assert components.predictor_names == model.predictor_names_
    assert all(value.shape == (len(frame), 4) for value in components.terms.values())
    assert np.isfinite(model.standard_errors(frame)).all()
    assert model.standard_errors(frame).shape == (len(frame), 5)
    assert model.standard_errors(frame, type="link").shape == (len(frame), 4)
    assert np.all(model.standard_errors(frame)[:, 4] > 0)
    # Compare the public covariance adapter to a coefficient-space numerical oracle.
    from nampy.gam.predict.general import _general_family_covariance_for_prediction
    from nampy.gam.predict.predictions import select_prediction_covariance_matrix

    P = model.lpmatrix(frame)
    covariance = _general_family_covariance_for_prediction(
        model.gam_, select_prediction_covariance_matrix(model.gam_, cov="bayes")
    )
    jj = _predictor_full_indices(model.gam_)
    beta = _fit_result(model.gam_).coef_full
    errors = []
    for row in range(3):
        derivative = approx_derivative(
            lambda b, row=row: model.gam_.family.predict_fitted(P, jj, b)[row], beta
        )
        errors.append(np.sqrt(np.diag(derivative @ covariance @ derivative.T)))
    assert_allclose(model.standard_errors(frame)[:3], errors, atol=1e-8)
    assert clone(model).family == "beinf_lss"
    path = tmp_path / "beinf.pkl"
    model.save_model(path)
    assert_allclose(GAMLSS.load_model(path).predict(frame), model.predict(frame))
    missing = frame.copy()
    missing.loc[3, "x"] = np.nan
    assert np.isnan(model.predict_distribution(missing).mean()[3])
    assert np.isnan(model.standard_errors(missing)[3]).all()


@pytest.mark.parametrize(
    "kind",
    [
        "negative",
        "above_one",
        "no_zero",
        "no_one",
        "no_interior",
        "constant_interior",
        "zero_weight_ones",
    ],
)
def test_absent_invalid_or_degenerate_components_are_explicit(kind):
    frame = _data()
    weights = np.ones(len(frame))
    if kind == "negative":
        frame.loc[0, "y"] = -0.1
    elif kind == "above_one":
        frame.loc[0, "y"] = 1.1
    elif kind == "no_zero":
        frame.loc[frame.y == 0, "y"] = 0.1
    elif kind == "no_one":
        frame.loc[frame.y == 1, "y"] = 0.9
    elif kind == "no_interior":
        frame.loc[(frame.y > 0) & (frame.y < 1), "y"] = 0
    elif kind == "constant_interior":
        frame.loc[(frame.y > 0) & (frame.y < 1), "y"] = 0.4
    else:
        weights[frame.y == 1] = 0
    with pytest.raises(ValueError, match="inside|components"):
        _model(optimize_smoothing=False).fit(frame, sample_weight=weights)


def test_saved_r_beinf_probabilities_unpenalized_fit_and_variance_discrepancy():
    root = Path(__file__).parents[1] / "reference_fixtures" / "inflated_beta"
    ref = pd.read_csv(root / "probability.csv")
    distribution = inflated_beta_prediction(
        ref[["mu", "phi", "pi0", "pi1", "pi_interior"]].to_numpy()
    )
    assert_allclose(distribution.log_prob(ref.y), ref.log_prob, atol=2e-13)
    assert_allclose(distribution.cdf(ref.y), ref.cdf, atol=2e-13)
    for row, p in enumerate(ref.probability):
        assert_allclose(
            distribution.quantile(p)[row, 0], ref["quantile"][row], atol=2e-13
        )
    assert_allclose(distribution.variance(), ref.mixture_variance, atol=1e-14)
    # The recorded R variance hook adds rather than subtracts the squared mean.
    # Density/CDF/quantile semantics agree; that inconsistent hook is not copied.
    assert_allclose(
        ref.reported_variance,
        ref.mixture_variance + 2 * distribution.mean() ** 2,
        atol=2e-14,
    )
    frame = pd.read_csv(root / "fit.csv")
    model = _model(optimize_smoothing=False).fit(frame)
    assert _fit_result(model.gam_).converged
    assert_allclose(
        model.predict(frame),
        frame[["mu", "phi", "pi0", "pi1", "pi_interior"]],
        atol=3e-5,
        rtol=0,
    )


def _all_smooth_model(**kwargs):
    formula = {
        name: ("y " if i == 0 else "") + '~ s(x,bs="cr",k=4)'
        for i, name in enumerate(InflatedBeta.parameter_names)
    }
    return GAMLSS(family="beinf_lss", formula=formula, **kwargs)


@pytest.mark.parametrize("weighted", [False, True])
def test_outer_reml_four_penalty_gradient_hessian_with_offsets(weighted):
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    frame = _data()
    weights = np.tile([0.0, 1.0, 2.0], 70) if weighted else None
    offsets = [
        0.08 * np.sin(frame.x.to_numpy()),
        np.full(len(frame), 0.03),
        np.linspace(-0.1, 0.1, len(frame)),
        np.linspace(0.05, -0.05, len(frame)),
    ]
    model = _all_smooth_model(
        optimize_smoothing=False, smoothing_params=[0.3, 0.5, 0.7, 0.9], irls_tol=1e-10
    ).fit(frame, sample_weight=weights, offset=offsets)
    gam = model.gam_
    rho = np.log(gam.smoothing_params)
    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient,
        approx_derivative(
            lambda z: np.array([criterion_value(gam, gam.y_, z, method="reml")]), rho
        ).ravel(),
        atol=2e-4,
        rtol=2e-4,
    )
    assert_allclose(
        hessian,
        approx_derivative(
            lambda z: criterion_gradient(gam, gam.y_, z, method="reml"), rho
        ),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_automatic_reml_with_all_four_predictors_smoothed(optimizer):
    frame = _data()
    model = _all_smooth_model(smoothing_optimizer=optimizer).fit(frame)
    assert _fit_result(model.gam_).converged
    assert model.gam_._optim_method == "reml"
    assert np.all(np.isfinite(model.gam_.smoothing_params))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
    assert np.isfinite(model.score(frame, frame.y))
    assert model.standard_errors(frame).shape == (len(frame), 5)
    assert_allclose(model.predict(frame)[:, 2:].sum(axis=1), 1, atol=3e-16)


def test_unsupported_routes_links_and_invalid_simplex_are_explicit():
    from types import SimpleNamespace

    from nampy.gam.families.registry import make_gam_family
    from nampy.gam.fit.capabilities import coerce_general_family_smoothing_method

    family = InflatedBeta()
    for method in ["ml", "laml", "gcv", "ubre", "ncv"]:
        assert (
            coerce_general_family_smoothing_method(
                SimpleNamespace(family=family, n_samples_=0), method
            )
            == "reml"
        )
    for optimizer in ["magic", "nlm.fd", "optim"]:
        with pytest.raises(NotImplementedError):
            _model(smooth=True, smoothing_optimizer=optimizer).fit(_data())
    with pytest.raises(ValueError, match="Unsupported"):
        make_gam_family({"name": "beinf_lss", "link": "identity"})
    with pytest.raises(ValueError, match="simplex"):
        family.point_prediction_from_parameters([[0.5, 4, 0.3, 0.4, 0.5]])
