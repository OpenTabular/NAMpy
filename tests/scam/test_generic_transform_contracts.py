from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.coefficients import (
    BlockCoefficientTransform,
    CoordinatewiseCoefficientTransform,
    IdentityCoefficientTransform,
    compose_coefficient_transforms,
)
from nampy.gam.fit.selection.criteria import criterion_gradient, criterion_value
from nampy.gam.observations import AR1ObservationTransform


def test_identity_and_block_coefficient_transforms_preserve_layout():
    identity = IdentityCoefficientTransform(2)
    positive = CoordinatewiseCoefficientTransform(
        np.array([True, False]), positive_map="exp"
    )
    transform = compose_coefficient_transforms([identity, positive])

    assert isinstance(transform, BlockCoefficientTransform)
    theta = np.array([1.0, -2.0, np.log(3.0), 4.0])
    expected = np.array([1.0, -2.0, 3.0, 4.0])
    np.testing.assert_allclose(transform.forward(theta), expected)
    np.testing.assert_allclose(transform.inverse(expected), theta)
    np.testing.assert_array_equal(transform.positive_mask, [False, False, True, False])


def test_covariance_transport_policy_is_owned_by_transform_block():
    theta = np.array([-0.4])
    covariance = np.array([[2.0]])
    generic = CoordinatewiseCoefficientTransform(
        np.array([True]), positive_map="softplus", covariance_transport="jacobian"
    )
    scam_compatible = CoordinatewiseCoefficientTransform(
        np.array([True]), positive_map="softplus", covariance_transport="prediction"
    )

    expected_generic = 2.0 * generic.derivative(theta, order=1)[0] ** 2
    expected_scam = 2.0 * scam_compatible.forward(theta)[0] ** 2
    assert generic.transport_covariance(theta, covariance)[0, 0] == pytest.approx(
        expected_generic
    )
    assert scam_compatible.transport_covariance(theta, covariance)[
        0, 0
    ] == pytest.approx(expected_scam)


def test_ar1_transform_applies_identically_to_vectors_and_matrix_columns():
    values = np.arange(12.0).reshape(6, 2)
    transform = AR1ObservationTransform(
        size=6,
        rho=0.35,
        starts=np.array([True, False, False, True, False, False]),
    )

    actual = transform.apply(values)
    by_column = np.column_stack(
        [transform.apply(values[:, column]) for column in range(values.shape[1])]
    )
    np.testing.assert_allclose(actual, by_column)
    np.testing.assert_array_equal(actual[[0, 3]], values[[0, 3]])


def test_ordinary_gaussian_ar1_supports_generic_gcv_smoothing():
    rng = np.random.default_rng(813)
    x = np.linspace(-1.0, 2.0, 72)
    data = pd.DataFrame(
        {"y": np.sin(1.3 * x) + rng.normal(scale=0.08, size=x.size), "x": x}
    )
    model = GAM(
        formula='y ~ s(x, bs="ps", k=8)',
        family="gaussian",
        ar1_rho=0.25,
        smoothing_method="gcv",
        smoothing_optimizer="bfgs",
        optimize_smoothing=True,
    ).fit(data=data)

    assert model.gam_result_.compiled_model.observation_transform.is_identity is False
    assert np.all(np.isfinite(model.predict(data, type="response")))
    assert np.all(np.isfinite(model.ar1_standardized_residuals()))


def test_ordinary_gaussian_ar1_gcv_gradient_uses_transformed_criterion_state():
    x = np.linspace(-1.2, 1.6, 64)
    data = pd.DataFrame({"y": np.cos(1.1 * x) + 0.05 * x, "x": x})
    model = GAM(
        formula='y ~ s(x, bs="ps", k=8)',
        family="gaussian",
        ar1_rho=0.3,
        smoothing_params=[0.9],
        optimize_smoothing=False,
    ).fit(data=data)
    log_sp = np.log(np.array([0.9]))
    actual = criterion_gradient(model, model.y_, log_sp, method="gcv")
    step = 1e-5
    expected = np.array(
        [
            (
                criterion_value(model, model.y_, log_sp + step, method="gcv")
                - criterion_value(model, model.y_, log_sp - step, method="gcv")
            )
            / (2.0 * step)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)


def test_observation_transform_rejects_unsupported_likelihoods_explicitly():
    data = pd.DataFrame({"y": np.arange(24) % 2, "x": np.linspace(-1.0, 1.0, 24)})
    with pytest.raises(NotImplementedError, match="Gaussian family"):
        GAM(
            formula="y ~ x",
            family="binomial",
            ar1_rho=0.2,
        ).fit(data=data)


@pytest.mark.parametrize("method", ["ml", "reml", "laml"])
def test_ar1_likelihood_routes_use_bam_correlation_determinant_terms(method):
    data = pd.DataFrame(
        {"y": np.linspace(-0.3, 0.8, 30), "x": np.linspace(-1.0, 1.0, 30)}
    )
    model = GAM(
        formula='y ~ s(x, bs="ps", k=7)',
        family="gaussian",
        ar1_rho=0.2,
        smoothing_method=method,
        optimize_smoothing=True,
    ).fit(data=data)
    assert np.isfinite(model.smoothing_score_)
    assert model._optim_method == method


@pytest.mark.parametrize("basis", ["bs", "ps", "cp"])
def test_pspline_exposes_exact_derivative_provider_at_new_data(basis):
    x = np.linspace(-1.5, 2.0, 60)
    data = pd.DataFrame({"y": np.sin(x), "x": x})
    model = GAM(
        formula=f'y ~ s(x, bs="{basis}", k=9, m=c(2, 2))',
        family="gaussian",
        smoothing_params=[0.8],
    ).fit(data=data)
    new_data = pd.DataFrame({"x": np.linspace(-1.2, 1.7, 19)})

    derivative = model.derivative(new_data, smooth_number=1, deriv=1)
    eps = 1e-6
    plus = new_data.copy()
    minus = new_data.copy()
    plus["x"] += eps
    minus["x"] -= eps
    finite_difference = (
        model.predict(plus, type="terms")[:, 0]
        - model.predict(minus, type="terms")[:, 0]
    ) / (2.0 * eps)

    np.testing.assert_allclose(
        derivative.derivative, finite_difference, rtol=2e-7, atol=2e-8
    )
    assert derivative.derivative_matrix.shape[0] == len(new_data)


@pytest.mark.parametrize("basis", ["bs", "ps", "cp", "cr", "cc"])
def test_linear_functional_smooth_is_available_through_generic_by_contract(basis):
    rng = np.random.default_rng(91)
    locations = np.tile(np.linspace(-1.0, 1.0, 11), (28, 1))
    weights = rng.normal(size=locations.shape)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=locations.shape[0]),
            "X": list(locations),
            "L": list(weights),
        }
    )
    model = GAM(
        formula=f'y ~ s(X, by=L, bs="{basis}", k=8)',
        family="gaussian",
        smoothing_params=[1.0],
    ).fit(data=data)

    term = model.gam_result_.compiled_model.compiled_terms[0]
    assert term.by_variable_info.handling == "linear_functional"
    assert term.basis_train.shape == (len(data), term.coef_slice.stop)
    prediction = model.predict(data.iloc[:5], type="response")
    assert prediction.shape == (5,)
    assert np.all(np.isfinite(prediction))


def test_shape_coefficient_transform_composes_across_lss_predictors(tmp_path):
    rng = np.random.default_rng(117)
    x = np.linspace(-1.4, 1.8, 90)
    y = 0.7 + np.exp(0.3 * x) + rng.normal(scale=0.15, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})
    model = GAM(
        formula=['y ~ s(x, bs="mpi", k=7)', '~ s(x, bs="cr", k=6)'],
        family="gaulss",
        smoothing_params=[1.0, 1.0],
        optimize_smoothing=False,
    ).fit(data=data)

    compiled = model.gam_result_.compiled_model
    assert len(compiled.predictors) == 2
    assert compiled.coefficient_transform.is_identity is False
    result = model.gam_result_.fit_core_solution.fit_result
    reduced_to_full = np.asarray(compiled.coef_reduced_to_full_idx, dtype=int)
    np.testing.assert_allclose(result.beta, result.coef_full[reduced_to_full])
    assert result.coef_optimization is not None
    mask = np.asarray(result.positive_coefficient_mask, dtype=bool)
    assert np.all(result.coef_full[mask] > 0.0)
    assert np.all(np.isfinite(result.mu))
    assert result.eta.shape == (len(data), 2)

    expected = model.predict(data.iloc[:12], type="response")
    term_values, term_se = model.predict(data.iloc[:12], type="terms", return_se=True)
    assert term_values.shape == term_se.shape
    assert np.all(np.isfinite(term_values))
    assert np.all(np.isfinite(term_se))
    derivative = model.derivative(smooth_number=1, deriv=1)
    assert derivative.derivative.shape == (len(data),)
    assert np.all(np.isfinite(derivative.se))
    summary = model.summary()
    assert summary.np == result.coef_full.size
    anova = model.anova()
    expected_labels = [str(term.label) for term in compiled.compiled_terms]
    assert list(anova.smooth_table["label"]) == expected_labels
    path = tmp_path / "transformed_lss.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(data.iloc[:12], type="response"), expected
    )
    restored_transform = restored.gam_result_.compiled_model.coefficient_transform
    assert restored_transform.is_identity is False
    np.testing.assert_array_equal(restored_transform.positive_mask, mask)


def test_transformed_lss_automatic_smoothing_has_explicit_derivative_boundary():
    x = np.linspace(-1.0, 1.0, 50)
    data = pd.DataFrame({"y": 1.5 + np.exp(0.2 * x), "x": x})
    with pytest.raises(NotImplementedError, match="higher-order transformed Laplace"):
        GAM(
            formula=['y ~ s(x, bs="mpi", k=7)', "~ 1"],
            family="gaulss",
            smoothing_method="reml",
            optimize_smoothing=True,
        ).fit(data=data)
