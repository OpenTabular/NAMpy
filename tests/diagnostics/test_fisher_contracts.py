"""Independent sampling experiments for shared mean-information diagnostics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nampy import FisherDiagnostics
from nampy.diagnostics.fisher import aggregate_fisher, normalize_fisher_diagnostics


def summarize(
    means,
    variances=None,
    group_variances=None,
    *,
    mode="fixed",
    population_size=None,
    retain=True,
    estimator=None,
):
    config = FisherDiagnostics(
        estimator=estimator
        or ("exact" if variances is None else "score_outer_product"),
        n_draws=12,
        input_sampling=mode,
        population_size=population_size,
    )
    return aggregate_fisher(
        (
            (
                mean,
                None if variances is None else variances[i],
                None if group_variances is None else group_variances[i],
            )
            for i, mean in enumerate(means)
        ),
        config=config,
        parameter_names=("coefficient",),
        parameter_shapes=((means.shape[1],),),
        parameter_slices=((0, means.shape[1]),),
        groups=(("term", np.arange(means.shape[1])),),
        backend="test",
        estimator=config.estimator,
        n_inputs=len(means),
        return_diagonal=retain,
    )


def test_mean_information_is_invariant_to_row_replication_and_arrays_are_readonly():
    means = np.array([[1.0, 2.0], [3.0, 7.0]])
    one = summarize(means)
    repeated = summarize(np.repeat(means, 5, axis=0))
    assert_allclose(one.diagonal, [2, 4.5])
    assert_allclose(one.diagonal, repeated.diagonal)
    assert one.normalization == "mean" and one.n_draws is None
    assert one.conditional_standard_error is None and one.total_standard_error is None
    with pytest.raises(ValueError):
        one.diagonal[0] = 10


def test_group_only_retains_no_parameter_sized_result_arrays():
    result = summarize(np.ones((3, 100)), retain=False, mode="iid")
    assert result.groups[0].trace == 100
    for name in (
        "diagonal",
        "output_sampling_variance",
        "estimator_variance",
        "conditional_standard_error",
        "input_estimator_variance",
        "total_estimator_variance",
        "total_standard_error",
    ):
        assert getattr(result, name) is None


def test_hessian_samples_remain_signed():
    result = summarize(
        np.array([[-2.0, 1.0], [-4.0, 2.0]]),
        np.ones((2, 2)),
        np.ones((2, 1)),
        estimator="negative_hessian",
    )
    assert_allclose(result.diagonal, [-3, 1.5])
    assert result.groups[0].trace == -1.5
    with pytest.raises(ValueError, match="nonnegative"):
        summarize(np.array([[-1.0]]))


@pytest.mark.parametrize("mode", ["fixed", "iid", "without_replacement"])
def test_gaussian_score_sampling_errors_match_independent_experiments(mode):
    # Known-scale linear Gaussian scores have q_j = J_j**2 * Z**2.
    # Both coordinates share the same Z, so group covariance is essential.
    rng = np.random.default_rng(142)
    population = np.linspace(0.4, 2.4, 24)
    b, r, repetitions = 8, 12, 1200
    estimates, reported, group_reported, input_reported = [], [], [], []
    for _ in range(repetitions):
        if mode == "fixed":
            scale = population[:b]
        elif mode == "iid":
            scale = rng.choice(population, b, replace=True)
        else:
            scale = rng.choice(population, b, replace=False)
        coordinate_scale = np.column_stack([scale, 2 * scale])
        q = coordinate_scale[:, None, :] * rng.normal(size=(b, r, 1)) ** 2
        result = summarize(
            q.mean(axis=1),
            q.var(axis=1, ddof=1),
            q.sum(axis=2).var(axis=1, ddof=1)[:, None],
            mode=mode,
            population_size=len(population) if mode == "without_replacement" else None,
        )
        estimates.append(result.diagonal)
        variance = (
            result.estimator_variance
            if mode == "fixed"
            else result.total_estimator_variance
        )
        reported.append(variance)
        group = result.groups[0]
        group_reported.append(
            group.estimator_variance
            if mode == "fixed"
            else group.total_estimator_variance
        )
        if mode != "fixed":
            input_reported.append(result.input_estimator_variance)
    empirical = np.var(estimates, axis=0, ddof=1)
    assert_allclose(np.mean(reported, axis=0), empirical, rtol=0.13)
    assert_allclose(
        np.mean(group_reported), np.var(np.sum(estimates, axis=1), ddof=1), rtol=0.13
    )
    reference = population[:b] if mode == "fixed" else population
    oracle = 2 * np.mean(reference**2) / (b * r)
    if mode != "fixed":
        input_oracle = (
            np.var(population, ddof=1 if mode == "without_replacement" else 0) / b
        )
        if mode == "without_replacement":
            input_oracle *= 1 - b / len(population)
        oracle += input_oracle
        assert_allclose(np.mean(input_reported, axis=0)[0], input_oracle, atol=0.006)
    assert_allclose(np.mean(reported, axis=0)[0], oracle, rtol=0.07)


def test_finite_population_census_has_no_input_sampling_component():
    result = summarize(
        np.array([[1.0], [10.0]]),
        np.array([[2.0], [8.0]]),
        np.array([[2.0], [8.0]]),
        mode="without_replacement",
        population_size=2,
    )
    assert_allclose(result.input_estimator_variance, 0)
    assert_allclose(result.total_estimator_variance, result.estimator_variance)


@pytest.mark.parametrize(
    "options",
    [
        {"n_inputs": True},
        {"n_draws": 1},
        {"estimator": "empirical_fisher"},
        {"regression_scale": 0},
        {"random_state": -1},
        {"population_size": 4},
    ],
)
def test_invalid_configuration_fails_early(options):
    with pytest.raises((TypeError, ValueError)):
        FisherDiagnostics(**options)


def test_boolean_opt_in_and_incomplete_input_population_are_explicit():
    assert normalize_fisher_diagnostics(False) is None
    assert normalize_fisher_diagnostics(True) == FisherDiagnostics()
    with pytest.raises(TypeError):
        normalize_fisher_diagnostics({})
    with pytest.raises(ValueError, match="at least two"):
        summarize(np.ones((1, 2)), mode="iid")
    with pytest.raises(ValueError, match="population"):
        summarize(np.ones((3, 2)), mode="without_replacement", population_size=2)
