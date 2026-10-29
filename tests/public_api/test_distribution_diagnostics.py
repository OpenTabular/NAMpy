"""Shared diagnostics depend on probability operations, not model identities."""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from nampy.diagnostics import (
    centile_calibration,
    prepare_centiles,
    prepare_worm_plot,
    quantile_residuals,
)
from nampy.distributions import DistributionPrediction, NegativeBinomialLaw, NormalLaw
from nampy.distributions.contracts import DistributionSpec
from nampy.plotting import plot_centiles, plot_worm, render_centiles, render_worm_plot


def _normal(n):
    return DistributionPrediction(
        NormalLaw(), {"loc": np.zeros(n), "scale": np.ones(n)}
    )


def test_continuous_residuals_recover_standardization_without_rng_consumption():
    y = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    distribution = DistributionPrediction(
        NormalLaw(), {"loc": np.arange(5.0), "scale": np.arange(1.0, 6.0)}
    )
    rng = np.random.default_rng(4)
    result = distribution.quantile_residuals(
        np.arange(5.0) + np.arange(1.0, 6.0) * y, random_state=rng
    )
    assert_allclose(result.residuals, y, atol=1e-13)
    assert not result.randomized.any()
    assert not result.clipped.any()
    assert rng.random() == np.random.default_rng(4).random()


def test_discrete_randomization_uses_cdf_jump_not_plain_cdf():
    distribution = DistributionPrediction(
        NegativeBinomialLaw(), {"mean": [1.0, 1.0], "dispersion": [1.0, 1.0]}
    )
    result = quantile_residuals(distribution, [0.0, 1.0], uniforms=[0.2, 0.8])
    assert_allclose(result.cdf_lower, [0.0, 0.5])
    assert_allclose(result.cdf_upper, [0.5, 0.75])
    assert_allclose(result.pit, [0.1, 0.7])
    assert_allclose(result.residuals, norm.ppf([0.1, 0.7]))
    assert result.randomized.all()
    assert_array_equal(
        distribution.quantile_residuals([0.0, 1.0], random_state=11).pit,
        distribution.quantile_residuals([0.0, 1.0], random_state=11).pit,
    )


@pytest.mark.parametrize("y", [[0.0, 1.2], [0.0, -1.0]])
def test_discrete_observation_support_is_distinct_from_cdf_threshold_queries(y):
    distribution = DistributionPrediction(
        NegativeBinomialLaw(), {"mean": [1.0, 1.0], "dispersion": [1.0, 1.0]}
    )
    assert np.isfinite(distribution.cdf(y)).all()
    with pytest.raises(ValueError, match="response support"):
        distribution.quantile_residuals(y)
    with pytest.raises(ValueError, match="response support"):
        centile_calibration(distribution, y)


@dataclass(frozen=True)
class _MixedLaw:
    spec = DistributionSpec(
        "external_mixed",
        ("p",),
        frozenset({"cdf", "cdf_left"}),
        "atom_plus_shifted_exponential_v1",
        "mixed",
    )

    def validate_parameters(self, parameters, *, context):
        assert np.all((parameters["p"] > 0) & (parameters["p"] < 1))

    def evaluate(self, operation, parameters, *, values=None, **kwargs):
        result = np.where(
            values < 2.5,
            0.0,
            parameters["p"] + (1 - parameters["p"]) * (-np.expm1(-(values - 2.5))),
        )
        return (
            np.where(values == 2.5, 0.0, result) if operation == "cdf_left" else result
        )


def test_external_mixed_distribution_randomizes_only_its_noninteger_atom():
    distribution = DistributionPrediction(_MixedLaw(), {"p": [0.3, 0.3, 0.3]})
    result = distribution.quantile_residuals(
        [2.5, 3.5, np.nan], uniforms=[0.4, 0.9, 0.8]
    )
    assert_allclose(result.pit, [0.12, 0.3 + 0.7 * (1 - np.exp(-1)), np.nan])
    assert_array_equal(result.randomized, [True, False, False])


def test_residual_endpoint_policy_reports_clipping_and_missing_rows():
    distribution = _normal(3)
    result = distribution.quantile_residuals([-40.0, 40.0, np.nan])
    assert_array_equal(result.residuals, [-np.inf, np.inf, np.nan])
    clipped = distribution.quantile_residuals([-40.0, 40.0, np.nan], clip=1e-6)
    assert_array_equal(clipped.clipped, [True, True, False])
    assert_allclose(clipped.residuals[:2], norm.ppf([1e-6, 1 - 1e-6]))
    assert_array_equal(clipped.cdf_upper, result.cdf_upper)
    assert not clipped.residuals.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uniforms": [0.2]},
        {"uniforms": [0.2, 1.2]},
        {"uniforms": [0.2, np.nan]},
        {"random_state": 1, "uniforms": [0.2, 0.3]},
        {"clip": 0},
        {"clip": 0.5},
        {"clip": np.nan},
        {"clip": 1e-20},
    ],
)
def test_bad_residual_requests_fail(kwargs):
    with pytest.raises(ValueError):
        _normal(2).quantile_residuals([0.0, 1.0], **kwargs)


def test_worm_plot_recovers_cubic_and_preserves_original_row_indices():
    n = 40
    theoretical = norm.ppf((np.arange(n) + 0.5) / n)
    coefficients = np.array([0.1, 0.05, 0.02, 0.01])
    deviation = np.polynomial.polynomial.polyval(theoretical, coefficients)
    permutation = np.random.default_rng(4).permutation(n)
    residuals = (theoretical + deviation)[permutation]
    data = prepare_worm_plot(residuals)
    panel = data.panels[0]
    assert_allclose(panel.theoretical, theoretical)
    assert_allclose(panel.deviation, deviation, atol=1e-14)
    assert_allclose(panel.cubic_coefficients, coefficients, atol=1e-14)
    assert_array_equal(residuals[panel.row_indices], theoretical + deviation)
    assert np.all(panel.envelope_upper > 0)
    assert_allclose(panel.envelope_lower, -panel.envelope_upper)


def test_grouped_worm_counts_nonfinite_and_small_groups_explicitly():
    data = prepare_worm_plot(
        [1.0, np.nan, np.inf, 0.0, 2.0], groups=["a", "a", "a", "b", None]
    )
    assert data.n_ungrouped == 1
    a, b = data.panels
    assert a.n_total == 3 and a.n_missing == 1 and a.n_infinite == 1
    assert a.cubic_coefficients is None and b.cubic_coefficients is None
    assert_array_equal(a.row_indices, [0])
    assert_array_equal(b.row_indices, [3])
    assert_array_equal(a.theoretical, [0.0])


def test_numeric_bins_use_declared_closure_keep_empty_bins_and_collapse_ties():
    data = prepare_worm_plot(
        [0.0, 1.0, 2.0, 3.0],
        x=[0.0, 0.5, 1.0, np.nan],
        breaks=[0.0, 0.2, 0.4, 0.6, 1.0],
    )
    assert [p.n_total for p in data.panels] == [1, 0, 1, 1]
    assert data.n_ungrouped == 1
    assert_array_equal(data.panels[-1].row_indices, [2])
    data = prepare_worm_plot([0.0, 1.0], x=[2.0, 2.0], n_bins=4)
    assert len(data.panels) == 1 and data.panels[0].n_total == 2
    assert len(prepare_worm_plot([], groups=[]).panels) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": ["a"], "x": [1.0]},
        {"groups": ["a", "b"]},
        {"breaks": [0, 1]},
        {"x": [0.0], "n_bins": 0},
        {"x": [0.0], "breaks": [1.0, 2.0]},
        {"x": [0.0], "breaks": [0.0, 0.0]},
        {"reference_level": 1.0},
    ],
)
def test_worm_invalid_grouping_is_rejected(kwargs):
    with pytest.raises(ValueError):
        prepare_worm_plot([0.0], **kwargs)


def test_centiles_keep_conditional_order_missing_gaps_and_row_identity():
    distribution = DistributionPrediction(
        NormalLaw(), {"loc": [2.0, 0.0, np.nan, 8.0], "scale": [1.0, 1.0, np.nan, 1.0]}
    )
    curves = prepare_centiles(distribution, [2.0, 0.0, 1.0, np.nan])
    assert_array_equal(curves.x, [0.0, 1.0, 2.0])
    assert_array_equal(curves.row_indices, [1, 2, 0])
    assert curves.n_excluded_x == 1
    assert_allclose(curves.quantiles[:, 1], [0.0, np.nan, 2.0])
    assert np.all(np.diff(curves.quantiles[[0, 2]], axis=1) > 0)


def test_discrete_centile_calibration_compares_attainable_probability():
    distribution = DistributionPrediction(
        NegativeBinomialLaw(), {"mean": [1.0] * 4, "dispersion": [1.0] * 4}
    )
    result = centile_calibration(
        distribution, [0.0, 1.0, 0.0, np.nan], probs=[0.05, 0.5, 0.9]
    )
    assert_allclose(result.observed, [2 / 3, 2 / 3, 1.0])
    assert_allclose(result.model_probability, [0.5, 0.5, 0.9375])
    assert_array_equal(result.n_used, [3, 3, 3])
    result = centile_calibration(_normal(2), [np.nan, np.nan])
    assert np.isnan(result.observed).all()
    assert_array_equal(result.n_used, [0, 0, 0])


@pytest.mark.parametrize("probs", [[], [0.0], [1.0], [0.7, 0.3], [0.5, 0.5], [np.nan]])
def test_invalid_centile_levels_are_rejected(probs):
    with pytest.raises(ValueError):
        prepare_centiles(_normal(2), [0.0, 1.0], probs=probs)


def test_shared_renderers_use_prepared_data_and_label_reference_envelopes():
    residuals = _normal(8).quantile_residuals(np.linspace(-1.0, 1.0, 8))
    worm = prepare_worm_plot(residuals, groups=["a"] * 4 + ["b"] * 4)
    centiles = prepare_centiles(_normal(8), np.arange(8.0))
    figures = [
        render_worm_plot(worm),
        plot_worm(residuals),
        render_centiles(centiles),
        plot_centiles(_normal(8), np.arange(8.0)),
    ]
    try:
        assert len(figures[0].axes) == 2
        assert (
            "pointwise normal reference"
            in figures[0].axes[0].get_legend_handles_labels()[1][0]
        )
        assert len(figures[2].axes[0].lines) == 3
        assert not figures[2].axes[0].collections  # no confidence-band shading
        for fig in figures:
            fig.canvas.draw()
    finally:
        for fig in figures:
            plt.close(fig)


def test_discrete_centile_rendering_and_empty_worm_status_are_explicit():
    distribution = DistributionPrediction(
        NegativeBinomialLaw(), {"mean": [1.0, 3.0], "dispersion": [1.0, 1.0]}
    )
    curves = prepare_centiles(distribution, [0.0, 1.0])
    assert curves.response_kind == "discrete"
    count_figure = render_centiles(curves)
    empty_figure = render_worm_plot(prepare_worm_plot([], groups=[]))
    try:
        assert all(
            line.get_drawstyle() == "steps-post" for line in count_figure.axes[0].lines
        )
        assert empty_figure.axes[0].texts[0].get_text() == "No grouped residuals"
        count_figure.canvas.draw()
        empty_figure.canvas.draw()
    finally:
        plt.close(count_figure)
        plt.close(empty_figure)
