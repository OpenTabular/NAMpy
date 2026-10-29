"""Shared survival observation and curve semantics."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy.distributions import DistributionPrediction, LogNormalLaw, WeibullLaw
from nampy.survival import (
    PiecewiseExponentialDesign,
    SurvivalPrediction,
    SurvivalTarget,
    piecewise_exponential_data,
)


def test_target_copies_and_rejects_ambiguous_observations():
    times = np.array([1.0, 2.0, 3.0])
    target = SurvivalTarget(times, [1, 0, 1], [0, 1, 2], ["a", "b", "c"])
    times[:] = 50
    assert_array_equal(target.time, [1, 2, 3])
    for value in (target.time, target.event, target.entry, target.subject_id):
        assert not value.flags.writeable
    assert_array_equal(target.take([2, 0]).subject_id, ["c", "a"])
    with pytest.raises(ValueError, match="0/1"):
        SurvivalTarget([1, 2], [1, 2])
    with pytest.raises(ValueError, match="entry"):
        SurvivalTarget([1, 2], [1, 0], [1, 0])
    with pytest.raises(ValueError, match="subject_id"):
        SurvivalTarget([1, 2], [1, 0], subject_id=["a", "a"])


def test_design_matches_existing_exposure_expansion_with_entry_and_cut_events():
    target = SurvivalTarget([1, 2.5, 3], [1, 0, 1], [0, 0.5, 1.5])
    design = PiecewiseExponentialDesign(target, [0, 1, 2, 3])
    frame = pd.DataFrame(
        {"time": target.time, "event": target.event, "entry": target.entry}
    )
    expanded = piecewise_exponential_data(
        frame, time="time", event="event", entry="entry", cuts=design.cuts
    )
    rows, intervals = np.nonzero(design.exposure)
    assert_array_equal(expanded.ped_id, rows)
    assert_array_equal(expanded.interval, intervals)
    assert_allclose(expanded.exposure, design.exposure[rows, intervals])
    assert_allclose(expanded.ped_event, design.events[rows, intervals])
    assert_allclose(design.exposure.sum(axis=1), target.time - target.entry)
    assert_array_equal(design.events.sum(axis=1), target.event)


def test_piecewise_likelihood_and_landmark_curves():
    target = SurvivalTarget([1, 2.5], [1, 0], [0, 0.5])
    design = PiecewiseExponentialDesign(target, [0, 1, 3])
    rates = np.array([[0.2, 0.4], [0.3, 0.5]])
    assert_allclose(
        design.log_likelihood(np.log(rates)),
        [np.log(0.2) - 0.2, -0.3 * 0.5 - 0.5 * 1.5],
    )
    prediction = SurvivalPrediction(rates, design.cuts)
    times = [0, 0.5, 1, 2, 3]
    assert_allclose(prediction.cumulative_hazard(times)[0], [0, 0.1, 0.2, 0.6, 1])
    assert_allclose(prediction.hazard([0, 1, 2, 3])[0], [0.2, 0.2, 0.4, 0.4])
    conditional = prediction.survival(times, conditional_on=[0.5, 1])
    assert np.isnan(conditional[0, 0])
    assert_allclose(conditional[0, 1:], np.exp(-np.array([0, 0.1, 0.5, 0.9])))
    assert_allclose(conditional[1, 2], 1)
    assert_allclose(prediction.rmst(1), -np.expm1(-rates[:, 0]) / rates[:, 0])
    assert_allclose(prediction.quantile(0), 0)
    assert_allclose(prediction.quantile(1 - np.exp(-0.6))[0], [2])
    with pytest.raises(NotImplementedError, match="tail"):
        prediction.survival([4])
    with pytest.raises(ValueError, match="increasing"):
        prediction.survival([1, 1])


@pytest.mark.parametrize(
    "law,parameters",
    [
        (LogNormalLaw(), {"loc": np.array([0.0, 1.0]), "scale": np.array([0.5, 1.0])}),
        (WeibullLaw(), {"scale": np.array([1.0, 2.0]), "shape": np.array([0.5, 1.0])}),
    ],
)
def test_latent_law_view_preserves_scalar_probability_contract(law, parameters):
    distribution = DistributionPrediction(law, parameters)
    prediction = SurvivalPrediction(distribution=distribution)
    values = prediction.survival([0, 1, 2])
    assert values.shape == (2, 3)
    assert_allclose(values[:, 0], 1)
    assert_allclose(values[:, 1], distribution.survival(1))
    assert_allclose(prediction.quantile([0.2, 0.5]), distribution.quantile([0.2, 0.5]))
    assert prediction.hazard([1, 2]).shape == (2, 2)


def test_statistical_and_shared_survival_imports_do_not_load_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import nampy.survival; from nampy.models import SurvivalGAM; assert 'torch' not in sys.modules; assert 'lightning' not in sys.modules",
        ],
        check=True,
    )
