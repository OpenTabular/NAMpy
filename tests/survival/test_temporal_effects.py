"""Temporal interaction identification, mgcv endpoints and exact integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import SurvivalGAM
from nampy.survival import SurvivalTarget, SurvivalTrajectory, predict_trajectory


@pytest.mark.parametrize("route", ["fixed", "reml"])
def test_temporal_tensor_effects_match_mgcv_with_exposure_entry_weights_and_offsets(
    route, tmp_path
):
    root = Path(__file__).parents[1] / "reference_fixtures"
    source = pd.read_csv(root / "survival_temporal_input.csv")
    expected = pd.read_csv(root / "survival_temporal.csv")
    expected = expected[expected.route == route]
    X, y = source[["x"]], SurvivalTarget(source.time, source.event, source.entry)
    model = SurvivalGAM(
        cuts=np.arange(6),
        baseline="smooth",
        k=4,
        formula="~x",
        temporal_formula="~ti(survival_time,x,k=c(4,4))",
        optimize_smoothing=route == "reml",
        smoothing_params=[0.4, 0.3, 0.2] if route == "fixed" else None,
        irls_tol=1e-10,
    ).fit(X, y, sample_weight=source.weight.to_numpy(), offset=source.offset.to_numpy())
    query = pd.DataFrame({"x": [-0.5, 0, 0.8]})
    eta = model.predict_link(query, offset=np.zeros(3))
    assert_allclose(eta.ravel(), expected.link, rtol=3e-6, atol=3e-6)
    assert_allclose(
        model.native_poisson_log_likelihood_,
        expected.loglik.iloc[0],
        rtol=3e-6,
        atol=3e-6,
    )
    if route == "reml":
        assert_allclose(
            model.gam_.smoothing_params,
            expected[["sp1", "sp2", "sp3"]].iloc[0],
            rtol=2e-4,
            atol=2e-4,
        )
    component = model.predict_components(query, offset=np.zeros(3)).prediction
    component.validate_additive_reconstruction()
    term = next(
        value for key, value in component.term_metadata.items() if key.startswith("ti(")
    )
    assert set(term.features) == {"survival_time", "x"}
    prediction = model.predict_distribution(query, offset=np.zeros(3))
    assert_allclose(
        prediction.survival([1, 2, 3, 4, 5]),
        np.exp(-np.cumsum(np.exp(eta), axis=1)),
        atol=2e-15,
    )
    restored = SurvivalGAM.load_model(model.save_model(tmp_path / "temporal.nampy"))
    assert_array_equal(restored.predict_link(query, offset=np.zeros(3)), eta)
    path = SurvivalTrajectory([0, 1.5], [1.5, 5], ["new", "new"])
    trajectory = predict_trajectory(model, query.iloc[[0, 2]], path, offset=np.zeros(2))
    assert_allclose(
        trajectory.hazard_rates,
        np.exp(eta[[0, 0, 2, 2, 2, 2], [0, 1, 1, 2, 3, 4]])[None, :],
    )


def test_temporal_effects_reject_ambiguous_baseline_and_unidentified_specifications():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 30)})
    y = SurvivalTarget(np.linspace(0.1, 4.9, 30), np.resize([1, 0, 1], 30))
    with pytest.raises(NotImplementedError, match="smooth baseline"):
        SurvivalGAM(cuts=np.arange(6), temporal_formula="~ti(survival_time,x)").fit(
            X, y
        )
    with pytest.raises(NotImplementedError, match="ti\\("):
        SurvivalGAM(
            cuts=np.arange(6),
            baseline="smooth",
            k=4,
            temporal_formula="~s(survival_time)",
        ).fit(X, y)
