"""Complete-history resampling, subject evaluation and trajectory uncertainty."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import (
    GradientTrainingConfig,
    LinRegSurvival,
    NAMSurvival,
    SurvivalGAM,
    SurvivalObjectiveConfig,
)
from nampy.survival import (
    CensoringDistribution,
    LongitudinalSurvivalTarget,
    SurvivalPrediction,
    SurvivalTrajectory,
    bootstrap_survival,
    brier_score,
    concordance_index,
    predict_trajectory,
    survival_curve,
    survival_residuals,
)


def data():
    frame = pd.read_csv(
        Path(__file__).parents[1] / "reference_fixtures/survival_longitudinal_input.csv"
    )
    return (
        frame[["x"]],
        LongitudinalSurvivalTarget(frame.start, frame.stop, frame.subject, frame.event),
        frame.weight.to_numpy(),
        frame.offset.to_numpy(),
    )


def test_whole_subject_split_resampling_alignment_and_fresh_occurrences():
    X, y, w, offset = data()
    train, val = y.split_subjects(validation_fraction=0.3, seed=19)
    assert not set(train.subject_ids) & set(val.subject_ids)
    assert train.n_subjects + val.n_subjects == y.n_subjects
    assert_array_equal(
        y.split_subjects(validation_fraction=0.3, seed=19)[0].subject_id,
        train.subject_id,
    )
    assert_array_equal(y.stop[y.record_indices(train.subject_ids)], train.stop)
    sampled, rows = y.resample_subjects([2, 2, 0])
    assert sampled.n_subjects == 3 and len(sampled) == 6
    assert len(set(sampled.subject_ids) & set(y.subject_ids)) == 0
    assert_array_equal(rows, [4, 5, 4, 5, 0, 1])
    assert_array_equal(sampled.stop, y.stop[rows])
    assert_array_equal(w[rows], [w[4], w[5], w[4], w[5], w[0], w[1]])
    assert_array_equal(
        sampled.subject_outcomes().time, [y.stop[5], y.stop[5], y.stop[1]]
    )
    with pytest.raises(ValueError, match="integer subject"):
        y.resample_subjects([0.5])


def test_subject_level_diagnostics_metrics_and_entry_restrictions():
    X, y, w, offset = data()
    model = SurvivalGAM(cuts=[0, 1, 3], formula="~x", optimize_smoothing=False).fit(
        X, y, sample_weight=w, offset=offset
    )
    record_prediction = model.predict_distribution(X, offset=offset)
    residuals = survival_residuals(record_prediction, y)
    record_increments = record_prediction.cumulative_hazard_at(
        y.stop, conditional_on=y.start
    )
    assert_allclose(
        residuals.cox_snell, np.add.reduceat(record_increments, np.arange(0, 16, 2))
    )
    assert_array_equal(residuals.event, y.event[1::2])
    assert_array_equal(residuals.subject_id, y.subject_ids)
    outcomes = y.subject_outcomes()
    for field in (
        "time",
        "at_risk",
        "events",
        "censored",
        "survival",
        "cumulative_hazard",
    ):
        assert_array_equal(
            getattr(survival_curve(y, sample_weight=w), field),
            getattr(survival_curve(outcomes, sample_weight=w[::2]), field),
        )
    censoring = CensoringDistribution.fit(y, sample_weight=w)
    assert_array_equal(
        censoring.probability,
        CensoringDistribution.fit(outcomes, sample_weight=w[::2]).probability,
    )
    prediction = SurvivalPrediction(np.full((8, 2), 0.3), [0, 1, 3])
    assert_array_equal(
        brier_score(
            prediction, y, [0.5, 1, 2], censoring=censoring, sample_weight=w
        ).score,
        brier_score(
            prediction, outcomes, [0.5, 1, 2], censoring=censoring, sample_weight=w[::2]
        ).score,
    )
    assert concordance_index(
        y, np.arange(8), higher_risk=True, sample_weight=w
    ) == concordance_index(
        outcomes, np.arange(8), higher_risk=True, sample_weight=w[::2]
    )
    entered = LongitudinalSurvivalTarget(
        y.start + 0.01, y.stop + 0.01, y.subject_id, y.event
    )
    with pytest.raises(NotImplementedError, match="Delayed entry"):
        CensoringDistribution.fit(entered)


@pytest.mark.parametrize("backend", ["statistical", "linreg", "nam"])
def test_bootstrap_complete_trajectories_preserves_prediction_paths_and_source(
    backend, tmp_path
):
    X, y, w, offset = data()
    kwargs = {}
    if backend == "statistical":
        model = SurvivalGAM(cuts=[0, 1, 3], formula="~x", optimize_smoothing=False)
        selected = np.array([[0, 0, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]])
    else:
        train, val = y.split_subjects(validation_fraction=0.3, seed=2)
        rows, validation_rows = (
            y.record_indices(train.subject_ids),
            y.record_indices(val.subject_ids),
        )
        kwargs = {
            "X_val": X.iloc[validation_rows],
            "y_val": val,
            "sample_weight_val": w[validation_rows],
            "offset_val": offset[validation_rows],
        }
        X, y, w, offset = X.iloc[rows], train, w[rows], offset[rows]
        options = {}
        if backend == "nam":
            from nampy.neural.configs import DefaultNAMConfig

            options["architecture_config"] = DefaultNAMConfig(
                layer_sizes=[4], dropout=0
            )
        cls = LinRegSurvival if backend == "linreg" else NAMSurvival
        model = cls(
            **options,
            objective=SurvivalObjectiveConfig(cuts=[0, 1, 3]),
            training=GradientTrainingConfig(
                max_epochs=1,
                batch_size=64,
                random_state=3,
                checkpoint_path=str(tmp_path),
                trainer_kwargs={
                    "logger": False,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
            ),
        )
        selected = np.stack([np.arange(y.n_subjects), np.arange(y.n_subjects)[::-1]])
    model.fit(X, y, sample_weight=w, offset=offset, **kwargs)
    path = SurvivalTrajectory(
        [0, 0.7, 0, 1.4], [0.7, 3, 1.4, 3], ["newA", "newA", "newB", "newB"]
    )
    query = pd.DataFrame({"x": [-0.5, 0.8, -0.2, 0.4]})
    before = predict_trajectory(model, query, path, offset=np.zeros(4))
    result = bootstrap_survival(
        model,
        X,
        y,
        query,
        resample_indices=selected,
        sample_weight=w,
        offset=offset,
        prediction_offset=np.zeros(4),
        prediction_trajectory=path,
        effects={"times": [0, 0.5, 1, 2, 3]},
        **kwargs,
    )
    assert not result.failures
    assert len(result.effect_samples) == 2
    assert result.effect_interval().intervals
    assert result.effect_interval(interval="simultaneous").intervals
    assert_array_equal(result.source_subject_id, y.subject_ids)
    assert_array_equal(result.resample_indices, selected)
    assert_array_equal(result.point_prediction.hazard_rates, before.hazard_rates)
    assert_array_equal(
        predict_trajectory(model, query, path, offset=np.zeros(4)).hazard_rates,
        before.hazard_rates,
    )
    for method in (result.percentile_interval, result.simultaneous_interval):
        interval = method(grid=[0, 0.5, 1, 2, 3])
        assert interval.estimate.shape == (2, 5)
        assert np.isfinite(interval.lower).all()
    if backend == "statistical":
        for indices, prediction in zip(selected, result.predictions, strict=True):
            target, rows = y.resample_subjects(indices)
            reference = SurvivalGAM(
                cuts=[0, 1, 3], formula="~x", optimize_smoothing=False
            ).fit(X.iloc[rows], target, sample_weight=w[rows], offset=offset[rows])
            assert_array_equal(
                prediction.hazard_rates,
                predict_trajectory(
                    reference, query, path, offset=np.zeros(4)
                ).hazard_rates,
            )
