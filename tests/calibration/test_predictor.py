"""Backend-owned fitted snapshots, data roles, offsets and trusted persistence."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nampy import GAMLSS, GAMRegressor, QuantileGAM
from nampy.calibration import (
    ConformalPredictor,
    calibrate_conformal,
    calibrate_regressor,
)


def data():
    rng = np.random.default_rng(727)
    X = pd.DataFrame({"x": np.linspace(-2, 2, 90)})
    return X, .5+X.x.to_numpy()+rng.normal(size=90)*(.5+.2*np.abs(X.x.to_numpy()))


def test_statistical_snapshot_offsets_refit_isolation_and_persistence(tmp_path):
    X, y = data()
    offset = np.linspace(.1, .3, 50)
    source = GAMRegressor(formula="~x", optimize_smoothing=False).fit(X.iloc[:50], y[:50], offset=offset)
    prediction = source.predict(X.iloc[50:75], offset=np.full(25, .4))
    before = source.predict(X.iloc[:4], offset=np.zeros(4))
    calibrated = calibrate_regressor(source, X.iloc[50:75], y[50:75], level=.8,
                                    predict_kwargs={"offset": np.full(25, .4)})
    expected = calibrate_conformal(y[50:75], prediction, level=.8)
    assert_array_equal(calibrated.calibration.scores, expected.scores)
    interval = calibrated.predict_interval(X.iloc[:4], offset=np.zeros(4))
    assert_allclose(interval.lower, before-expected.correction)
    assert_array_equal(source.predict(X.iloc[:4], offset=np.zeros(4)), before)
    source.fit(X.iloc[:50], y[:50]+100)
    assert_array_equal(calibrated.predict_interval(X.iloc[:4], offset=np.zeros(4)).lower, interval.lower)
    restored = ConformalPredictor.load_model(calibrated.save_model(tmp_path/"calibrated.nampy"))
    assert_array_equal(restored.predict_interval(X.iloc[:4], offset=np.zeros(4)).lower, interval.lower)
    assert not restored.calibration.scores.flags.writeable
    with pytest.raises(FrozenInstanceError):
        calibrated.calibration = expected
    with pytest.raises(ValueError, match="supported"):
        old = tmp_path/"old.nampy"
        old.write_bytes(b"NAMPY-ESTIMATOR 1\n")
        ConformalPredictor.load_model(old)


@pytest.mark.parametrize("method", ["absolute", "quantile"])
def test_statistical_distribution_adapter_reuses_shared_law(method):
    X, y = data()
    source = GAMLSS(family="normal", formula={"mu": "~x", "sigma": "~1"}, optimize_smoothing=False).fit(X.iloc[:50], y[:50])
    calibrated = calibrate_regressor(source, X.iloc[50:75], y[50:75], method=method,
                                    prediction="distribution", level=.8)
    explicit = calibrate_conformal(y[50:75], source.predict_distribution(X.iloc[50:75]), method=method, level=.8)
    assert_array_equal(calibrated.calibration.scores, explicit.scores)
    assert_array_equal(calibrated.predict_interval(X.iloc[75:]).lower,
                       explicit.interval(source.predict_distribution(X.iloc[75:])).lower)


def test_separate_quantile_gams_and_failed_calibration_leave_sources_unchanged():
    X, y = data()
    models = [QuantileGAM(formula="~x", level=q, smoothing_params=0).fit(X.iloc[:50], y[:50]) for q in (.1, .9)]
    pair = np.column_stack([model.predict(X.iloc[50:75]) for model in models])
    expected = calibrate_conformal(y[50:75], pair, method="quantile", level=.8)
    fitted = calibrate_regressor(models[0], X.iloc[50:75], y[50:75], method="quantile",
                                level=.8, upper_estimator=models[1])
    assert_array_equal(fitted.calibration.scores, expected.scores)
    before = models[0].predict(X.iloc[75:])
    with pytest.raises(ValueError, match="cross"):
        calibrate_regressor(models[1], X.iloc[50:75], y[50:75], method="quantile", upper_estimator=models[0])
    assert_array_equal(models[0].predict(X.iloc[75:]), before)


@pytest.mark.parametrize("kind", ["regression", "distribution"])
def test_neural_snapshot_prediction_and_execution_exclusion(kind, tmp_path):
    from nampy import (
        GradientTrainingConfig,
        LinRegLSS,
        LinRegRegressor,
        ObjectiveConfig,
    )
    X, y = data()
    training = GradientTrainingConfig(max_epochs=1, batch_size=128, random_state=18,
        checkpoint_path=str(tmp_path), trainer_kwargs={"logger": False, "enable_progress_bar": False,
                                                     "enable_model_summary": False})
    cls = LinRegRegressor if kind == "regression" else LinRegLSS
    kwargs = {} if kind == "regression" else {"objective": ObjectiveConfig(family="normal")}
    model = cls(training=training, **kwargs).fit(X.iloc[:40], y[:40], X_val=X.iloc[40:50], y_val=y[40:50])
    route = "response" if kind == "regression" else "distribution"
    method = "absolute" if kind == "regression" else "quantile"
    prediction = model.predict(X.iloc[50:75]) if kind == "regression" else model.predict_distribution(X.iloc[50:75])
    expected = calibrate_conformal(y[50:75], prediction, level=.8, method=method)
    calibrated = calibrate_regressor(model, X.iloc[50:75], y[50:75], prediction=route, method=method, level=.8)
    assert_array_equal(calibrated.calibration.scores, expected.scores)
    snapshot = calibrated._predictors[0]
    assert snapshot.model is not model.model
    assert snapshot.fitted_preprocessing_ is not model.fitted_preprocessing_
    assert not hasattr(snapshot, "trainer") and not hasattr(snapshot, "data_module")
    interval = calibrated.predict_interval(X.iloc[75:])
    model.fit(X.iloc[:40], y[:40]+10, X_val=X.iloc[40:50], y_val=y[40:50]+10)
    assert_array_equal(calibrated.predict_interval(X.iloc[75:]).lower, interval.lower)
    restored = ConformalPredictor.load_model(calibrated.save_model(tmp_path/f"{kind}.nampy"))
    assert_array_equal(restored.predict_interval(X.iloc[75:]).lower, interval.lower)


def test_survival_classification_unfitted_and_ambiguous_routes_are_rejected():
    from nampy import GAMClassifier, SurvivalGAM
    X, y = data()
    for model in (GAMClassifier(), SurvivalGAM(cuts=[0, 1, 3])):
        with pytest.raises(NotImplementedError, match="unsupported"):
            calibrate_regressor(model, X, y)
    with pytest.raises(ValueError):
        calibrate_regressor(GAMRegressor(), X, y)
    with pytest.raises(ValueError, match="explicit lower/upper"):
        calibrate_regressor(QuantileGAM(), X, y, method="quantile")


def test_neural_quantile_loss_cqr_needs_no_probability_law(tmp_path):
    from nampy import GradientTrainingConfig, LinRegLSS, ObjectiveConfig
    X, y = data()
    model = LinRegLSS(objective=ObjectiveConfig(family="quantile", distributional_kwargs={"quantiles": [.1, .5, .9]}),
        training=GradientTrainingConfig(max_epochs=1, random_state=22, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})).fit(X.iloc[:40], y[:40])
    parameters = model.predict(X.iloc[50:75])
    assert_array_equal(model.predict_quantiles(X.iloc[50:75], [.9, .1, .9]), parameters[:, [2, 0, 2]])
    assert model.predict_quantiles(X.iloc[50:75], []).shape == (25, 0)
    with pytest.raises(NotImplementedError, match="interpolation"):
        model.predict_quantiles(X.iloc[50:75], [.2])
    with pytest.raises(ValueError, match="probabilities"):
        model.predict_quantiles(X.iloc[50:75], [np.nan])
    calibrated = calibrate_regressor(model, X.iloc[50:75], y[50:75], method="quantile", prediction="quantiles", level=.8)
    expected = calibrate_conformal(y[50:75], parameters[:, [0, 2]], method="quantile", level=.8)
    assert_array_equal(calibrated.calibration.scores, expected.scores)
    assert_array_equal(calibrated.predict_interval(X.iloc[75:]).lower,
                       expected.interval(model.predict_quantiles(X.iloc[75:], [.1, .9])).lower)


def test_distribution_quantile_operation_matches_shared_law_and_point_route(tmp_path):
    from nampy import GradientTrainingConfig, LinRegLSS, ObjectiveConfig
    X, y = data()
    model = LinRegLSS(objective=ObjectiveConfig(family="normal"),
        training=GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path),
            trainer_kwargs={"logger": False, "enable_progress_bar": False, "enable_model_summary": False})).fit(X.iloc[:40], y[:40])
    rows = X.iloc[50:75]
    assert_array_equal(model.predict_quantiles(rows, [.1, .9]), model.predict_distribution(rows).quantile([.1, .9]))
    calibrated = calibrate_regressor(model, rows, y[50:75], prediction="point")
    explicit = calibrate_conformal(y[50:75], model.predict_point(rows))
    assert_array_equal(calibrated.calibration.scores, explicit.scores)
