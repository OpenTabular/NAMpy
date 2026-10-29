"""Real neural training completion and shared cross-backend held-out scoring."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from nampy import GAMLSS
from nampy.distributions import DistributionCandidate, compare_distributions
from nampy.models import LinRegLSS
from nampy.neural.configuration import (
    GradientTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)


def _data():
    rng = np.random.default_rng(71)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    V = pd.DataFrame({"x": np.linspace(-0.9, 0.9, 18)})
    return (
        X,
        0.3 + 0.4 * X.x.to_numpy() + rng.normal(0, 0.3, len(X)),
        V,
        0.3 + 0.4 * V.x.to_numpy() + rng.normal(0, 0.3, len(V)),
    )


def _training(tmp_path, *, epochs=1):
    return GradientTrainingConfig(max_epochs=epochs, patience=1, random_state=7,
        checkpoint_path=str(tmp_path), trainer_kwargs={"logger": False,
        "enable_progress_bar": False, "enable_model_summary": False,
        "num_sanity_val_steps": 0})


def test_cross_backend_comparison_uses_shared_probabilities_and_distinct_status_semantics(
    tmp_path,
):
    X, y, V, z = _data()
    neural = LinRegLSS(objective=ObjectiveConfig(family='normal'), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
    neural.set_params(training=_training(tmp_path))
    assert neural.distribution_fit_status().state == "unknown"
    statistical = GAMLSS(
        family="normal",
        formula={"mu": "y ~ x", "sigma": "~1"},
        optimize_smoothing=False,
    )
    validation_weight = np.tile([0.0, 1.0, 2.0], 6)
    result = compare_distributions(
        {
            "statistical": statistical,
            "neural": DistributionCandidate(
                neural,
                predict_kwargs={"batch_size": 5},
            ),
        },
        X,
        y,
        V,
        z,
        sample_weight=np.tile([1.0, 2.0], 30),
        validation_weight=validation_weight,
    )
    assert len(result.ranking) == 2
    rows = {row.name: row for row in result.results}
    assert rows["statistical"].fit_status.state == "converged"
    assert rows["neural"].fit_status.state == "completed"
    assert "not a statistical convergence claim" in rows["neural"].fit_status.message
    assert rows["neural"].estimator.n_optimizer_steps_ > 0
    for row in result.results:
        assert row.n_validation == row.n_scored == 12
        expected = row.estimator.predict_distribution(V).log_prob(z)
        assert_allclose(
            row.mean_log_prob,
            np.average(expected, weights=validation_weight),
            atol=2e-14,
        )
        assert_allclose(
            row.mean_log_prob,
            row.estimator.score(V, z, sample_weight=validation_weight),
            atol=3e-6,
        )
    assert neural.model is None and not hasattr(statistical, "gam_")
    fitted = result.models["neural"]
    assert clone(fitted).distribution_fit_status().state == "unknown"
    path = tmp_path / "completed.pkl"
    fitted.save_model(path)
    assert (
        LinRegLSS.load_model(path).distribution_fit_status()
        == rows["neural"].fit_status
    )


def test_zero_optimizer_steps_do_not_become_a_completed_rankable_neural_fit(tmp_path):
    X, y, V, z = _data()
    result = compare_distributions({'untrained': DistributionCandidate(LinRegLSS(objective=ObjectiveConfig(family='normal'), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'), training=_training(tmp_path, epochs=0)))}, X, y, V, z)
    assert result.ranking == ()
    assert result.results[0].fit_status.state == "failed"
    assert "without an optimizer step" in result.results[0].fit_status.message


def test_failed_neural_refit_preserves_previous_completion_evidence(tmp_path):
    X, y, V, z = _data()
    model = LinRegLSS(objective=ObjectiveConfig(family='normal'), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'), training=_training(tmp_path)).fit(X, y)
    assert model.distribution_fit_status().state == "completed"
    with pytest.raises(ValueError, match="negative"):
        model.fit(X, y, sample_weight=-np.ones(len(X)), )
    assert model.distribution_fit_status().eligible
