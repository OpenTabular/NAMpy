"""Joint objective optimization, weighted identifiability and retained projection."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from nampy import (
    GradientTrainingConfig,
    ObjectiveConfig,
    SemiStructuredLSS,
    SemiStructuredRegressor,
    SemiStructuredTrainingConfig,
)
from nampy.neural.configs import DefaultSemiStructuredConfig


def dataset():
    rng = np.random.default_rng(67)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 80), "z": rng.uniform(-1, 1, 80)})
    y = (
        0.5
        + np.sin(3 * X.x.to_numpy())
        + 0.6 * X.x.to_numpy() * X.z.to_numpy()
        + rng.normal(0, 0.1, 80)
    )
    return X, y


@pytest.mark.parametrize("cls", [SemiStructuredRegressor, SemiStructuredLSS])
def test_joint_fit_projection_reconstruction_new_data_and_persistence(cls, tmp_path):
    X, y = dataset()
    config = DefaultSemiStructuredConfig(
        structured_formula="~s(x,k=5)+z",
        neural_features=("x", "z"),
        hidden_dims=(6,),
        representation_dim=4,
    )
    training = SemiStructuredTrainingConfig(
        epochs=20, lr=0.03, patience=20, random_state=18, smoothing_params=[0.2]
    )
    model = cls(architecture_config=config, training=training)
    weight = np.resize([1.0, 2.0, 0.0, 0.5], 60)
    model.fit(
        X.iloc[:60], y[:60], X_val=X.iloc[60:], y_val=y[60:], sample_weight=weight
    )
    architecture = model.model.model
    assert model.orthogonality_error_ < 1e-10
    assert architecture.neural_weight_change_ > 0
    assert (
        architecture.design_.response is None
        and architecture.design_.matrix.shape[0] == 0
    )
    assert not hasattr(architecture, "optimizer")
    assert model.training is training and model.architecture_config is config
    components = model.predict_components(X, center=True)
    assert_allclose(
        components.intercept + sum(components.terms.values()),
        components.link,
        atol=1e-11,
    )
    assert components.term_metadata["neural_component"].features == ("x", "z")
    assert components.term_metadata["neural_component"].kind == "global"
    # The projection is fitted on training data. Prediction must be independent
    # of unrelated query rows and inference batch partitioning.
    full = model.predict(X, batch_size=80)
    assert_allclose(
        model.predict(X.iloc[:3], batch_size=1), full[:3], atol=1e-12, rtol=1e-12
    )
    restored = cls.load_model(model.save_model(tmp_path / "composition.nampy"))
    assert_array_equal(restored.predict(X, batch_size=80), full)
    assert_array_equal(
        restored.model.model.projection.detach(), architecture.projection.detach()
    )
    assert clone(model).training == model.training
    if cls is SemiStructuredLSS:
        assert model.distribution_fit_status().state == "completed"
        assert np.isfinite(model.predict_distribution(X).mean()).all()
    # Re-fitting under an unsupported execution route leaves the published
    # successful predictor intact, without corrupting its learned projection.
    model.training = GradientTrainingConfig(max_epochs=1, checkpoint_path=str(tmp_path))
    with pytest.raises(NotImplementedError, match="full-cohort"):
        model.fit(X.iloc[:60], y[:60], X_val=X.iloc[60:], y_val=y[60:])
    assert_array_equal(model.predict(X, batch_size=80), full)


def test_linear_neural_representation_in_structured_span_is_removed():
    X, y = dataset()
    model = SemiStructuredRegressor(
        architecture_config=DefaultSemiStructuredConfig(
            structured_formula="~x+z", hidden_dims=(), representation_dim=3
        ),
        training=SemiStructuredTrainingConfig(epochs=3, lr=0.02),
    ).fit(X.iloc[:60], y[:60], X_val=X.iloc[60:], y_val=y[60:])
    component = model.predict_components(X.iloc[:60])
    assert_allclose(component.terms["neural_component"], 0, atol=1e-13)


def test_unsupported_objectives_inputs_and_projection_rank_are_explicit():
    X, y = dataset()
    with pytest.raises(NotImplementedError, match="full-rank"):
        SemiStructuredRegressor(
            architecture_config=DefaultSemiStructuredConfig(structured_formula="~x+z"),
            training=SemiStructuredTrainingConfig(epochs=1),
        ).fit(X.assign(z=X.x), y)
    with pytest.raises(ValueError, match="predictor broadcasting"):
        SemiStructuredLSS(training=SemiStructuredTrainingConfig(epochs=1)).fit(
            X, y, offset=np.ones(len(y))
        )


@pytest.mark.parametrize("classes", [2, 3])
def test_classification_and_categorical_projection_are_query_independent(
    classes, tmp_path
):
    from nampy import SemiStructuredClassifier

    X, y = dataset()
    X["category"] = np.resize(["a", "b", "c"], len(X))
    labels = np.arange(len(X)) % classes
    model = SemiStructuredClassifier(
        architecture_config=DefaultSemiStructuredConfig(
            structured_formula="~s(x,k=4)+category",
            hidden_dims=(4,),
            representation_dim=3,
        ),
        training=SemiStructuredTrainingConfig(epochs=4, random_state=3),
    ).fit(X.iloc[:60], labels[:60], X_val=X.iloc[60:], y_val=labels[60:])
    assert model.orthogonality_error_ < 1e-10
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), classes)
    assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert_allclose(
        model.predict_proba(X.iloc[:3], batch_size=1), probabilities[:3], atol=1e-12
    )
    components = model.predict_components(X)
    assert components.link.shape == (len(X), 1 if classes == 2 else classes)
    assert_allclose(
        components.intercept + sum(components.terms.values()),
        components.link,
        atol=1e-12,
    )
    restored = model.load_model(model.save_model(tmp_path / "class.nampy"))
    assert_array_equal(restored.predict_proba(X), probabilities)
    with pytest.raises(ValueError, match="unknown categories"):
        model.predict(X.iloc[:2].assign(category="unseen"))


@pytest.mark.parametrize(
    "family", ["normal", "gamma", "beta", "studentt", "lognormal", "weibull"]
)
def test_scalar_distributional_composition_offsets_and_shared_predictions(family):
    X, y = dataset()
    y = (
        (y - y.min() + 0.2)
        if family != "beta"
        else 0.15 + 0.7 * (y - y.min()) / (y.max() - y.min())
    )
    width = 3 if family == "studentt" else 2
    offset = np.tile(np.linspace(-0.04, 0.04, width), (len(X), 1))
    model = SemiStructuredLSS(
        objective=ObjectiveConfig(family=family),
        architecture_config=DefaultSemiStructuredConfig(
            structured_formula="~s(x,k=4)+z", hidden_dims=(4,), representation_dim=3
        ),
        training=SemiStructuredTrainingConfig(epochs=3, random_state=3),
    ).fit(
        X.iloc[:60],
        y[:60],
        X_val=X.iloc[60:],
        y_val=y[60:],
        offset=offset[:60],
        offset_val=offset[60:],
    )
    raw = model.predict(X, raw=True)
    shifted = model.predict(X, raw=True, offset=offset)
    assert_allclose(shifted, raw + offset, atol=1e-12)
    components = model.predict_components(
        X,
        offset=offset,
        center=True,
        reference_X=X.iloc[:20],
        reference_offset=offset[:20],
    )
    assert_allclose(
        components.intercept + components.offset + sum(components.terms.values()),
        shifted,
        atol=1e-12,
    )
    law = model.predict_distribution(X, offset=offset)
    assert np.isfinite(law.quantile([0.2, 0.5, 0.8])).all()
    assert np.isfinite(model.score(X, y, offset=offset))
    assert model.orthogonality_error_ < 1e-10


def test_regression_offsets_are_supplied_again_at_prediction_and_refit_is_transactional():
    X, y = dataset()
    offset = np.linspace(-0.2, 0.2, len(X))
    model = SemiStructuredRegressor(
        training=SemiStructuredTrainingConfig(epochs=3, random_state=12)
    ).fit(
        X.iloc[:60],
        y[:60],
        X_val=X.iloc[60:],
        y_val=y[60:],
        offset=offset[:60],
        offset_val=offset[60:],
    )
    baseline = model.predict(X)
    assert_allclose(model.predict(X, offset=offset), baseline + offset, atol=1e-12)
    components = model.predict_components(X, offset=offset)
    assert_allclose(
        components.intercept + components.offset + sum(components.terms.values()),
        (baseline + offset)[:, None],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="predictor broadcasting"):
        model.fit(X, y, offset=np.zeros((len(X), 2)))
    assert_array_equal(model.predict(X), baseline)
