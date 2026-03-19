import numpy as np

from nampy.models import NGBSurvival, NGBoostClassifier, NGBoostRegressor


def test_ngboost_regressor_fit_predict(regression_data):
    X, y = regression_data
    model = NGBoostRegressor(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        n_estimators=5,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
        base_learner_kwargs={"max_depth": 2},
    )

    model.fit(X, y, val_size=0.2, random_state=0)
    preds = model.predict(X)
    dist = model.pred_dist(X)
    scores = model.evaluate(X, y)

    assert preds.shape == (len(X),)
    assert dist.params["loc"].shape[0] == len(X)
    assert "Mean Squared Error" in scores


def test_ngboost_regressor_supports_poisson_distribution(regression_data):
    X, y = regression_data
    counts = np.clip(np.round(np.abs(y) * 3), 0, None).astype(int)

    model = NGBoostRegressor(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        distribution="poisson",
        n_estimators=5,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
    )

    model.fit(X, counts, val_size=0.2, random_state=0)
    dist = model.pred_dist(X)

    assert dist.params["mu"].shape[0] == len(X)


def test_ngboost_classifier_fit_predict_proba(classification_data):
    X, y = classification_data
    model = NGBoostClassifier(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        n_estimators=5,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
        base_learner_kwargs={"max_depth": 2},
    )

    model.fit(X, y, val_size=0.2, random_state=0)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    scores = model.evaluate(X, y)

    assert preds.shape == (len(X),)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert "Accuracy" in scores


def test_ngboost_survival_fit_predict(regression_data):
    X, y = regression_data
    times = np.abs(y) + 0.5
    events = (np.arange(len(times)) % 3 != 0).astype(int)

    model = NGBSurvival(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        n_estimators=5,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
        base_learner_kwargs={"max_depth": 2},
    )

    model.fit(X, times, events, val_size=0.2, random_state=0)
    preds = model.predict(X)
    dist = model.pred_dist(X)
    scores = model.evaluate(X, times, events)

    assert preds.shape == (len(X),)
    assert dist.params["scale"].shape[0] == len(X)
    assert "NLL" in scores


def test_ngboost_survival_supports_exponential_distribution(regression_data):
    X, y = regression_data
    times = np.abs(y) + 0.5
    events = (np.arange(len(times)) % 2).astype(int)

    model = NGBSurvival(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        distribution="exponential",
        n_estimators=5,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
    )

    model.fit(X, times, events, val_size=0.2, random_state=0)
    dist = model.pred_dist(X)

    assert dist.params["scale"].shape[0] == len(X)
