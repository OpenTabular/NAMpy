from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from nampy.models.ensemble_treenam import (
    EnsembleTreeNAMClassifier,
    EnsembleTreeNAMLSS,
    EnsembleTreeNAMRegressor,
)
from nampy.models.gpnam import GPNAMLSS, GPNAMClassifier, GPNAMRegressor
from nampy.models.linreg import LinRegClassifier, LinRegLSS, LinRegRegressor
from nampy.models.nam import NAMLSS, NAMClassifier, NAMRegressor
from nampy.models.namformer import (
    NAMformerClassifier,
    NAMformerLSS,
    NAMformerRegressor,
)
from nampy.models.natt import NATTLSS, NATTClassifier, NATTRegressor
from nampy.models.nbm import NBMLSS, NBMClassifier, NBMRegressor
from nampy.models.nodegam import (
    NodeGAMClassifier,
    NodeGAMLSS,
    NodeGAMRegressor,
)
from nampy.models.qnam import QNAM
from nampy.models.sklearn_lss import SklearnBaseLSS
from nampy.models.snam import SNAMLSS, SNAMClassifier, SNAMRegressor
from nampy.models.spline_nam import SplineNAMRegressor
from nampy.models.treenam import TreeNAMClassifier, TreeNAMLSS, TreeNAMRegressor

ALL_NEURAL_ESTIMATORS = (
    NAMRegressor,
    NAMClassifier,
    NAMLSS,
    SNAMRegressor,
    SNAMClassifier,
    SNAMLSS,
    LinRegRegressor,
    LinRegClassifier,
    LinRegLSS,
    GPNAMRegressor,
    GPNAMClassifier,
    GPNAMLSS,
    NBMRegressor,
    NBMClassifier,
    NBMLSS,
    NATTRegressor,
    NATTClassifier,
    NATTLSS,
    NAMformerRegressor,
    NAMformerClassifier,
    NAMformerLSS,
    TreeNAMRegressor,
    TreeNAMClassifier,
    TreeNAMLSS,
    EnsembleTreeNAMRegressor,
    EnsembleTreeNAMClassifier,
    EnsembleTreeNAMLSS,
    NodeGAMRegressor,
    NodeGAMClassifier,
    NodeGAMLSS,
    QNAM,
    SplineNAMRegressor,
)


@pytest.mark.parametrize("estimator_class", ALL_NEURAL_ESTIMATORS)
def test_all_neural_estimators_satisfy_constructor_clone_contract(estimator_class):
    estimator = estimator_class()

    cloned = clone(estimator)

    assert cloned.__class__ is estimator_class
    if "Classifier" in estimator_class.__name__:
        assert cloned.preprocessor.task == "classification"


@pytest.mark.parametrize("estimator_class", [NAMRegressor, NAMClassifier, NAMLSS])
def test_neural_estimators_clone_config_and_preprocessor_parameters(estimator_class):
    estimator = estimator_class(
        dropout=0.23,
        n_bins=17,
        numerical_preprocessing="standardization",
    )

    params = estimator.get_params(deep=False)
    cloned = clone(estimator)

    assert params["dropout"] == pytest.approx(0.23)
    assert params["n_bins"] == 17
    assert params["numerical_preprocessing"] == "standardization"
    assert cloned.get_params(deep=False)["dropout"] == pytest.approx(0.23)
    assert cloned.get_params(deep=False)["n_bins"] == 17
    assert cloned.preprocessor.numerical_preprocessing == "standardization"


@pytest.mark.parametrize("estimator_class", [NAMRegressor, NAMClassifier, NAMLSS])
def test_neural_estimators_set_default_config_and_preprocessor_parameters(
    estimator_class,
):
    estimator = estimator_class()

    returned = estimator.set_params(dropout=0.31, n_bins=19)

    assert returned is estimator
    assert estimator.config.dropout == pytest.approx(0.31)
    assert estimator.preprocessor.n_bins == 19
    assert estimator.get_params(deep=False)["dropout"] == pytest.approx(0.31)
    assert estimator.get_params(deep=False)["n_bins"] == 19


def test_spline_nam_clone_preserves_overlapping_n_knots_parameter():
    estimator = SplineNAMRegressor(n_knots=7, n_bins=13)

    cloned = clone(estimator)

    assert cloned.config.n_knots == 7
    assert cloned.preprocessor.n_knots == 7
    assert cloned.preprocessor.n_bins == 13


def test_qnam_fit_returns_self_and_owns_quantile_family(monkeypatch):
    captured = {}

    def fake_fit(self, X, y, **kwargs):
        captured.update(kwargs)
        return self

    monkeypatch.setattr(SklearnBaseLSS, "fit", fake_fit)
    estimator = QNAM()

    returned = estimator.fit([[0.0], [1.0]], [0.0, 1.0])

    assert returned is estimator
    assert captured["family"] == "quantile"
    assert captured["distributional_kwargs"] == {
        "quantiles": [0.25, 0.5, 0.75]
    }


def test_classifier_evaluate_accepts_positional_array_after_dataframe_fit(tmp_path):
    x = np.linspace(-1.0, 1.0, 36)
    data = pd.DataFrame({"first": x, "second": np.cos(2.0 * x)})
    labels = (x > 0.0).astype(int)
    estimator = LinRegClassifier(numerical_preprocessing="standardization")
    estimator.fit(
        data,
        labels,
        max_epochs=1,
        batch_size=12,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    predictions = estimator.predict(data.to_numpy())
    scores = estimator.evaluate(data.to_numpy(), labels)

    assert predictions.shape == labels.shape
    assert scores["Accuracy"] == pytest.approx(accuracy_score(labels, predictions))
