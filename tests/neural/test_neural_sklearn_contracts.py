from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.metrics import accuracy_score

import nampy
from nampy import models
from nampy.models.ensemble_treenam import (
    EnsembleTreeNAMClassifier,
    EnsembleTreeNAMLSS,
    EnsembleTreeNAMRegressor,
)
from nampy.models.gpnam import GPNAMLSS, GPNAMClassifier, GPNAMRegressor
from nampy.models.igann import IGANNLSS, IGANNClassifier, IGANNRegressor
from nampy.models.linreg import LinRegClassifier, LinRegLSS, LinRegRegressor
from nampy.models.nam import NAMLSS, NAMClassifier, NAMRegressor
from nampy.models.namformer import (
    NAMformerClassifier,
    NAMformerLSS,
    NAMformerRegressor,
)
from nampy.models.natt import NATTLSS, NATTClassifier, NATTRegressor
from nampy.models.nbm import NBMLSS, NBMClassifier, NBMRegressor
from nampy.models.nbm_spam import NBMSPAMLSS, NBMSPAMClassifier, NBMSPAMRegressor
from nampy.models.nodegam import (
    NodeGAMClassifier,
    NodeGAMLSS,
    NodeGAMRegressor,
)
from nampy.models.qnam import QNAMLSS
from nampy.models.sian import SIANLSS, SIANClassifier, SIANRegressor
from nampy.models.snam import SNAMLSS, SNAMClassifier, SNAMRegressor
from nampy.models.spam import SPAMLSS, SPAMClassifier, SPAMRegressor
from nampy.models.spline_nam import SplineNAMRegressor
from nampy.models.treenam import TreeNAMClassifier, TreeNAMLSS, TreeNAMRegressor
from nampy.neural import architectures, configs
from nampy.neural.configs.nam_config import DefaultNAMConfig
from nampy.neural.configs.spline_nam_config import DefaultSplineNAMConfig
from nampy.neural.configuration import (
    GradientTrainingConfig,
    ObjectiveConfig,
    PreprocessingConfig,
)
from nampy.neural.families import NormalDistribution

ALL_NEURAL_ESTIMATORS = (
    NAMRegressor,
    NAMClassifier,
    NAMLSS,
    SIANRegressor,
    SIANClassifier,
    SIANLSS,
    SNAMRegressor,
    SNAMClassifier,
    SNAMLSS,
    LinRegRegressor,
    LinRegClassifier,
    LinRegLSS,
    GPNAMRegressor,
    GPNAMClassifier,
    GPNAMLSS,
    IGANNRegressor,
    IGANNClassifier,
    IGANNLSS,
    NBMRegressor,
    NBMClassifier,
    NBMLSS,
    NBMSPAMRegressor,
    NBMSPAMClassifier,
    NBMSPAMLSS,
    SPAMRegressor,
    SPAMClassifier,
    SPAMLSS,
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
    QNAMLSS,
    SplineNAMRegressor,
)


def test_public_packages_export_supported_neural_surfaces():
    for estimator_class in ALL_NEURAL_ESTIMATORS:
        assert getattr(models, estimator_class.__name__) is estimator_class
        assert getattr(nampy, estimator_class.__name__) is estimator_class

    expected_base_models = {
        "SIAN": architectures.SIAN,
        "SNAM": architectures.SNAM,
        "EnsembleTreeNAM": architectures.EnsembleTreeNAM,
        "NBMSPAM": architectures.NBMSPAM,
        "SPAM": architectures.SPAM,
    }
    expected_configs = {
        "DefaultSIANConfig": configs.DefaultSIANConfig,
        "DefaultSNAMConfig": configs.DefaultSNAMConfig,
        "DefaultGPNAMConfig": configs.DefaultGPNAMConfig,
        "DefaultIGANNConfig": configs.DefaultIGANNConfig,
        "DefaultNBMSPAMConfig": configs.DefaultNBMSPAMConfig,
        "DefaultSPAMConfig": configs.DefaultSPAMConfig,
        "DefaultQNAMConfig": configs.DefaultQNAMConfig,
        "DefaultTreeNAMConfig": configs.DefaultTreeNAMConfig,
        "DefaultEnsembleTreeNAMConfig": configs.DefaultEnsembleTreeNAMConfig,
    }

    assert expected_base_models.keys() <= set(architectures.__all__)
    assert expected_configs.keys() <= set(configs.__all__)


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
        architecture_config=DefaultNAMConfig(dropout=0.23),
        preprocessing=PreprocessingConfig(n_bins=17, numerical_preprocessing="standardization"),
    )

    params = estimator.get_params()
    cloned = clone(estimator)

    assert params["architecture_config__dropout"] == pytest.approx(0.23)
    assert params["preprocessing__n_bins"] == 17
    assert params["preprocessing__numerical_preprocessing"] == "standardization"
    assert cloned.get_params()["architecture_config__dropout"] == pytest.approx(0.23)
    assert cloned.get_params()["preprocessing__n_bins"] == 17
    assert cloned.preprocessor.numerical_preprocessing == "standardization"


@pytest.mark.parametrize("estimator_class", [NAMRegressor, NAMClassifier, NAMLSS])
def test_neural_estimators_set_default_config_and_preprocessor_parameters(
    estimator_class,
):
    estimator = estimator_class()

    returned = estimator.set_params(architecture_config__dropout=0.31, preprocessing__n_bins=19)

    assert returned is estimator
    assert estimator.architecture_config.dropout == pytest.approx(0.31)
    assert estimator.preprocessor.n_bins == 19
    assert estimator.get_params()["architecture_config__dropout"] == pytest.approx(0.31)
    assert estimator.get_params()["preprocessing__n_bins"] == 19


def test_spline_nam_clone_separates_architecture_and_preprocessor_widths():
    estimator = SplineNAMRegressor(architecture_config=DefaultSplineNAMConfig(n_knots=7), preprocessing=PreprocessingConfig(n_bins=13))

    cloned = clone(estimator)

    assert cloned.architecture_config.n_knots == 7
    assert cloned.preprocessor.n_bins == 13


def test_flat_parameters_and_old_preprocessing_aliases_are_not_translated():
    with pytest.raises(TypeError):
        NAMRegressor(dropout=0.1)
    with pytest.raises(ValueError, match="Unknown estimator parameter"):
        NAMRegressor().set_params(preprocessor__n_bins=12)


def test_lss_family_configuration_is_cloneable():
    estimator = NAMLSS(objective=ObjectiveConfig(family='poisson', distributional_kwargs={'eps': 1e-05}), preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))

    cloned = clone(estimator)

    assert cloned.objective.family == "poisson"
    assert cloned.objective.distributional_kwargs == {"eps": 1e-5}
    assert cloned.get_params()["objective__family"] == "poisson"


def test_lss_predict_point_uses_family_conditional_point(monkeypatch):
    estimator = NAMLSS()
    estimator.family_ = NormalDistribution()
    raw = torch.tensor([[1.5, -0.2], [-0.4, 0.7]], dtype=torch.float32)
    monkeypatch.setattr(
        estimator, "_predict", lambda X, batch_size=None, offset=None: {"output": raw}
    )

    np.testing.assert_allclose(estimator.predict_point([[0.0], [1.0]]), [1.5, -0.4])


def test_qnam_constructor_owns_quantile_family():
    estimator = QNAMLSS()

    assert estimator.objective.family == "quantile"
    assert estimator.objective.distributional_kwargs == {
        "quantiles": [0.25, 0.5, 0.75],
        "enforce_monotonic": False,
    }
    cloned = clone(estimator)
    assert cloned.objective.family == "quantile"
    assert cloned.objective.distributional_kwargs == {
        "quantiles": [0.25, 0.5, 0.75],
        "enforce_monotonic": False,
    }


def test_lss_rejects_fit_time_family_configuration():
    estimator = NAMLSS()

    with pytest.raises(TypeError, match="unexpected keyword"):
        estimator.fit([[0.0], [1.0]], [0.0, 1.0], family="normal")


def test_classifier_evaluate_accepts_positional_array_after_dataframe_fit(tmp_path):
    x = np.linspace(-1.0, 1.0, 36)
    data = pd.DataFrame({"first": x, "second": np.cos(2.0 * x)})
    labels = (x > 0.0).astype(int)
    estimator = LinRegClassifier(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
    estimator.set_params(training=replace(estimator.training or GradientTrainingConfig(), max_epochs=1, batch_size=12, checkpoint_path=tmp_path, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0})).fit(data, labels)

    predictions = estimator.predict(data.to_numpy())
    scores = estimator.evaluate(data.to_numpy(), labels)

    assert predictions.shape == labels.shape
    assert scores["Accuracy"] == pytest.approx(accuracy_score(labels, predictions))


def test_fitted_neural_estimator_persistence_round_trip(tmp_path):
    x = np.linspace(-1.0, 1.0, 24)
    data = pd.DataFrame({"first": x, "second": np.cos(2.0 * x)})
    labels = (x > 0.0).astype(int)
    estimator = LinRegClassifier(preprocessing=PreprocessingConfig(numerical_preprocessing='standardization'))
    estimator.set_params(training=replace(estimator.training or GradientTrainingConfig(), max_epochs=1, batch_size=len(data), checkpoint_path=tmp_path / 'checkpoints', trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False, 'num_sanity_val_steps': 0, 'fast_dev_run': True})).fit(data, labels)
    expected = estimator.predict_proba(data)

    model_path = estimator.save_model(tmp_path / "classifier.nampy")
    restored = LinRegClassifier.load_model(model_path)

    assert model_path.is_file()
    np.testing.assert_allclose(restored.predict_proba(data), expected)
    with pytest.raises(TypeError, match="not LinRegRegressor"):
        LinRegRegressor.load_model(model_path)


@pytest.mark.parametrize(
    "estimator_class",
    [models.NeuralRegressor, models.NeuralClassifier, models.NeuralLSS],
)
def test_generic_neural_estimators_clone_constructor_declarations(estimator_class):
    estimator = estimator_class(
        model=architectures.NAM, config_class=configs.DefaultNAMConfig, architecture_config=DefaultNAMConfig(dropout=0.23)
    )
    estimator.set_params(preprocessing__n_bins=13)
    rebuilt = clone(estimator)
    assert type(rebuilt) is estimator_class
    assert rebuilt.base_model is architectures.NAM
    assert rebuilt._config_class is configs.DefaultNAMConfig
    assert rebuilt.architecture_config.dropout == pytest.approx(0.23)
    assert rebuilt.get_params()["preprocessing__n_bins"] == 13
    rebuilt.set_params(
        model=architectures.NAM, config_class=configs.DefaultNAMConfig, architecture_config=DefaultNAMConfig(dropout=0.31)
    )
    assert rebuilt.get_params()["architecture_config__dropout"] == pytest.approx(0.31)
