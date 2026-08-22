from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

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
from nampy.models.nbm_spam import NBMSPAMLSS, NBMSPAMClassifier, NBMSPAMRegressor
from nampy.models.nodegam import (
    NodeGAMClassifier,
    NodeGAMLSS,
    NodeGAMRegressor,
)
from nampy.models.qnam import QNAMLSS
from nampy.models.snam import SNAMLSS, SNAMClassifier, SNAMRegressor
from nampy.models.spam import SPAMLSS, SPAMClassifier, SPAMRegressor
from nampy.models.spline_nam import SplineNAMRegressor
from nampy.models.treenam import TreeNAMClassifier, TreeNAMLSS, TreeNAMRegressor


@dataclass(frozen=True)
class _EstimatorCase:
    name: str
    estimator_class: type
    task: str
    model_kwargs: dict
    categorical_method: str = "int"
    n_classes: int = 2


_ARCHITECTURES = (
    (
        "linreg",
        LinRegRegressor,
        LinRegClassifier,
        LinRegLSS,
        {},
    ),
    (
        "nam",
        NAMRegressor,
        NAMClassifier,
        NAMLSS,
        {"layer_sizes": [8], "dropout": 0.0},
    ),
    (
        "snam",
        SNAMRegressor,
        SNAMClassifier,
        SNAMLSS,
        {
            "layer_sizes": [8],
            "dropout": 0.0,
            "group_lasso_lambda": 0.01,
        },
    ),
    (
        "gpnam",
        GPNAMRegressor,
        GPNAMClassifier,
        GPNAMLSS,
        {"rff_num_feat": 8},
    ),
    (
        "nbm",
        NBMRegressor,
        NBMClassifier,
        NBMLSS,
        {
            "layer_sizes": [8],
            "dropout": 0.0,
            "bases_dropout": 0.0,
            "num_bases": 4,
            "output_penalty": 0.01,
        },
    ),
    (
        "spam",
        SPAMRegressor,
        SPAMClassifier,
        SPAMLSS,
        {"ranks": [4], "dropout": 0.0},
    ),
    (
        "nbm_spam",
        NBMSPAMRegressor,
        NBMSPAMClassifier,
        NBMSPAMLSS,
        {
            "layer_sizes": [8],
            "num_bases": 4,
            "ranks": [4],
            "batch_norm": False,
        },
    ),
    (
        "natt",
        NATTRegressor,
        NATTClassifier,
        NATTLSS,
        {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "attn_dropout": 0.0,
            "transformer_dim_feedforward": 16,
            "head_layer_sizes": (),
            "head_dropout": 0.0,
        },
    ),
    (
        "namformer",
        NAMformerRegressor,
        NAMformerClassifier,
        NAMformerLSS,
        {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "attn_dropout": 0.0,
            "transformer_dim_feedforward": 16,
            "head_layer_sizes": (),
            "head_dropout": 0.0,
        },
    ),
    (
        "treenam",
        TreeNAMRegressor,
        TreeNAMClassifier,
        TreeNAMLSS,
        {"tree_depth": 2, "tree_lamda": 0.01},
    ),
    (
        "ensemble_treenam",
        EnsembleTreeNAMRegressor,
        EnsembleTreeNAMClassifier,
        EnsembleTreeNAMLSS,
        {"tree_depth": 2, "tree_lamda": 0.01, "num_estimators": 2},
    ),
    (
        "nodegam",
        NodeGAMRegressor,
        NodeGAMClassifier,
        NodeGAMLSS,
        {
            "num_trees": 4,
            "num_layers": 1,
            "depth": 2,
            "last_dropout": 0.0,
            "colsample_bytree": 1.0,
            "l2_lambda": 0.01,
            "anneal_steps": 10,
            "interaction_degree": 1,
        },
    ),
)


ESTIMATOR_CASES = tuple(
    _EstimatorCase(f"{name}_{task}", estimator_class, task, model_kwargs)
    for name, regressor, classifier, lss, model_kwargs in _ARCHITECTURES
    for task, estimator_class in (
        ("regression", regressor),
        ("classification", classifier),
        ("lss", lss),
    )
) + (
    _EstimatorCase(
        "spline_nam_regression",
        SplineNAMRegressor,
        "regression",
        {"n_knots": 5, "smoothing": 0.01},
    ),
    _EstimatorCase(
        "qnam_lss",
        QNAMLSS,
        "quantile",
        {"layer_sizes": [8], "dropout": 0.0},
    ),
)
ESTIMATOR_CASES += tuple(
    _EstimatorCase(
        f"{case.name}_one_hot",
        case.estimator_class,
        case.task,
        case.model_kwargs,
        categorical_method="one-hot",
    )
    for case in ESTIMATOR_CASES
    if case.estimator_class is not SplineNAMRegressor
)

MULTIOUTPUT_REGRESSION_CASES = tuple(
    _EstimatorCase(f"{name}_multioutput", regressor, "regression", model_kwargs)
    for name, regressor, _classifier, _lss, model_kwargs in _ARCHITECTURES
) + (
    _EstimatorCase(
        "spline_nam_multioutput",
        SplineNAMRegressor,
        "regression",
        {"n_knots": 5, "smoothing": 0.01},
    ),
)
ESTIMATOR_CASES += tuple(
    _EstimatorCase(
        f"{case.name}_multiclass",
        case.estimator_class,
        case.task,
        case.model_kwargs,
        categorical_method=case.categorical_method,
        n_classes=3,
    )
    for case in ESTIMATOR_CASES
    if case.task == "classification"
    and case.categorical_method == "int"
)


@pytest.mark.parametrize("case", ESTIMATOR_CASES, ids=lambda case: case.name)
def test_public_neural_estimator_fits_and_predicts(case, tmp_path):
    x = np.linspace(0.05, 0.95, 24)
    groups = np.resize(np.array(["a", "b", "c"]), x.size)
    data = pd.DataFrame(
        {
            "x": x,
            "z": np.cos(2.0 * np.pi * x),
            "group": groups,
        }
    )
    regression_target = (
        np.sin(2.0 * np.pi * x) + 0.25 * x + 0.2 * (groups == "b")
    )
    classification_target = np.resize(np.arange(case.n_classes), x.size)

    estimator_kwargs = {
        "numerical_method": "minmax",
        "categorical_method": case.categorical_method,
        **case.model_kwargs,
    }
    if case.task == "lss":
        estimator_kwargs["family"] = "normal"
    estimator = case.estimator_class(**estimator_kwargs)
    target = (
        classification_target
        if case.task == "classification"
        else regression_target
    )
    fit_kwargs = {
        "max_epochs": 1,
        "batch_size": 20,
        "shuffle": False,
        "checkpoint_path": tmp_path / case.name,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "fast_dev_run": True,
    }
    returned = estimator.fit(data, target, **fit_kwargs)
    assert returned is estimator
    contributions = estimator._predict(data)
    assert {"output", "x", "z"} <= set(contributions)
    assert "group" in contributions or any(
        name.startswith("group[") for name in contributions
    )
    if case.name == "spam_regression":
        local = estimator.local_term_importance(data.iloc[:3], top_k=2)
        assert len(local) == 3
        # ``top_k`` is an upper bound: [0, 1] preprocessing can map a row
        # to the all-zero origin, for which SPAM has no active source term.
        assert all(len(row) <= 2 for row in local)
        nonempty = next(row for row in local if row)
        assert {"term", "order", "contribution"} <= set(nonempty[0])

    if case.task == "classification":
        predictions = estimator.predict(data)
        probabilities = estimator.predict_proba(data)
        assert predictions.shape == (len(data),)
        assert probabilities.shape == (len(data), case.n_classes)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    else:
        predictions = estimator.predict(data, raw=True) if case.task in {"lss", "quantile"} else estimator.predict(data)
        expected_width = 3 if case.task == "quantile" else (2 if case.task == "lss" else None)
        expected_shape = (len(data), expected_width) if expected_width else (len(data),)
        assert predictions.shape == expected_shape
        assert np.isfinite(predictions).all()


@pytest.mark.parametrize(
    "case", MULTIOUTPUT_REGRESSION_CASES, ids=lambda case: case.name
)
def test_public_neural_regressors_support_multioutput_targets(case, tmp_path):
    x = np.linspace(0.05, 0.95, 20)
    data = pd.DataFrame({"x": x, "z": np.cos(2.0 * np.pi * x)})
    targets = np.column_stack(
        (
            np.sin(2.0 * np.pi * x) + 0.25 * x,
            np.cos(np.pi * x) - 0.1 * x,
        )
    )
    estimator = case.estimator_class(
        numerical_method="minmax", **case.model_kwargs
    )

    estimator.fit(
        data,
        targets,
        max_epochs=1,
        batch_size=len(data),
        shuffle=False,
        checkpoint_path=tmp_path / case.name,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        fast_dev_run=True,
    )

    predictions = estimator.predict(data)
    assert estimator.n_outputs_ == 2
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()
