import pytest
from sklearn.metrics import roc_auc_score

from nampy.models import (
    GPNAMClassifier,
    LinRegClassifier,
    NAMClassifier,
    NAMformerClassifier,
    NATTClassifier,
    NBMClassifier,
    NodeGAMClassifier,
)

CLASSIFICATION_MODELS = [
    (NAMClassifier, {"layer_sizes": (4,), "dropout": 0.0}),
    (GPNAMClassifier, {"layer_sizes": (4,), "dropout": 0.0}),
    (LinRegClassifier, {"lr": 1e-3}),
    (
        NBMClassifier,
        {
            "hidden_dims": (4,),
            "num_bases": 4,
            "num_subnets": 1,
            "dropout": 0.0,
            "bases_dropout": 0.0,
            "batch_norm": False,
        },
    ),
    (
        NATTClassifier,
        {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 1,
            "transformer_dim_feedforward": 16,
            "head_layer_sizes": (),
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "dropout": 0.0,
        },
    ),
    (
        NAMformerClassifier,
        {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 1,
            "transformer_dim_feedforward": 16,
            "head_layer_sizes": (),
            "attn_dropout": 0.0,
            "ff_dropout": 0.0,
            "dropout": 0.0,
        },
    ),
    (
        NodeGAMClassifier,
        {
            "num_trees": 2,
            "num_layers": 1,
            "depth": 2,
            "interaction_degree": 1,
            "output_dropout": 0.0,
            "last_dropout": 0.0,
            "colsample_bytree": 1.0,
        },
    ),
]


@pytest.mark.parametrize("model_cls, model_kwargs", CLASSIFICATION_MODELS)
def test_classification_models_fit_predict(
    model_cls, model_kwargs, classification_data, tmp_path
):
    X, y = classification_data
    model = model_cls(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        **model_kwargs,
    )
    model.fit(
        X,
        y,
        max_epochs=1,
        batch_size=8,
        val_size=0.2,
        checkpoint_path=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    preds = model.predict(X)
    assert len(preds) == len(X)

    scores = model.evaluate(X, y)
    assert "Accuracy" in scores


def test_classifier_predict_outputs_numpy_and_predict_proba_binary_shape(
    classification_data, tmp_path
):
    X, y = classification_data
    model = LinRegClassifier(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        lr=1e-3,
    )
    model.fit(
        X,
        y,
        max_epochs=1,
        batch_size=8,
        val_size=0.2,
        checkpoint_path=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype.kind in {"i", "u"}

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert (abs(probs.sum(axis=1) - 1.0) < 1e-6).all()


def test_classifier_evaluate_probability_metrics_binary(classification_data, tmp_path):
    X, y = classification_data
    model = LinRegClassifier(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        lr=1e-3,
    )
    model.fit(
        X,
        y,
        max_epochs=1,
        batch_size=8,
        val_size=0.2,
        checkpoint_path=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    metrics = {"AUC": (roc_auc_score, True)}
    scores = model.evaluate(X, y, metrics=metrics)
    assert "AUC" in scores
    assert 0.0 <= scores["AUC"] <= 1.0


def test_classifier_predict_feature_vals_returns_dict(classification_data, tmp_path):
    X, y = classification_data
    model = LinRegClassifier(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        lr=1e-3,
    )
    model.fit(
        X,
        y,
        max_epochs=1,
        batch_size=8,
        val_size=0.2,
        checkpoint_path=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    pred_dict = model.predict_feature_vals(X)
    assert isinstance(pred_dict, dict)
    assert "output" in pred_dict
