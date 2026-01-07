import pytest

from nampy.models import (
    GPNAMLSS,
    LinRegLSS,
    NAMLSS,
    NAMformerLSS,
    NATTLSS,
    NBMLSS,
    NodeGAMLSS,
    QNAM,
)


LSS_MODELS = [
    (NAMLSS, {"layer_sizes": (4,), "dropout": 0.0}),
    (GPNAMLSS, {"layer_sizes": (4,), "dropout": 0.0}),
    (LinRegLSS, {"lr": 1e-3}),
    (
        NBMLSS,
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
        NATTLSS,
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
        NAMformerLSS,
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
        NodeGAMLSS,
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


@pytest.mark.parametrize("model_cls, model_kwargs", LSS_MODELS)
def test_lss_models_fit_predict(model_cls, model_kwargs, regression_data, tmp_path):
    X, y = regression_data
    model = model_cls(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        **model_kwargs,
    )
    model.fit(
        X,
        y,
        family="normal",
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
    assert preds.shape[0] == len(X)

    raw_preds = model.predict(X, raw=True)
    assert raw_preds.shape[0] == len(X)

    scores = model.evaluate(X, y, distribution_family="normal")
    assert "NLL" in scores


def test_qnam_fit_predict(regression_data, tmp_path):
    X, y = regression_data
    model = QNAM(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        layer_sizes=(4,),
        dropout=0.0,
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
        distributional_kwargs={"quantiles": [0.25, 0.5, 0.75]},
    )

    preds = model.predict(X)
    assert preds.shape[0] == len(X)
    assert preds.shape[1] == 3
