import pytest

from nampy.models import (
    GPNAMRegressor,
    LinRegRegressor,
    NAMRegressor,
    NAMformerRegressor,
    NATTRegressor,
    NBMRegressor,
    NodeGAMRegressor,
    SNAMRegressor,
    TreeNAMRegressor,
)


REGRESSION_MODELS = [
    (NAMRegressor, {"layer_sizes": (4,), "dropout": 0.0}),
    (GPNAMRegressor, {"layer_sizes": (4,), "dropout": 0.0}),
    (LinRegRegressor, {"lr": 1e-3}),
    (
        NBMRegressor,
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
        NATTRegressor,
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
        NAMformerRegressor,
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
    (TreeNAMRegressor, {"n_estimators": 2, "tree_depth": 2, "lr": 0.1}),
    (
        SNAMRegressor,
        {"n_knots": 4, "smoothing": 0.0, "learn_knots": False, "dropout": 0.0},
    ),
    (
        NodeGAMRegressor,
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


@pytest.mark.parametrize("model_cls, model_kwargs", REGRESSION_MODELS)
def test_regression_models_fit_predict(model_cls, model_kwargs, regression_data, tmp_path):
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
    assert "Mean Squared Error" in scores
