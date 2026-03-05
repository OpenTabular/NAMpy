"""
Comprehensive tests for the NAM model family:
  - NAMRegressor
  - NAMClassifier
  - NAMLSS (every distribution family)

Coverage:
  - Basic fit / predict / evaluate
  - Every LSS distribution: normal, robustnormal, poisson, gamma, beta, dirichlet,
    studentt, negativebinom, inversegamma, categorical, quantile
  - Config variants: layer_sizes, dropout, activation, skip_connections, batch_norm,
    layer_norm, use_glu
  - Interaction terms (interaction_degree=None vs. interaction_degree=2)
  - Mixed data (numerical + categorical features)
  - Plotting for every model type (non-interactive Agg backend, plt.show mocked)
  - Sklearn compatibility: get_params / set_params idempotency
  - Error handling: predict before fit, incomplete validation inputs
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend – must come before pyplot imports

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from nampy.models import NAMClassifier, NAMRegressor, NAMLSS


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_FAST_FIT = dict(
    max_epochs=1,
    batch_size=8,
    val_size=0.2,
    limit_train_batches=1,
    limit_val_batches=1,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)

_BASE_KWARGS = dict(
    layer_sizes=(4,),
    dropout=0.0,
    numerical_preprocessing="standardization",
    categorical_preprocessing="int",
    cat_cutoff=0.1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = 0.5 * X["f1"].to_numpy() - 0.2 * X["f2"].to_numpy() + rng.normal(scale=0.1, size=60)
    return X, y


@pytest.fixture
def cls_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    raw = X["f1"].to_numpy() + X["f2"].to_numpy()
    y = np.digitize(raw, bins=np.percentile(raw, [33, 67])).astype(int)
    # Guarantee all three classes present
    y[0], y[1], y[2] = 0, 1, 2
    return X, y


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=60),
            "num2": rng.uniform(-1, 1, size=60),
            "int_cat": rng.integers(0, 3, size=60),
            "str_cat": rng.choice(["a", "b", "c"], size=60),
        }
    )
    y = (X["num1"] * 0.3 + rng.normal(scale=0.1, size=60)).to_numpy()
    return X, y


@pytest.fixture
def positive_data():
    """Strictly positive targets (>0) for Gamma / InverseGamma."""
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = np.abs(0.5 * X["f1"].to_numpy() - 0.2 * X["f2"].to_numpy()) + 0.5
    return X, y


@pytest.fixture
def count_data():
    """Non-negative integer counts for Poisson / NegativeBinomial."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = rng.poisson(lam=3, size=60).astype(float)
    return X, y


@pytest.fixture
def beta_data():
    """Targets in (0, 1) for Beta."""
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = rng.beta(2, 5, size=60)
    return X, y


@pytest.fixture
def dirichlet_data():
    """Simplex targets shape (n, 3) for Dirichlet."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = rng.dirichlet(alpha=np.ones(3), size=60)
    return X, y


@pytest.fixture
def cat_lss_data():
    """Integer class labels (0/1/2) for CategoricalDistribution."""
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.uniform(-1, 1, size=60)})
    y = rng.integers(0, 3, size=60).astype(int)
    y[0], y[1], y[2] = 0, 1, 2  # ensure all classes present
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(cls, **extra):
    return cls(**{**_BASE_KWARGS, **extra})


def _fit(model, X, y, tmp_path, **extra_fit):
    kw = {**_FAST_FIT, "checkpoint_path": tmp_path, **extra_fit}
    model.fit(X, y, **kw)
    return model


def _fit_lss(X, y, family, tmp_path, dist_kwargs=None, **model_extra):
    model = _make(NAMLSS, **model_extra)
    kw = {**_FAST_FIT, "checkpoint_path": tmp_path}
    if dist_kwargs:
        kw["distributional_kwargs"] = dist_kwargs
    model.fit(X, y, family=family, **kw)
    return model


# ===========================================================================
# 1. NAMRegressor
# ===========================================================================


class TestNAMRegressorBasics:
    def test_fit_predict_shape(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_returns_numpy_array(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        assert isinstance(model.predict(X), np.ndarray)

    def test_evaluate_default_mse(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        scores = model.evaluate(X, y)
        assert "Mean Squared Error" in scores
        assert np.isfinite(scores["Mean Squared Error"])

    def test_evaluate_custom_metrics(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        scores = model.evaluate(
            X, y, metrics={"MAE": mean_absolute_error, "R2": r2_score}
        )
        assert "MAE" in scores and "R2" in scores
        assert np.isfinite(scores["MAE"]) and np.isfinite(scores["R2"])

    def test_predict_feature_vals_has_output_key(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        d = model.predict_feature_vals(X)
        assert isinstance(d, dict)
        assert "output" in d

    def test_predict_before_fit_raises(self, reg_data):
        X, _ = reg_data
        with pytest.raises(ValueError, match="not been fitted"):
            _make(NAMRegressor).predict(X)

    def test_mixed_data(self, mixed_data, tmp_path):
        X, y = mixed_data
        model = NAMRegressor(
            layer_sizes=(4,),
            dropout=0.0,
            numerical_preprocessing="standardization",
            categorical_preprocessing="int",
            cat_cutoff=0.1,
        )
        _fit(model, X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    def test_explicit_validation_set(self, reg_data, tmp_path):
        X, y = reg_data
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = _make(NAMRegressor)
        model.fit(
            X_tr,
            y_tr,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=tmp_path,
            max_epochs=1,
            batch_size=8,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        assert model.predict(X_val).shape[0] == len(X_val)

    def test_raises_if_only_x_val_provided(self, reg_data, tmp_path):
        X, y = reg_data
        X_tr, X_val, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        with pytest.raises(ValueError, match="X_val and y_val must be provided together"):
            _make(NAMRegressor).fit(
                X_tr,
                y_tr,
                X_val=X_val,
                y_val=None,
                checkpoint_path=tmp_path,
                max_epochs=1,
                batch_size=8,
                limit_train_batches=1,
                limit_val_batches=1,
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )


class TestNAMRegressorConfigs:
    @pytest.mark.parametrize("skip_connections", [False, True])
    def test_skip_connections(self, reg_data, tmp_path, skip_connections):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, skip_connections=skip_connections), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    @pytest.mark.parametrize("batch_norm", [False, True])
    def test_batch_norm(self, reg_data, tmp_path, batch_norm):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, batch_norm=batch_norm), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    @pytest.mark.parametrize("layer_norm", [False, True])
    def test_layer_norm(self, reg_data, tmp_path, layer_norm):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, layer_norm=layer_norm), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    @pytest.mark.parametrize("use_glu", [False, True])
    def test_use_glu(self, reg_data, tmp_path, use_glu):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, use_glu=use_glu), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    def test_relu_activation(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, activation=nn.ReLU()), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    def test_selu_activation(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, activation=nn.SELU()), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    def test_larger_layer_sizes(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, layer_sizes=(16, 8)), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    def test_feature_dropout(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, feature_dropout=0.1), X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)

    @pytest.mark.parametrize("num_prep", ["standardization", "minmax"])
    def test_numerical_preprocessing(self, reg_data, tmp_path, num_prep):
        X, y = reg_data
        model = NAMRegressor(
            layer_sizes=(4,),
            dropout=0.0,
            numerical_preprocessing=num_prep,
            categorical_preprocessing="int",
            cat_cutoff=0.1,
        )
        _fit(model, X, y, tmp_path)
        assert model.predict(X).shape[0] == len(X)


class TestNAMRegressorInteractions:
    def test_no_interaction_produces_no_colon_keys(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, interaction_degree=None), X, y, tmp_path)
        d = model.predict_feature_vals(X)
        assert all(":" not in k for k in d)

    def test_interaction_degree_2_produces_colon_keys(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, interaction_degree=2), X, y, tmp_path)
        d = model.predict_feature_vals(X)
        interaction_keys = [k for k in d if ":" in k]
        assert len(interaction_keys) > 0, "Expected pairwise interaction keys"

    def test_interaction_predict_shape_matches(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, interaction_degree=2), X, y, tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X),)


class TestNAMRegressorPlotting:
    def test_plot_all_features(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)

    def test_plot_single_named_feature(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, feature_name="f1")
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, feature_name="f2")

    def test_plot_invalid_feature_raises(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor), X, y, tmp_path)
        with pytest.raises(ValueError, match="not found"):
            model.plot(X, y, feature_name="does_not_exist")

    def test_plot_with_interactions(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit(_make(NAMRegressor, interaction_degree=2), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, plot_interactions=True)


class TestNAMRegressorSklearnCompat:
    def test_get_params_does_not_mutate_config_keys(self):
        model = _make(NAMRegressor)
        original = set(model.config_kwargs.keys())
        p1 = model.get_params(deep=True)
        p2 = model.get_params(deep=True)
        assert set(model.config_kwargs.keys()) == original
        assert not any(k.startswith("preprocessor__") for k in model.config_kwargs)
        assert p1 == p2

    def test_set_params_updates_config(self):
        model = _make(NAMRegressor)
        model.set_params(dropout=0.3)
        assert model.config.dropout == pytest.approx(0.3)

    def test_set_params_preprocessor_prefix(self):
        model = _make(NAMRegressor)
        model.set_params(**{"preprocessor__numerical_preprocessing": "minmax"})
        # Should not raise; verifying no crash is sufficient


# ===========================================================================
# 2. NAMClassifier
# ===========================================================================


class TestNAMClassifierBasics:
    def test_fit_predict_binary(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_integer_dtype(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        assert model.predict(X).dtype.kind in {"i", "u"}

    def test_predict_proba_binary_shape_and_valid(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert (probs >= 0).all() and (probs <= 1).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_multiclass_shape_and_valid(self, multiclass_data, tmp_path):
        X, y = multiclass_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3)
        assert (probs >= 0).all() and (probs <= 1).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_accuracy(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        scores = model.evaluate(X, y)
        assert "Accuracy" in scores
        assert 0.0 <= scores["Accuracy"] <= 1.0

    def test_evaluate_auc(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        scores = model.evaluate(X, y, metrics={"AUC": (roc_auc_score, True)})
        assert "AUC" in scores
        assert 0.0 <= scores["AUC"] <= 1.0

    def test_predict_feature_vals_has_output_key(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        d = model.predict_feature_vals(X)
        assert isinstance(d, dict) and "output" in d

    def test_predict_before_fit_raises(self, cls_data):
        X, _ = cls_data
        with pytest.raises(ValueError, match="not been fitted"):
            _make(NAMClassifier).predict(X)

    def test_explicit_validation_set(self, cls_data, tmp_path):
        X, y = cls_data
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = _make(NAMClassifier)
        model.fit(
            X_tr,
            y_tr,
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=tmp_path,
            max_epochs=1,
            batch_size=8,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        assert model.predict(X_val).shape[0] == len(X_val)

    def test_raises_if_only_x_val_provided(self, cls_data, tmp_path):
        X, y = cls_data
        X_tr, X_val, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        with pytest.raises(ValueError, match="X_val and y_val must be provided together"):
            _make(NAMClassifier).fit(
                X_tr,
                y_tr,
                X_val=X_val,
                y_val=None,
                checkpoint_path=tmp_path,
                max_epochs=1,
                batch_size=8,
                limit_train_batches=1,
                limit_val_batches=1,
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )


class TestNAMClassifierInteractions:
    def test_no_interaction_produces_no_colon_keys(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier, interaction_degree=None), X, y, tmp_path)
        assert all(":" not in k for k in model.predict_feature_vals(X))

    def test_interaction_degree_2_produces_colon_keys(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier, interaction_degree=2), X, y, tmp_path)
        d = model.predict_feature_vals(X)
        assert len([k for k in d if ":" in k]) > 0


class TestNAMClassifierPlotting:
    def test_plot_binary_all_features(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)

    def test_plot_single_feature(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, feature_name="f1")

    def test_plot_multiclass(self, multiclass_data, tmp_path):
        X, y = multiclass_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)

    def test_plot_with_interactions(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier, interaction_degree=2), X, y, tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, plot_interactions=True)

    def test_plot_invalid_feature_raises(self, cls_data, tmp_path):
        X, y = cls_data
        model = _fit(_make(NAMClassifier), X, y, tmp_path)
        with pytest.raises(ValueError, match="not found"):
            model.plot(X, y, feature_name="nonexistent_xyz")


class TestNAMClassifierSklearnCompat:
    def test_get_params_does_not_mutate_config_keys(self):
        model = _make(NAMClassifier)
        original = set(model.config_kwargs.keys())
        p1 = model.get_params(deep=True)
        p2 = model.get_params(deep=True)
        assert set(model.config_kwargs.keys()) == original
        assert not any(k.startswith("preprocessor__") for k in model.config_kwargs)
        assert p1 == p2

    def test_set_params_updates_config(self):
        model = _make(NAMClassifier)
        model.set_params(dropout=0.25)
        assert model.config.dropout == pytest.approx(0.25)


# ===========================================================================
# 3. NAMLSS – all distribution families
# ===========================================================================


class TestNAMLSSNormal:
    def test_fit_predict_shape(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2), "Normal returns [mean, scale]"

    def test_predict_raw_shape(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        raw = model.predict(X, raw=True)
        assert raw.shape[0] == len(X)

    def test_predict_scale_positive(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        preds = model.predict(X)
        assert (preds[:, 1] > 0).all(), "Normal scale must be positive"

    def test_evaluate_contains_nll_mse_crps(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "MSE" in scores and np.isfinite(scores["MSE"])
        assert "CRPS" in scores and np.isfinite(scores["CRPS"])

    def test_plot(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)

    def test_plot_single_feature(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, feature_name="f1")


class TestNAMLSSRobustNormal:
    def test_fit_predict_shape(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "robustnormal", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2)

    def test_evaluate_nll_finite(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "robustnormal", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "MSE" in scores

    def test_plot(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "robustnormal", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSPoisson:
    def test_fit_predict_shape_and_positivity(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "poisson", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 1), "Poisson returns [rate]"
        assert (preds >= 0).all(), "Poisson rate must be non-negative"

    def test_evaluate_nll_and_deviance(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "poisson", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Poisson Deviance" in scores and np.isfinite(scores["Poisson Deviance"])

    def test_plot(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "poisson", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSGamma:
    def test_fit_predict_shape_and_positivity(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "gamma", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2), "Gamma returns [shape, rate]"
        assert (preds > 0).all(), "Gamma parameters must be positive"

    def test_evaluate_nll_and_deviance(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "gamma", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Gamma Deviance" in scores and np.isfinite(scores["Gamma Deviance"])

    def test_plot(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "gamma", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSBeta:
    def test_fit_predict_shape_and_positivity(self, beta_data, tmp_path):
        X, y = beta_data
        model = _fit_lss(X, y, "beta", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2), "Beta returns [alpha, beta]"
        assert (preds > 0).all(), "Beta concentration parameters must be positive"

    def test_evaluate_nll_and_mse(self, beta_data, tmp_path):
        X, y = beta_data
        model = _fit_lss(X, y, "beta", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Beta Mean MSE" in scores and np.isfinite(scores["Beta Mean MSE"])

    def test_plot(self, beta_data, tmp_path):
        X, y = beta_data
        model = _fit_lss(X, y, "beta", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSDirichlet:
    def test_fit_predict_shape(self, dirichlet_data, tmp_path):
        X, y = dirichlet_data
        model = _fit_lss(X, y, "dirichlet", tmp_path)
        preds = model.predict(X)
        assert preds.shape[0] == len(X)
        assert preds.shape[1] == 3, "Dirichlet with K=3 returns 3 concentrations"

    def test_infers_n_dim_from_y(self, dirichlet_data, tmp_path):
        X, y = dirichlet_data
        model = _fit_lss(X, y, "dirichlet", tmp_path)
        assert getattr(model.family, "n_dim", None) == 3

    def test_predict_concentrations_positive(self, dirichlet_data, tmp_path):
        X, y = dirichlet_data
        model = _fit_lss(X, y, "dirichlet", tmp_path)
        preds = model.predict(X)
        assert (preds > 0).all(), "Dirichlet concentrations must be positive"

    def test_evaluate_nll_and_error(self, dirichlet_data, tmp_path):
        X, y = dirichlet_data
        model = _fit_lss(X, y, "dirichlet", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Dirichlet Error" in scores and np.isfinite(scores["Dirichlet Error"])

    def test_plot(self, dirichlet_data, tmp_path):
        X, y = dirichlet_data
        model = _fit_lss(X, y, "dirichlet", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSStudentT:
    def test_fit_predict_shape(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "studentt", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 3), "StudentT returns [df, loc, scale]"

    def test_predict_df_and_scale_positive(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "studentt", tmp_path)
        preds = model.predict(X)
        assert (preds[:, 0] > 0).all(), "df must be positive"
        assert (preds[:, 2] > 0).all(), "scale must be positive"

    def test_evaluate_nll_and_studentt_nll(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "studentt", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Student-T NLL" in scores and np.isfinite(scores["Student-T NLL"])

    def test_plot(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "studentt", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSNegativeBinomial:
    def test_fit_predict_shape_and_positivity(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "negativebinom", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2), "NB returns [mean, dispersion]"
        assert (preds > 0).all(), "NB parameters must be positive"

    def test_evaluate_nll_and_deviance(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "negativebinom", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Negative Binomial Deviance" in scores

    def test_plot(self, count_data, tmp_path):
        X, y = count_data
        model = _fit_lss(X, y, "negativebinom", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSInverseGamma:
    def test_fit_predict_shape_and_positivity(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "inversegamma", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 2), "InverseGamma returns [shape, rate]"
        assert (preds > 0).all(), "InverseGamma parameters must be positive"

    def test_evaluate_nll_and_ig_nll(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "inversegamma", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Inverse Gamma NLL" in scores and np.isfinite(scores["Inverse Gamma NLL"])

    def test_plot(self, positive_data, tmp_path):
        X, y = positive_data
        model = _fit_lss(X, y, "inversegamma", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSCategorical:
    def test_fit_predict_shape_and_probabilities(self, cat_lss_data, tmp_path):
        X, y = cat_lss_data
        model = _fit_lss(X, y, "categorical", tmp_path)
        preds = model.predict(X)
        assert preds.shape == (len(X), 3), "Categorical(K=3) returns (n, 3) probs"
        assert (preds >= 0).all() and (preds <= 1).all()
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-4)

    def test_infers_num_classes(self, cat_lss_data, tmp_path):
        X, y = cat_lss_data
        model = _fit_lss(X, y, "categorical", tmp_path)
        assert getattr(model.family, "num_classes", None) == 3

    def test_evaluate_nll_and_accuracy(self, cat_lss_data, tmp_path):
        X, y = cat_lss_data
        model = _fit_lss(X, y, "categorical", tmp_path)
        scores = model.evaluate(X, y)
        assert "NLL" in scores and np.isfinite(scores["NLL"])
        assert "Accuracy" in scores and 0.0 <= scores["Accuracy"] <= 1.0

    def test_plot(self, cat_lss_data, tmp_path):
        X, y = cat_lss_data
        model = _fit_lss(X, y, "categorical", tmp_path)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSQuantile:
    def test_fit_predict_three_quantiles(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(
            X, y, "quantile", tmp_path, dist_kwargs={"quantiles": [0.1, 0.5, 0.9]}
        )
        preds = model.predict(X)
        assert preds.shape == (len(X), 3)

    def test_fit_predict_custom_quantile_count(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(
            X, y, "quantile", tmp_path, dist_kwargs={"quantiles": [0.25, 0.75]}
        )
        assert model.predict(X).shape == (len(X), 2)

    def test_monotone_quantiles_non_decreasing(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(
            X,
            y,
            "quantile",
            tmp_path,
            dist_kwargs={"quantiles": [0.1, 0.5, 0.9], "enforce_monotonic": True},
        )
        preds = model.predict(X)
        # q0.1 <= q0.5 <= q0.9 (allow tiny numerical slack)
        assert (preds[:, 0] <= preds[:, 1] + 1e-5).all()
        assert (preds[:, 1] <= preds[:, 2] + 1e-5).all()

    def test_evaluate_pinball_and_median_mae(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(
            X, y, "quantile", tmp_path, dist_kwargs={"quantiles": [0.25, 0.5, 0.75]}
        )
        scores = model.evaluate(X, y)
        assert "NLL" in scores
        assert "Pinball Loss" in scores and np.isfinite(scores["Pinball Loss"])
        assert "Median MAE" in scores and np.isfinite(scores["Median MAE"])

    def test_plot(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(
            X, y, "quantile", tmp_path, dist_kwargs={"quantiles": [0.25, 0.5, 0.75]}
        )
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y)


class TestNAMLSSUnsupportedFamily:
    def test_unsupported_family_raises_value_error(self, reg_data, tmp_path):
        X, y = reg_data
        with pytest.raises(ValueError, match="Unsupported family"):
            _fit_lss(X, y, "not_a_real_distribution", tmp_path)


class TestNAMLSSInteractions:
    def test_no_interaction(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path, interaction_degree=None)
        assert all(":" not in k for k in model.predict_feature_vals(X))

    def test_interaction_degree_2(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path, interaction_degree=2)
        d = model.predict_feature_vals(X)
        assert len([k for k in d if ":" in k]) > 0

    def test_lss_interaction_plot(self, reg_data, tmp_path):
        X, y = reg_data
        model = _fit_lss(X, y, "normal", tmp_path, interaction_degree=2)
        with patch("matplotlib.pyplot.show"):
            model.plot(X, y, plot_interactions=True)


class TestNAMLSSSklearnCompat:
    def test_get_params_does_not_mutate_config_keys(self):
        model = _make(NAMLSS)
        original = set(model.config_kwargs.keys())
        p1 = model.get_params(deep=True)
        p2 = model.get_params(deep=True)
        assert set(model.config_kwargs.keys()) == original
        assert not any(k.startswith("preprocessor__") for k in model.config_kwargs)
        assert p1 == p2

    def test_explicit_validation_set(self, reg_data, tmp_path):
        X, y = reg_data
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = _make(NAMLSS)
        model.fit(
            X_tr,
            y_tr,
            family="normal",
            X_val=X_val,
            y_val=y_val,
            checkpoint_path=tmp_path,
            max_epochs=1,
            batch_size=8,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        preds = model.predict(X_val)
        assert preds.shape[0] == len(X_val)

    def test_raises_if_only_x_val_provided(self, reg_data, tmp_path):
        X, y = reg_data
        X_tr, X_val, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        with pytest.raises(ValueError, match="X_val and y_val must be provided together"):
            _make(NAMLSS).fit(
                X_tr,
                y_tr,
                family="normal",
                X_val=X_val,
                y_val=None,
                checkpoint_path=tmp_path,
                max_epochs=1,
                batch_size=8,
                limit_train_batches=1,
                limit_val_batches=1,
                logger=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )

    def test_predict_before_fit_raises(self, reg_data):
        X, _ = reg_data
        with pytest.raises(ValueError, match="not been fitted"):
            _make(NAMLSS).predict(X)

    def test_set_params_updates_config(self):
        model = _make(NAMLSS)
        model.set_params(dropout=0.2)
        assert model.config.dropout == pytest.approx(0.2)
