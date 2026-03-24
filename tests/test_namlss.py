"""Focused smoke tests for NAMLSS (distributional regression with NAM back-end)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from nampy.models import NAMLSS

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

_BASE = dict(
    layer_sizes=(4,),
    dropout=0.0,
    numerical_preprocessing="standardization",
    categorical_preprocessing="int",
    cat_cutoff=0.1,
)


def test_namlss_normal_fit_predict_evaluate(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(size=50), "f2": rng.uniform(-1, 1, size=50)})
    y = 0.5 * X["f1"].to_numpy() - 0.2 * X["f2"].to_numpy() + rng.normal(
        scale=0.1, size=50
    )

    model = NAMLSS(**_BASE)
    model.fit(X, y, family="normal", checkpoint_path=tmp_path, **_FAST_FIT)

    preds = model.predict(X)
    assert preds.shape == (len(X), 2)
    assert np.isfinite(preds).all()
    assert (preds[:, 1] > 0).all(), "normal scale must be positive"

    scores = model.evaluate(X, y)
    assert "NLL" in scores and np.isfinite(scores["NLL"])


def test_namlss_predict_before_fit_raises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=10)})
    model = NAMLSS(**_BASE)
    with pytest.raises(ValueError, match="not been fitted"):
        model.predict(X)
