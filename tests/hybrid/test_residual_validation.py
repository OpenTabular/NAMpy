"""GAMResidual explicit validation sets: offsets and data hygiene."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.hybrid import GAMResidualRegressor
from nampy.models.linreg import LinRegRegressor

_KW = {
    "max_epochs": 3,
    "patience": 3,
    "batch_size": 64,
    "logger": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


def _split_frames(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"x0": rng.uniform(size=n), "x3": rng.normal(size=n)})
    data["y"] = (
        np.sin(3.0 * data["x0"]) + data["x3"]
        + rng.normal(scale=0.1, size=n)
    )
    train = data.iloc[: n - 50].reset_index(drop=True)
    val = data.iloc[n - 50 :].reset_index(drop=True)
    return train, val


def _fit(train, val, tmp_path):
    estimator = GAMResidualRegressor(
        "y ~ s(x0, k=5)",
        LinRegRegressor(numerical_preprocessing="standardization"),
    )
    estimator.fit(
        train,
        neural_features=["x3"],
        val_data=val,
        neural_fit_kwargs=dict(_KW, checkpoint_path=str(tmp_path)),
    )
    return estimator


def test_validation_offset_is_gam_link_on_val_rows(tmp_path):
    train, val = _split_frames()
    estimator = _fit(train, val, tmp_path)

    np.testing.assert_allclose(
        np.asarray(estimator.neural_.data_module.offset_val, dtype=float),
        np.asarray(estimator.gam_.predict(val, type="link"), dtype=float),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(estimator.neural_.data_module.y_val),
        val["y"].to_numpy(),
    )


def test_preprocessor_fit_on_training_rows_only(tmp_path):
    train, val = _split_frames()
    benign = _fit(train, val, tmp_path)
    shifted = _fit(train, val.assign(x3=val["x3"] * 100.0), tmp_path)

    probe = train[["x3"]].head(20)
    benign_out = benign.neural_.data_module.preprocessor.transform(probe)
    shifted_out = shifted.neural_.data_module.preprocessor.transform(probe)
    for key in benign_out:
        np.testing.assert_array_equal(benign_out[key], shifted_out[key])


def test_val_data_missing_response_column_raises(tmp_path):
    train, val = _split_frames(n=120)
    estimator = GAMResidualRegressor(
        "y ~ s(x0, k=5)",
        LinRegRegressor(numerical_preprocessing="standardization"),
    )
    with pytest.raises(ValueError, match="response column"):
        estimator.fit(
            train,
            neural_features=["x3"],
            val_data=val.drop(columns=["y"]),
            neural_fit_kwargs=dict(_KW, checkpoint_path=str(tmp_path)),
        )
