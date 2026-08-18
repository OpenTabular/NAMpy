"""Data-handling contracts: no validation leakage, stratified auto-split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pretab.preprocessor import Preprocessor

from nampy.neural.data.datamodule import NAMpyDataModule


def _make_frames(seed=0):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(
        {"x": rng.normal(size=120), "z": rng.normal(size=120)}
    )
    y_train = (
        1.5 * X_train["x"].to_numpy()
        - 0.5 * X_train["z"].to_numpy()
        + rng.normal(scale=0.1, size=120)
    )
    X_val = pd.DataFrame({"x": rng.normal(size=40), "z": rng.normal(size=40)})
    y_val = rng.normal(size=40)
    return X_train, y_train, X_val, y_val


def _fit_datamodule(X_train, y_train, X_val, y_val):
    data_module = NAMpyDataModule(
        preprocessor=Preprocessor(numerical_preprocessing="ple"),
        batch_size=32,
        shuffle=False,
        regression=True,
    )
    data_module.setup_data(X_train, y_train, X_val=X_val, y_val=y_val)
    return data_module


def test_preprocessor_state_is_independent_of_validation_data():
    X_train, y_train, X_val, y_val = _make_frames()

    benign = _fit_datamodule(X_train, y_train, X_val, y_val)
    # Same training data, radically different validation data: fitted
    # preprocessor statistics must not change.
    shifted = _fit_datamodule(X_train, y_train, X_val * 100.0, y_val * 100.0)

    probe = X_train.head(20)
    benign_out = benign.preprocessor.transform(probe)
    shifted_out = shifted.preprocessor.transform(probe)

    assert benign_out.keys() == shifted_out.keys()
    for key in benign_out:
        np.testing.assert_array_equal(benign_out[key], shifted_out[key])


def test_out_of_range_validation_values_transform_without_error():
    X_train, y_train, X_val, y_val = _make_frames()
    data_module = _fit_datamodule(X_train, y_train, X_val * 100.0, y_val)
    data_module.setup("fit")
    assert len(data_module.val_dataset) == len(X_val)


def test_stratified_auto_split_preserves_class_ratio():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200)})
    y = np.array([0] * 160 + [1] * 40)

    data_module = NAMpyDataModule(
        preprocessor=Preprocessor(numerical_preprocessing="standardization"),
        batch_size=32,
        shuffle=False,
        regression=False,
    )
    data_module.setup_data(X, y, val_size=0.25, stratify=y)

    train_ratio = float(np.mean(np.asarray(data_module.y_train)))
    val_ratio = float(np.mean(np.asarray(data_module.y_val)))
    assert abs(train_ratio - 0.2) < 0.02
    assert abs(val_ratio - 0.2) < 0.02
