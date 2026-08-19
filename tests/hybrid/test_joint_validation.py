"""GAMNet explicit validation sets: the design rides the passthrough_val channel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nampy.hybrid import GAMNetRegressor
from nampy.hybrid.net import GAM_DESIGN_KEY
from nampy.neural.configs.linreg_config import DefaultLinRegConfig
from nampy.neural.modules.linreg import LinReg

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


def _regressor():
    return GAMNetRegressor(
        "y ~ s(x0, k=6)",
        LinReg,
        DefaultLinRegConfig,
        lam=[0.5],
        numerical_preprocessing="standardization",
    )


def test_val_design_equals_compiled_design_on_val_rows(tmp_path):
    train, val = _split_frames()
    estimator = _regressor()
    estimator.fit(
        train,
        neural_features=["x3"],
        val_data=val,
        checkpoint_path=str(tmp_path),
        **_KW,
    )

    val_dataset = estimator.data_module.val_dataset
    stacked = torch.stack(
        [val_dataset[i][1][GAM_DESIGN_KEY] for i in range(len(val_dataset))]
    ).numpy()
    np.testing.assert_allclose(
        stacked,
        estimator.gam_terms_.design(val).astype(np.float32),
        atol=1e-6,
    )
    # The validation batches carry no offsets (zeros).
    _, _, _, offset = val_dataset[0]
    assert float(offset.abs().sum()) == 0.0
    # Validation targets come from the formula response column.
    np.testing.assert_array_equal(
        np.asarray(estimator.data_module.y_val), val["y"].to_numpy()
    )


def test_x_val_rejected_in_favor_of_val_data(tmp_path):
    train, val = _split_frames(n=100)
    estimator = _regressor()
    with pytest.raises(ValueError, match="val_data"):
        estimator.fit(
            train,
            neural_features=["x3"],
            X_val=val[["x3"]],
            checkpoint_path=str(tmp_path),
            **_KW,
        )
