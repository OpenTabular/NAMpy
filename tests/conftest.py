import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seeds():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.uniform(-1, 1, size=40),
        }
    )
    y = (
        0.5 * X["f1"].to_numpy()
        - 0.2 * X["f2"].to_numpy()
        + rng.normal(scale=0.1, size=40)
    )
    return X, y


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=40),
            "f2": rng.uniform(-1, 1, size=40),
        }
    )
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    return X, y


@pytest.fixture
def mixed_data():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(
        {
            "num1": rng.normal(size=40),
            "num2": rng.uniform(-1, 1, size=40),
            "int_cat": rng.integers(0, 3, size=40),
            "str_cat": rng.choice(["a", "b", "c"], size=40),
        }
    )
    y = (X["num1"] * 0.3 + rng.normal(scale=0.1, size=40)).to_numpy()
    return X, y
