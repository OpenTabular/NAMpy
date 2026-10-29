"""Fitted PLE coordinates depend on learned intervals, never query batches."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pretab.preprocessor import Preprocessor
from pretab.transformers.ple.ple import PLETransformer

from nampy import GradientTrainingConfig, LinRegRegressor
from nampy.neural.data.datamodule import NAMpyDataModule
from nampy.neural.data.preprocessing import _FittedPiecewiseLinearEncoder


def test_fitted_ple_preserves_reference_boundaries_and_extrapolation():
    X = np.linspace(-3, 5, 80).reshape(-1, 1)
    original = PLETransformer(n_bins=5).fit(X, np.sin(X[:, 0]))
    stable = _FittedPiecewiseLinearEncoder.from_fitted(original)
    cuts = np.asarray(stable.knots[0])
    probe = np.sort(
        np.concatenate(([cuts[0] - 2, cuts[-1] + 2], cuts, (cuts[1:] + cuts[:-1]) / 2))
    ).reshape(-1, 1)
    reference = original.transform(probe)
    np.testing.assert_array_equal(stable.transform(probe), reference)
    np.testing.assert_array_equal(
        np.vstack([stable.transform(row.reshape(1, 1)) for row in probe]), reference
    )
    # Constant query batches, including exact boundaries, retain fitted widths.
    for row, expected in zip(probe, reference, strict=True):
        np.testing.assert_array_equal(
            stable.transform(np.repeat(row.reshape(1, 1), 3, axis=0)),
            np.repeat(expected.reshape(1, -1), 3, axis=0),
        )


def test_fitted_ple_handles_single_cut_and_rejects_missing_cuts():
    stable = _FittedPiecewiseLinearEncoder.from_fitted(
        SimpleNamespace(
            conditions_=[["(feature <= 2.0)", "(feature > 2.0)"]],
            pattern=r"-?\d+\.?\d*",
        )
    )
    np.testing.assert_array_equal(
        stable.transform([[1], [2], [3]]), [[1, 0], [2, 0], [1, 3]]
    )
    with pytest.raises(ValueError, match="finite split thresholds"):
        _FittedPiecewiseLinearEncoder.from_fitted(
            SimpleNamespace(conditions_=[["True"]], pattern=r"-?\d+\.?\d*")
        )
    with pytest.raises(ValueError, match="fitted feature width"):
        stable.transform([[1, 2]])


def test_actual_preprocessing_is_invariant_to_query_partition_and_order():
    X = pd.DataFrame(
        {"x": np.linspace(-3, 5, 80), "constant": pd.Categorical(["same"] * 80)}
    )
    y = np.sin(X.x.to_numpy())
    data = NAMpyDataModule(
        preprocessor=Preprocessor(numerical_preprocessing="ple"),
        batch_size=8,
        shuffle=False,
        regression=True,
    )
    data.setup_data(X, y, X_val=X.iloc[:8], y_val=y[:8])
    expected = data.preprocessor.transform(X)
    permutation = np.random.default_rng(7).permutation(len(X))
    permuted = data.preprocessor.transform(X.iloc[permutation])
    chunks = [
        data.preprocessor.transform(X.iloc[start : start + 3])
        for start in range(0, len(X), 3)
    ]
    for key, reference in expected.items():
        np.testing.assert_array_equal(permuted[key][np.argsort(permutation)], reference)
        np.testing.assert_array_equal(
            np.concatenate([chunk[key] for chunk in chunks]), reference
        )


def test_ple_predictions_survive_batching_checkpoints_and_persistence(tmp_path):
    X = pd.DataFrame({"x": np.linspace(-3, 5, 40)})
    y = np.sin(X.x.to_numpy())
    estimator = LinRegRegressor(
        training=GradientTrainingConfig(
            max_epochs=2,
            batch_size=8,
            random_state=7,
            checkpoint_path=str(tmp_path / "checkpoints"),
            trainer_kwargs={
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )
    ).fit(X, y)
    expected = estimator.predict(X)
    np.testing.assert_array_equal(estimator.predict(X, batch_size=3), expected)
    permutation = np.random.default_rng(7).permutation(len(X))
    np.testing.assert_array_equal(
        estimator.predict(X.iloc[permutation], batch_size=3)[np.argsort(permutation)],
        expected,
    )
    restored = LinRegRegressor.load_model(estimator.save_model(tmp_path / "ple.nampy"))
    np.testing.assert_array_equal(restored.predict(X, batch_size=3), expected)
