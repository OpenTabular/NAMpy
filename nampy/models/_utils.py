from __future__ import annotations

from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint


def coerce_feature_frame(X, feature_names=None) -> pd.DataFrame:
    """Return X as a DataFrame with stable string feature names."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        frame.columns = [str(column) for column in frame.columns]
    else:
        frame = pd.DataFrame(X)
        if feature_names is None:
            frame.columns = [f"feature_{idx}" for idx in range(frame.shape[1])]
        else:
            if frame.shape[1] != len(feature_names):
                raise ValueError(
                    "X has a different number of features than the fitted data: "
                    f"got {frame.shape[1]}, expected {len(feature_names)}."
                )
            frame.columns = list(feature_names)

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Feature names must be unique. Duplicates: {duplicates}.")

    if feature_names is not None:
        feature_names = list(feature_names)
        missing = [name for name in feature_names if name not in frame.columns]
        extra = [name for name in frame.columns if name not in feature_names]
        if missing or extra:
            raise ValueError(
                "X feature names do not match the fitted data. "
                f"Missing: {missing}; extra: {extra}."
            )
        frame = frame.loc[:, feature_names]

    return frame


def prepare_fit_frames(estimator, X, X_val=None):
    X = coerce_feature_frame(X)
    estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
    estimator.n_features_in_ = X.shape[1]

    if X_val is not None:
        X_val = coerce_feature_frame(X_val, estimator.feature_names_in_)

    return X, X_val


def prepare_predict_frame(estimator, X) -> pd.DataFrame:
    feature_names = getattr(estimator, "feature_names_in_", None)
    return coerce_feature_frame(X, feature_names)


@contextmanager
def checkpoint_callback_context(
    checkpoint_path,
    monitor: str,
    mode: str,
    trainer_kwargs: dict,
) -> Iterator[Optional[ModelCheckpoint]]:
    """Create a scoped checkpoint callback or disable checkpointing."""
    if checkpoint_path is False or trainer_kwargs.get("enable_checkpointing") is False:
        trainer_kwargs["enable_checkpointing"] = False
        yield None
        return

    def make_callback(path):
        return ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            dirpath=path,
            filename="best_model",
        )

    if checkpoint_path is None:
        with TemporaryDirectory(prefix="nampy-checkpoints-") as tmpdir:
            yield make_callback(tmpdir)
    else:
        yield make_callback(checkpoint_path)
