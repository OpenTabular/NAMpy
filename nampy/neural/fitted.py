"""Inference state without a Lightning trainer or training observations."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn


def neural_artifact_state(estimator):
    """Persist prediction state and configuration without execution objects."""
    from .configuration import GradientTrainingConfig

    state = dict(estimator.__dict__)
    for name in ("training", "training_config_"):
        configuration = state.get(name)
        if not isinstance(configuration, GradientTrainingConfig):
            continue
        options = dict(configuration.trainer_kwargs)
        options.pop("callbacks", None)
        for key in ("logger", "profiler", "plugins", "strategy"):
            value = options.get(key)
            if value is not None and not isinstance(value, (str, bool, int, float)):
                options.pop(key)
        state[name] = replace(configuration, trainer_kwargs=options)
    return state


def preprocess_tensors(preprocessor, cat_feature_info, num_feature_info, X):
    """Apply the fitted schema without fitting or retaining input batches."""
    transformed = preprocessor.transform(X)
    categorical, numerical = {}, {}

    def dense(value):
        return np.asarray(value.toarray() if hasattr(value, "toarray") else value)

    for key, info in cat_feature_info.items():
        source = "cat_" + key
        if source not in transformed:
            continue
        array = dense(transformed[source])
        onehot = (
            "onehot" in info.get("preprocessing", "").lower()
            or info.get("dimension", 1) > 1
        )
        if not onehot and array.dtype.kind == "f":
            array = array.astype("int64")
        categorical[key] = torch.tensor(
            array, dtype=torch.float32 if onehot else torch.long
        )
    for key in num_feature_info:
        source = "num_" + key
        if source in transformed:
            numerical[key] = torch.tensor(
                dense(transformed[source]), dtype=torch.float32
            )
    return categorical, numerical


@dataclass(frozen=True)
class FittedPreprocessing:
    preprocessor: object
    cat_feature_info: dict
    num_feature_info: dict

    def preprocess_tensors(self, X):
        return preprocess_tensors(
            self.preprocessor, self.cat_feature_info, self.num_feature_info, X
        )


class NeuralPredictionModule(nn.Module):
    """The fitted architecture and objective, independent of training machinery."""

    def __init__(self, architecture, objective):
        super().__init__()
        self.model = architecture
        self.objective = objective

    def forward(self, num_features, cat_features):
        return self.model(num_features=num_features, cat_features=cat_features)
