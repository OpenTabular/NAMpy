"""Runnable reference example: neural_reference/methods/training_history."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
from nampy.models import IGANNRegressor

torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
y = np.sin(3 * X.x.to_numpy()) + rng.normal(0, 0.1, n)
from nampy.neural.configs.igann_config import DefaultIGANNConfig
from nampy.neural.configuration import IGANNTrainingConfig
model = IGANNRegressor(architecture_config=DefaultIGANNConfig(n_estimators=2), training=IGANNTrainingConfig(batch_size=32, random_state=7))
with TemporaryDirectory() as directory:
    model.fit(X, y)
prediction = model.predict(X)
result = {'prediction': prediction}
components = model.predict_components(X)
components.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)
result['components'] = components
print({key: type(value).__name__ for key, value in result.items()})
result = model.training_history()
print(type(result).__name__)
