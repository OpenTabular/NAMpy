"""Runnable reference example: neural_reference/estimators/SNAMClassifier."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
from nampy.models import SNAMClassifier

torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
y = (np.arange(n) % 2).astype(int)
from nampy.neural.configs.snam_config import DefaultSNAMConfig
from nampy.neural.configuration import GradientTrainingConfig
model = SNAMClassifier(architecture_config=DefaultSNAMConfig(layer_sizes=[12, 8], interactions=[]), training=GradientTrainingConfig(batch_size=32, random_state=7, max_epochs=2, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False}))
with TemporaryDirectory() as directory:
    model.set_params(training__checkpoint_path=directory)
    model.fit(X, y)
prediction = model.predict(X)
result = {'prediction': prediction}
probability = model.predict_proba(X)
result['probabilities'] = probability
components = model.predict_components(X)
components.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)
result['components'] = components
print({key: type(value).__name__ for key, value in result.items()})
