"""Runnable reference example: neural_reference/estimators/NodeGAMRegressor."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
from nampy.models import NodeGAMRegressor

torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
y = np.sin(3 * X.x.to_numpy()) + rng.normal(0, 0.1, n)
from nampy.neural.configs.nodegam_config import DefaultNodeGAMConfig
from nampy.neural.configuration import GradientTrainingConfig
model = NodeGAMRegressor(architecture_config=DefaultNodeGAMConfig(num_trees=3, depth=2, num_layers=1, anneal_steps=1, last_dropout=0.0, interaction_degree=1), training=GradientTrainingConfig(batch_size=32, random_state=7, max_epochs=2, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False}))
with TemporaryDirectory() as directory:
    model.set_params(training__checkpoint_path=directory)
    model.fit(X, y)
prediction = model.predict(X)
result = {'prediction': prediction}
components = model.predict_components(X)
components.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)
result['components'] = components
print({key: type(value).__name__ for key, value in result.items()})
