"""Runnable reference example: neural_reference/estimators/SPAMClassifier."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
from nampy.models import SPAMClassifier

torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
y = (np.arange(n) % 2).astype(int)
model = SPAMClassifier(n_knots=6)
with TemporaryDirectory() as directory:
    model.fit(X, y, max_epochs=2, batch_size=32, random_state=7,
              checkpoint_path=directory, logger=False, enable_progress_bar=False,
              enable_model_summary=False)
prediction = model.predict(X)
result = {'prediction': prediction}
probability = model.predict_proba(X)
result['probabilities'] = probability
components = model.predict_components(X)
components.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)
result['components'] = components
print({key: type(value).__name__ for key, value in result.items()})
