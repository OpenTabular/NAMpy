"""Runnable reference example: neural_reference/methods/local_term_importance."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
from nampy.models import SPAMRegressor

torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
y = np.sin(3 * X.x.to_numpy()) + rng.normal(0, 0.1, n)
model = SPAMRegressor(n_knots=6)
with TemporaryDirectory() as directory:
    model.fit(X, y, max_epochs=2, batch_size=32, random_state=7,
              checkpoint_path=directory, logger=False, enable_progress_bar=False,
              enable_model_summary=False)
prediction = model.predict(X)
assert np.asarray(prediction).shape[0] == n
assert np.isfinite(prediction).all()
result = {'prediction': prediction}
components = model.predict_components(X)
components.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)
result['components'] = components
print({key: type(value).__name__ for key, value in result.items()})
result = model.local_term_importance(X, target=0, top_k=3)
print(type(result).__name__)
