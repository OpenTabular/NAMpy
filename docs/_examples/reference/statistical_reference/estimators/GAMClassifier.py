"""Runnable reference example: statistical_reference/estimators/GAMClassifier."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
from nampy.models import GAMClassifier
data['y'] = (np.arange(n) % 2).astype(int)
model = GAMClassifier(optimize_smoothing=False, k=6).fit(data[['x', 'z']], data.y)
result = {'prediction': model.predict(data[['x', 'z']]), 'components': model.predict_components(data[['x', 'z']])}
result['probabilities'] = model.predict_proba(data[['x', 'z']])
print({key: type(value).__name__ for key, value in result.items()})
