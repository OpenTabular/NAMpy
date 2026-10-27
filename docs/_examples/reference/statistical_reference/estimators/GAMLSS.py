"""Runnable reference example: statistical_reference/estimators/GAMLSS."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
from nampy.models import GAMLSS
model = GAMLSS(family='gaulss', optimize_smoothing=False, k=6).fit(data[['x', 'z']], data.y)
result = {'natural_parameters': model.predict(data[['x', 'z']]), 'raw': model.predict(data[['x', 'z']], raw=True), 'point': model.predict_point(data[['x', 'z']])}
print({key: type(value).__name__ for key, value in result.items()})
