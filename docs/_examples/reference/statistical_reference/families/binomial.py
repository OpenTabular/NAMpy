"""Runnable reference example: statistical_reference/families/binomial."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = rng.binomial(1, 1 / (1 + np.exp(-2 * (data.x - 0.5))))
model = GAM(formula="y ~ s(x, bs='cr', k=6)", family='binomial', optimize_smoothing=False).fit(data=data)
prediction = model.predict(data, type='response')
raw = model.predict(data, type='link')
result = {'response_prediction': np.asarray(prediction), 'link_prediction': np.asarray(raw)}
print({key: value.shape for key, value in result.items()})
