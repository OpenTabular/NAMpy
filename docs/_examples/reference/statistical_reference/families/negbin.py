"""Runnable reference example: statistical_reference/families/negbin."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = rng.negative_binomial(2, 2 / (2 + np.exp(data.x)))
model = GAM(formula="y ~ s(x, bs='cr', k=6)", family={'name': 'negbin', 'theta': 2.0}, optimize_smoothing=False).fit(data=data)
prediction = model.predict(data, type='response')
raw = model.predict(data, type='link')
result = {'response_prediction': np.asarray(prediction), 'link_prediction': np.asarray(raw)}
print({key: value.shape for key, value in result.items()})
