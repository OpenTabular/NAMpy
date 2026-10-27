"""Runnable reference example: statistical_reference/families/clog."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 2 + data.x + rng.logistic(0, 0.3, n)
data['bound'] = data.y.astype(float)
data.loc[:5, 'bound'] = np.inf
model = GAM(formula="cbind(y, bound) ~ s(x, bs='cr', k=6)", family={'name': 'clog', 'theta': 0.3}, optimize_smoothing=False).fit(data=data)
prediction = model.predict(data, type='response')
raw = model.predict(data, type='link')
assert np.asarray(prediction).shape[0] == n
assert np.isfinite(prediction).all()
result = {'response_prediction': np.asarray(prediction), 'link_prediction': np.asarray(raw)}
print({key: value.shape for key, value in result.items()})
