"""Runnable reference example: statistical_reference/families/gammals."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = rng.gamma(3, np.exp(0.3 * data.x) / 3)
model = GAM(formula=["y ~ s(x, bs='cr', k=6)", '~ 1'], family='gammals', optimize_smoothing=False).fit(data=data)
raw = model.predict(data, type='link')
parameters = model.family.distribution_parameters_from_eta(np.asarray(raw))
source_response = np.asarray(model.predict(data, type='response'))
result = {'link_predictors': np.asarray(raw), 'natural_parameters': parameters, 'source_response': source_response}
print({key: value.shape for key, value in result.items()})
