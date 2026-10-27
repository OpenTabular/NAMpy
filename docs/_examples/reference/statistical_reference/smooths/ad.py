"""Runnable reference example: statistical_reference/smooths/ad."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ad', k=12)", optimize_smoothing=False).fit(data=data)
prediction = model.predict(data)
design = model.lpmatrix(data)
covariance = model.vcov()
assert prediction.shape == (n,)
assert design.shape[0] == n
assert covariance.shape == (design.shape[1], design.shape[1])
assert np.isfinite(prediction).all()
result = {'prediction': prediction, 'full_model_design': design,
          'coefficient_covariance': covariance}
print({key: value.shape for key, value in result.items()})
