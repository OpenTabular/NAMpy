"""Runnable reference example: statistical_reference/smooths/sf."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
boundary = [{'x': [0., 1., 1., 0., 0.], 'z': [0., 0., 1., 1., 0.]}]
formula = "y ~ s(x, z, bs='sf', k=8, xt={'bnd': " + repr(boundary) + ", 'nmax': 25})"
model = GAM(formula=formula, optimize_smoothing=False, knots={'x': [0.3, 0.7, 0.3, 0.7], 'z': [0.3, 0.3, 0.7, 0.7]}).fit(data=data)
prediction = model.predict(data)
design = model.lpmatrix(data)
covariance = model.vcov()
result = {'prediction': prediction, 'full_model_design': design,
          'coefficient_covariance': covariance}
print({key: value.shape for key, value in result.items()})
