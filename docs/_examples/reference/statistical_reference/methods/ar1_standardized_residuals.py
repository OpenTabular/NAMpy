"""Runnable reference example: statistical_reference/methods/ar1_standardized_residuals."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            ar1_rho=0.3, optimize_smoothing=False, smoothing_method='reml').fit(data=data)
result = model.ar1_standardized_residuals()
print(type(result).__name__)
