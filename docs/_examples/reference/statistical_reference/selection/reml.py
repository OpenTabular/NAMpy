"""Runnable reference example: statistical_reference/selection/reml."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='cr', k=6)", smoothing_method='reml', optimize_smoothing=True).fit(data=data)
result = np.asarray(model.smoothing_params)
print(result.shape)
