"""Runnable reference example: statistical_reference/extensions/linear_functional_interval."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)

from nampy.gam.extensions import linear_functional_interval
left = data.copy()
right = data.copy()
left['x'] = .2
right['x'] = .8
contrast = model.lpmatrix(right) - model.lpmatrix(left)
result = linear_functional_interval(model, contrast, n_draws=2000, seed=4)
