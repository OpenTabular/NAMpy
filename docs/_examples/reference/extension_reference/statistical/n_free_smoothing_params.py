"""Runnable reference example: extension_reference/statistical/n_free_smoothing_params."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)
from nampy.gam.fit.selection import n_free_smoothing_params
partly_fixed = GAM(formula="y ~ s(x, bs='cr', k=6, sp=0.2) + s(z, bs='cr', k=6)", optimize_smoothing=False).fit(data=data)
result = {'all_free': n_free_smoothing_params(model), 'partly_fixed': n_free_smoothing_params(partly_fixed), 'fixed_mask': partly_fixed.smoothing_fixed_mask_}
