"""Runnable reference example: extension_reference/statistical/optimizer_endpoint_diagnostics."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)
from nampy.gam.fit.selection import optimizer_endpoint_diagnostics
diagnostic = optimizer_endpoint_diagnostics(model)
result = {'projected_gradient': np.asarray(diagnostic['projected_gradient']), 'stationary': diagnostic['stationary_by_projected_gradient'], 'flat_ridge_suspected': diagnostic['flat_ridge_suspected'], 'free_parameters': diagnostic['n_free_smoothing_params']}
