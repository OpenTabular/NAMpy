"""Runnable reference example: api/generated/nampy.diagnostics.prepare_worm_plot."""

import numpy as np
from nampy.distributions import DistributionPrediction, NormalLaw

n = 24
x = np.linspace(0, 1, n)
distribution = DistributionPrediction(
    NormalLaw(), {'loc': 0.4 + np.sin(3 * x), 'scale': np.full(n, 0.2)})
y = 0.4 + np.sin(3 * x) + 0.05 * np.cos(np.arange(n))
from nampy.diagnostics import prepare_worm_plot, quantile_residuals

residuals = quantile_residuals(distribution, y, random_state=7)
result = prepare_worm_plot(residuals, x=x, n_bins=4)
print(type(result).__name__)
