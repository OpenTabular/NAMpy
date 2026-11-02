"""Runnable reference example: api/generated/nampy.diagnostics.centile_calibration."""

import numpy as np
from nampy.distributions import DistributionPrediction, NormalLaw

n = 24
x = np.linspace(0, 1, n)
distribution = DistributionPrediction(
    NormalLaw(), {'loc': 0.4 + np.sin(3 * x), 'scale': np.full(n, 0.2)})
y = 0.4 + np.sin(3 * x) + 0.05 * np.cos(np.arange(n))
from nampy.diagnostics import centile_calibration

result = centile_calibration(distribution, y, probs=(0.05, 0.5, 0.95))
print(type(result).__name__)
