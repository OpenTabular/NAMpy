"""Runnable reference example: api/generated/nampy.inspection.DistributionalEffects."""

import numpy as np
from nampy.inspection import DistributionalEffects

n = 24
x = np.linspace(0, 1, n)
effects = DistributionalEffects(feature_values={'x': x},
                               point_effects={'x': np.sin(3*x)},
                               conditional_sd={'x': 0.1 + 0.2*x})
from nampy.inspection import DistributionalEffects
result = effects
print(type(result).__name__)
