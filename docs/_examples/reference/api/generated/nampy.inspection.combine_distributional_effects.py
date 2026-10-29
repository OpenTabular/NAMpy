"""Runnable reference example: api/generated/nampy.inspection.combine_distributional_effects."""

import numpy as np
from nampy.inspection import DistributionalEffects

n = 24
x = np.linspace(0, 1, n)
effects = DistributionalEffects(feature_values={'x': x},
                               point_effects={'x': np.sin(3*x)},
                               conditional_sd={'x': 0.1 + 0.2*x})
from nampy.inspection import combine_distributional_effects
result = combine_distributional_effects([effects, effects])
print(type(result).__name__)
