"""Runnable reference example: api/generated/nampy.plotting.render_distributional_effects."""

import numpy as np
from nampy.inspection import DistributionalEffects

n = 24
x = np.linspace(0, 1, n)
effects = DistributionalEffects(feature_values={'x': x},
                               point_effects={'x': np.sin(3*x)},
                               conditional_sd={'x': 0.1 + 0.2*x})
from nampy.plotting import render_distributional_effects
result = render_distributional_effects(effects, density_bins=8)
print(type(result).__name__)
