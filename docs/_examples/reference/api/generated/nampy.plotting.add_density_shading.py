"""Runnable reference example: api/generated/nampy.plotting.add_density_shading."""

import numpy as np
from nampy.inspection import DistributionalEffects

n = 24
x = np.linspace(0, 1, n)
effects = DistributionalEffects(feature_values={'x': x},
                               point_effects={'x': np.sin(3*x)},
                               conditional_sd={'x': 0.1 + 0.2*x})
import matplotlib.pyplot as plt
from nampy.plotting import add_density_shading

figure, ax = plt.subplots()
ax.plot(x, np.sin(3*x))
result = add_density_shading(ax, x, bins=8)
