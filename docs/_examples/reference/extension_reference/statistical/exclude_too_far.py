"""Runnable reference example: extension_reference/statistical/exclude_too_far."""

import numpy as np
from nampy.gam.diagnostics.plots import exclude_too_far

gx, gz = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
observed_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
observed_z = observed_x.copy()
mask = exclude_too_far(gx.ravel(), gz.ravel(), observed_x, observed_z, dist=0.2)
surface_mask = mask.reshape(gx.shape)
result = {"excluded": int(mask.sum()), "retained": int((~mask).sum()), "surface_mask": surface_mask}
