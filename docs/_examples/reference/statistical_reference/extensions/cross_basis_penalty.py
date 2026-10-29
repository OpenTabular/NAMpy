"""Runnable reference example: statistical_reference/extensions/cross_basis_penalty."""

import numpy as np
from nampy.gam.structured.lags import cross_basis_penalty
result = cross_basis_penalty(np.diag([0.,1.]), np.diag([0.,1.,1.]), exposure_weight=.1, lag_weight=.5)
