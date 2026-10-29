"""Runnable reference example: statistical_reference/extensions/DistributedLagBasis."""

import numpy as np
from nampy.gam.structured import DistributedLagBasis
history = np.random.default_rng(4).normal(size=(30,4))
basis = DistributedLagBasis(max_lag=3, exposure_formula='~exposure',lag_formula='~lag').fit(history)
result = {'matrix': basis.transform(history), 'penalty': basis.penalty(), 'contrast': basis.cumulative_contrast([0.,1.],0.)}
