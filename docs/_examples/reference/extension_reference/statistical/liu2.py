"""Runnable reference example: extension_reference/statistical/liu2."""

import numpy as np
from nampy.gam.inference.chi_square_mixtures import liu2

thresholds = np.array([1.0, 4.0, 8.0])
weights = np.array([0.5, 1.5])
degrees = np.array([1, 2])
upper = liu2(thresholds, weights, df=degrees)
lower = liu2(thresholds, weights, df=degrees, lower_tail=True)
result = {"thresholds": thresholds, "upper_tail": upper, "lower_tail": lower}
