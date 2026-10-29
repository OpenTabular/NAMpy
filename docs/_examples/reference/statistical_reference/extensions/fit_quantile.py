"""Runnable reference example: statistical_reference/extensions/fit_quantile."""

import numpy as np
from nampy.gam.losses.objectives import fit_quantile

x = np.linspace(-1, 1, 40)
design = np.column_stack([np.ones(len(x)), x, x**2])
y = .4 + .5*x + .2*np.sin(np.arange(len(x)))
penalty = np.diag([0., .1, .2])
result = {'exact': fit_quantile(design, y, level=.8, penalty=penalty), 'smooth': fit_quantile(design, y, level=.8, penalty=penalty, bandwidth=.1)}
