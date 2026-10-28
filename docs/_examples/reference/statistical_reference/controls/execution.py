"""Runnable reference example: statistical_reference/controls/execution."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
from nampy.gam import gam_control

control = gam_control(trace=False, nthreads=1, ncv_threads=1)
model = GAM(formula="y ~ s(x, bs='cr', k=6)", control=control, optimize_smoothing=False, smoothing_method="reml").fit(data=data)
result = model.predict(data)[:3]
print(result)
