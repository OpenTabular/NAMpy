"""Runnable reference example: api/generated/nampy.plotting.render_model_check."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)
from nampy.gam.diagnostics import prepare_gam_check_data
from nampy.plotting import render_model_check

prepared = prepare_gam_check_data(model, seed=7)
result = render_model_check(prepared)
print(len(result))
