"""Runnable reference example: statistical_reference/estimators/CalibratedQuantileGAM."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.4, n)
from nampy import CalibratedQuantileGAM
model = CalibratedQuantileGAM(formula='~s(x,k=4)+z', level=.3,
    bandwidth=.2, smoothing_params=[.5], log_scale_bounds=(-3,2)).fit(data[['x','z']], data.y)
interval = model.predict_interval(data[['x','z']])
result = {'prediction': model.predict(data[['x','z']]),
          'interval': interval, 'components': model.predict_components(data[['x','z']])}
print(interval.coverage, model.loss_scale_)
