"""Runnable reference example: statistical_reference/families/weibull_observation_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.distributions import ObservationTarget

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': np.exp(.2+.3*x)*rng.weibull(1.8,n)})
data = data.loc[(data.y>0.2) & (data.y<3)].copy()
n = len(data)
target = ObservationTarget(
    np.where(data.y<0.7, -np.inf, np.minimum(data.y, 1.8)),
    np.where(data.y>1.8, np.inf, np.maximum(data.y, 0.7)))
model = GAMLSS(family={'name': 'weibull_observation_lss', 'lower': 0.2, 'upper': 3.0}, formula={'scale': '~x', 'shape': '~1'},
               optimize_smoothing=False).fit(data[['x']], target)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
