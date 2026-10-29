"""Runnable reference example: statistical_reference/families/lognormal_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.lognormal(-.2+.5*x, .35, n)})
model = GAMLSS(family='lognormal_lss', formula={'loc':'y ~ s(x,k=5)','scale':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
