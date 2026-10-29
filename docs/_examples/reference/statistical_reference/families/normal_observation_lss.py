"""Runnable reference example: statistical_reference/families/normal_observation_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.distributions import ObservationTarget

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.normal(.3+.4*x,.4,n)})
target = ObservationTarget(np.where(data.y<0,-np.inf,data.y),data.y)
model = GAMLSS(family='normal_observation_lss', formula={'mean':'y ~ s(x,k=5)','scale':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data,target)
parameters = model.predict_parameters(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
