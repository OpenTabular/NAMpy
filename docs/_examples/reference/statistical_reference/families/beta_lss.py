"""Runnable reference example: statistical_reference/families/beta_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.beta(3,6,n)})
model = GAMLSS(family='beta_lss', formula={'mu':'y ~ s(x,k=5)','phi':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data)
parameters = model.predict_parameters(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
