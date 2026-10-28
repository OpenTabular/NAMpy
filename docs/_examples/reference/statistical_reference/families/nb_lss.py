"""Runnable reference example: statistical_reference/families/nb_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.negative_binomial(3,1/(1+np.exp(.3+.5*x)/3))})
model = GAMLSS(family='nb_lss', formula={'mu':'y ~ s(x,k=5)','alpha':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
