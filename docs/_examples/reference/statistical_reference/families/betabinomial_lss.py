"""Runnable reference example: statistical_reference/families/betabinomial_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.binomial(10, rng.beta(2,3,n))})
model = GAMLSS(family='betabinomial_lss', formula={'mu':'y ~ s(x,k=5)','kappa':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data, context={'trials': np.full(n,10)})
parameters = model.predict_parameters(data)
point = model.predict_point(data, context={'trials': np.full(n,10)})
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
