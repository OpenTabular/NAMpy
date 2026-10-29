"""Runnable reference example: statistical_reference/families/bccg_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.distributions import BCCGLaw, DistributionPrediction

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': DistributionPrediction(BCCGLaw(), {'mu': np.exp(.2+.3*x), 'sigma': np.full(n,.25), 'nu': np.ones(n)}).sample(1, random_state=rng)[0]})
model = GAMLSS(family='bccg_lss', formula={'mu': 'y ~ s(x,k=5)', 'sigma': '~1', 'nu': '~1'},
               smoothing_optimizer='outer_newton').fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
