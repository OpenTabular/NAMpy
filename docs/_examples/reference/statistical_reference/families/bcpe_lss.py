"""Runnable reference example: statistical_reference/families/bcpe_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.distributions import BCPELaw, DistributionPrediction

rng = np.random.default_rng(41)
n = 160
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': np.exp(.3+.2*x)*DistributionPrediction(BCPELaw(), {'mu': [1.], 'sigma': [.7], 'nu': [1.2], 'tau': [1.4]}).quantile(rng.permutation((np.arange(n)+.5)/n))[0]})
model = GAMLSS(family='bcpe_lss', formula={'mu': 'y~x', 'sigma': '~1', 'nu': '~1', 'tau': '~1'},
               optimize_smoothing=False).fit(data)
parameters = model.predict_parameters(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
