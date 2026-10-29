"""Runnable reference example: statistical_reference/families/bct_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.distributions import BCTLaw, DistributionPrediction

rng = np.random.default_rng(41)
n = 160
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': np.exp(.3+.2*x)*DistributionPrediction(BCTLaw(), {'mu': [1.], 'sigma': [.7], 'nu': [1.2], 'tau': [4.]}).quantile(rng.permutation((np.arange(n)+.5)/n))[0]})
model = GAMLSS(family='bct_lss', formula={'mu': 'y~x', 'sigma': '~1', 'nu': '~1', 'tau': '~1'},
               optimize_smoothing=False).fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
