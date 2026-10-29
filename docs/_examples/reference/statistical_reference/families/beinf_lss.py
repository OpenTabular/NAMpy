"""Runnable reference example: statistical_reference/families/beinf_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': np.where(rng.uniform(size=n)<.2, 0., np.where(rng.uniform(size=n)<.25, 1., rng.beta(3,5,n)))})
model = GAMLSS(family='beinf_lss', formula={'mu':'y ~ s(x,k=5)','phi':'~x','pi0':'~x','pi1':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
