"""Runnable reference example: statistical_reference/families/student_lss."""

import numpy as np
import pandas as pd
from nampy import GAMLSS

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({'x': x, 'y': rng.standard_t(5, n)*np.exp(-.5+.2*x)+.3+.5*x})
model = GAMLSS(family='student_lss', formula={'mu':'y ~ s(x,k=5)','sigma':'~x'},
               optimize_smoothing=False, smoothing_params=[.2]).fit(data)
parameters = model.predict(data)
point = model.predict_point(data)
result = {'parameters': parameters, 'point': point, 'components': model.predict_components(data)}
