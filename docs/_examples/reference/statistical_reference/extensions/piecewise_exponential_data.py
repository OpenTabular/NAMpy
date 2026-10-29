"""Runnable reference example: statistical_reference/extensions/piecewise_exponential_data."""

import numpy as np
import pandas as pd
from nampy import GAMRegressor
from nampy.survival import piecewise_exponential_data, survival_from_hazard

rng = np.random.default_rng(24)
x = rng.normal(size=100)
event_time = rng.exponential(scale=np.exp(-.3*x))
censor_time = rng.uniform(.5, 4, size=100)
subjects = pd.DataFrame({'x': x, 'time': np.minimum(event_time, censor_time),
                         'event': (event_time <= censor_time).astype(int)})
cuts = np.linspace(0, 4, 9)
ped = piecewise_exponential_data(subjects, time='time', event='event', cuts=cuts)
model = GAMRegressor(formula="ped_event ~ s(tend, bs='cr', k=5) + x", family='poisson',
                     optimize_smoothing=False, smoothing_params=[.2]).fit(ped, offset=ped.offset)
profile = pd.DataFrame({'tend': cuts[1:], 'x': np.zeros(8)})
hazard = model.predict(profile, offset=np.zeros(8))[None, :]
curves = survival_from_hazard(hazard, cuts)
result = {'expanded': ped, 'hazard': hazard, **curves}
