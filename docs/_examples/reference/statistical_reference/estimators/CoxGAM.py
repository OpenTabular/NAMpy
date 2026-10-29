"""Runnable reference example: statistical_reference/estimators/CoxGAM."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
from nampy import CoxGAM
from nampy.survival import SurvivalTarget
record = np.arange(n)
target = SurvivalTarget(.2+((record*13)%27)/10, record%4!=1)
model = CoxGAM(formula='~x+z', tol=1e-8).fit(data[['x','z']], target)
prediction = model.predict_distribution(data[['x','z']].iloc[:4])
result = {'survival': prediction.survival([.5,1,2]),
          'rmst': prediction.rmst(2),
          'components': model.predict_components(data[['x','z']].iloc[:4])}
print({key: type(value).__name__ for key,value in result.items()})
