"""Runnable reference example: statistical_reference/families/bcg."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = np.exp(0.3 + 0.2 * data.x + rng.normal(0, 0.15, n))
from nampy.gam.families.registry import make_gam_family
family = make_gam_family({'name': 'bcg'})
family.putTheta([0.0, np.log(0.3)])
family.n_theta = 0
model = GAM(formula="y ~ s(x, bs='cr', k=6)", family=family, optimize_smoothing=False).fit(data=data)
prediction = model.predict(data, type='response')
raw = model.predict(data, type='link')
result = {'response_prediction': np.asarray(prediction), 'link_prediction': np.asarray(raw)}
print({key: value.shape for key, value in result.items()})
