"""Runnable reference example: statistical_reference/estimators/QuantileGAM_smooth."""

import numpy as np
import pandas as pd
from nampy import QuantileGAM

rng = np.random.default_rng(19)
x = np.linspace(-1, 1, 60)
data = pd.DataFrame({'x': x, 'y': np.sin(3*x) + rng.normal(0, .25, len(x))})
model = QuantileGAM(formula='y ~ s(x, k=6)', level=.8, smoothing_params=[.1], bandwidth=.1).fit(data)
prediction = model.predict(data)
components = model.predict_components(data)
figures = model.plot(data)
result = {'prediction': prediction, 'components': components, 'fit': model.fit_result_}
score = model.score(data, data.y)
design = model.lpmatrix(data)
components.validate_additive_reconstruction()
print(score)
