"""Runnable reference example: statistical_reference/uncertainty."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': np.linspace(0, 1, n)})
data['y'] = np.sin(3 * data.x) + rng.normal(0, 0.15, n)
model = GAM(formula="y ~ s(x, bs='cr', k=8)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)
new_data = pd.DataFrame({'x': np.linspace(0.1, 0.9, 20)})
mean, se = model.predict(new_data, type='link', return_se=True, unconditional=True)
coefficient_covariance = model.vcov(unconditional=True)
pointwise_interval = np.column_stack([mean - 1.96 * se, mean + 1.96 * se])
result = {'mean': mean, 'standard_error': se, 'pointwise_interval': pointwise_interval,
          'coefficient_covariance': coefficient_covariance}
print({key: value.shape for key, value in result.items()})
