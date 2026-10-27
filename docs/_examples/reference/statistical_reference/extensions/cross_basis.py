"""Runnable reference example: statistical_reference/extensions/cross_basis."""

import numpy as np
import pandas as pd
from nampy import GAMRegressor
from nampy.gam.extensions import exposure_history, cross_basis, cumulative_lag_matrix

rng = np.random.default_rng(42)
exposure = rng.normal(size=100)
history = exposure_history(exposure, max_lag=3)
lag = np.column_stack([np.ones(4), np.arange(4)])
def exposure_basis(x):
    return np.column_stack([x, x*x])
basis = cross_basis(history, exposure_basis=exposure_basis, lag_basis=lag)
complete = np.all(np.isfinite(basis), axis=1)
columns = ['b0','b1','b2','b3']
data = pd.DataFrame(basis[complete], columns=columns)
data['y'] = basis[complete] @ np.array([.2,-.03,.1,-.01]) + rng.normal(0,.2, complete.sum())
model = GAMRegressor(formula='y ~ b0+b1+b2+b3', optimize_smoothing=False).fit(data)
contrast = cumulative_lag_matrix([0.,1.,2.], 0., n_lags=4,
                                exposure_basis=exposure_basis, lag_basis=lag)
result = {'history': history, 'basis': basis, 'predictions': model.predict(data), 'cumulative_design': contrast}
