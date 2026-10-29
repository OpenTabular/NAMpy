"""Fixed-bandwidth, fixed-smoothing qgam calFast inference."""

import numpy as np
import pandas as pd

from nampy import CalibratedQuantileGAM

rng = np.random.default_rng(7)
X = pd.DataFrame({"x": rng.uniform(0, 1, 100), "z": rng.normal(size=100)})
y = 0.4 + np.sin(3 * X.x) + 0.3 * X.z + rng.normal(0, 0.4, 100)
model = CalibratedQuantileGAM(
    formula="~s(x,k=4)+z",
    level=0.3,
    bandwidth=0.2,
    smoothing_params=[0.5],
    log_scale_bounds=(-3, 2),
).fit(X, y)
interval = model.predict_interval(X.iloc[:5], level=0.95)
components = model.predict_components(X.iloc[:5])
components.validate_additive_reconstruction()
print("Calibrated loss scale:", model.loss_scale_)
print(interval.coverage, interval.lower, interval.upper)
