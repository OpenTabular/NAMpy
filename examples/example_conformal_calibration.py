"""Independent training/calibration/test cohorts; no training uses calibration."""

import numpy as np
import pandas as pd

from nampy import GAMLSS, GAMRegressor
from nampy.calibration import calibrate_regressor

rng = np.random.default_rng(18)
X = pd.DataFrame({"x": rng.uniform(-2, 2, 180)})
y = np.sin(X.x.to_numpy()) + rng.normal(size=len(X)) * (.4+.2*np.abs(X.x.to_numpy()))
train, calibration, test = np.arange(100), np.arange(100, 150), np.arange(150, 180)

model = GAMRegressor(formula="~s(x,k=6)", optimize_smoothing=False).fit(X.iloc[train], y[train])
absolute = calibrate_regressor(model, X.iloc[calibration], y[calibration], level=.9)

lss = GAMLSS(family="normal", formula={"mu": "~s(x,k=6)", "sigma": "~1"},
            optimize_smoothing=False).fit(X.iloc[train], y[train])
quantile = calibrate_regressor(lss, X.iloc[calibration], y[calibration],
                              method="quantile", prediction="distribution", level=.9)
result = {"absolute": absolute.predict_interval(X.iloc[test]),
          "quantile": quantile.predict_interval(X.iloc[test])}
print({name: {"level": interval.level, "n_calibration": interval.n_calibration,
              "empty_sets": int(interval.empty.sum())} for name, interval in result.items()})
