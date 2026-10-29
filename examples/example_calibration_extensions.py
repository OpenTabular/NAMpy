"""Distinct prediction targets: covariate shift, new groups and censored times."""

import numpy as np
import pandas as pd

from nampy import GAMRegressor, SurvivalGAM
from nampy.calibration import calibrate_regressor, calibrate_survival_estimator
from nampy.survival import SurvivalTarget

rng = np.random.default_rng(83)
X = pd.DataFrame({"x": rng.uniform(-1, 1, 180)})
y = X.x.to_numpy()+rng.normal(size=180)
model = GAMRegressor(formula="~x", optimize_smoothing=False).fit(X.iloc[:80], y[:80])
# Illustrative known target/source covariate ratios, on a common scale.
ratio = np.exp(.5*X.x.to_numpy())
weighted = calibrate_regressor(model, X.iloc[80:140], y[80:140],
                              density_ratio=ratio[80:140], level=.9)
weighted_interval = weighted.predict_interval(X.iloc[140:], density_ratio=ratio[140:])

# Complete exchangeable groups are the calibration unit; this example assigns
# fixed group sizes. Real repeated-record data must be split by whole groups.
clustered = calibrate_regressor(model, X.iloc[80:140], y[80:140],
                               groups=np.repeat(np.arange(30), 2), level=.9)
cluster_interval = clustered.predict_interval(X.iloc[140:], groups=np.repeat(np.arange(100, 120), 2))

latent = np.exp(.3+.3*X.x.to_numpy())*rng.weibull(2., len(X))*2
potential_censoring = rng.choice([2., 4.], size=len(X), p=[.3, .7])
target = SurvivalTarget(np.minimum(latent, potential_censoring), latent <= potential_censoring)
survival = SurvivalGAM(cuts=[0, 1, 2, 3, 4], baseline="smooth", k=4, formula="~x",
                       optimize_smoothing=False, smoothing_params=[.4]).fit(X.iloc[:80], target.take(np.arange(80)))
# Every subject's potential censoring time is known in this simulation.
# cutoff=3 is fixed before calibration; P(C>=3|X)=.7 is known by construction.
calibrated_survival = calibrate_survival_estimator(survival, X.iloc[80:140], target.take(np.arange(80, 140)),
    level=.9, cutoff=3., censoring_time=potential_censoring[80:140], selection_probability=np.full(60, .7))
survival_interval = calibrated_survival.predict_interval(X.iloc[140:], selection_probability=np.full(40, .7))
result = {"weighted": weighted_interval, "complete_group": cluster_interval, "event_time_lower": survival_interval}
print({name: value.coverage for name, value in result.items()})
