"""Subject histories stay together; future covariate paths are supplied explicitly."""

import numpy as np
import pandas as pd

from nampy import SurvivalGAM
from nampy.survival import (
          LongitudinalSurvivalTarget,
          SurvivalTrajectory,
          predict_trajectory,
)

rng = np.random.default_rng(74)
n = 60
change = rng.uniform(.1, .4, n)
exit_time = rng.uniform(.5, 2.9, n)
start = np.column_stack([np.zeros(n), change]).ravel()
stop = np.column_stack([change, exit_time]).ravel()
event = np.column_stack([np.zeros(n), np.arange(n)%3!=0]).ravel()
ids = np.repeat(np.arange(n), 2)
target = LongitudinalSurvivalTarget(start=start, stop=stop, subject_id=ids, event=event)
X = pd.DataFrame({'exposure': rng.uniform(-1, 1, 2*n)})
model = SurvivalGAM(cuts=[0, 1, 3], formula='~exposure', optimize_smoothing=False).fit(X, target)

# A hypothetical full path from zero through the requested horizon. Covariate
# changes at .7, distinct from model cuts; both appear in integration cells.
path = SurvivalTrajectory(start=[0, .7], stop=[.7, 3], subject_id=['new', 'new'])
future = pd.DataFrame({'exposure': [-.3, .8]})
prediction = predict_trajectory(model, future, path)
result = {'survival': prediction.survival([0, .5, 1, 2, 3]),
          'rmst': prediction.rmst(3), 'integration_cuts': prediction.cuts}
print(result)
