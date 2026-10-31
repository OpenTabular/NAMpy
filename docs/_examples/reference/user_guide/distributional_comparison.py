"""Runnable reference example: user_guide/distributional_comparison."""

import numpy as np
import pandas as pd
from nampy import GAMLSS
from nampy.models import DistributionCandidate, compare_distributions

rng = np.random.default_rng(31)
X = pd.DataFrame({'x': np.linspace(-1, 1, 90)})
y = rng.lognormal(.2 + .3 * X.x, .4)
result = compare_distributions(
    {'normal': DistributionCandidate(GAMLSS(family='normal',
        formula={'mu': '~x', 'sigma': '~1'}, optimize_smoothing=False)),
     'lognormal': GAMLSS(family='lognormal_lss',
        formula={'loc': '~x', 'scale': '~1'}, optimize_smoothing=False)},
    X.iloc[:60], y[:60], X.iloc[60:], y[60:],
)
print(result.to_frame()[['name', 'status', 'mean_log_prob']])
