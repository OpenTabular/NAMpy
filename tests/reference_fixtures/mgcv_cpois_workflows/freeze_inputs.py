"""Frozen inputs for actual mgcv censored-Poisson workflow captures."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
rng = np.random.default_rng(764)
x = np.linspace(-1, 1, 80)
y = rng.poisson(np.exp(1.0 + .5 * np.sin(2.5 * x))) + 1
bounds = y.astype(object).tolist()
for i in range(len(y)):
    if i % 4 == 1:
        bounds[i] = '-Inf'
    elif i % 4 == 2:
        bounds[i] = 'Inf'
    elif i % 4 == 3:
        bounds[i] = int(y[i] + 2)
output = {'x': x.tolist(), 'y': y.tolist(), 'bound': [str(v) for v in bounds],
          'off': (.08 * np.cos(x)).tolist(), 'w': np.linspace(.5, 2, len(x)).tolist()}
(root / 'inputs.json').write_text(json.dumps(output, indent=2) + '\n')
