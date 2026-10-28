"""Runnable reference example: statistical_reference/extensions/CompiledDesign."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)

from nampy.gam import compile_design

design = compile_design(data, formula='y ~ s(x,k=6)+offset(z)', smoothing_params=[.2])
evaluation = design.evaluate(data.assign(z=data.z+.5))
result = {'matrix': design.matrix, 'penalty': design.penalty,
          'training_offset': design.training_offset, 'evaluation': evaluation,
          'terms': design.terms}
