"""Runnable reference example: statistical_reference/penalties."""

import numpy as np
import pandas as pd
from nampy.gam import GAM
from nampy.gam.compiler.construct import smooth_con
from nampy.gam.specs import TermSpec, build_smooth_spec

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
model = GAM(formula="y ~ x + s(z, bs='cr', k=7)",
            para_pen={'x': {'penalties': [np.eye(1)], 'sp': [0.5]}},
            optimize_smoothing=False).fit(data=data)
term = TermSpec(kind='smooth', features=('x',), label='s(x)',
                smooth_spec=build_smooth_spec(special='s', bs='cr', k=7))
raw = smooth_con(term, data[['x']].to_numpy(), ['x'], absorb_cons=False)
constraint = raw.fit_constraint_operator
constrained = smooth_con(term, data[['x']].to_numpy(), ['x'], constraints=constraint)
matrix = constrained.compiled_term.predict_matrix(data[['x']].to_numpy())
assert matrix.shape[0] == n
assert np.isfinite(matrix).all()
result = {'prediction': model.predict(data)[:3], 'constraint_shape': constraint.shape,
          'constrained_design': matrix}
print(result['prediction'], matrix.shape)
