"""Runnable reference example: statistical_reference/smooths/tesmi1."""

import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
from nampy.gam.smooths import make_smooth_term

# Build the term's identified basis before choosing a coefficient optimizer.
term = make_smooth_term('tesmi1', feature=['x', 'z'], k=[5, 6], m=[2, 1])
term.fit(data[['x', 'z']].to_numpy(), ['x', 'z'])
basis = term.basis_train
penalties = term.penalties
coefficients = np.linspace(0.05, 0.2, basis.shape[1])
surface = basis @ coefficients
result = {'basis': basis, 'penalties': penalties, 'surface': surface}
print(basis.shape, [matrix.shape for matrix in penalties])
