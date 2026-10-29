"""Runnable reference example: api/generated/nampy.diagnostics.FisherGroupSummary."""

from nampy import FisherDiagnostics
config = FisherDiagnostics(n_inputs=12, random_state=7)
import numpy as np
from nampy.gam import GAM

X = np.linspace(-1, 1, 24)[:, None]
y = np.sin(X[:, 0]) + 0.1 * np.cos(np.arange(24))
model = GAM(k=5, optimize_smoothing=False, fisher_diagnostics=config).fit(X, y)
result = model.fisher_information_.groups[0]
print(type(result).__name__)
