"""Runnable reference example: api/generated/nampy.diagnostics.FisherDiagnostics."""

from nampy import FisherDiagnostics
config = FisherDiagnostics(n_inputs=12, random_state=7)
result = config
print(result.estimator)
