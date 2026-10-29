"""Runnable reference example: api/generated/nampy.contracts.CapabilityStatus."""

from nampy import GAMRegressor
result = GAMRegressor().capabilities()['predict_link']
print(result)
