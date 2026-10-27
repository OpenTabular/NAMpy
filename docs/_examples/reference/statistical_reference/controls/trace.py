"""Runnable reference example: statistical_reference/controls/trace."""

from nampy.gam import gam_control

control = gam_control(trace=False)
result = {'trace': control.trace}
assert hasattr(control, 'trace')
print(control.trace)
