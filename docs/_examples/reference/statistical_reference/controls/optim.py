"""Runnable reference example: statistical_reference/controls/optim."""

from nampy.gam import gam_control

control = gam_control(optim={'factr': 10000000.0})
result = {'optim': control.optim}
assert hasattr(control, 'optim')
print(control.optim)
