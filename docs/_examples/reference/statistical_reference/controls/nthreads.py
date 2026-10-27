"""Runnable reference example: statistical_reference/controls/nthreads."""

from nampy.gam import gam_control

control = gam_control(nthreads=1)
result = {'nthreads': control.nthreads}
assert hasattr(control, 'nthreads')
print(control.nthreads)
