"""Runnable reference example: statistical_reference/controls/maxit."""

from nampy.gam import gam_control

control = gam_control(maxit=200)
result = control
assert hasattr(control, 'maxit')
print(control.maxit)
