"""Runnable reference example: statistical_reference/controls/mgcv_half."""

from nampy.gam import gam_control

control = gam_control(mgcv_half=15)
result = control
assert hasattr(control, 'mgcv_half')
print(control.mgcv_half)
