"""Runnable reference example: statistical_reference/controls/scale_est."""

from nampy.gam import gam_control

control = gam_control(scale_est='fletcher')
result = control
assert hasattr(control, 'scale_est')
print(control.scale_est)
