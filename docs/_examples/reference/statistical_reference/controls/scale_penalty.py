"""Runnable reference example: statistical_reference/controls/scale_penalty."""

from nampy.gam import gam_control

control = gam_control(scale_penalty=True)
result = {'scale_penalty': control.scale_penalty}
print(control.scale_penalty)
