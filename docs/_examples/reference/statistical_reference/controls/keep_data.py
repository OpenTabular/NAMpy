"""Runnable reference example: statistical_reference/controls/keep_data."""

from nampy.gam import gam_control

control = gam_control(keep_data=False)
result = control
assert hasattr(control, 'keep_data')
print(control.keep_data)
