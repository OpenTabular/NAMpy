"""Runnable reference example: statistical_reference/controls/scam_max_half."""

from nampy.gam import gam_control

control = gam_control(scam_max_half=30)
result = {'scam_max_half': control.scam_max_half}
print(control.scam_max_half)
