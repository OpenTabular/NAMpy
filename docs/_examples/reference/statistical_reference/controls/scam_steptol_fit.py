"""Runnable reference example: statistical_reference/controls/scam_steptol_fit."""

from nampy.gam import gam_control

control = gam_control(scam_steptol_fit=1e-07)
result = control
assert hasattr(control, 'scam_steptol_fit')
print(control.scam_steptol_fit)
