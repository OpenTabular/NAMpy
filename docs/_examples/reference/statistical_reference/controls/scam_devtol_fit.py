"""Runnable reference example: statistical_reference/controls/scam_devtol_fit."""

from nampy.gam import gam_control

control = gam_control(scam_devtol_fit=1e-07)
result = {'scam_devtol_fit': control.scam_devtol_fit}
assert hasattr(control, 'scam_devtol_fit')
print(control.scam_devtol_fit)
