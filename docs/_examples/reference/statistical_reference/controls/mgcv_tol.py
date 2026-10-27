"""Runnable reference example: statistical_reference/controls/mgcv_tol."""

from nampy.gam import gam_control

control = gam_control(mgcv_tol=1e-07)
result = {'mgcv_tol': control.mgcv_tol}
assert hasattr(control, 'mgcv_tol')
print(control.mgcv_tol)
