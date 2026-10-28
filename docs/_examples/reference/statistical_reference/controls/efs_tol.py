"""Runnable reference example: statistical_reference/controls/efs_tol."""

from nampy.gam import gam_control

control = gam_control(efs_tol=0.1)
result = {'efs_tol': control.efs_tol}
print(control.efs_tol)
