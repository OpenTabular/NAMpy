"""Runnable reference example: statistical_reference/controls/efs_lspmax."""

from nampy.gam import gam_control

control = gam_control(efs_lspmax=15.0)
result = control
assert hasattr(control, 'efs_lspmax')
print(control.efs_lspmax)
