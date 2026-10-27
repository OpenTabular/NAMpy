"""Runnable reference example: statistical_reference/controls/irls_reg."""

from nampy.gam import gam_control

control = gam_control(irls_reg=0.0)
result = control
assert hasattr(control, 'irls_reg')
print(control.irls_reg)
