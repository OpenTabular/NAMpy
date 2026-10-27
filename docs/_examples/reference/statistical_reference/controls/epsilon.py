"""Runnable reference example: statistical_reference/controls/epsilon."""

from nampy.gam import gam_control

control = gam_control(epsilon=1e-07)
result = control
assert hasattr(control, 'epsilon')
print(control.epsilon)
