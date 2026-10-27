"""Runnable reference example: statistical_reference/controls/scam_b_notexp."""

from nampy.gam import gam_control

control = gam_control(scam_b_notexp=1.0)
result = control
assert hasattr(control, 'scam_b_notexp')
print(control.scam_b_notexp)
