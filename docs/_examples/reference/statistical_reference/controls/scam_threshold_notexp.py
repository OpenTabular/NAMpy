"""Runnable reference example: statistical_reference/controls/scam_threshold_notexp."""

from nampy.gam import gam_control

control = gam_control(scam_threshold_notexp=20.0)
result = control
assert hasattr(control, 'scam_threshold_notexp')
print(control.scam_threshold_notexp)
