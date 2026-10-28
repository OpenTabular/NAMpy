"""Runnable reference example: statistical_reference/controls/scam_print_warn."""

from nampy.gam import gam_control

control = gam_control(scam_print_warn=False)
result = {'scam_print_warn': control.scam_print_warn}
print(control.scam_print_warn)
