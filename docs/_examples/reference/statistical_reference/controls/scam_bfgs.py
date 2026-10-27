"""Runnable reference example: statistical_reference/controls/scam_bfgs."""

from nampy.gam import gam_control

control = gam_control(scam_bfgs={'check_analytical': False, 'del': 0.0001, 'steptol_bfgs': 1e-07, 'gradtol_bfgs': 1e-06, 'max_n_step': 5.0, 'max_half': 30})
result = {'scam_bfgs': control.scam_bfgs}
assert hasattr(control, 'scam_bfgs')
print(control.scam_bfgs)
