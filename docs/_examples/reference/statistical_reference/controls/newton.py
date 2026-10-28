"""Runnable reference example: statistical_reference/controls/newton."""

from nampy.gam import gam_control

control = gam_control(newton={'conv_tol': 1e-06, 'max_n_step': 5.0, 'max_s_step': 2.0, 'max_half': 30, 'use_svd': False})
result = {'newton': control.newton}
print(control.newton)
