"""Runnable reference example: statistical_reference/controls/nlm."""

from nampy.gam import gam_control

control = gam_control(nlm={'ndigit': 7, 'gradtol': 1e-06, 'stepmax': 2.0, 'steptol': 0.0001, 'iterlim': 200, 'check_analyticals': False})
result = {'nlm': control.nlm}
assert hasattr(control, 'nlm')
print(control.nlm)
