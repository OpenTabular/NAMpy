"""Runnable reference example: statistical_reference/controls/ncv_threads."""

from nampy.gam import gam_control

control = gam_control(ncv_threads=1)
result = {'ncv_threads': control.ncv_threads}
assert hasattr(control, 'ncv_threads')
print(control.ncv_threads)
