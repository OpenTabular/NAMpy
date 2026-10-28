"""Runnable reference example: statistical_reference/controls/rank_tol."""

from nampy.gam import gam_control

control = gam_control(rank_tol=1.4901161193847656e-08)
result = {'rank_tol': control.rank_tol}
print(control.rank_tol)
