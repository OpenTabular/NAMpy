"""Runnable reference example: statistical_reference/controls/edge_correct."""

from nampy.gam import gam_control

control = gam_control(edge_correct=False)
result = {'edge_correct': control.edge_correct}
print(control.edge_correct)
