"""Runnable reference example: statistical_reference/controls/id_links_bases."""

from nampy.gam import gam_control

control = gam_control(id_links_bases=True)
result = {'id_links_bases': control.id_links_bases}
assert hasattr(control, 'id_links_bases')
print(control.id_links_bases)
