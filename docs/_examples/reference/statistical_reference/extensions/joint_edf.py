"""Runnable reference example: statistical_reference/extensions/joint_edf."""

import numpy as np
from nampy.gam.extensions import joint_edf
information = np.array([[4.,1.3],[1.3,2.]])
penalty = np.diag([.2,3.])
edf, total, edf1 = joint_edf(information, penalty)
result = {'edf': edf, 'total': total, 'edf1': edf1}
assert total < 2
