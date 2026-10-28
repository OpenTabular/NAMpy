"""Runnable reference example: statistical_reference/internals/covariance_correction."""

import numpy as np
from nampy.gam.fit.solvers.general_family.newton import _vb_corr_root

p = 3
information_root = 2.0 * np.eye(p)
penalty = np.diag([0.0, 1.0, 1.0])
result = _vb_corr_root(information_root, L=None, lsp0=None,
                       S_blocks=[penalty], off=[1], rho=np.array([0.0]),
                       Vr=np.array([[0.1]]))
print(result.shape)
