"""Runnable reference example: extension_reference/statistical/DaviesAlgorithm."""

import numpy as np
from nampy.gam.inference.chi_square_mixtures import DaviesAlgorithm

probability, trace, status = DaviesAlgorithm().davies(
    lb=np.array([0.5, 1.5]), nc=np.zeros(2), n=np.array([1, 2]),
    r=2, sigma=0.0, c_val=4.0, lim=100000, acc=2e-5,
)
result = {"lower_tail_probability": probability, "status": status, "trace": trace}
