"""Runnable reference example: statistical_reference/links/cloglog."""

import numpy as np
from nampy.gam.families.registry import make_gam_family

family = make_gam_family({'name': 'binomial', 'link': 'cloglog'})
mean = np.array([0.1, 0.5, 0.9])
predictor = family.link(mean)
recovered_mean = family.inverse_link(predictor)
inverse_derivative = family.mu_eta(predictor)
second_forward_derivative = family.d2link(mean)
result = {"predictor": predictor, "recovered_mean": recovered_mean,
          "inverse_derivative": inverse_derivative,
          "second_forward_derivative": second_forward_derivative}
