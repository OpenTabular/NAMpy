"""Runnable reference example: neural_reference/families/mvnormdiag."""

import numpy as np
import torch
from nampy.neural.distributions.distributions import MultivariateNormalDiagDistribution

family = MultivariateNormalDiagDistribution(n_dim=3)
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.ones((n, 3))
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
assert parameters.shape[0] == n
assert loss.ndim == 0 and torch.isfinite(loss)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
