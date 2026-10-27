"""Runnable reference example: neural_reference/families/inversegamma."""

import numpy as np
import torch
from nampy.neural.distributions.distributions import InverseGammaDistribution

family = InverseGammaDistribution()
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.ones(n)
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
assert parameters.shape[0] == n
assert loss.ndim == 0 and torch.isfinite(loss)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
