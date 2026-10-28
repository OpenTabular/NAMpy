"""Runnable reference example: neural_reference/families/dirichlet."""

import numpy as np
import torch
from nampy.neural.distributions.distributions import DirichletDistribution

family = DirichletDistribution(n_dim=3)
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.full((n, 3), 1 / 3)
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
