"""Runnable reference example: neural_reference/families/normal_observation."""

import numpy as np
import torch
from nampy.neural.families.observations import NormalObservationsDistribution

family = NormalObservationsDistribution()
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.column_stack([torch.ones(n),torch.ones(n),torch.zeros(n)])
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
