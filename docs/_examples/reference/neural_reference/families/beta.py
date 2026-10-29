"""Runnable reference example: neural_reference/families/beta."""

import numpy as np
import torch
from nampy.neural.families.continuous import BetaDistribution

family = BetaDistribution()
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.full((n,), 0.4)
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
