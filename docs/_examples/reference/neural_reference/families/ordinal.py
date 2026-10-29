"""Runnable reference example: neural_reference/families/ordinal."""

import numpy as np
import torch
from nampy.neural.families.categorical import OrdinalCumulativeLogitDistribution

family = OrdinalCumulativeLogitDistribution(num_classes=3)
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
target = torch.arange(n) % 3
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
