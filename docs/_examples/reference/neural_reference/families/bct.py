"""Runnable reference example: neural_reference/families/bct."""

import numpy as np
import torch
from nampy.neural.families.boxcox_tail import BCTDistribution

family = BCTDistribution()
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
raw[:, 1] = -1
raw[:, 2] = 1
raw[:, 3] = np.log(5)
target = torch.linspace(1.1, 1.8, n)
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
