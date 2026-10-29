"""Runnable reference example: neural_reference/families/lognormal_observation."""

import numpy as np
import torch
from nampy.neural.families.observations import LogNormalObservationsDistribution

family = LogNormalObservationsDistribution(lower=0.2, upper=3.0)
n = 8
raw = torch.zeros((n, family.param_count), dtype=torch.float32)
from nampy.distributions import ObservationTarget
observation = ObservationTarget([0.6, -np.inf, 1.5, 0.7] * 2, [0.6, 0.5, np.inf, 1.1] * 2)
target = family.encode_targets(observation)
parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
