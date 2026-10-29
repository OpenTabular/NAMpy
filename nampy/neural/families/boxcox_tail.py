"""Neural BCT and BCPE fitting with normalized likelihoods.

Torch owns the response density and predictor gradients. The support
normalizer uses checked vector SciPy quadrature on CPU; its analytical first
derivatives are returned on the input device. Higher autograd derivatives
are deliberately unsupported. Prediction reuses the shared probability laws.

References
----------
- Rigby, R. A., and Stasinopoulos, D. M. (2004). Smooth centile curves for skew and
  kurtotic data modelled using the Box-Cox power exponential distribution.
  https://doi.org/10.1002/sim.1861
- Rigby, R. A., and Stasinopoulos, D. M. (2006). Using the Box-Cox t distribution in
  GAMLSS to model skewness and kurtosis.
  https://doi.org/10.1191/1471082X06st122oa
"""

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from ...distributions._boxcox_normalizer import normalizer_with_scores
from .gamlss import _boxcox_standardized, _BoxCoxDistribution, _raw, _target


class _Normalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_sigma, nu, log_tau, kind):
        arrays = [
            value.detach().double().cpu().numpy() for value in (log_sigma, nu, log_tau)
        ]
        values = normalizer_with_scores(*arrays, kind=kind)
        tensors = [torch.as_tensor(v, device=nu.device, dtype=nu.dtype) for v in values]
        ctx.save_for_backward(*tensors[1:])
        return tensors[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return *(gradient * score for score in ctx.saved_tensors), None


class _BoxCoxTailDistribution(_BoxCoxDistribution):
    """Four predictors with log/log/identity/log links and positive targets."""

    kind: str

    def __init__(self):
        super().__init__(self.kind.upper(), ["mu", "sigma", "nu", "tau"])

    def forward(self, predictions):
        p = _raw(predictions, 4)
        return torch.stack(
            [p[:, 0].exp(), p[:, 1].exp(), p[:, 2], p[:, 3].exp()], dim=1
        )

    def compute_loss_per_sample(self, predictions, y_true):
        p = _raw(predictions, 4)
        y = _target(y_true, p, 0)
        if torch.any(y <= 0):
            raise ValueError("Box-Cox families require positive responses.")
        lm, ls, nu, lt = p.T
        r = torch.log(y) - lm
        z = _boxcox_standardized(r, ls, nu)
        tau = lt.exp()
        if self.kind == "bct":
            latent = (
                torch.lgamma((tau + 1) / 2)
                - torch.lgamma(tau / 2)
                - (lt + np.log(np.pi)) / 2
                - (tau + 1) / 2 * torch.log1p(z.square() / tau)
            )
        else:
            # |z|**tau need not be differentiable at zero for tau<=1.
            if torch.any((z == 0) & (tau <= 1)):
                raise NotImplementedError("BCPE gradient at y=mu requires tau>1.")
            inverse = 1 / tau
            alpha = (torch.lgamma(inverse) - torch.lgamma(3 * inverse)) / 2
            latent = (
                lt
                - np.log(2)
                - alpha
                - torch.lgamma(inverse)
                - (z.abs() * torch.exp(-alpha)).pow(tau)
            )
        result = (
            -nu * r
            + ls
            + torch.log(y)
            - latent
            + _Normalizer.apply(ls, nu, lt, self.kind)
        )
        if not torch.isfinite(result).all():
            raise NotImplementedError(
                "Box-Cox neural likelihood exceeds the supported numerical range."
            )
        return result


class BCTDistribution(_BoxCoxTailDistribution):
    """Normalized Box-Cox Student t; tau is fitted degrees of freedom."""

    kind = "bct"


class BCPEDistribution(_BoxCoxTailDistribution):
    """Normalized variance-one Box-Cox power exponential; tau is fitted shape."""

    kind = "bcpe"
