"""Differentiable normalized GAMLSS likelihoods, reusing shared NumPy inference.

These are neural objectives, not GAMLSS working-curvature algorithms. Beta-binomial
trials are fixed per model; observation-specific trials remain unsupported here.
"""

from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F

from ..objectives.parameterized import ParameterizedObjective


def _raw(value, width):
    value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    if value.ndim != 2 or value.shape[1] != width or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite raw predictors (n, {width}).")
    return value


def _target(value, predictions, lower, upper=None, *, integer=False):
    value = torch.as_tensor(
        value, device=predictions.device, dtype=predictions.dtype
    ).reshape(-1)
    if (
        len(value) != len(predictions)
        or not torch.isfinite(value).all()
        or torch.any(value < lower)
    ):
        raise ValueError(
            "Targets must be finite, aligned and inside the response support."
        )
    if upper is not None and torch.any(value > upper):
        raise ValueError("Targets exceed the response support.")
    if integer and torch.any(value != torch.floor(value)):
        raise ValueError("Count responses must be integers.")
    return value


class BetaBinomialDistribution(ParameterizedObjective):
    trials: int
    point_summary = "mean"

    def __init__(self, *, trials: int):
        if isinstance(trials, bool) or not isinstance(trials, Integral) or trials < 2:
            raise ValueError(
                "Neural beta-binomial requires fixed integer trials >=2; row-specific trials are unsupported."
            )
        super().__init__("BetaBinomial", ["mu", "kappa"])
        self.trials = int(trials)

    def forward(self, predictions):
        p = _raw(predictions, 2)
        return torch.stack([torch.sigmoid(p[:, 0]), torch.exp(p[:, 1])], dim=1)

    def compute_loss_per_sample(self, predictions, y_true):
        p = self(predictions)
        mu, kappa = p.T
        y = _target(y_true, p, 0, self.trials, integer=True)
        n = torch.full_like(y, self.trials)
        a, b = mu * kappa, (1 - mu) * kappa
        return -(
            torch.lgamma(n + 1)
            - torch.lgamma(y + 1)
            - torch.lgamma(n - y + 1)
            + torch.lgamma(y + a)
            + torch.lgamma(n - y + b)
            - torch.lgamma(n + kappa)
            + torch.lgamma(kappa)
            - torch.lgamma(a)
            - torch.lgamma(b)
        )

    def predict_point(self, predictions, transformed=False):
        p = predictions if transformed else self(predictions)
        return self.trials * p[:, 0]

    def predict_spread(self, predictions, transformed=False):
        p = predictions if transformed else self(predictions)
        mu, kappa = p.T
        return torch.sqrt(
            self.trials * mu * (1 - mu) * (self.trials + kappa) / (1 + kappa)
        )


class ZeroAdjustedGammaDistribution(ParameterizedObjective):
    point_summary = "mean"

    def __init__(self):
        super().__init__("ZAGA", ["mu", "sigma", "pi0"])

    def forward(self, predictions):
        p = _raw(predictions, 3)
        return torch.stack(
            [torch.exp(p[:, 0]), torch.exp(p[:, 1]), torch.sigmoid(p[:, 2])], dim=1
        )

    def compute_loss_per_sample(self, predictions, y_true):
        p = _raw(predictions, 3)
        y = _target(y_true, p, 0)
        positive = y > 0
        result = F.softplus(-p[:, 2])
        mu, sigma, _ = self(p[positive]).T
        shape = 1 / sigma.square()
        rate = shape / mu
        yy = y[positive]
        result[positive] = F.softplus(p[positive, 2]) - (
            shape * torch.log(rate)
            + (shape - 1) * torch.log(yy)
            - rate * yy
            - torch.lgamma(shape)
        )
        return result

    def predict_point(self, predictions, transformed=False):
        mu, sigma, pi0 = (predictions if transformed else self(predictions)).T
        return (1 - pi0) * mu

    def predict_spread(self, predictions, transformed=False):
        mu, sigma, pi0 = (predictions if transformed else self(predictions)).T
        return torch.sqrt((1 - pi0) * mu.square() * (sigma.square() + pi0))


class InflatedBetaDistribution(ParameterizedObjective):
    point_summary = "mean"

    def __init__(self):
        super().__init__("BEINF", ["mu", "phi", "pi0", "pi1"])

    def forward(self, predictions):
        p = _raw(predictions, 4)
        masses = torch.softmax(
            torch.cat([p[:, 2:], torch.zeros_like(p[:, :1])], dim=1).double(), dim=1
        )
        return torch.cat([torch.sigmoid(p[:, :1]), torch.exp(p[:, 1:2]), masses], dim=1)

    def compute_loss_per_sample(self, predictions, y_true):
        p = _raw(predictions, 4)
        y = _target(y_true, p, 0, 1)
        logits = torch.cat([p[:, 2:], torch.zeros_like(p[:, :1])], dim=1)
        log_mass = torch.log_softmax(logits.double(), dim=1)
        result = -torch.where(y == 0, log_mass[:, 0], log_mass[:, 1])
        interior = (y > 0) & (y < 1)
        mu, phi = self(p[interior])[:, :2].T
        a, b = mu * phi, (1 - mu) * phi
        yy = y[interior].to(dtype=mu.dtype)
        density = (
            (a - 1) * torch.log(yy)
            + (b - 1) * torch.log1p(-yy)
            + torch.lgamma(phi)
            - torch.lgamma(a)
            - torch.lgamma(b)
        )
        result[interior] = -log_mass[interior, 2] - density
        return result

    def predict_point(self, predictions, transformed=False):
        mu, phi, p0, p1, pc = (predictions if transformed else self(predictions)).T
        return p1 + pc * mu

    def predict_spread(self, predictions, transformed=False):
        mu, phi, p0, p1, pc = (predictions if transformed else self(predictions)).T
        mean = p1 + pc * mu
        return torch.sqrt(
            p1 + pc * (mu.square() + mu * (1 - mu) / (phi + 1)) - mean.square()
        )


def _boxcox_standardized(log_ratio, log_sigma, nu):
    """Positive Box-Cox transform with its differentiable nu=0 limit."""
    h = nu * log_ratio
    safe = torch.where(h == 0, torch.ones_like(h), h)
    ratio = torch.where(
        torch.abs(h) < 1e-4,
        1 + h / 2 + h**2 / 6 + h**3 / 24 + h**4 / 120 + h**5 / 720 + h**6 / 5040,
        torch.expm1(h) / safe,
    )
    return log_ratio * torch.exp(-log_sigma) * ratio


class _BoxCoxDistribution(ParameterizedObjective):
    """Shared probability-backed summaries for positive Box-Cox families."""

    point_summary = "mean"

    def _moment(self, predictions, transformed, operation):
        from .distribution_adapter import distribution_prediction

        p = predictions if transformed else self(predictions)
        law = distribution_prediction(self, p.detach().cpu().numpy())
        value = law.mean() if operation == "mean" else np.sqrt(law.variance())
        return torch.as_tensor(value, device=p.device, dtype=p.dtype)

    def predict_point(self, predictions, transformed=False):
        return self._moment(predictions, transformed, "mean")

    def predict_spread(self, predictions, transformed=False):
        return self._moment(predictions, transformed, "spread")


class BCCGDistribution(_BoxCoxDistribution):
    def __init__(self):
        super().__init__("BCCG", ["mu", "sigma", "nu"])

    def forward(self, predictions):
        p = _raw(predictions, 3)
        return torch.cat([torch.exp(p[:, :2]), p[:, 2:]], dim=1)

    def compute_loss_per_sample(self, predictions, y_true):
        p = _raw(predictions, 3)
        y = _target(y_true, p, 0)
        if torch.any(y <= 0):
            raise ValueError("BCCG requires positive responses.")
        lm, ls, nu = p.T
        r = torch.log(y) - lm
        z = _boxcox_standardized(r, ls, nu)
        t = torch.exp(ls) * nu
        a = 1 / torch.where(t == 0, torch.ones_like(t), torch.abs(t))
        normalizer = torch.where(t == 0, torch.zeros_like(t), torch.special.log_ndtr(a))
        result = (
            -nu * r
            + ls
            + 0.5 * z.square()
            + torch.log(y)
            + 0.5 * np.log(2 * np.pi)
            + normalizer
        )
        if not torch.isfinite(result).all():
            raise NotImplementedError(
                "BCCG neural likelihood exceeds the supported numerical range."
            )
        return result

