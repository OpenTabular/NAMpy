"""Nonlinear maps between optimization and prediction coefficients.

The fitting layer distinguishes unconstrained ``optimization`` coordinates
from ``prediction`` coefficients on which the additive-model design acts.
Most GAMs use an identity map; constrained and multi-predictor models can use
coordinatewise maps and independent block composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

PositiveMap = Literal["exp", "softplus"]
CovarianceScale = Literal["jacobian", "prediction"]


@runtime_checkable
class CoefficientTransform(Protocol):
    """Contract for separable optimization-to-prediction coefficient maps."""

    @property
    def size(self) -> int: ...

    @property
    def is_identity(self) -> bool: ...

    def forward(self, optimization_coefficients) -> np.ndarray: ...

    def inverse(self, prediction_coefficients) -> np.ndarray: ...

    def derivative(
        self, optimization_coefficients, *, order: int = 1
    ) -> np.ndarray: ...

    def jacobian(self, optimization_coefficients) -> np.ndarray: ...

    def covariance_scale(self, optimization_coefficients) -> np.ndarray: ...

    def transport_covariance(
        self, optimization_coefficients, covariance
    ) -> np.ndarray: ...

    def subset(self, indices) -> "CoefficientTransform": ...


def _validate_coefficient_array(coefficients, size: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != int(size):
        raise ValueError(
            "Coefficient arrays must have a final dimension equal to the "
            f"transform size ({size}); got {values.shape}."
        )
    return values


@dataclass(frozen=True)
class IdentityCoefficientTransform:
    """Identity map used by ordinary GAM coefficient blocks."""

    size: int

    def __post_init__(self) -> None:
        if int(self.size) < 0:
            raise ValueError("Transform size must be non-negative.")
        object.__setattr__(self, "size", int(self.size))

    @property
    def is_identity(self) -> bool:
        return True

    def forward(self, optimization_coefficients) -> np.ndarray:
        return _validate_coefficient_array(optimization_coefficients, self.size).copy()

    transform = forward

    def inverse(self, prediction_coefficients) -> np.ndarray:
        return _validate_coefficient_array(prediction_coefficients, self.size).copy()

    def derivative(self, optimization_coefficients, *, order: int = 1) -> np.ndarray:
        if order not in {1, 2, 3}:
            raise ValueError("Only derivative orders 1, 2, and 3 are supported.")
        beta = _validate_coefficient_array(optimization_coefficients, self.size)
        return np.ones_like(beta) if order == 1 else np.zeros_like(beta)

    def jacobian(self, optimization_coefficients) -> np.ndarray:
        beta = _validate_coefficient_array(optimization_coefficients, self.size)
        if beta.ndim != 1:
            raise ValueError("jacobian() requires a one-dimensional vector.")
        return np.eye(self.size, dtype=np.float64)

    def covariance_scale(self, optimization_coefficients) -> np.ndarray:
        return np.ones_like(
            _validate_coefficient_array(optimization_coefficients, self.size)
        )

    def transport_covariance(self, optimization_coefficients, covariance) -> np.ndarray:
        beta = _validate_coefficient_array(optimization_coefficients, self.size)
        if beta.ndim != 1:
            raise ValueError(
                "transport_covariance() requires one-dimensional coefficients."
            )
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (self.size, self.size):
            raise ValueError(
                f"Covariance must have shape {(self.size, self.size)}, got {cov.shape}."
            )
        return cov.copy()

    def subset(self, indices) -> "IdentityCoefficientTransform":
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(index < 0) or np.any(index >= self.size):
            raise IndexError("Coefficient-transform subset index out of range.")
        return IdentityCoefficientTransform(index.size)


@dataclass(frozen=True)
class CoordinatewiseCoefficientTransform:
    """Identity/positive coordinate map for one coefficient block."""

    positive_mask: np.ndarray
    positive_map: PositiveMap = "softplus"
    softplus_beta: float = 1.0
    softplus_threshold: float = 20.0
    covariance_transport: CovarianceScale = "jacobian"

    def __post_init__(self) -> None:
        mask = np.asarray(self.positive_mask, dtype=bool).reshape(-1).copy()
        mask.setflags(write=False)
        object.__setattr__(self, "positive_mask", mask)
        if self.positive_map not in {"exp", "softplus"}:
            raise ValueError("positive_map must be 'exp' or 'softplus'.")
        if not np.isfinite(self.softplus_beta) or self.softplus_beta <= 0.0:
            raise ValueError("softplus_beta must be finite and strictly positive.")
        if not np.isfinite(self.softplus_threshold):
            raise ValueError("softplus_threshold must be finite.")
        if self.covariance_transport not in {"jacobian", "prediction"}:
            raise ValueError(
                "covariance_transport must be 'jacobian' or 'prediction'."
            )

    @property
    def size(self) -> int:
        return int(self.positive_mask.size)

    @property
    def is_identity(self) -> bool:
        return not bool(np.any(self.positive_mask))

    def _validate(self, coefficients) -> np.ndarray:
        return _validate_coefficient_array(coefficients, self.size)

    def forward(self, optimization_coefficients) -> np.ndarray:
        beta = self._validate(optimization_coefficients)
        out = beta.copy()
        x = beta[..., self.positive_mask]
        if self.positive_map == "exp":
            out[..., self.positive_mask] = np.exp(x)
        else:
            out[..., self.positive_mask] = self._softplus(x)
        return out

    transform = forward

    def inverse(self, prediction_coefficients) -> np.ndarray:
        beta = self._validate(prediction_coefficients)
        out = beta.copy()
        x = beta[..., self.positive_mask]
        if np.any(x <= 0.0):
            raise ValueError("Positive prediction coefficients must be strictly positive.")
        if self.positive_map == "exp":
            out[..., self.positive_mask] = np.log(x)
        else:
            linear = self.softplus_beta * x >= self.softplus_threshold
            inverse = x.copy()
            active = ~linear
            inverse[active] = np.log(
                np.expm1(self.softplus_beta * x[active])
            ) / self.softplus_beta
            out[..., self.positive_mask] = inverse
        return out

    def derivative(self, optimization_coefficients, *, order: int = 1) -> np.ndarray:
        if order not in {1, 2, 3}:
            raise ValueError("Only derivative orders 1, 2, and 3 are supported.")
        beta = self._validate(optimization_coefficients)
        out = np.ones_like(beta) if order == 1 else np.zeros_like(beta)
        x = beta[..., self.positive_mask]
        if self.positive_map == "exp":
            out[..., self.positive_mask] = np.exp(x)
            return out
        out[..., self.positive_mask] = self._softplus_derivative(x, order=order)
        return out

    def jacobian(self, optimization_coefficients) -> np.ndarray:
        beta = self._validate(optimization_coefficients)
        if beta.ndim != 1:
            raise ValueError("jacobian() requires a one-dimensional vector.")
        return np.diag(self.derivative(beta, order=1))

    def covariance_scale(self, optimization_coefficients) -> np.ndarray:
        beta = self._validate(optimization_coefficients)
        if self.covariance_transport == "jacobian":
            return self.derivative(beta, order=1)
        out = np.ones_like(beta)
        predicted = self.forward(beta)
        out[..., self.positive_mask] = predicted[..., self.positive_mask]
        return out

    def transport_covariance(self, optimization_coefficients, covariance) -> np.ndarray:
        beta = self._validate(optimization_coefficients)
        if beta.ndim != 1:
            raise ValueError(
                "transport_covariance() requires one-dimensional coefficients."
            )
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (self.size, self.size):
            raise ValueError(
                f"Covariance must have shape {(self.size, self.size)}, got {cov.shape}."
            )
        scale = self.covariance_scale(beta)
        return np.asarray(scale[:, None] * cov * scale[None, :], dtype=np.float64)

    def subset(self, indices) -> CoefficientTransform:
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(index < 0) or np.any(index >= self.size):
            raise IndexError("Coefficient-transform subset index out of range.")
        mask = self.positive_mask[index]
        if not np.any(mask):
            return IdentityCoefficientTransform(index.size)
        return CoordinatewiseCoefficientTransform(
            positive_mask=mask,
            positive_map=self.positive_map,
            softplus_beta=self.softplus_beta,
            softplus_threshold=self.softplus_threshold,
            covariance_transport=self.covariance_transport,
        )

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        bx = self.softplus_beta * x
        out = x.copy()
        active = bx < self.softplus_threshold
        out[active] = np.log1p(np.exp(bx[active])) / self.softplus_beta
        return out

    def _softplus_derivative(self, x: np.ndarray, *, order: int) -> np.ndarray:
        bx = self.softplus_beta * x
        out = np.ones_like(x) if order == 1 else np.zeros_like(x)
        active = bx < self.softplus_threshold
        exp_bx = np.exp(bx[active])
        denom = 1.0 + exp_bx
        if order == 1:
            out[active] = exp_bx / denom
        elif order == 2:
            out[active] = self.softplus_beta * exp_bx / denom**2
        else:
            out[active] = (
                self.softplus_beta**2
                * exp_bx
                * (1.0 - exp_bx**2)
                / denom**4
            )
        return out


@dataclass(frozen=True)
class BlockCoefficientTransform:
    """Concatenation of independent coefficient-transform blocks."""

    blocks: tuple[CoefficientTransform, ...]

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if any(not isinstance(block, CoefficientTransform) for block in blocks):
            raise TypeError("Every block must satisfy CoefficientTransform.")
        object.__setattr__(self, "blocks", blocks)

    @property
    def size(self) -> int:
        return int(sum(block.size for block in self.blocks))

    @property
    def is_identity(self) -> bool:
        return all(block.is_identity for block in self.blocks)

    @property
    def block_slices(self) -> tuple[slice, ...]:
        slices = []
        start = 0
        for block in self.blocks:
            slices.append(slice(start, start + block.size))
            start += block.size
        return tuple(slices)

    @property
    def positive_mask(self) -> np.ndarray:
        masks = []
        for block in self.blocks:
            mask = getattr(block, "positive_mask", None)
            masks.append(
                np.zeros(block.size, dtype=bool)
                if mask is None
                else np.asarray(mask, dtype=bool)
            )
        return np.concatenate(masks) if masks else np.zeros(0, dtype=bool)

    def _apply(self, coefficients, method: str, **kwargs) -> np.ndarray:
        values = _validate_coefficient_array(coefficients, self.size)
        outputs = [
            getattr(block, method)(values[..., sl], **kwargs)
            for block, sl in zip(self.blocks, self.block_slices, strict=True)
        ]
        return np.concatenate(outputs, axis=-1) if outputs else values.copy()

    def forward(self, optimization_coefficients) -> np.ndarray:
        return self._apply(optimization_coefficients, "forward")

    transform = forward

    def inverse(self, prediction_coefficients) -> np.ndarray:
        return self._apply(prediction_coefficients, "inverse")

    def derivative(self, optimization_coefficients, *, order: int = 1) -> np.ndarray:
        return self._apply(optimization_coefficients, "derivative", order=order)

    def jacobian(self, optimization_coefficients) -> np.ndarray:
        beta = _validate_coefficient_array(optimization_coefficients, self.size)
        if beta.ndim != 1:
            raise ValueError("jacobian() requires a one-dimensional vector.")
        return np.diag(self.derivative(beta, order=1))

    def covariance_scale(self, optimization_coefficients) -> np.ndarray:
        return self._apply(optimization_coefficients, "covariance_scale")

    def transport_covariance(self, optimization_coefficients, covariance) -> np.ndarray:
        beta = _validate_coefficient_array(optimization_coefficients, self.size)
        if beta.ndim != 1:
            raise ValueError(
                "transport_covariance() requires one-dimensional coefficients."
            )
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (self.size, self.size):
            raise ValueError(
                f"Covariance must have shape {(self.size, self.size)}, got {cov.shape}."
            )
        scale = self.covariance_scale(beta)
        return np.asarray(scale[:, None] * cov * scale[None, :], dtype=np.float64)

    def subset(self, indices) -> CoefficientTransform:
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(index < 0) or np.any(index >= self.size):
            raise IndexError("Coefficient-transform subset index out of range.")
        mask = self.positive_mask[index]
        if not np.any(mask):
            return IdentityCoefficientTransform(index.size)
        nonidentity = [block for block in self.blocks if not block.is_identity]
        settings = {
            (
                getattr(block, "positive_map", None),
                getattr(block, "softplus_beta", None),
                getattr(block, "softplus_threshold", None),
                getattr(block, "covariance_transport", None),
            )
            for block in nonidentity
        }
        if len(settings) == 1:
            positive_map, beta, threshold, covariance_transport = settings.pop()
            return CoordinatewiseCoefficientTransform(
                mask,
                positive_map=positive_map,
                softplus_beta=beta,
                softplus_threshold=threshold,
                covariance_transport=covariance_transport,
            )
        if np.array_equal(index, np.arange(self.size, dtype=np.int64)):
            return self
        raise NotImplementedError(
            "Subsetting heterogeneous nonlinear transform blocks is unsupported."
        )


def compose_coefficient_transforms(
    transforms: tuple[CoefficientTransform, ...] | list[CoefficientTransform],
) -> CoefficientTransform:
    """Compose independent blocks, collapsing all-identity layouts."""
    blocks = tuple(transforms)
    if not blocks:
        return IdentityCoefficientTransform(0)
    if len(blocks) == 1:
        return blocks[0]
    if all(block.is_identity for block in blocks):
        return IdentityCoefficientTransform(sum(block.size for block in blocks))
    return BlockCoefficientTransform(blocks)


__all__ = [
    "BlockCoefficientTransform",
    "CoefficientTransform",
    "CoordinatewiseCoefficientTransform",
    "CovarianceScale",
    "IdentityCoefficientTransform",
    "PositiveMap",
    "compose_coefficient_transforms",
]
