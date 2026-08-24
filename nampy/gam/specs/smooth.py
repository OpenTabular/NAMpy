from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence, Union


@dataclass(frozen=True)
class BaseSmoothSpec:
    special: str
    bs: Any
    k: Any = -1
    fx: Any = False
    select: bool = False
    sp: Any = None
    knots: Any = None

    def to_basis_options(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CubicRegressionSmoothSpec(BaseSmoothSpec):
    bs: str = "cr"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class CyclicCubicRegressionSmoothSpec(BaseSmoothSpec):
    bs: str = "cc"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class CubicShrinkageSmoothSpec(BaseSmoothSpec):
    bs: str = "cs"
    constraint_mode: str = "auto"
    shared_basis_setup: Any = None
    pc: Any = None


@dataclass(frozen=True)
class PSplineSmoothSpec(BaseSmoothSpec):
    bs: str = "ps"
    m: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class DerivativeBSplineSmoothSpec(BaseSmoothSpec):
    bs: str = "bs"
    m: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class DuchonSplineSmoothSpec(BaseSmoothSpec):
    bs: str = "ds"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class GaussianProcessSmoothSpec(BaseSmoothSpec):
    bs: str = "gp"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class ShapeConstrainedSmoothSpec(BaseSmoothSpec):
    """SCAM SCOP-spline specification for a named shape basis code."""

    bs: str = "mpi"
    m: Any = None
    xt: Any = None


@dataclass(frozen=True)
class ThinPlateSmoothSpec(BaseSmoothSpec):
    bs: str = "tp"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class ThinPlateShrinkageSmoothSpec(BaseSmoothSpec):
    bs: str = "ts"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"
    pc: Any = None


@dataclass(frozen=True)
class RandomEffectSmoothSpec(BaseSmoothSpec):
    bs: str = "re"
    xt: Any = None


@dataclass(frozen=True)
class FactorSmoothInteractionSpec(BaseSmoothSpec):
    bs: str = "fs"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"


@dataclass(frozen=True)
class SumToZeroFactorSmoothSpec(BaseSmoothSpec):
    bs: str = "sz"
    m: Any = None
    xt: Any = None
    constraint_mode: str = "auto"


@dataclass(frozen=True)
class TensorProductSmoothSpec(BaseSmoothSpec):
    special: str = "te"
    bs: Any = "cr"
    m: Any = None
    xt: Any = None
    d: Any = None
    pc: Any = None


@dataclass(frozen=True)
class TensorInteractionSmoothSpec(BaseSmoothSpec):
    special: str = "ti"
    bs: Any = "cr"
    m: Any = None
    xt: Any = None
    mc: Any = None
    d: Any = None
    pc: Any = None


SmoothSpec = Union[
    CubicRegressionSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    DerivativeBSplineSmoothSpec,
    DuchonSplineSmoothSpec,
    GaussianProcessSmoothSpec,
    CubicShrinkageSmoothSpec,
    PSplineSmoothSpec,
    ShapeConstrainedSmoothSpec,
    ThinPlateSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    RandomEffectSmoothSpec,
    FactorSmoothInteractionSpec,
    SumToZeroFactorSmoothSpec,
    TensorProductSmoothSpec,
    TensorInteractionSmoothSpec,
]


def replace_smooth_spec(spec: SmoothSpec, **changes: Any) -> SmoothSpec:
    return replace(spec, **changes)


def tensor_basis_list(
    spec: Union[TensorProductSmoothSpec, TensorInteractionSmoothSpec],
    n_features: int,
) -> list[Any]:
    basis = spec.bs
    if isinstance(basis, str):
        return [basis] * n_features
    if isinstance(basis, Sequence):
        basis_list = list(basis)
        if len(basis_list) != n_features:
            raise ValueError(f"Expected length {n_features}, got {len(basis_list)}.")
        return basis_list
    return [basis] * n_features
