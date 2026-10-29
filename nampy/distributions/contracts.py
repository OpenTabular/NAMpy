"""Probability-law and fit-status contracts, independent of either backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class DistributionSpec:
    """Probability-law metadata, independent of either fitting backend.

    ``operations`` explicitly lists implemented probability operations; it says
    nothing about fitting, smoothing selection, or statistical inference.
    ``parameterization`` versions the meaning of the named natural parameters.
    Scalar continuous, discrete, and mixed responses have ``event_shape=()``.
    ``context_names`` lists fixed rowwise inputs, separate from parameters and
    observation weights.
    ``atom_locations`` declares the reference atoms for mixed-law comparison.
    None keeps atom metadata unknown; probability queries remain available,
    but cross-family scoring cannot assume a common mixed reference measure.
    """

    name: str
    parameter_names: tuple[str, ...]
    operations: frozenset[str]
    parameterization: str
    response_kind: Literal["continuous", "discrete", "mixed"]
    event_shape: tuple[int, ...] = ()
    context_names: tuple[str, ...] = ()
    atom_locations: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Distribution parameter names must be nonempty and unique."
            )
        if self.response_kind not in {"continuous", "discrete", "mixed"}:
            raise ValueError("Unknown distribution response kind.")
        context_names = tuple(self.context_names)
        if any(not isinstance(name, str) or not name for name in context_names) or len(
            set(context_names)
        ) != len(context_names):
            raise ValueError("Distribution context names must be nonempty and unique.")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "operations", frozenset(self.operations))
        object.__setattr__(self, "event_shape", tuple(self.event_shape))
        object.__setattr__(self, "context_names", context_names)
        if self.atom_locations is not None:
            atoms = tuple(float(value) for value in self.atom_locations)
            if (
                self.response_kind != "mixed"
                or not atoms
                or not np.all(np.isfinite(atoms))
                or any(a >= b for a, b in zip(atoms, atoms[1:], strict=False))
            ):
                raise ValueError(
                    "Declared atoms require a mixed law and finite, strictly increasing locations."
                )
            object.__setattr__(self, "atom_locations", atoms)


@dataclass(frozen=True)
class DistributionFitStatus:
    """Backend-owned eligibility evidence, separate from probability support.

    Statistical convergence and completed neural training are distinct states.
    Unknown, failed and nonconverged fits are not eligible for family ranking.
    """

    backend: str
    state: Literal["converged", "completed", "nonconverged", "failed", "unknown"]
    message: str = ""

    def __post_init__(self):
        if not self.backend or self.state not in {
            "converged",
            "completed",
            "nonconverged",
            "failed",
            "unknown",
        }:
            raise ValueError("Invalid distribution fit status.")

    @property
    def eligible(self) -> bool:
        return self.state in {"converged", "completed"}


class DistributionLaw(Protocol):
    """Structural interface consumed by distribution predictions.

    Implementations own parameter domains and probability mathematics. The
    prediction object supplies finite, row-aligned parameter vectors (missing
    prediction rows are removed before dispatch). ``evaluate`` receives only
    operations explicitly declared by ``spec.operations``. Quantile values have
    shape ``(n_probs,)``; other evaluation values have shape ``(n_rows,)``.
    Results follow the public distribution-prediction shape conventions.
    ``context`` contains row-aligned fixed inputs such as trial counts, sliced
    with the same missing-row mask as parameters. Backend-dependent fixed
    configuration belongs to the law instance, not a solver.
    """

    @property
    def spec(self) -> DistributionSpec: ...

    def validate_parameters(
        self,
        parameters: Mapping[str, np.ndarray],
        *,
        context: Mapping[str, np.ndarray],
    ) -> None: ...

    def evaluate(
        self,
        operation: str,
        parameters: Mapping[str, np.ndarray],
        *,
        context: Mapping[str, np.ndarray],
        values: np.ndarray | None = None,
        n_draws: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...

