"""Transient engine state and penalized-system contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PenalizedSystem:
    X: np.ndarray | None = None
    A: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    P: np.ndarray | None = None
    penalty_matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None
    penalized_system_rank: int | None = None
    dropped_column_indices: np.ndarray | None = None


@dataclass(frozen=True)
class FitState:
    X: np.ndarray | None = None
    A: np.ndarray | None = None
    A_inv: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    P: np.ndarray | None = None
    penalty_matrix: np.ndarray | None = None
    working_weights: np.ndarray | None = None
    fisher_weights: np.ndarray | None = None
    working_response: np.ndarray | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None
    penalized_system_rank: int | None = None
    dropped_column_indices: np.ndarray | None = None
    scale: float | None = None

    def to_penalized_system(self) -> PenalizedSystem:
        return PenalizedSystem(
            X=self.X,
            A=self.A,
            XtWX=self.XtWX,
            P=self.P,
            penalty_matrix=self.penalty_matrix,
            offset=self.offset,
            log_det_XtWX_plus_penalty=self.log_det_XtWX_plus_penalty,
            penalized_system_rank=self.penalized_system_rank,
            dropped_column_indices=self.dropped_column_indices,
        )


__all__ = ["PenalizedSystem", "FitState"]
