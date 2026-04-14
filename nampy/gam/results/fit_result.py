"""Stable fit-result structures consumed outside the numerical engine."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TermFitResult:
    label: str
    term_type: str
    basis_name: str
    coef_slice: tuple[int, int]
    n_coef: int
    edf: float | None = None
    smoothing_indices: list[int] = field(default_factory=list)
    smoothing_ids: list[str | None] = field(default_factory=list)
    smoothing_values: list[float] = field(default_factory=list)
    deleted_columns: list[int] = field(default_factory=list)
    kept_columns: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "label": self.label,
            "term_type": self.term_type,
            "basis_name": self.basis_name,
            "coef_slice": list(self.coef_slice),
            "n_coef": int(self.n_coef),
            "edf": None if self.edf is None else float(self.edf),
            "smoothing_indices": [int(i) for i in self.smoothing_indices],
            "smoothing_ids": list(self.smoothing_ids),
            "smoothing_values": [float(v) for v in self.smoothing_values],
            "deleted_columns": [int(i) for i in self.deleted_columns],
            "kept_columns": [int(i) for i in self.kept_columns],
            "metadata": dict(self.metadata),
        }


@dataclass
class GAMFitResult:
    family_name: str
    link_name: str
    criterion_name: str | None
    criterion_value: float | None
    coef_full: np.ndarray
    intercept: float
    smoothing_params: np.ndarray
    edf_total: float
    edf_by_term: np.ndarray
    trace_H: float
    scale: float
    rss: float | None
    deviance: float
    cov_bayes: np.ndarray | None = None
    cov_freq: np.ndarray | None = None
    side_condition_reports: list[dict[str, Any]] | None = None
    term_results: list[TermFitResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_covariances=False):
        out = {
            "family_name": self.family_name,
            "link_name": self.link_name,
            "criterion_name": self.criterion_name,
            "criterion_value": (
                None if self.criterion_value is None else float(self.criterion_value)
            ),
            "coef_full": np.asarray(self.coef_full, dtype=np.float64).tolist(),
            "intercept": float(self.intercept),
            "smoothing_params": np.asarray(
                self.smoothing_params, dtype=np.float64
            ).tolist(),
            "edf_total": float(self.edf_total),
            "edf_by_term": np.asarray(self.edf_by_term, dtype=np.float64).tolist(),
            "trace_H": float(self.trace_H),
            "scale": float(self.scale),
            "rss": None if self.rss is None else float(self.rss),
            "deviance": float(self.deviance),
            "side_condition_reports": (
                None
                if self.side_condition_reports is None
                else list(self.side_condition_reports)
            ),
            "term_results": [term.to_dict() for term in self.term_results],
            "metadata": dict(self.metadata),
        }

        if include_covariances:
            out["cov_bayes"] = (
                None
                if self.cov_bayes is None
                else np.asarray(self.cov_bayes, dtype=np.float64).tolist()
            )
            out["cov_freq"] = (
                None
                if self.cov_freq is None
                else np.asarray(self.cov_freq, dtype=np.float64).tolist()
            )

        return out


__all__ = ["TermFitResult", "GAMFitResult"]
