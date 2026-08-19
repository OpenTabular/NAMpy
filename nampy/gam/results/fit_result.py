"""Stable fit-result structures consumed outside the fit subsystem."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .solution import FitResult


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
    """Presentation-level fit summary.

    The numeric record is held once, by reference, in ``core`` (the solver's
    :class:`FitResult`); this class adds only derived presentation data.
    """

    family_name: str
    link_name: str
    criterion_name: str | None
    criterion_value: float | None
    core: FitResult
    smoothing_params: np.ndarray
    side_condition_reports: list[dict[str, Any]] | None = None
    term_results: list[TermFitResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def coef_full(self) -> np.ndarray:
        return self.core.coef_full

    @property
    def intercept(self) -> float:
        return self.core.intercept

    @property
    def edf_total(self) -> float:
        return self.core.edf

    @property
    def edf_by_term(self) -> np.ndarray | None:
        return self.core.edf_by_term

    @property
    def trace_H(self) -> float:
        return self.core.trace_H

    @property
    def scale(self) -> float:
        return self.core.scale

    @property
    def rss(self) -> float | None:
        return self.core.rss

    @property
    def deviance(self) -> float:
        return self.core.deviance

    @property
    def cov_bayes(self) -> np.ndarray | None:
        return self.core.cov_bayes

    @property
    def cov_freq(self) -> np.ndarray | None:
        return self.core.cov_freq

    @property
    def cov_unconditional(self) -> np.ndarray | None:
        return self.core.cov_unconditional

    @property
    def cov_unconditional_space(self) -> str | None:
        return self.core.cov_unconditional_space

    @property
    def edf2(self) -> np.ndarray | None:
        return self.core.edf2

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
            "edf2": (
                None
                if self.edf2 is None
                else np.asarray(self.edf2, dtype=np.float64).tolist()
            ),
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
            out["cov_unconditional"] = (
                None
                if self.cov_unconditional is None
                else np.asarray(self.cov_unconditional, dtype=np.float64).tolist()
            )
            out["cov_unconditional_space"] = self.cov_unconditional_space

        return out


__all__ = ["TermFitResult", "GAMFitResult"]
