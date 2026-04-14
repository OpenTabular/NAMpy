"""
Formula intent extraction.

This stage mirrors mgcv's interpret.gam bookkeeping: parsed AST -> one extracted
predictor per linear predictor, with declarative parametric/smooth requests but
without defaults, data attachment, or runtime/spec construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parse import ParsedGAMFormula, ParsedParametricTerm, ParsedSmoothTerm


@dataclass(frozen=True)
class ExtractedParametricTerm:
    variables: tuple[str, ...]
    raw_label: str


@dataclass(frozen=True)
class ExtractedSmoothTerm:
    kind: str
    features: tuple[str, ...]
    kwargs: dict
    raw_label: str


ExtractedTerm = ExtractedParametricTerm | ExtractedSmoothTerm


@dataclass(frozen=True)
class ExtractedPredictor:
    predictor_index: int
    predictor_name: str
    response_name: str | None
    intercept: bool
    terms: tuple[ExtractedTerm, ...]
    offset_name: str | None = None
    raw_formula: str = ""


def extract_formula_terms(
    parsed: ParsedGAMFormula, *, drop_intercept=None
) -> list[ExtractedPredictor]:
    n_pred = len(parsed.predictors)
    if drop_intercept is None:
        flags = [False] * n_pred
    elif np.isscalar(drop_intercept):
        flags = [bool(drop_intercept)] * n_pred
    else:
        flags = [bool(v) for v in drop_intercept]
        if len(flags) != n_pred:
            raise ValueError(
                f"drop_intercept must have length {n_pred} for a list-of-formula model, "
                f"got {len(flags)}."
            )

    extracted = []
    for i, (pf, drop_flag) in enumerate(zip(parsed.predictors, flags)):
        terms: list[ExtractedTerm] = []
        for term in pf.terms:
            if isinstance(term, ParsedParametricTerm):
                terms.append(
                    ExtractedParametricTerm(
                        variables=tuple(str(v) for v in term.variables),
                        raw_label=str(term.raw_label),
                    )
                )
                continue

            if isinstance(term, ParsedSmoothTerm):
                terms.append(
                    ExtractedSmoothTerm(
                        kind=str(term.kind),
                        features=tuple(str(f) for f in term.features),
                        kwargs=dict(term.kwargs),
                        raw_label=str(term.raw_label),
                    )
                )
                continue

            raise TypeError(f"Unknown parsed term type: {type(term)}")

        extracted.append(
            ExtractedPredictor(
                predictor_index=i,
                predictor_name=f"eta{i+1}",
                response_name=pf.response_name,
                intercept=(False if drop_flag else bool(pf.intercept)),
                terms=tuple(terms),
                offset_name=pf.offset_name,
                raw_formula=pf.raw_formula,
            )
        )

    return extracted


__all__ = [
    "ExtractedPredictor",
    "ExtractedTerm",
    "ExtractedParametricTerm",
    "ExtractedSmoothTerm",
    "extract_formula_terms",
]
