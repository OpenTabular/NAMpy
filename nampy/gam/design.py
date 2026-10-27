"""Compile identified bases, penalties and offsets without fitting a model.

This interface shares the formula compiler, prediction schema, smoothing
resolution and known-offset kernels with GAM. It owns no likelihood,
coefficients, covariance or inference state.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .compiler.compile_model import compile_model
from .compiler.structures import CompiledModel
from .data import (
    coerce_formula_design_inputs,
    coerce_optional_offset,
    coerce_X,
    combine_offsets,
)
from .fit.offsets import compiled_offsets
from .fit.penalized_system import build_full_design
from .fit.smoothing_params import resolve_compiled_smoothing_params
from .predict.linear_predictor_matrix import compiled_prediction_matrices
from .specs.modeling import make_predictor_specs, prepare_formula_inputs

__all__ = ["compile_design", "CompiledDesign", "DesignEvaluation", "DesignTerm"]


def response_vector(value, *, size=None, name="y"):
    """Normalize a finite (n,) or (n,1) target; reject other output shapes."""
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (n,) or (n,1).")
    if size is not None and vector.shape != (size,):
        raise ValueError(f"{name} must contain {size} observations.")
    return vector


def observation_weights(value, size):
    weights = (
        np.ones(size)
        if value is None
        else response_vector(value, size=size, name="sample_weight")
    )
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("sample_weight must be nonnegative with positive sum.")
    return weights


@dataclass(frozen=True)
class _Options:
    k: int
    basis: str
    fit_intercept: bool
    knots: object = None
    select: bool = False


@dataclass(frozen=True)
class DesignEvaluation:
    """A float basis matrix (n,p) and known additive offset vector (n,)."""

    matrix: np.ndarray
    offset: np.ndarray


@dataclass(frozen=True)
class DesignTerm:
    """One term's label, coefficient indices (q,), and source feature names."""

    label: str
    term_id: str
    coefficient_indices: np.ndarray
    features: tuple[str, ...]


@dataclass(frozen=True)
class CompiledDesign:
    """Reusable single-predictor design in identified fit coordinates.

    ``matrix`` is float (n,p), ``penalty`` float (p,p), ``training_offset``
    float (n,), ``sample_weight`` float (n,), and optional ``response`` float
    (n,). ``smoothing_params`` is float (s,). ``terms`` describes coefficient
    ownership; ``intercept_indices`` identifies separate intercept columns.
    Compiler-owned bases and preprocessing state are retained for evaluation,
    but no statistical fit is performed. Treat arrays as read-only.
    """

    matrix: np.ndarray
    penalty: np.ndarray
    training_offset: np.ndarray
    sample_weight: np.ndarray
    response: np.ndarray | None
    smoothing_params: np.ndarray
    terms: tuple[DesignTerm, ...]
    intercept_indices: np.ndarray
    feature_names: tuple[str, ...]
    _compiled: CompiledModel
    _preprocess_state: dict | None
    _fit_intercept: bool

    def evaluate(self, X, *, offset=None):
        """Return DesignEvaluation for DataFrame/array X (m,d).

        Formula schemas require a DataFrame. Formula and smooth-owned offsets
        are evaluated on X; explicit numeric offset (m,) is added to them.
        Separate training offsets are never carried to new data. Column order
        for DataFrames follows the fitted schema, not the caller's ordering.
        """
        if self._preprocess_state is not None:
            raw, _, formula_offset = coerce_formula_design_inputs(
                X, self._preprocess_state, self._compiled
            )
        else:
            if isinstance(X, pd.DataFrame):
                X = X.loc[:, list(self.feature_names)]
            raw, _ = coerce_X(None, X)
            if raw.shape[1] != len(self.feature_names):
                raise ValueError("Prediction columns differ from the fitted design.")
            formula_offset = None
        _, matrix = compiled_prediction_matrices(
            self._compiled, raw, fit_intercept=self._fit_intercept, fit_coordinates=True
        )
        known = combine_offsets(
            formula_offset,
            compiled_offsets(self._compiled, raw),
            coerce_optional_offset(offset, len(raw)),
        )
        if known is None:
            known = np.zeros(len(raw))
        return DesignEvaluation(matrix, coerce_optional_offset(known, len(raw)))


def compile_design(
    X=None,
    y=None,
    *,
    data=None,
    formula=None,
    k=10,
    basis="tp",
    fit_intercept=True,
    smoothing_params=None,
    knots=None,
    sample_weight=None,
    offset=None,
):
    """Compile a single unconstrained predictor without fitting a likelihood.

    Parameters
    ----------
    X, data : DataFrame or array-like of shape (n,d), optional
        Supply exactly one input. Formula mode requires a DataFrame; data is
        an alias for X. Supplying both raises instead of mixing their rows.
    y : array-like of shape (n,) or (n,1), optional
        Response, or omit to use the formula response column. Response-free
        formulas may compile bases with response=None.
    formula : str or None, default None
        Single NAMpy predictor formula. None creates main-effect smooths.
    k : int, default 10
        Default implicit basis dimension.
    basis : str, default 'tp'
        Default ordinary smooth basis; positive coefficient maps are rejected.
    fit_intercept : bool, default True
        Array-mode intercept policy; an explicit formula owns its intercept.
    smoothing_params : scalar, sequence, mapping or None, default None
        Fixed nonnegative strengths; None uses ones. Formula fixed overrides
        and linked smoothing IDs use the same resolution as GAM.
    knots : mapping or None, default None
        Constructor knots using GAM's existing knot contract.
    sample_weight : array-like (n,) or str or None, default None
        Nonnegative observation weights with positive sum. A string selects a
        column in formula data. None means ones.
    offset : array-like (n,) or None, default None
        Known training-only offset, added to formula/smooth-owned offsets.

    Returns
    -------
    CompiledDesign
        Identified matrix, penalty, offsets, response, weights and term layout,
        with evaluate(X, offset=None) for new data. No solver is called.
    """
    if X is not None and data is not None:
        raise ValueError("Supply either X or data, not both.")
    frame = data if data is not None else X
    if frame is None:
        raise ValueError("X or data is required.")
    options = _Options(k, basis, fit_intercept, knots)
    state = None
    if formula is not None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Formula compilation requires a DataFrame.")
        _, specs, raw, names, target, _, formula_offset, state = prepare_formula_inputs(
            options, frame, formula, y=y, knots=knots
        )
        if len(specs) != 1:
            raise ValueError("compile_design requires one linear predictor.")
        intercept = bool(specs[0].has_intercept)
    else:
        raw, names = coerce_X(None, frame)
        specs = make_predictor_specs(options, names, knots=knots)
        target, formula_offset, intercept = y, None, bool(fit_intercept)
    response = None if target is None else response_vector(target, size=len(raw))
    if isinstance(sample_weight, str):
        if formula is None:
            raise ValueError("Named sample weights require formula data.")
        sample_weight = frame[sample_weight]
    weights = observation_weights(sample_weight, len(raw))
    compiled = compile_model(raw, names, specs, fit_intercept=intercept)
    if len(compiled.predictors) != 1 or np.any(compiled.positive_coefficient_mask):
        raise NotImplementedError(
            "compile_design supports a single unconstrained predictor."
        )
    strengths, _, _ = resolve_compiled_smoothing_params(
        compiled, smoothing_params, allow_zero=True
    )
    matrix = build_full_design(compiled.design_matrix, intercept)
    penalty = np.zeros((matrix.shape[1], matrix.shape[1]))
    mapping = compiled.coef_reduced_to_full_idx
    for block in compiled.compiled_penalties:
        indices = mapping[block.coef_slice]
        penalty[np.ix_(indices, indices)] += (
            strengths[block.smoothing_index] * block.matrix
        )
    terms = tuple(
        DesignTerm(
            term.label,
            term.term_id,
            mapping[term.coef_slice].copy(),
            tuple(term.feature_info.feature_names),
        )
        for term in compiled.compiled_terms
    )
    known = combine_offsets(
        formula_offset,
        compiled_offsets(compiled),
        coerce_optional_offset(offset, len(raw)),
    )
    if known is None:
        known = np.zeros(len(raw))
    return CompiledDesign(
        matrix,
        penalty,
        coerce_optional_offset(known, len(raw)),
        weights,
        response,
        strengths,
        terms,
        np.array([0], dtype=int) if intercept else np.empty(0, dtype=int),
        tuple(names),
        compiled,
        state,
        intercept,
    )
