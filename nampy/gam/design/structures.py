"""
Core data structures for the compiled GAM predictor.

These dataclasses represent the output of stages 3–5 of the fit pipeline:

Stage 3  Term construction wrapper  -> ConstructedTerm  (gam/design/constructed.py)
Stage 4  Predictor compilation      -> CompiledTerm, CompiledPenalty, CompiledPredictor
Stage 5  Global side conditions     -> updated CompiledPredictor (same types)

None of these classes contain basis-construction logic.  They are containers
that carry pre-computed matrices and metadata forward into fitting and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PenaltySpec:
    """
    Normalised description of one penalty component for a single smooth term.

    Produced by ``get_penalty_definitions()`` on a runtime term and then
    normalised through ``gam/penalties/subsystem.py:normalize_penalty_spec``.

    Attributes
    ----------
    matrix : np.ndarray
        Square, symmetric penalty matrix in the term's current coefficient space.
    smoothing_id : str or None
        Identifier used to share a smoothing parameter across terms (e.g. linked
        basis, factor-by replicates).  None means auto-assign.
    kind : str
        Semantic class: ``"smooth"``, ``"null_space"``, ``"random_effect"``, etc.
    rank : int or None
        Rank of ``matrix``; filled by ``normalize_penalty_spec``.
    null_space_dim : int or None
        Dimension of the null space of ``matrix``; filled by ``normalize_penalty_spec``.
    is_null_space_penalty : bool
        True for selection penalties that penalise the null space of the main penalty.
    sp_mode : str or None
        ``"fixed"`` to pin the smoothing parameter; ``"estimate"`` or None to optimise.
    sp_value : float or None
        Fixed value when ``sp_mode == "fixed"``.
    metadata : dict
        Provenance information (term type, feature names, constraint state, etc.).
    """

    matrix: np.ndarray
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False
    sp_mode: str | None = None
    sp_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledPenalty:
    """
    A penalty component placed within the global coefficient vector of a predictor.

    Produced during stage 4 (``compile_predictor_designs``) and updated by
    stage 5 (``apply_global_side_conditions``).

    Attributes
    ----------
    label : str
        Human-readable term label; matches the owning ``CompiledTerm.label``.
    coef_slice : slice
        Range of columns in the predictor design matrix that this penalty covers.
    matrix : np.ndarray
        Square penalty matrix aligned with the columns in ``coef_slice``.
        shape[0] == shape[1] == len(coef_slice).
    smoothing_index : int
        Index into the model's smoothing-parameter vector.
    term_index : int
        Position of the owning ``CompiledTerm`` in ``CompiledPredictor.compiled_terms``.
        Set at compile time so that side-condition code can match penalties to
        terms without relying on ``coef_slice`` equality (which is ambiguous for
        zero-width terms).
    smoothing_id : str or None
        Smoothing-parameter group identifier.
    kind, rank, null_space_dim, is_null_space_penalty, sp_mode, sp_value, metadata
        Copied from the originating ``PenaltySpec``; see that class for semantics.
    """

    label: str
    coef_slice: slice
    matrix: np.ndarray
    smoothing_index: int
    term_index: int = -1
    smoothing_id: str | None = None
    kind: str = "smooth"
    rank: int | None = None
    null_space_dim: int | None = None
    is_null_space_penalty: bool = False
    sp_mode: str | None = None
    sp_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledTerm:
    """
    A smooth or parametric term placed within a compiled predictor.

    Produced during stage 4 and updated (column-filtered) by stage 5.

    The central invariant is::

        smooth.predict_matrix(X_new) @ basis_transform

    produces the correct prediction columns for this term.  ``basis_transform``
    is the *only* coefficient-space mapping used at prediction time; no other
    transform is applied outside this object.

    Attributes
    ----------
    label : str
        Human-readable term identifier.
    coef_slice : slice
        Range of the global coefficient vector occupied by this term.
        len(coef_slice) == basis_train.shape[1].
    smooth : ConstructedTerm
        The stage-3 wrapper object, used for prediction via ``smooth.predict_matrix``.
    basis_train : np.ndarray
        Final training design block for this term, shape (n_obs, n_coef).
    basis_transform : np.ndarray or None
        Full coefficient transform from the term's stage-4 constructed
        coefficient space to the final fitted coefficient space.
        Prediction always starts from ``smooth.predict_matrix(X_new)``, which is
        already in the constructed-term space.
        Shape: (constructed_n_coef, n_coef).
        None is interpreted as the identity (no transform applied).
    kept_columns : np.ndarray or None
        Indices (in the pre-side-condition space) of surviving columns.
        None when centering was absorbed (original indices non-trivially remapped).
    deleted_columns : np.ndarray or None
        Indices (in the pre-side-condition space) of deleted columns.
    smoothing_indices : list[int]
        Indices into the model smoothing-parameter vector for each penalty.
    smoothing_ids : list[str or None]
        Smoothing-parameter group ids, one per penalty.
    n_penalties : int
        Number of penalties associated with this term.
    term_type : str
        Term family: ``"smooth"``, ``"parametric"``, ``"random_effect"``, etc.
    basis_name : str
        Basis identifier: ``"cr"``, ``"tp"``, ``"ps"``, etc.
    term_id : str
        Stable unique identity assigned at TermSpec creation and carried through
        all pipeline stages.  Used as the canonical key in ``term_index_map`` and
        contribution dicts; never duplicated even when labels collide.
    smoothing_group_id : str or None
        User-supplied smoothing-id from the formula / TermSpec; used to share a
        smoothing parameter across terms (linked basis, factor-by replicates).
    metadata : dict
        Provenance metadata including ``constructor_metadata`` from stage 3.
    """

    label: str
    coef_slice: slice
    smooth: object
    basis_train: np.ndarray
    basis_transform: np.ndarray | None = None
    kept_columns: np.ndarray | None = None
    deleted_columns: np.ndarray | None = None
    smoothing_indices: list[int] = field(default_factory=list)
    smoothing_ids: list[str | None] = field(default_factory=list)
    n_penalties: int = 0
    term_type: str = "smooth"
    basis_name: str = "unknown"
    term_id: str = ""
    smoothing_group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledPredictor:
    """
    A fully assembled linear predictor ready for fitting.

    Produced by stage 4 (``compile_predictor_designs``) and updated by
    stage 5 (``apply_global_side_conditions``).

    The ``design_matrix`` and all ``compiled_terms`` / ``compiled_penalties``
    are mutually consistent: every term's ``coef_slice`` refers to columns of
    ``design_matrix``, and every penalty's ``matrix`` is aligned with those
    same columns.

    Prediction entry point
    ----------------------
    ``build_new_matrix(X_new)`` is the canonical predictor-level design builder
    for new data.  For each term it calls::

        smooth.predict_matrix(X_new) @ basis_transform

    and concatenates the results.  No fit-time transform is re-derived at
    prediction time.
    """

    name: str
    design_matrix: np.ndarray
    compiled_terms: tuple
    compiled_penalties: tuple
    smoothing_parameter_map: dict[str, int]
    n_coef: int
    n_smoothing_params: int
    has_intercept: bool = False
    term_index_map: dict[str, int] = field(default_factory=dict)
    side_condition_Q: np.ndarray | None = None
    smoothing_override_modes: list[str | None] = field(default_factory=list)
    smoothing_override_values: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_new_matrix(self, X_new):
        """
        Build the predictor design matrix for new observations.

        For each compiled term, evaluates the stage-3 constructed-term basis at
        ``X_new`` via ``smooth.predict_matrix(X_new)`` and applies
        ``basis_transform`` — the canonical mapping from constructed-term
        coordinates to final fitted coordinates. No fit-time transforms are
        re-derived.
        """
        if len(self.compiled_terms) == 0:
            return np.empty((len(X_new), 0), dtype=np.float64)

        blocks = []
        for tb in self.compiled_terms:
            B = np.asarray(tb.smooth.predict_matrix(X_new), dtype=np.float64)
            C = tb.basis_transform
            if C is not None:
                B = B @ C
            blocks.append(B)

        return np.column_stack(blocks)


__all__ = ["PenaltySpec", "CompiledPenalty", "CompiledTerm", "CompiledPredictor"]
