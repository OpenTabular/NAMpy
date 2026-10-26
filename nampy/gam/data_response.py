"""Response initialization shared by the facade and array-facing fit core."""

import warnings

import numpy as np


def initialize_binomial_response(model, y, sample_weight, n_rows):
    """Port stats::binomInitialize's two-column response branch.

    The response used by PIRLS is a proportion, prior weights include trial
    counts, and initialization/likelihood retain the original denominators.
    """
    model.binomial_trials_ = None
    model.grouped_response_ = None
    if str(getattr(model.family, "name", "")) not in {"binomial", "quasibinomial"}:
        return y, sample_weight
    model.family._trial_counts = None
    values = np.asarray(y, dtype=np.float64)
    if values.ndim <= 1 or (values.ndim == 2 and values.shape[1] == 1):
        # stats::binomInitialize modifies the response itself at zero weights.
        values = values.copy()
        if sample_weight is not None:
            values[np.asarray(sample_weight) == 0] = 0
        return values, sample_weight
    if values.ndim != 2 or values.shape != (int(n_rows), 2):
        raise ValueError(
            "Binomial y must be a proportion vector or an (n, 2) matrix "
            "of success and failure counts."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Binomial success and failure counts must be finite and non-negative."
        )
    if model.family.name == "binomial" and np.any(np.abs(values - np.rint(values)) > 0.001):
        warnings.warn("non-integer counts in a binomial glm!", stacklevel=3)
    trials = values.sum(axis=1)
    weights = (
        np.ones(n_rows)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        weights.shape != (n_rows,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError(
            "sample_weight must be a finite non-negative vector with one value per row."
        )
    proportion = np.divide(
        values[:, 0], trials, out=np.zeros(n_rows), where=trials != 0
    )
    model.binomial_trials_ = trials.copy()
    model.grouped_response_ = values.copy()
    model.family._trial_counts = trials.copy()
    return proportion, weights * trials
