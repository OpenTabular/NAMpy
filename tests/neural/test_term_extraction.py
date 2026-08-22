"""Unit tests for the generic additive-term extraction functions."""

import numpy as np
import pandas as pd
import pytest

from nampy.neural.architectures.components.term_extraction import (
    aggregate_term_values,
    build_terms_frame,
    center_main_effects,
    purify_interactions,
    terms_from_feature_selectors,
)

TERMS = [0, 1, (0, 1)]


def _toy_data():
    rng = np.random.default_rng(0)
    x0 = rng.integers(0, 3, size=200)
    x1 = rng.integers(0, 2, size=200)
    X = pd.DataFrame({"a": x0, "b": x1})
    # Per-term outputs are deterministic functions of the feature values.
    f0 = 0.5 * x0.astype(float)
    f1 = -1.0 + 2.0 * x1.astype(float)
    f01 = 0.25 * x0 * x1 + 0.1
    results = np.stack([f0, f1, f01], axis=1)[:, :, None]
    return X, results


def _reconstruction(vals, X):
    """Per-row sum of main and interaction contributions plus intercept."""
    total = np.full(len(X), float(vals.get(-1, 0.0)))
    total += vals[0].loc[X["a"]].values
    total += vals[1].loc[X["b"]].values
    total += np.array(
        [vals[(0, 1)].loc[r, c] for r, c in zip(X["a"].values, X["b"].values, strict=True)]
    )
    return total


def test_terms_from_feature_selectors():
    import torch

    # 4 features, 3 selectors, depth 2: selector 0 uses feature 1, selector 1
    # uses features 0 and 2 (an interaction), selector 2 duplicates selector 0.
    fs = torch.zeros(4, 3, 2)
    fs[1, 0, 0] = 0.7
    fs[0, 1, 0] = 0.4
    fs[2, 1, 1] = 0.6
    fs[1, 2, 1] = 0.3

    terms = terms_from_feature_selectors(fs)
    assert terms == [1, (0, 2)]

    terms, inverse = terms_from_feature_selectors(fs, return_inverse=True)
    assert terms == [1, (0, 2)]
    # selectors 0 and 2 map to the same term, selector 1 to the other
    assert inverse[0] == inverse[2] != inverse[1]


def test_aggregate_term_values_groups_by_unique_value():
    X, results = _toy_data()
    vals, counts = aggregate_term_values(results, X, TERMS)

    np.testing.assert_allclose(vals[0].loc[[0, 1, 2]].values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(vals[1].loc[[0, 1]].values, [-1.0, 1.0])
    assert vals[(0, 1)].shape == (3, 2)
    np.testing.assert_allclose(vals[(0, 1)].loc[2, 1], 0.25 * 2 * 1 + 0.1)
    assert counts[0].sum() == len(X)
    assert counts[(0, 1)].values.sum() == len(X)


def test_purify_interactions_preserves_predictions_and_zeroes_margins():
    X, results = _toy_data()
    vals, counts = aggregate_term_values(results, X, TERMS)
    before = _reconstruction(vals, X)

    purify_interactions(vals, counts, tol=1e-10)

    np.testing.assert_allclose(_reconstruction(vals, X), before, atol=1e-8)
    inter, w = vals[(0, 1)], counts[(0, 1)]
    row_means = (inter * w).sum(axis=1).values / w.sum(axis=1).values
    col_means = (inter * w).sum(axis=0).values / w.sum(axis=0).values
    np.testing.assert_allclose(row_means, 0.0, atol=1e-9)
    np.testing.assert_allclose(col_means, 0.0, atol=1e-9)


def test_center_main_effects_moves_weighted_means_to_intercept():
    X, results = _toy_data()
    vals, counts = aggregate_term_values(results, X, TERMS)
    vals[-1] = 0.0
    before = _reconstruction(vals, X)

    center_main_effects(vals, counts, bias=1.5)

    np.testing.assert_allclose(_reconstruction(vals, X), before + 1.5, atol=1e-8)
    for t in (0, 1):
        assert abs(np.average(vals[t].values, weights=counts[t].values)) < 1e-12


def test_build_terms_frame_shape_and_order():
    X, results = _toy_data()
    vals, counts = aggregate_term_values(results, X, TERMS)
    vals[-1] = 0.3

    df = build_terms_frame(vals, counts, X.columns)

    assert list(df["feat_idx"]) == [-1, 0, 1, (0, 1)]
    assert list(df["feat_name"]) == ["offset", "a", "b", "a_b"]
    row = df[df["feat_idx"] == 0].iloc[0]
    assert row["x"] == [0, 1, 2]
    expected_imp = np.average(np.abs(row["y"]), weights=row["counts"])
    assert row["importance"] == pytest.approx(expected_imp)
