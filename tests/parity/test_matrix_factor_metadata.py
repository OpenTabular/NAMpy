"""Categorical matrix cells retain pointwise levels for smooth constructors."""

import numpy as np
import pandas as pd

from nampy.gam.data import feature_values_for_levels
from nampy.gam.specs.build import _factor_levels_metadata_for_features


def test_categorical_matrix_levels_are_pooled_without_changing_cells():
    frame = pd.DataFrame({"F": [["b", "a"], ["c", "b"]]})
    metadata = _factor_levels_metadata_for_features(frame, ["F"])
    assert metadata == {"F": {"levels": ["a", "b", "c"], "ordered": False}}
    assert frame.F.tolist() == [["b", "a"], ["c", "b"]]
    np.testing.assert_array_equal(
        feature_values_for_levels(frame.F), ["b", "a", "c", "b"]
    )


def test_scalar_ordered_factor_levels_keep_declared_order():
    frame = pd.DataFrame(
        {"F": pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True)}
    )
    assert _factor_levels_metadata_for_features(frame, ["F"]) == {
        "F": {"levels": ["b", "a"], "ordered": True}
    }


def test_numeric_looking_matrix_labels_remain_distinct_factor_levels():
    frame = pd.DataFrame({"F": [["01", "1"], ["2", "01"]]})
    assert _factor_levels_metadata_for_features(frame, ["F"]) == {
        "F": {"levels": ["01", "1", "2"], "ordered": False}
    }


def test_boolean_and_explicit_categorical_matrix_cells_keep_factor_metadata():
    frame = pd.DataFrame(
        {
            "logical": [[True, False], [False, True]],
            "coded": [pd.Categorical([1, 2]), pd.Categorical([2, 1])],
        }
    )
    metadata = _factor_levels_metadata_for_features(frame, ["logical", "coded"])
    assert metadata["logical"]["levels"] == [False, True]
    assert metadata["coded"]["levels"] == [1, 2]
