"""Tests for the dependency-light contracts shared by NAMpy backends."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.api import (
    AdditivePrediction,
    Capabilities,
    FeatureSchema,
    PersistableModel,
    SupportsCapabilities,
)


def test_feature_schema_round_trip_for_dataframe_and_ndarray():
    frame = pd.DataFrame(
        {
            "height": pd.Series([1.0, 2.0], dtype="float64"),
            "group": pd.Series([1, 2], dtype="int64"),
        }
    )
    frame_schema = FeatureSchema.from_data(frame)

    assert frame_schema.feature_names == ("height", "group")
    assert frame_schema.dtypes == ("float64", "int64")
    assert frame_schema.n_features == 2
    assert frame_schema.validate(frame.copy()) is None

    array = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    array_schema = FeatureSchema.from_data(array)

    assert array_schema.feature_names == ("x0", "x1")
    assert array_schema.dtypes == ("float64", "float64")
    assert array_schema.n_features == 2
    assert array_schema.validate(array.copy()) is None


def test_feature_schema_rejects_feature_name_and_count_mismatches():
    schema = FeatureSchema.from_data(
        pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]})
    )

    with pytest.raises(ValueError, match="Feature name mismatch"):
        schema.validate(pd.DataFrame({"x": [1.0], "renamed": [3.0]}))

    with pytest.raises(ValueError, match="Feature count mismatch"):
        schema.validate(np.ones((2, 3), dtype=np.float64))


def test_additive_prediction_preserves_link_scale_components():
    response = np.asarray([0.25, 0.75], dtype=np.float64)
    link = np.asarray([-1.1, 1.1], dtype=np.float64)
    terms = {
        "x0": np.asarray([-0.8, 0.8], dtype=np.float64),
        "x1": np.asarray([-0.3, 0.3], dtype=np.float64),
    }
    offset = np.asarray([0.1, 0.1], dtype=np.float64)
    prediction = AdditivePrediction(
        response=response,
        link=link,
        terms=terms,
        intercept=0.0,
        backend="gam",
        offset=offset,
    )

    assert prediction.response is response
    assert prediction.link is link
    assert prediction.terms is terms
    assert prediction.intercept == 0.0
    assert prediction.backend == "gam"
    assert prediction.offset is offset
    with pytest.raises(FrozenInstanceError):
        prediction.backend = "neural"  # type: ignore[misc]


class _ContractModel:
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_predict_proba=False,
            supports_standard_errors=False,
            supports_lpmatrix=False,
            supports_term_contributions=True,
        )

    def save_model(self, path: str | Path) -> Path:
        return Path(path)

    @classmethod
    def load_model(cls, path: str | Path) -> _ContractModel:
        del path
        return cls()


def test_runtime_checkable_protocols_are_structural():
    model = _ContractModel()

    assert isinstance(model, SupportsCapabilities)
    assert isinstance(model, PersistableModel)
    assert not isinstance(object(), SupportsCapabilities)
    assert not isinstance(object(), PersistableModel)
