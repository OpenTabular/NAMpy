"""Smoke tests for preprocessing via pretab.

NAMpy delegates all preprocessing to the pretab library. These tests ensure
that the pretab Preprocessor works with NAMpyDataModule (fit, transform,
get_feature_info(verbose=False) returning num_info, cat_info, emb_info).
"""
import pytest

from pretab.preprocessor import Preprocessor


def test_pretab_preprocessor_fit_transform(mixed_data):
    """Pretab Preprocessor fits and transforms mixed data."""
    X, y = mixed_data
    preprocessor = Preprocessor(
        task="regression",
        n_bins=8,
    )
    preprocessor.fit(X, y)
    transformed = preprocessor.transform(X)
    assert transformed is not None
    assert len(transformed) > 0
    for arr in transformed.values():
        assert arr.shape[0] == len(X)


def test_pretab_preprocessor_get_feature_info(mixed_data):
    """Pretab get_feature_info(verbose=False) returns (num_info, cat_info, emb_info)."""
    X, y = mixed_data
    preprocessor = Preprocessor(task="regression", n_bins=8)
    preprocessor.fit(X, y)
    # NAMpyDataModule expects get_feature_info(verbose=False) -> (num, cat, emb)
    result = preprocessor.get_feature_info(verbose=False)
    assert isinstance(result, (tuple, list)), "get_feature_info should return a sequence"
    assert len(result) >= 3, (
        "get_feature_info(verbose=False) should return (num, cat, emb) for NAMpyDataModule"
    )
    num_info, cat_info, emb_info = result[0], result[1], result[2]
    assert isinstance(num_info, dict)
    assert isinstance(cat_info, dict)
    assert emb_info is None or isinstance(emb_info, dict)


def test_pretab_preprocessor_get_set_params(mixed_data):
    """Pretab Preprocessor supports get_params and set_params for sklearn compatibility."""
    X, y = mixed_data
    preprocessor = Preprocessor(task="regression", n_bins=5)
    params = preprocessor.get_params()
    assert isinstance(params, dict)
    # set_params should not raise; used by sklearn-style models
    preprocessor.set_params(n_bins=7)
