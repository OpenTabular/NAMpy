import pytest

from nampy.basemodels.model_output import (
    make_model_output,
    merge_terms,
    validate_feature_names,
)


def test_validate_feature_names_rejects_colon_separator():
    with pytest.raises(ValueError, match="cannot contain ':'"):
        validate_feature_names(["age:income"])


def test_validate_feature_names_rejects_generated_term_collisions():
    with pytest.raises(ValueError, match="generated model term names"):
        validate_feature_names(["transformer_context"], reserved_terms=["transformer_context"])


def test_merge_terms_rejects_duplicate_term_names():
    with pytest.raises(ValueError, match="Duplicate model term name"):
        merge_terms({"age": 1}, {"age": 2})


def test_make_model_output_returns_canonical_sections():
    result = make_model_output(prediction=1)

    assert result == {
        "prediction": 1,
        "terms": {},
        "intercept": None,
        "regularization": {},
        "extras": {},
    }
