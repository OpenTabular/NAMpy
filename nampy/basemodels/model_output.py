from collections.abc import Iterable, Mapping
from typing import Any


def validate_feature_names(
    feature_names: Iterable[str],
    *,
    reserved_terms: Iterable[str] = (),
) -> None:
    """Validate feature names for the flat terms namespace."""
    feature_names = [str(name) for name in feature_names]

    bad_separator_names = sorted({name for name in feature_names if ":" in name})
    if bad_separator_names:
        raise ValueError(
            "Feature names cannot contain ':', because ':' is used to name "
            f"interaction terms. Invalid names: {bad_separator_names}."
        )

    reserved = set(reserved_terms)
    collisions = sorted(set(feature_names) & reserved)
    if collisions:
        raise ValueError(
            "Feature names collide with generated model term names: "
            f"{collisions}."
        )


def merge_terms(*term_maps: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> dict:
    """Merge term mappings and fail on duplicate flat term names."""
    terms = {}
    for term_map in term_maps:
        if not term_map:
            continue
        items = term_map.items() if isinstance(term_map, Mapping) else term_map
        for name, value in items:
            if not isinstance(name, str):
                raise TypeError(f"Term names must be strings, got {type(name)!r}.")
            if name in terms:
                raise ValueError(f"Duplicate model term name {name!r}.")
            terms[name] = value
    return terms


def _validate_mapping(name: str, values: Mapping[str, Any]) -> dict:
    values = dict(values)
    for key in values:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings, got {type(key)!r}.")
    return values


def make_model_output(
    *,
    prediction,
    terms: Mapping[str, Any] | None = None,
    intercept=None,
    regularization: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict:
    """Build the canonical NAMpy model output dictionary."""
    return {
        "prediction": prediction,
        "terms": _validate_mapping("terms", terms or {}),
        "intercept": intercept,
        "regularization": _validate_mapping("regularization", regularization or {}),
        "extras": _validate_mapping("extras", extras or {}),
    }
