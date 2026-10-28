"""Shared estimator artifact handling, independent of either fitting backend."""

import pickle
from pathlib import Path


def save_estimator(estimator, path):
    """Write the current NAMpy estimator envelope to a trusted pickle file."""
    destination = Path(path)
    payload = {
        "format": "nampy-estimator",
        "version": 1,
        "estimator_class": type(estimator).__name__,
        "estimator": estimator,
    }
    with destination.open("wb") as handle:
        pickle.dump(payload, handle)
    return destination


def load_estimator(path, expected_class):
    """Load a trusted artifact and reject unsupported formats or estimator types."""
    source = Path(path)
    with source.open("rb") as handle:
        loaded = pickle.load(handle)
    if (
        not isinstance(loaded, dict)
        or loaded.get("format") != "nampy-estimator"
        or loaded.get("version") != 1
    ):
        raise ValueError(f"{source} is not a supported NAMpy estimator artifact.")
    estimator = loaded.get("estimator")
    if not isinstance(estimator, expected_class):
        raise TypeError(
            f"{source} contains {type(estimator).__name__}, not {expected_class.__name__}."
        )
    return estimator
