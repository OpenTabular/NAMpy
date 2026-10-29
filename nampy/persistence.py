"""Versioned trusted artifacts containing backend-owned fitted state."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"NAMPY-ESTIMATOR 5\n"


@dataclass(frozen=True)
class EstimatorState:
    estimator_class: type
    attributes: dict


def _estimator_state(estimator):
    exporter = getattr(estimator, "_artifact_state", None)
    attributes = dict(estimator.__dict__) if exporter is None else exporter()
    return EstimatorState(type(estimator), attributes)


def _restore_estimator(state):
    estimator = state.estimator_class.__new__(state.estimator_class)
    restore = getattr(estimator, "__setstate__", None)
    if restore is None:
        estimator.__dict__.update(state.attributes)
    else:
        restore(state.attributes)
    return estimator


def snapshot_estimator(estimator):
    """Isolate current fitted state using existing backend artifact ownership.

    The in-memory round trip shares no mutable fitted arrays/modules with the
    source and uses the same execution-state exclusions as save/load.
    """
    state = pickle.loads(pickle.dumps(_estimator_state(estimator)))
    return _restore_estimator(state)


def save_estimator(estimator, path):
    """Write current fitted state; runtime trainers and caches are excluded."""
    destination = Path(path)
    state = _estimator_state(estimator)
    header = {
        "format": "nampy-estimator",
        "version": 5,
        "estimator_class": type(estimator).__name__,
    }
    with destination.open("wb") as handle:
        handle.write(_MAGIC)
        handle.write(json.dumps(header, sort_keys=True).encode("utf-8") + b"\n")
        pickle.dump(state, handle)
    return destination


def load_estimator(path, expected_class):
    """Check the format before unpickling a trusted current-version artifact."""
    source = Path(path)
    invalid = f"{source} is not a supported NAMpy estimator artifact."
    with source.open("rb") as handle:
        if handle.readline(len(_MAGIC) + 1) != _MAGIC:
            raise ValueError(invalid)
        try:
            header = json.loads(handle.readline(16384))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(invalid) from error
        if (
            not isinstance(header, dict)
            or header.get("format") != "nampy-estimator"
            or header.get("version") != 5
        ):
            raise ValueError(invalid)
        if header.get("estimator_class") != expected_class.__name__:
            raise TypeError(
                f"{source} contains {header.get('estimator_class')}, not {expected_class.__name__}."
            )
        state = pickle.load(handle)
    if not isinstance(state, EstimatorState) or not issubclass(
        state.estimator_class, expected_class
    ):
        raise TypeError(f"{source} contains an unexpected estimator state.")
    return _restore_estimator(state)
