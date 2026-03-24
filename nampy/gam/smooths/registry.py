import numpy as np

_SMOOTH_REGISTRY = {}


def register_smooth(name: str):
    name = str(name).lower()

    def decorator(cls):
        _SMOOTH_REGISTRY[name] = cls
        return cls

    return decorator


def available_smooths():
    return dict(_SMOOTH_REGISTRY)


def make_smooth_term(kind: str, *args, **kwargs):
    key = str(kind).lower()
    if key not in _SMOOTH_REGISTRY:
        raise ValueError(
            f"Unknown smooth kind {kind!r}. "
            f"Available smooths: {sorted(_SMOOTH_REGISTRY)}"
        )
    return _SMOOTH_REGISTRY[key](*args, **kwargs)
