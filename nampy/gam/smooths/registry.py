from __future__ import annotations

from ..basis_registry import register_basis_runtime, require_basis_descriptor

_SMOOTH_REGISTRY = {}


def register_smooth(name: str):
    name = str(name).lower()

    def decorator(cls):
        _SMOOTH_REGISTRY[name] = cls
        register_basis_runtime(name, cls)
        descriptor = require_basis_descriptor(name)
        # Backwards-compatible class attribute; descriptor metadata is the
        # authoritative capability source.
        cls.supports_tensor_marginal = descriptor.supports_tensor
        return cls

    return decorator


def make_smooth_term(kind: str, *args, **kwargs):
    key = str(kind).lower()
    if key not in _SMOOTH_REGISTRY:
        raise ValueError(
            f"Unknown smooth kind {kind!r}. "
            f"Available smooths: {sorted(_SMOOTH_REGISTRY)}"
        )
    return _SMOOTH_REGISTRY[key](*args, **kwargs)


def make_basis_term(
    kind: str,
    *,
    feature,
    k=-1,
    m=None,
    xt=None,
    pc=None,
    knots=None,
    shared_basis_setup=None,
    label=None,
    term_id=None,
    smoothing_id=None,
    by=None,
    sp=None,
    select=False,
    fixed=False,
    constraint_mode="auto",
    metadata=None,
):
    """Instantiate a regular smooth through its declarative descriptor."""
    descriptor = require_basis_descriptor(kind)
    runtime_class = descriptor.runtime_class
    if runtime_class is None:
        raise RuntimeError(f"Smooth basis {descriptor.name!r} has no runtime class.")

    features = [feature] if isinstance(feature, (str, int)) else list(feature)
    descriptor.validate_feature_count(len(features), context="Smooth")
    runtime_feature = features[0] if descriptor.max_features == 1 else features

    kwargs = {
        "feature": runtime_feature,
        "k": k,
        "label": label,
        "term_id": term_id,
        "smoothing_id": smoothing_id,
        "by": by,
        "sp": sp,
        "select": select,
        "fixed": fixed,
        "constraint_mode": constraint_mode,
        "knots": knots,
        "metadata": metadata,
    }
    option_values = {
        "basis": descriptor.name,
        "m": m,
        "xt": xt,
        "pc": pc,
        "shared_basis_setup": shared_basis_setup,
    }
    for option in descriptor.runtime_options:
        kwargs[option] = option_values[option]
    return runtime_class(**kwargs)


__all__ = ["make_basis_term", "make_smooth_term", "register_smooth"]
