"""NAMpy: interpretable additive GAM and neural models.

Public estimators are resolved lazily so importing the dependency-light GAM
backend does not initialize PyTorch or Lightning.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .__version__ import __version__
from .models import __all__ as _MODEL_EXPORTS

_SUBMODULE_EXPORTS = frozenset({"gam", "models", "neural"})

if TYPE_CHECKING:
    from . import gam, models, neural  # noqa: F401
    from .models import *  # noqa: F401,F403


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_EXPORTS:
        value = import_module(f".{name}", __name__)
    elif name in _MODEL_EXPORTS:
        value = getattr(import_module(".models", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "gam",
    "models",
    "neural",
    *_MODEL_EXPORTS,
    "__version__",
]
