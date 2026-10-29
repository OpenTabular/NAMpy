"""NAMpy: interpretable additive GAM and neural models.

Public estimators are resolved lazily so importing the dependency-light GAM
backend does not initialize PyTorch or Lightning.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .__version__ import __version__
from .models import __all__ as _MODEL_EXPORTS

_SUBMODULE_EXPORTS = frozenset(
    {
        "distributions",
        "gam",
        "models",
        "neural",
        "diagnostics",
        "survival",
        "calibration",
    }
)
_CONFIG_EXPORTS = (
    "ObjectiveConfig",
    "SurvivalObjectiveConfig",
    "PreprocessingConfig",
    "GradientTrainingConfig",
    "FixedDesignTrainingConfig",
    "NativeTrainingConfig",
    "IGANNTrainingConfig",
    "SemiStructuredTrainingConfig",
)

_FISHER_EXPORTS = (
    "FisherDiagnostics",
    "FisherGroupSummary",
    "FisherInformationResult",
    "FisherSnapshot",
)

if TYPE_CHECKING:
    from . import distributions, gam, models, neural  # noqa: F401
    from .models import *  # noqa: F401,F403


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_EXPORTS:
        value = import_module(f".{name}", __name__)
    elif name in _FISHER_EXPORTS:
        value = getattr(import_module(".diagnostics.fisher", __name__), name)
    elif name in _CONFIG_EXPORTS:
        value = getattr(import_module(".neural.configuration", __name__), name)
    elif name in _MODEL_EXPORTS:
        value = getattr(import_module(".models", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "distributions",
    "gam",
    "models",
    "neural",
    *_MODEL_EXPORTS,
    *_CONFIG_EXPORTS,
    *_FISHER_EXPORTS,
    "diagnostics",
    "survival",
    "calibration",
    "__version__",
]
