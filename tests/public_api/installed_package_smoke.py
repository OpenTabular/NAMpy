"""Smoke-test the installed NAMpy distribution and its public imports.

Run this script from outside the repository after installing a built wheel with
its ``all`` extra. It intentionally rejects imports resolved from the source
checkout, so a passing result certifies the installed artifact rather than the
working tree.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import nampy
import nampy.gam as gam
import nampy.models as models
import nampy.neural as neural
from nampy import GAMLSS, LinRegRegressor, SNAMClassifier
from nampy.neural.configs import DefaultSNAMConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GAM_EXPORTS = [
    "GAM",
    "fit_model_core",
    "solve_fit",
    "FitCoreSolution",
    "GAMControl",
    "gam_control",
    "normalize_nei",
]


def main() -> None:
    package_path = Path(nampy.__file__).resolve()
    if package_path.is_relative_to(REPOSITORY_ROOT):
        raise AssertionError(
            f"Imported NAMpy from the source checkout instead of the installed "
            f"distribution: {package_path}"
        )

    assert version("nampy") == nampy.__version__
    assert gam.__all__ == EXPECTED_GAM_EXPORTS
    assert gam.GAM.__module__.startswith("nampy.gam.")
    assert not hasattr(nampy, "GAM")

    assert nampy.models is models
    assert nampy.neural is neural
    assert models.LinRegRegressor is LinRegRegressor
    assert models.GAMLSS is GAMLSS
    assert neural.configs.DefaultSNAMConfig is DefaultSNAMConfig
    assert hasattr(LinRegRegressor(), "save_model")
    SNAMClassifier()
    DefaultSNAMConfig()

    print(f"Installed NAMpy {nampy.__version__} smoke test passed: {package_path}")


if __name__ == "__main__":
    main()
