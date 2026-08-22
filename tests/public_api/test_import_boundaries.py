"""Import-boundary contracts for optional NAMpy backends."""

from __future__ import annotations

import subprocess
import sys


def test_gam_public_import_does_not_initialize_neural_backend():
    code = """
import sys

from nampy.gam import GAM
from nampy import GAMRegressor

assert GAM.__module__.startswith("nampy.gam.")
assert GAMRegressor.__module__ == "nampy.models.gam"
assert "nampy.neural" not in sys.modules
assert "torch" not in sys.modules
assert "lightning" not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
