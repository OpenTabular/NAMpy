"""Run the five remaining CLI examples without an R runtime."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("family", ["bcg", "gumbls", "ziplss", "multinom", "gevlss"])
def test_remaining_family_example_runs(family):
    runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / f"example_{family}_gam.py"),
        run_name="__main__",
    )
