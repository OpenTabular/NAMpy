"""Execute the documented soap-film workflow with the installed public API."""

from pathlib import Path
from textwrap import dedent

import numpy as np


def test_soap_reference_example():
    source = (
        Path(__file__).resolve().parents[2] / "docs/api/gam_smooths.rst"
    ).read_text()
    section = source.split(".. _soap-film-smooths:", 1)[1]
    block = section.split(".. code-block:: python", 1)[1].lstrip("\n")
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    namespace = {}
    exec(compile(dedent("\n".join(lines)), "soap documentation example", "exec"), namespace)
    assert namespace["prediction"].shape == (150,)
    assert np.isfinite(namespace["prediction"]).all()
    assert np.isfinite(namespace["standard_error"]).all()
    assert np.all(namespace["standard_error"] >= 0)
