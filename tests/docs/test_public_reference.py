"""Reference generation must work before autosummary creates its output folder."""

import runpy
from pathlib import Path


def test_reference_generation_from_clean_checkout(tmp_path):
    extension = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "docs/_ext/public_reference.py")
    )
    generated = tmp_path / "docs/api/generated"
    assert not generated.exists()

    extension["generate"](tmp_path)

    page = generated / "nampy.models.LinRegRegressor.rst"
    assert ".. autoclass:: LinRegRegressor" in page.read_text()
    reference = tmp_path / "docs/_generated/reference/LinRegRegressor.inc"
    assert reference.is_file()
    # The same builder hook also runs for repeated documentation builds.
    before = page.read_text()
    extension["generate"](tmp_path)
    assert page.read_text() == before
