from pathlib import Path


def test_python_support_excludes_python_313_until_numpy_pin_changes():
    pyproject = Path("pyproject.toml").read_text()

    assert 'requires-python = ">=3.10,<3.13"' in pyproject
    assert "Programming Language :: Python :: 3.13" not in pyproject
