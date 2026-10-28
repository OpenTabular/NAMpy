"""Windows-style Git checkouts must preserve upstream fixture identities."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_fixture_hashes_survive_autocrlf_checkout(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to exercise checkout line-ending conversion")

    fixture = Path("tests/reference_fixtures/mgcv_shash")
    manifest = json.loads((ROOT / fixture / "manifest.json").read_text())
    compressed = fixture / next(iter(manifest["fixture_sha256"]))
    sources = [
        fixture / "inputs.json",
        Path("tests/families/test_shash_family_contracts.py"),
        next((ROOT / "tests/smooths/probes").glob("*.R")).relative_to(ROOT),
        Path("tests/parity/probes/mgcv_parametric_penalty_maps.R"),
        compressed,
    ]
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in [Path(".gitattributes"), *sources]:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (repository / "control.txt").write_bytes(b"unprotected text\n")

    def run(*args):
        subprocess.run([git, *args], cwd=repository, check=True, capture_output=True)

    run("init", "--quiet")
    run("config", "core.autocrlf", "true")
    run("add", ".")
    checkout = tmp_path / "checkout"
    run("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")

    assert (checkout / "control.txt").read_bytes() == b"unprotected text\r\n"
    for relative in sources:
        assert (checkout / relative).read_bytes() == (ROOT / relative).read_bytes(), (
            relative
        )
    assert (
        hashlib.sha256((checkout / fixture / "inputs.json").read_bytes()).hexdigest()
        == manifest["inputs_sha256"]
    )
    assert (
        hashlib.sha256((checkout / compressed).read_bytes()).hexdigest()
        == manifest["fixture_sha256"][compressed.name]
    )
