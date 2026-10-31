"""Local hooks and CI enforce the same declared repository quality gates."""

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _target(name):
    text = (ROOT / "Makefile").read_text()
    return shlex.split(text.split(f"\n{name}:\n", 1)[1].splitlines()[0].strip())


def _hooks():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    return {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}


def test_native_hooks_match_enforced_make_targets():
    hooks = _hooks()
    for hook, target in (("mypy", "type-check"), ("repository-hygiene", "hygiene")):
        assert shlex.split(hooks[hook]["entry"]) == _target(target)
        assert hooks[hook]["pass_filenames"] is False
    typing = _target("type-check")
    assert "--follow-imports=skip" in typing
    assert {value for value in typing if value.endswith(".py")} == {
        "nampy/contracts.py", "nampy/neural/contracts.py", "nampy/gam/fit/state.py",
    }
    checker = _target("hygiene")[-1]
    assert (ROOT / checker).is_file()
    assert _target("type-check-all") == ["mypy", "nampy"]


def test_ruff_hook_matches_ci_lint_roots():
    hook = _hooks()["ruff"]
    command = shlex.split(hook["entry"])
    assert _target("lint")[:len(command)] == command
    roots = _target("lint")[len(command):]
    assert set(roots) == {"nampy", "tests", "examples"}
    assert all(re.search(hook["files"], root + "/example.py") for root in roots)
    assert not re.search(hook["files"], "docs/_examples/reference/example.py")


def test_ci_and_publication_use_canonical_make_targets():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    type_steps = ci["jobs"]["type-check"]["steps"]
    assert any(step.get("run") == "make type-check" for step in type_steps)
    publish = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())
    commands = [step.get("run", "") for job in publish["jobs"].values() for step in job["steps"]]
    assert any("make quality" in command.splitlines() for command in commands)
