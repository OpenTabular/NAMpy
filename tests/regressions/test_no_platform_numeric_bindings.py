from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "nampy"
_FORBIDDEN_MODULES = {
    "ctypes",
    "cffi",
    "scipy.linalg.blas",
    "scipy.linalg.lapack",
}
_FORBIDDEN_NAMES = {"get_lapack_funcs"}


def test_production_code_has_no_direct_native_numeric_bindings():
    """Keep production numerics on NumPy/SciPy's supported public interfaces."""
    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES:
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _FORBIDDEN_MODULES:
                    violations.append(f"{path}:{node.lineno}: from {module}")
                for alias in node.names:
                    if alias.name in _FORBIDDEN_NAMES:
                        violations.append(
                            f"{path}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.Call):
                if any(
                    keyword.arg in {"driver", "use_scipy"}
                    for keyword in node.keywords
                ):
                    violations.append(
                        f"{path}:{node.lineno}: backend-specific eigensolver driver"
                    )
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else function.attr
                    if isinstance(function, ast.Attribute)
                    else ""
                )
                if name in _FORBIDDEN_NAMES:
                    violations.append(f"{path}:{node.lineno}: call {name}")

    assert not violations, "\n".join(violations)
