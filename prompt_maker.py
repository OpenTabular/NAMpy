"""
Combine all GAM/GSM/spline-related code into a single text file.

This script intentionally discovers files dynamically (via glob patterns) so it
doesn't break when packages are reorganized (e.g., `nampy/gam/families/` became
multiple modules).
"""

from __future__ import annotations

from pathlib import Path

repo_root = Path(__file__).resolve().parent
output_file = "combined.txt"


def _rel_posix(p: Path) -> str:
    return p.relative_to(repo_root).as_posix()


def _sort_key(rel_posix: str) -> tuple:
    """
    Deterministic, "readable" ordering:
    - package order: gam, gsm, splines, then wrappers/configs
    - within a package, `__init__.py` first, then other modules lexicographically
    """
    if rel_posix.startswith("nampy/gam/"):
        group = 0
    elif rel_posix.startswith("nampy/gsm/"):
        group = 1
    elif rel_posix.startswith("nampy/splines/"):
        group = 2
    elif rel_posix.startswith("nampy/basemodels/") or rel_posix.startswith("nampy/models/") or rel_posix.startswith(
        "nampy/configs/"
    ):
        group = 3
    else:
        group = 9
    init_bias = 0 if rel_posix.endswith("__init__.py") else 1
    return (group, rel_posix.count("/"), init_bias, rel_posix)


def collect_input_files(root: Path) -> list[str]:
    patterns = [
        "nampy/gam/**/*.py",
        "nampy/gsm/**/*.py",
        "nampy/splines/**/*.py",
        # GAM-facing wrappers/configs (keep list explicit so we don't pull unrelated models/configs)
        "nampy/basemodels/gam.py",
        "nampy/models/gam.py",
        "nampy/configs/gam_config.py",
    ]

    files: set[Path] = set()
    for pat in patterns:
        matches = list(root.glob(pat))
        if matches:
            files.update(p for p in matches if p.is_file())
        else:
            # allow missing optional wrappers/configs during refactors
            candidate = root / pat
            if candidate.is_file():
                files.add(candidate)

    rels = sorted((_rel_posix(p) for p in files if p.name != "__init__.py"), key=_sort_key)
    return rels


def render_tree(file_paths: list[str]) -> str:
    # Build a directory tree from the file list, then render it completely.
    root_label = "nampy"

    tree: dict = {}
    for rel in file_paths:
        if not rel.startswith("nampy/"):
            continue
        parts = rel.split("/")[1:]  # drop "nampy"
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", set()).add(parts[-1])

    def _render(node: dict, prefix: str = "") -> list[str]:
        dirs = sorted(k for k in node.keys() if k != "__files__")
        files = sorted(node.get("__files__", set()))
        entries: list[tuple[str, str]] = [(d, "dir") for d in dirs] + [(f, "file") for f in files]

        lines: list[str] = []
        for i, (name, kind) in enumerate(entries):
            last = i == (len(entries) - 1)
            branch = "└─ " if last else "├─ "
            lines.append(f"{prefix}{branch}{name}{'/' if kind == 'dir' else ''}")
            if kind == "dir":
                child_prefix = prefix + ("   " if last else "│  ")
                lines.extend(_render(node[name], child_prefix))
        return lines

    lines = [f"{root_label}/"]
    lines.extend(_render(tree))
    return "\n".join(lines)


input_files = collect_input_files(repo_root)

with (repo_root / output_file).open("w", encoding="utf-8") as outfile:
    outfile.write(
        '''You are a senior computational statistician, numerical software architect, and Python package maintainer with deep expertise in generalized additive models, penalized regression splines, and Simon Wood’s **mgcv** framework. 
        I am building a Python reimplementation of the complete **mgcv** ecosystem as a submodule of my package **nampy**. The codebase is already substantial, and I want you to help me turn the current implementation into a disciplined, publication-grade roadmap. 
        Using the official docs from mgcv, rdrr io, stat eth zurich and the relevant papers from Simon Wood, review my code for correctness, bugs and limitations compared to the real mgcv implementation.
        '''
    )
    outfile.write("Directory structure:\n")
    outfile.write(render_tree(input_files))
    outfile.write("\n\n")

    for rel_path in input_files:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        with abs_path.open("r", encoding="utf-8") as infile:
            outfile.write(f"# --- Start of {rel_path} ---\n")
            outfile.write(infile.read())
            outfile.write(f"\n# --- End of {rel_path} ---\n\n")

print(f"Combined {len(input_files)} files into {output_file}")