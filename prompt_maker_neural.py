"""
Combine core neural NAM files into a single handoff bundle for LLM review.

This script writes one output:
- combined_neural_core.txt

The output contains:
- an instruction preface
- a tree view for the included files
- full concatenated file contents
"""

from __future__ import annotations

from pathlib import Path

repo_root = Path(__file__).resolve().parent
output_file = "combined_neural_core.txt"


def _rel_posix(p: Path) -> str:
    return p.relative_to(repo_root).as_posix()


def _sort_key(rel_posix: str) -> tuple:
    """Deterministic ordering for the core neural bundle."""
    if rel_posix.startswith("nampy/basemodels/"):
        group = 0
    elif rel_posix.startswith("nampy/models/"):
        group = 1
    elif rel_posix.startswith("nampy/configs/"):
        group = 2
    else:
        group = 9
    return (group, rel_posix.count("/"), rel_posix)


def collect_input_files(root: Path) -> list[str]:
    target_files = [
        "nampy/basemodels/basemodel.py",
        "nampy/basemodels/lightning_wrapper.py",
        "nampy/models/sklearn_regressor.py",
        "nampy/models/sklearn_classifier.py",
        "nampy/models/sklearn_lss.py",
        "nampy/models/nam.py",
        "nampy/basemodels/nam.py",
        "nampy/configs/nam_config.py",
        "nampy/utils/distributions.py",
        "nampy/utils/distributional_metrics.py",
    ]

    files: list[Path] = []
    for rel in target_files:
        candidate = root / rel
        if candidate.is_file():
            files.append(candidate)
        else:
            raise FileNotFoundError(f"Required file not found: {rel}")

    rels = sorted((_rel_posix(p) for p in files), key=_sort_key)
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
    outfile.write("""You are a senior computational statistician, numerical software architect, and Python package maintainer with deep expertise in neural additive models and Python package design. I am building a python packagw which offers a variety of neural models like NAM, NBM, GPNAM, NODE-GAM, MonotonicNAM etc as a single package. Review this core neural bundle, I have only included the example of NAM here. I wish to incorporate the ideas of the neurips paper into this as it can be done for any neural model like NAM or NBM.""")
    outfile.write("Directory structure (core neural scope):\n")
    outfile.write(render_tree(input_files))
    outfile.write("\n\n")
    outfile.write(f"Included files: {len(input_files)}\n\n")

    for rel_path in input_files:
        abs_path = repo_root / rel_path
        with abs_path.open("r", encoding="utf-8") as infile:
            outfile.write(f"# --- Start of {rel_path} ---\n")
            outfile.write(infile.read())
            outfile.write(f"\n# --- End of {rel_path} ---\n\n")

print(f"Wrote {output_file} with {len(input_files)} files.")
print(f"Done. Generated single bundle: {output_file}")