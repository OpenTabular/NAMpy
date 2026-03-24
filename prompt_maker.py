"""
Combine mgcv-replication code into coherent phase files for LLM handoff.

By default this script writes 4 outputs:
- combined_phase1.txt
- combined_phase2.txt
- combined_phase3.txt
- combined_phase4.txt

Each phase contains:
- an instruction preface
- a tree view for files in that phase
- full concatenated file contents
"""

from __future__ import annotations

from pathlib import Path

repo_root = Path(__file__).resolve().parent
phase_output_files = {
    1: "combined_phase1.txt",
    2: "combined_phase2.txt",
    3: "combined_phase3.txt",
    4: "combined_phase4.txt",
}


def _rel_posix(p: Path) -> str:
    return p.relative_to(repo_root).as_posix()


def _sort_key(rel_posix: str) -> tuple:
    """
    Deterministic, "readable" ordering:
    - package order: gam, splines, then wrappers/configs
    - within a package, `__init__.py` first, then other modules lexicographically
    """
    if rel_posix.startswith("nampy/gam/"):
        group = 0
    elif rel_posix.startswith("nampy/splines/"):
        group = 1
    elif rel_posix.startswith("nampy/basemodels/") or rel_posix.startswith("nampy/models/") or rel_posix.startswith(
        "nampy/configs/"
    ):
        group = 2
    else:
        group = 9
    init_bias = 0 if rel_posix.endswith("__init__.py") else 1
    return (group, rel_posix.count("/"), init_bias, rel_posix)


def _phase_for_path(rel_posix: str) -> int:
    """
    Assign each file to one of 4 coherent review phases.

    Phase 1: Foundational APIs and interfaces
      - package exports, family definitions, configs, base models
      - formula/spec/runtime interfaces and smooth base/registry
      - low-level splines/utils foundations

    Phase 2: GAM core fitting + smoothness internals
      - GAM fitting/orchestration/smoother optimization internals
      - GAM-facing spline math used heavily by fit/reparam pathways

    Phase 3: GAM design/construction + concrete smooth terms
      - formula preprocessing, term materialization, design compilation
      - univariate/tensor/categorical smooth runtime implementations

    Phase 4: Validation/parity/predict/results/integration
      - parity/snapshot/trace, prediction/results wrappers, leftovers
      - anything uncategorized falls here
    """
    # Phase 1: core interfaces and foundations
    phase1_prefixes = (
        "nampy/gam/__init__.py",
        "nampy/gam/families/",
        "nampy/gam/smoothness/__init__.py",
        "nampy/gam/formula/parser.py",
        "nampy/gam/formula/compiler.py",
        "nampy/gam/specs/",
        "nampy/gam/runtime/__init__.py",
        "nampy/gam/smooths/__init__.py",
        "nampy/gam/smooths/base.py",
        "nampy/gam/smooths/registry.py",
        "nampy/splines/__init__.py",
        "nampy/splines/constraints.py",
        "nampy/splines/penalty_scaling.py",
        "nampy/splines/univariate_bases.py",
        "nampy/configs/gam_config.py",
        "nampy/basemodels/gam.py",
        "nampy/models/gam.py",
    )
    if any(rel_posix == p or rel_posix.startswith(p) for p in phase1_prefixes):
        return 1

    # Phase 2: fitting internals + smooth implementations
    phase2_prefixes = (
        "nampy/gam/fit/",
        "nampy/gam/smoothness/criteria/",
        "nampy/gam/smoothness/optimize/",
        "nampy/gam/smoothness/reparam.py",
        "nampy/splines/cubic.py",
        "nampy/splines/cubic_basis.py",
    )
    if any(rel_posix == p or rel_posix.startswith(p) for p in phase2_prefixes):
        return 2

    # Phase 3: GAM construction/design + concrete smooth implementations
    phase3_prefixes = (
        "nampy/gam/construction/",
        "nampy/gam/design/",
        "nampy/gam/runtime/factory.py",
        "nampy/gam/runtime/terms/",
        "nampy/gam/smooths/univariate/",
        "nampy/gam/smooths/tensor/",
        "nampy/gam/smooths/categorical/",
        "nampy/gam/formula/extract.py",
        "nampy/gam/formula/preprocess.py",
        "nampy/splines/pspline.py",
        "nampy/splines/thin_plate.py",
        "nampy/splines/thin_plate_basis.py",
        "nampy/splines/gaussian_process.py",
        "nampy/splines/mrf.py",
    )
    if any(rel_posix == p or rel_posix.startswith(p) for p in phase3_prefixes):
        return 3

    # Phase 4: parity/predict/results/integration and uncategorized files
    return 4


def collect_input_files(root: Path) -> list[str]:
    patterns = [
        "nampy/gam/**/*.py",
        "nampy/splines/**/*.py",
        "nampy/utils/**/*.py",
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


def split_into_phases(file_paths: list[str]) -> dict[int, list[str]]:
    phased: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for rel in file_paths:
        phase = _phase_for_path(rel)
        phased[phase].append(rel)
    for phase in phased:
        phased[phase] = sorted(phased[phase], key=_sort_key)
    return phased


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
phased_files = split_into_phases(input_files)

phase_titles = {
    1: "Phase 1 - Foundations and public interfaces",
    2: "Phase 2 - GAM fit and smoothness internals",
    3: "Phase 3 - GAM design/construction/smooth runtime internals",
    4: "Phase 4 - Parity, prediction, and integration",
}

for phase in (1, 2, 3, 4):
    out_name = phase_output_files[phase]
    phase_paths = phased_files[phase]
    tree_paths = input_files if phase == 1 else phase_paths
    with (repo_root / out_name).open("w", encoding="utf-8") as outfile:
        outfile.write(
            f"""You are a senior computational statistician, numerical software architect, and Python package maintainer with deep expertise in generalized additive models, penalized regression splines, and Simon Wood's mgcv framework.
I am building a Python reimplementation of the mgcv ecosystem as a submodule of my package nampy.
You are reviewing {phase_titles[phase]}.
This is part {phase}/4. Keep continuity with prior parts when available, and produce structured feedback for correctness, redundancy elimination, and architecture cleanup.

"""
        )
        if phase == 1:
            outfile.write("Directory structure (full collected scope):\n")
        else:
            outfile.write("Directory structure (this phase):\n")
        outfile.write(render_tree(tree_paths))
        outfile.write("\n\n")
        outfile.write(f"Included files in this phase: {len(phase_paths)}\n\n")

        for rel_path in phase_paths:
            abs_path = repo_root / rel_path
            if not abs_path.exists():
                continue
            with abs_path.open("r", encoding="utf-8") as infile:
                outfile.write(f"# --- Start of {rel_path} ---\n")
                outfile.write(infile.read())
                outfile.write(f"\n# --- End of {rel_path} ---\n\n")

    print(f"Wrote {out_name} with {len(phase_paths)} files.")

print(
    "Done. Generated 4 phased bundles: "
    + ", ".join(phase_output_files[i] for i in (1, 2, 3, 4))
)