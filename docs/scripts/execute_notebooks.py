"""Execute documented notebook sources from clean kernels and record provenance.

Usage: python docs/scripts/execute_notebooks.py [notebook paths ...]
No paths executes the teaching notebooks. Pass validation reports explicitly.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from execution_support import clean_outputs, source_state
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "docs/notebooks").glob("*.ipynb"))
    versions = {
        name: importlib.metadata.version(name)
        for name in ["nampy", "numpy", "scikit-learn", "torch", "lightning"]
    }
    state = source_state(ROOT)
    commit = state["revision"]
    for path in paths:
        path = path.resolve()
        notebook = nbformat.read(path, as_version=4)
        notebook.cells = [
            c
            for c in notebook.cells
            if "execution-provenance" not in c.metadata.get("tags", [])
        ]
        with tempfile.TemporaryDirectory(prefix="nampy-doc-execution-") as directory:
            environment = dict(
                os.environ,
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MPLBACKEND="module://matplotlib_inline.backend_inline",
            )
            environment["PATH"] = (
                str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
            )
            started = time.perf_counter()
            NotebookClient(
                notebook,
                timeout=900,
                kernel_name="python3",
                resources={"metadata": {"path": directory}},
            ).execute(env=environment)
            seconds = time.perf_counter() - started
        clean_outputs(notebook)
        note = nbformat.v4.new_markdown_cell(
            "**Execution environment:** "
            + ", ".join(f"{name} {version}" for name, version in versions.items())
            + f". Python {sys.version.split()[0]}; source revision `{commit}`; source-content SHA256 `{state['source_sha256']}` (includes local source changes); elapsed {seconds:.1f} seconds on this CPU. Runtime varies by hardware.",
            metadata={"tags": ["execution-provenance"]},
        )
        notebook.cells.insert(1, note)
        notebook.metadata["nampy_execution"] = {
            "versions": versions,
            "commit": commit,
            **state,
            "seconds": seconds,
            "python": sys.version.split()[0],
            "code_sha256": hashlib.sha256(
                json.dumps(
                    [
                        cell.source
                        for cell in notebook.cells
                        if cell.cell_type == "code"
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest(),
        }
        nbformat.write(notebook, path)
        print(f"PASS {path.relative_to(ROOT)} ({seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
