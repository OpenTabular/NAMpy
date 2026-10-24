"""Source provenance and display cleanup for executable documentation."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import nbformat


def source_state(root):
    """Identify the exact source content, including uncommitted local changes."""
    root = Path(root)
    files = [root / "pyproject.toml"]
    files += list((root / "nampy").rglob("*.py"))
    files += list((root / "examples").glob("*.py"))
    files += [
        p
        for p in (root / "docs").rglob("*")
        if p.suffix in {".py", ".rst", ".inc", ".ipynb"}
        and not {
            "_build",
            "_generated",
            "generated",
            ".ipynb_checkpoints",
        }.intersection(p.parts)
    ]
    manifest = {}
    for path in sorted(set(files)):
        if path.suffix == ".ipynb":
            notebook = nbformat.read(path, 4)
            payload = json.dumps(
                [c.source for c in notebook.cells if c.cell_type == "code"],
                ensure_ascii=False,
            ).encode()
        else:
            payload = path.read_bytes()
        manifest[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    return {
        "revision": revision,
        "source_sha256": fingerprint,
        "source_manifest": manifest,
    }


def clean_outputs(notebook):
    """Remove incidental training noise while retaining figures, tables and warnings."""
    noise = re.compile(
        r"^(?:INFO: )?(?:GPU available:|TPU available:|💡|`Trainer.fit` stopped:|.*_pytree\.py:[0-9]+:.*deprecated)"
    )
    notebook.metadata["nampy_display_cleanup_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        keep = []
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                output.text = re.sub(
                    r"(?m)^.*(?:FutureWarning: `torch.jit.script` is deprecated|UserWarning: enable_nested_tensor is True)[^\n]*\n(?:[ \t]+[^\n]*\n)?",
                    "",
                    output.text,
                )
                output.text = "".join(
                    line
                    for line in output.text.splitlines(keepends=True)
                    if not noise.match(line)
                )
                if not output.text.strip():
                    continue
            text = output.get("data", {}).get("text/plain", "")
            if output.output_type == "execute_result" and (
                text.startswith("GAMSummary(")
                or re.match(
                    r"^[A-Za-z_][A-Za-z_0-9]*(?:Regressor|Classifier|LSS)\(", text
                )
            ):
                continue
            keep.append(output)
        cell.outputs = keep
