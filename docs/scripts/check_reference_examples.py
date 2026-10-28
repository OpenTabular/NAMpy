"""Execute the exact downloadable reference examples and record output schemas.

Run from the repository root. --match selects pages by substring, and --resume
reuses successful records only when their complete source fingerprint matches.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/_examples/reference"


def write_json_atomic(path, value):
    """Keep an existing manifest readable until its complete replacement exists."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_schema(value, path="result", depth=0):
    import numpy as np
    import pandas as pd
    import torch

    rows = []
    cls = type(value)
    name = f"{cls.__module__}.{cls.__qualname__}"
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return [(path, f"{name}[{value.dtype}]", str(tuple(value.shape)))]
    if isinstance(value, (pd.DataFrame, pd.Series)):
        rows.append((path, name, str(value.shape)))
        if isinstance(value, pd.DataFrame):
            for column in value.columns:
                rows.append(
                    (
                        f"{path}[{str(column)!r}]",
                        str(value[column].dtype),
                        str((len(value),)),
                    )
                )
        return rows
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return [(path, name, "()" if value is not None else "not applicable")]
    if isinstance(value, type):
        return [(path, "type", "class object")]
    if dataclasses.is_dataclass(value):
        rows.append((path, name, "scalar record"))
        if depth < 3:
            for field in dataclasses.fields(value):
                rows.extend(
                    output_schema(
                        getattr(value, field.name), f"{path}.{field.name}", depth + 1
                    )
                )
        return rows
    if isinstance(value, dict):
        rows.append((path, "dict", f"{len(value)} named fields"))
        if depth < 3:
            for key, item in value.items():
                rows.extend(output_schema(item, f"{path}[{str(key)!r}]", depth + 1))
        return rows
    if isinstance(value, (list, tuple)):
        rows.append((path, cls.__name__, f"length {len(value)}"))
        if depth < 3 and value:
            for index, item in enumerate(value[:3]):
                rows.extend(output_schema(item, f"{path}[{index}]", depth + 1))
        return rows
    return [(path, name, "scalar object")]


PREVIEW_VERSION = 2


def result_preview(value, depth=0, *, field_name="", columns=None):
    """Shape-preserving samples; every omitted value is explicitly accounted for."""
    import numpy as np
    import pandas as pd
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        result = {"kind": "array", "shape": list(value.shape)}
        if any(
            word in field_name.lower()
            for word in ("covariance", "vcov", "lpmatrix", "design", "penalt", "basis")
        ):
            return {**result, "values": "omitted; see shape contract"}
        limits = (2, 4) if value.ndim >= 2 else (6,)
        slices = tuple(
            slice(0, limits[min(axis, len(limits) - 1)]) for axis in range(value.ndim)
        )
        sample = value[slices]
        result["values"] = (
            np.round(sample, 5).tolist()
            if np.issubdtype(sample.dtype, np.number)
            else sample.tolist()
        )
        result["omitted_by_axis"] = [
            size - shown for size, shown in zip(value.shape, sample.shape, strict=True)
        ]
        labels = (columns or {}).get(field_name)
        if labels and value.ndim == 2:
            result["columns"] = list(labels[: sample.shape[1]])
        return result
    if isinstance(value, (pd.DataFrame, pd.Series)):
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        sample = frame.iloc[:2, :4]
        return {
            "kind": "table",
            "shape": list(frame.shape),
            "columns": [str(c) for c in sample.columns],
            "values": [
                [
                    cell.item()
                    if isinstance(cell, np.generic)
                    else cell
                    if cell is None or isinstance(cell, (str, int, float, bool))
                    else str(cell)
                    for cell in row
                ]
                for row in sample.to_numpy().tolist()
            ],
            "omitted_by_axis": [
                len(frame) - len(sample),
                len(frame.columns) - len(sample.columns),
            ],
        }
    if isinstance(value, dict):
        items = list(value.items())
        if depth >= 4:
            return {"kind": "mapping", "omitted_fields": [str(k) for k in value]}
        result = {
            str(k): result_preview(v, depth + 1, field_name=str(k), columns=columns)
            for k, v in items[:4]
        }
        if len(items) > 4:
            result["omitted_fields"] = [str(k) for k, _ in items[4:]]
        return result
    if isinstance(value, (list, tuple)):
        limit = (1 if "local_term_importance" in field_name else 2) if depth < 4 else 0
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [
                result_preview(v, depth + 1, columns=columns) for v in value[:limit]
            ],
            "omitted": max(0, len(value) - limit),
        }
    if dataclasses.is_dataclass(value):
        selected = [
            f
            for f in dataclasses.fields(value)
            if isinstance(getattr(value, f.name), (str, int, float, bool, np.generic))
        ][:4]
        return {
            "kind": type(value).__name__,
            "fields": {
                f.name: result_preview(getattr(value, f.name), depth + 1)
                for f in selected
            },
            "omitted_fields": [
                f.name for f in dataclasses.fields(value) if f not in selected
            ],
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def distribution_preview_columns(namespace):
    family = namespace.get("family")
    if not hasattr(family, "param_names"):
        return None
    names = list(family.param_names)
    parameters = namespace.get("parameters")
    if parameters is None:
        return None
    width = parameters.shape[1]
    return {
        "raw": names,
        "parameters": names
        if len(names) == width
        else [f"class_{i}" for i in range(width)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    import matplotlib.pyplot as plt

    record_path = EXAMPLES / "verification.json"
    schemas_path = EXAMPLES / "output_schemas.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    schemas = json.loads(schemas_path.read_text()) if schemas_path.exists() else {}
    preview_path = EXAMPLES / "output_previews.json"
    previews = json.loads(preview_path.read_text()) if preview_path.exists() else {}
    failures = []
    passed = 0
    files = [
        path
        for path in sorted(EXAMPLES.rglob("*.py"))
        if args.match in path.relative_to(EXAMPLES).as_posix()
    ]
    cache = {}
    for path in files:
        key = path.relative_to(EXAMPLES).with_suffix("").as_posix()
        source = path.read_text()
        digest = hashlib.sha256(source.encode()).hexdigest()
        if (
            args.resume
            and records.get(key, {}).get("sha256") == digest
            and records[key].get("status") == "passed"
            and key in previews
            and records[key].get("preview_version") == PREVIEW_VERSION
        ):
            passed += 1
            continue
        start = time.monotonic()
        namespace = {"__name__": "__reference_example__"}
        # Architecture landing pages embed the same example as their estimator.
        code = source.split("\n\n", 1)[1]
        try:
            if code in cache:
                schemas[key], previews[key] = cache[code]
            else:
                with tempfile.TemporaryDirectory(
                    prefix="nampy-reference-"
                ) as directory:
                    previous = Path.cwd()
                    os.chdir(directory)
                    try:
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            exec(compile(source, str(path), "exec"), namespace)
                        if "result" not in namespace:
                            raise AssertionError(
                                "Example must expose result for output verification"
                            )
                        schemas[key] = output_schema(namespace["result"])
                        previews[key] = result_preview(
                            namespace["result"],
                            field_name=key,
                            columns=distribution_preview_columns(namespace),
                        )
                        cache[code] = (schemas[key], previews[key])
                    finally:
                        os.chdir(previous)
            records[key] = {
                "sha256": digest,
                "status": "passed",
                "preview_version": PREVIEW_VERSION,
            }
            passed += 1
            print(f"PASS {key} ({time.monotonic() - start:.2f}s)", flush=True)
        except Exception as error:
            failures.append(key)
            records[key] = {
                "sha256": digest,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"FAIL {key}: {records[key]['error']}", flush=True)
            traceback.print_exc()
        finally:
            plt.close("all")
            namespace.clear()
            write_json_atomic(preview_path, previews)
            write_json_atomic(schemas_path, schemas)
            write_json_atomic(record_path, records)
    print(f"{passed}/{len(files)} passed; {len(failures)} failed", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
