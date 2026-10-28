"""Independent static references; generation must be explicitly enabled."""

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

from tests.mgcv_parity_utils import _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_registered_pipeline.R"


def _json_input(value):
    """Encode censoring infinities as explicit strings, never invalid JSON."""
    if isinstance(value, dict):
        return {k: _json_input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_input(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Inf" if value > 0 else "-Inf" if value < 0 else "NaN"
    return value


def pipeline_reference(
    name,
    frame,
    newdata,
    formulas,
    family,
    *,
    knots=None,
    scale=None,
    engine="mgcv",
    sp=(0.4,),
    start=None,
    namespace="registered_seven_stage",
    probe=PROBE,
    extra_payload=None,
):
    payload = {
        "name": name,
        "data": frame.to_dict("list"),
        "newdata": newdata.to_dict("list"),
        "formula": [_normalize_python_formula_text(f) for f in formulas],
        "family": family,
        "knots": knots,
        "scale": scale,
        "engine": engine,
        "sp": list(sp),
        "start": start,
    }
    payload.update(extra_payload or {})
    # jsonlite coerces mixed numeric/infinity arrays to shorter strings, turning
    # exactly observed rows into tiny intervals. Encode the entire bound column
    # uniformly so its finite entries retain the same exact doubles as y.
    for block in ("data", "newdata"):
        if "bound" in payload[block]:
            payload[block]["bound"] = [str(v) for v in payload[block]["bound"]]
    probe_identity = {
        "payload": payload,
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    }
    if probe != PROBE:
        probe_identity["base_probe_sha256"] = hashlib.sha256(
            PROBE.read_bytes()
        ).hexdigest()
    key = reference_key(
        "registered-seven-stage-v1",
        probe_identity,
        normalize_floats=True,
    )
    result = load_reference(namespace, key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            inputs, output = [
                Path(directory) / f for f in ("inputs.json", "output.json")
            ]
            inputs.write_text(json.dumps(_json_input(payload), allow_nan=False))
            run = subprocess.run(
                ["Rscript", str(probe), str(inputs), str(output)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference(namespace, key, result)
    return result
