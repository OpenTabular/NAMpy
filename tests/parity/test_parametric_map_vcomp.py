"""Variance component names and full-SP SDs follow source map coordinates."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.parity.test_mgcv_parametric_penalty_maps import fit_python, payload
from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_parametric_map_vcomp.R"


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("state", ["mixed", "selected"])
def test_mapped_variance_component_names_and_full_penalty_scales(family, state):
    inputs = payload(family=family, state=state)
    key = reference_key("parametric-map-vcomp", {
        "inputs": inputs, "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
    })
    expected = load_reference("mgcv_parametric_map_vcomp", key)
    if expected is None:
        with tempfile.TemporaryDirectory() as folder:
            inp, out = [Path(folder) / name for name in ("in.json", "out.json")]
            inp.write_text(json.dumps(inputs))
            run = subprocess.run(["Rscript", str(PROBE), str(inp), str(out)],
                                 capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            expected = json.loads(out.read_text())
        save_reference("mgcv_parametric_map_vcomp", key, expected)
    model, _, _ = fit_python(inputs)
    actual = model.gam_vcomp(rescale=False)
    assert actual["names"] == expected["names"]
    assert actual["all_names"] == expected["all_names"]
    np.testing.assert_allclose(np.asarray(actual["vc"])[:, 0], expected["sd"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(actual["all"], expected["all"], atol=2e-7, rtol=2e-7)
