"""Soap matrix fits on repeated, nonuniformly sampled location pairs."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.smooths.test_mgcv_soap_functional import (
    test_soap_functional_fit_prediction_and_upstream_offset_boundary as _compare,
)

PROBE = Path(__file__).with_name("probes") / "mgcv_soap_repeated_functional.R"
KEY = reference_key("soap-repeated-functions", hashlib.sha256(PROBE.read_bytes()).hexdigest())


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_soap_repeated_functional", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(["Rscript", str(PROBE), str(output)], check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
        save_reference("mgcv_soap_repeated_functional", KEY, result)
    return result["cases"]


@pytest.mark.parametrize("basis", ["so", "sf", "sw"])
@pytest.mark.parametrize("kind", ["unit", "varying", "constant"])
def test_repeated_soap_locations_match_mgcv(references, basis, kind):
    _compare(references, basis, kind, "fixed", "FALSE")
