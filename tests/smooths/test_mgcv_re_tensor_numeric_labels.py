"""Numeric-looking categorical labels remain distinct during tensor pooling."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference
from tests.smooths.test_mgcv_re_tensor import (
    test_re_tensor_reference as check_reference,
)

PROBE = Path(__file__).with_name("probes") / "mgcv_re_tensor_numeric_labels.R"
KEY = reference_key(
    "re-tensor-numeric-labels", hashlib.sha256(PROBE.read_bytes()).hexdigest()
)


@pytest.fixture(scope="module")
def references():
    result = load_reference("mgcv_re_tensor_numeric_labels", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            subprocess.run(
                ["Rscript", str(PROBE), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
        save_reference("mgcv_re_tensor_numeric_labels", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


@pytest.mark.parametrize(
    "kind,repeated", [("factor", False), ("mixed", False), ("mixed", True)]
)
@pytest.mark.parametrize("tensor", ["te", "ti", "t2"])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_numeric_string_factor_parity(references, kind, repeated, tensor, method):
    key = f"TRUE/{kind}/{tensor}/{method}"
    ref = references[key + ("/repeated" if repeated else "")]
    check_reference({key: ref}, True, kind, tensor, method)
