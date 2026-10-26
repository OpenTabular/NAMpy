"""Upstream capability inventory, not claims of NAMpy runtime support.

The saved R probe exercises smoothCon and PredictMat independently of NAMpy.
Refresh explicitly; normal tests inspect the committed mgcv 1.9-4 observations.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.reference_fixtures import load_reference, reference_key, save_reference

PROBE = Path(__file__).with_name("probes") / "mgcv_wider_composition_support.R"
KEY = reference_key(
    "smoothCon-PredictMat-support",
    {"source_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
)


@pytest.fixture(scope="module")
def inventory():
    result = load_reference("mgcv_composition_support", KEY)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            subprocess.run(
                ["Rscript", str(PROBE), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
        save_reference("mgcv_composition_support", KEY, result)
    assert result["version"] == "1.9.4"
    return result["cases"]


@pytest.mark.parametrize(
    "basis", ["tp", "ts", "cr", "cs", "cc", "ps", "cp", "bs", "ds", "gp", "re", "ad"]
)
@pytest.mark.parametrize("weighted", ["FALSE", "TRUE"])
def test_upstream_scalar_bases_accept_matrix_arguments(inventory, basis, weighted):
    assert inventory[f"matrix/{basis}/{weighted}"]["accepted"]


@pytest.mark.parametrize("basis", ["sos", "so", "sf", "sw", "mrf", "factor_re"])
def test_upstream_special_bases_accept_matrix_arguments(inventory, basis):
    assert inventory[f"matrix/{basis}"]["accepted"]


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize("basis", ["tp", "cr", "ps", "bs", "ds", "gp", "re"])
def test_upstream_matrix_tensor_acceptance(inventory, fun, basis):
    assert inventory[f"matrix_tensor/{fun}/{basis}"]["accepted"]


@pytest.mark.parametrize("fun", ["te", "ti", "t2"])
@pytest.mark.parametrize("kind", ["soap_tensor", "scalar_soap_tensor"])
def test_upstream_soap_tensor_separates_single_and_multiple_penalties(
    inventory, fun, kind
):
    for basis in ["sf", "sw"]:
        assert inventory[f"{kind}/{fun}/{basis}"]["accepted"]
    assert "multiple penalties" in inventory[f"{kind}/{fun}/so"]["error"]


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_upstream_factor_smooth_nesting_boundaries(inventory, basis):
    assert inventory[f"matrix_factor/{basis}"]["accepted"]
    for marginal in ["cr", "ps", "tp", "ds", "gp"]:
        assert inventory[f"factor_base/{basis}/{marginal}"]["accepted"]
    assert "multiply penalized" in inventory[f"adaptive_factor/{basis}"]["error"]
    for fun in ["te", "ti", "t2"]:
        assert (
            "unsuitable marginal smooth class"
            in inventory[f"factor_tensor/{fun}/{basis}"]["error"]
        )
        assert (
            "unsuitable marginal smooth class"
            in inventory[f"adaptive_tensor/{fun}"]["error"]
        )
    assert inventory[f"nested_factor/{basis}"]["accepted"] == (basis == "sz")
    assert "factor `by'" in inventory["matrix/factor_by"]["error"]


def test_upstream_prediction_reconstruction_and_t2_sf_alternative_constraint(inventory):
    for name, case in inventory.items():
        if not case["accepted"]:
            continue
        assert case["dim"] == case["prediction_dim"]
        if name == "scalar_soap_tensor/t2/sf":
            # t2 uses C on its null space and Cp for prediction. The saved
            # mgcv_t2_soap_constraint_probe.R confirms modCon=1/2 and fitted
            # gam predictions agree: this difference is intentional.
            assert max(case["prediction_error"]) > 1
        else:
            assert max(case["prediction_error"]) < 2e-8, name
