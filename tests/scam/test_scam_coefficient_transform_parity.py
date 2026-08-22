"""Layer-one parity for SCAM nonlinear coefficient transforms."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.coefficients import CoordinatewiseCoefficientTransform
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import R_SCRIPT, _build_r_command


def _run_scam_transform_reference(values, *, beta, threshold):
    library = os.environ.get("SCAM_LIB_PATH")
    if not library:
        pytest.skip("Set SCAM_LIB_PATH to an R library containing vendored SCAM.")
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
x <- as.numeric(fromJSON(args[[2]]))
b <- as.numeric(args[[3]])
threshold <- as.numeric(args[[4]])
out <- list(
  value=vapply(x, scam:::notExp, numeric(1), b=b, threshold=threshold),
  d1=vapply(x, scam:::DnotExp, numeric(1), b=b, threshold=threshold),
  d2=vapply(x, scam:::D2notExp, numeric(1), b=b, threshold=threshold),
  d3=vapply(x, scam:::D3notExp, numeric(1), b=b, threshold=threshold)
)
write_json(out, args[[5]], digits=17, auto_unbox=FALSE)
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "transform.R"
        output = root / "transform.json"
        script.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script,
                library,
                json.dumps(list(values)),
                str(beta),
                str(threshold),
                str(output),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for SCAM parity")
def test_softplus_value_and_three_derivatives_match_scam_notexp():
    values = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 19.99, 20.0, 30.0])
    expected = _run_scam_transform_reference(values, beta=1.0, threshold=20.0)
    transform = CoordinatewiseCoefficientTransform(
        np.ones(values.size, dtype=bool), positive_map="softplus"
    )

    np.testing.assert_allclose(
        transform.transform(values), expected["value"], rtol=0.0, atol=2e-15
    )
    for order in (1, 2, 3):
        np.testing.assert_allclose(
            transform.derivative(values, order=order),
            expected[f"d{order}"],
            rtol=2e-15,
            atol=2e-15,
        )


def test_mixed_coordinate_transform_and_covariance_transport_contract():
    beta = np.array([-0.5, 0.2, 1.1, -2.0])
    mask = np.array([False, True, False, True])
    transform = CoordinatewiseCoefficientTransform(mask, positive_map="exp")
    prediction_beta = transform.transform(beta)

    np.testing.assert_allclose(
        prediction_beta, [beta[0], np.exp(beta[1]), beta[2], np.exp(beta[3])]
    )
    d1 = transform.derivative(beta, order=1)
    np.testing.assert_allclose(d1, [1.0, np.exp(beta[1]), 1.0, np.exp(beta[3])])
    np.testing.assert_allclose(
        transform.derivative(beta, order=2), [0.0, np.exp(beta[1]), 0.0, np.exp(beta[3])]
    )

    covariance = np.arange(16, dtype=np.float64).reshape(4, 4)
    covariance = covariance @ covariance.T
    expected = np.diag(d1) @ covariance @ np.diag(d1)
    np.testing.assert_allclose(
        transform.transport_covariance(beta, covariance), expected, rtol=0.0, atol=0.0
    )


def test_subset_preserves_transform_controls_and_reindexes_positive_mask():
    transform = CoordinatewiseCoefficientTransform(
        [True, False, True, False],
        positive_map="softplus",
        softplus_beta=2.5,
        softplus_threshold=11.0,
    )
    subset = transform.subset([3, 0, 2])
    np.testing.assert_array_equal(subset.positive_mask, [False, True, True])
    assert subset.positive_map == "softplus"
    assert subset.softplus_beta == 2.5
    assert subset.softplus_threshold == 11.0
