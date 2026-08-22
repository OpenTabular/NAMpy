"""summary.gam option parity: freq=, dispersion=, re.test= vs live mgcv.

The default summary surface is owned by test_mgcv_summary_parity.py; the three
optional arguments (mgcv/R/mgcv.r:3858 ``summary.gam(object, dispersion=NULL,
re.test=TRUE, freq=FALSE, ...)``) previously had only self-consistency
coverage. This file compares each option's p-table / s-table directly against
``summary.gam`` run with the same argument.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.summary import summary_gam
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _make_random_effect_data_noisy,
)
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = [pytest.mark.surface_output]

R_SCRIPT = shutil.which("Rscript")

_R_DRIVER = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = TRUE)
mode <- args[[3]]

serialize_summary <- function(s) {
  list(
    p_table = unname(as.matrix(s$p.table)),
    s_table = if (is.null(s$s.table)) NULL else unname(as.matrix(s$s.table)),
    s_labels = if (is.null(s$s.table)) NULL else rownames(s$s.table),
    residual_df = s$residual.df,
    scale = s$scale,
    covariance = if (isTRUE(s$freq)) "freq" else "bayes"
  )
}

if (mode == "options") {
  b <- gam(y ~ x0 + s(x1, bs = "cr", k = 8), data = d, method = "REML")
  payload <- list(
    default = serialize_summary(summary(b)),
    freq = serialize_summary(summary(b, freq = TRUE)),
    dispersion = serialize_summary(summary(b, dispersion = 2.5))
  )
} else {
  b <- gam(y ~ s(f, bs = "re"), data = d, method = "REML")
  payload <- list(
    default = serialize_summary(summary(b)),
    no_re_test = serialize_summary(summary(b, re.test = FALSE))
  )
}
write_json(payload, args[[2]], auto_unbox = TRUE, digits = 17)
"""


def _run_summary_driver(data: pd.DataFrame, mode: str) -> dict:
    key = reference_key(
        "summary_options", {"data": data.to_csv(index=False), "mode": mode}
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        csv_path = tmp / "d.csv"
        json_path = tmp / "out.json"
        script_path = tmp / "summary_options.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(_R_DRIVER, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), mode],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _p_table_matrix(summary) -> np.ndarray:
    return summary.p_table.to_numpy(dtype=np.float64)


def _assert_table_close(actual: np.ndarray, expected, *, atol=1e-8, rtol=1e-5):
    expected_arr = np.atleast_2d(np.asarray(expected, dtype=np.float64))
    assert actual.shape == expected_arr.shape
    np.testing.assert_allclose(actual, expected_arr, atol=atol, rtol=rtol)


def test_summary_freq_and_dispersion_options_match_mgcv():
    """freq=TRUE switches to Ve; dispersion= rescales SEs and test statistics."""
    data = _make_gaussian_data(seed=433, n=160)
    expected = _run_summary_driver(data, "options")

    gam = GAM(
        family="gaussian",
        formula='y ~ x0 + s(x1, bs="cr", k=8)',
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)

    s_default = summary_gam(gam)
    _assert_table_close(_p_table_matrix(s_default), expected["default"]["p_table"])

    s_freq = summary_gam(gam, freq=True)
    _assert_table_close(_p_table_matrix(s_freq), expected["freq"]["p_table"])
    _assert_table_close(
        s_freq.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=np.float64
        ),
        expected["freq"]["s_table"],
        rtol=1e-4,
    )

    s_disp = summary_gam(gam, dispersion=2.5)
    _assert_table_close(_p_table_matrix(s_disp), expected["dispersion"]["p_table"])
    assert s_disp.scale == pytest.approx(
        float(expected["dispersion"]["scale"]), rel=1e-9
    )
    _assert_table_close(
        s_disp.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=np.float64
        ),
        expected["dispersion"]["s_table"],
        rtol=1e-4,
    )


def test_summary_re_test_false_drops_random_effect_rows_like_mgcv():
    """re.test=FALSE removes reTest-eligible smooth rows exactly like mgcv."""
    data = _make_random_effect_data_noisy(seed=29, n_draws=45)
    expected = _run_summary_driver(data, "re")

    gam = GAM(
        family="gaussian",
        formula='y ~ s(f, bs="re")',
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)

    s_default = summary_gam(gam)
    assert expected["default"]["s_table"] is not None
    _assert_table_close(
        s_default.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=np.float64
        ),
        expected["default"]["s_table"],
        rtol=1e-4,
    )

    s_no_re = summary_gam(gam, re_test=False)
    # jsonlite writes R NULL as [] — mgcv's s.table is absent either way.
    assert expected["no_re_test"]["s_table"] in (None, [])
    assert len(s_no_re.s_table) == 0
