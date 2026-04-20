from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_value,
)
from tests.mgcv_parity_utils import (
    _REPO_ROOT,
    _assert_exact_mgcv_snapshot_parity,
    _build_r_command,
    _family_specs,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _normalize_python_formula_text,
    _run_mgcv_fixed_sp_score,
    _run_mgcv_snapshot,
)

R_SCRIPT = shutil.which("Rscript")

pytestmark = [
    pytest.mark.surface_regression,
    pytest.mark.surface_derivatives,
    pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity"),
]


def _make_binomial_data(seed=456, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) - 0.45 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=n).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_gaulss_data(seed=11, n=120):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.25, 1.25, n)
    mu = 0.3 + np.sin(np.pi * x)
    sigma = np.exp(-0.35 + 0.25 * x)
    y = rng.normal(mu, sigma, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _run_mgcv_fixed_sp_jackknife(data, formula, family, method, smoothing_params, nei):
    _family_nampy, family_token = _family_specs(family)
    del _family_nampy
    sp_list = np.asarray(smoothing_params, dtype=np.float64).tolist()
    formula_r = _normalize_python_formula_text(formula)

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
formula_text <- args[[2]]
family_name <- tolower(args[[3]])
method_name <- args[[4]]
sp <- as.numeric(fromJSON(args[[5]]))
nei <- fromJSON(args[[6]], simplifyVector = TRUE)
out <- args[[7]]
coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}
family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  gaulss = mgcv::gaulss(),
  stop(sprintf("Unsupported family for jackknife NCV test: %s", family_name))
)
formula_obj <- coerce_formula(formula_text)
fit <- gam(
  formula = formula_obj,
  data = d,
  family = family_obj,
  method = method_name,
  sp = sp,
  nei = nei
)
write_json(
  list(
    criterion_value = unname(as.numeric(fit$gcv.ubre)),
    dd = unname(attr(fit$gcv.ubre, "dd")),
    eta_cv = unname(attr(fit$gcv.ubre, "eta.cv"))
  ),
  out,
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "jackknife.json"
        script_path = tmpdir_path / "jackknife_ncv.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                formula_r,
                family_token,
                method,
                json.dumps(sp_list),
                json.dumps(nei),
                str(json_path),
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def test_gaussian_fixed_sp_ncv_matches_mgcv():
    data = _make_gaussian_data(seed=321, n=60)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.7, 1.2], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
    y = gam.family.validate_y(gam.y_)
    log_sp = np.log(sp)
    expected = _run_mgcv_fixed_sp_score(data, formula, "gaussian", "NCV", sp)

    actual = float(criterion_value(gam, y, log_sp, method="ncv"))
    actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="ncv"))

    np.testing.assert_allclose(
        actual,
        float(expected["criterion_value"]),
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual_grad,
        np.asarray(expected["gradient"], dtype=np.float64),
        atol=1e-9,
        rtol=0.0,
    )


def test_binomial_fixed_sp_qncv_matches_mgcv():
    data = _make_binomial_data(seed=456, n=80)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.8, 1.6], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, "binomial", sp)
    y = gam.family.validate_y(gam.y_)
    log_sp = np.log(sp)
    expected = _run_mgcv_fixed_sp_score(data, formula, "binomial", "QNCV", sp)

    actual = float(criterion_value(gam, y, log_sp, method="qncv"))
    actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="qncv"))

    np.testing.assert_allclose(
        actual,
        float(expected["criterion_value"]),
        atol=5e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual_grad,
        np.asarray(expected["gradient"], dtype=np.float64),
        atol=2e-4,
        rtol=0.0,
    )


def test_negbin_fixed_sp_ncv_matches_mgcv():
    data = _make_negbin_data(seed=77, n=70)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    sp = np.array([0.9, 1.4], dtype=np.float64)
    family = {"name": "negbin", "theta": 2.5}

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp)
    y = gam.family.validate_y(gam.y_)
    log_sp = np.log(sp)
    expected = _run_mgcv_fixed_sp_score(data, formula, "negbin:2.5", "NCV", sp)

    actual = float(criterion_value(gam, y, log_sp, method="ncv"))
    actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="ncv"))

    np.testing.assert_allclose(
        actual,
        float(expected["criterion_value"]),
        atol=3e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        actual_grad,
        np.asarray(expected["gradient"], dtype=np.float64),
        atol=1e-7,
        rtol=0.0,
    )


def test_poisson_outer_ncv_matches_mgcv():
    data = _make_poisson_data(seed=789, n=90)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "NCV")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "NCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-8,
        pred_rtol=0.0,
        edf_atol=1e-7,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=1e-7,
        sp_rtol=0.0,
        log_sp_atol=3e-8,
    )


def test_binomial_outer_qncv_single_smooth_matches_mgcv():
    data = _make_binomial_data(seed=456, n=100)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "binomial", "QNCV")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "QNCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=1e-7,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=1e-2,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )


def test_gamma_outer_qncv_single_smooth_matches_mgcv():
    data = _make_gamma_data(seed=123, n=100)
    formula = 'y ~ s(x0, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gamma", "QNCV")
    expected = _run_mgcv_snapshot(data, formula, "gamma", "QNCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=1e-7,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=1e-6,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )


def test_gaulss_outer_ncv_matches_mgcv():
    data = _make_gaulss_data(seed=11, n=90)
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]

    actual = _fit_nampy_snapshot(data, formula, "gaulss", "NCV")
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "NCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=2e-6,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=1e-6,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )


def test_gaulss_outer_qncv_matches_mgcv():
    data = _make_gaulss_data(seed=13, n=90)
    formula = ['y ~ s(x, bs="cr", k=6)', '~ s(x, bs="cr", k=5)']

    actual = _fit_nampy_snapshot(data, formula, "gaulss", "QNCV")
    expected = _run_mgcv_snapshot(data, formula, "gaulss", "QNCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=2e-6,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=1e-6,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )


def test_negbin_est_outer_ncv_matches_mgcv():
    data = _make_negbin_data(seed=93, n=90)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 1.8, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "NCV")
    expected = _run_mgcv_snapshot(data, formula, family, "NCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=2e-6,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=2e-5,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )
    np.testing.assert_allclose(
        float(actual["fit"]["family_theta"]),
        float(expected["fit"]["family_theta"]),
        atol=2e-6,
        rtol=0.0,
    )


def test_negbin_est_outer_qncv_matches_mgcv():
    data = _make_negbin_data(seed=94, n=90)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    family = {"name": "negbin", "theta": 1.8, "estimate_theta": True}

    actual = _fit_nampy_snapshot(data, formula, family, "QNCV")
    expected = _run_mgcv_snapshot(data, formula, family, "QNCV")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-7,
        pred_rtol=0.0,
        edf_atol=2e-6,
        criterion_atol=1e-8,
        criterion_rtol=0.0,
        sp_atol=2e-5,
        sp_rtol=0.0,
        log_sp_atol=1e-8,
    )
    np.testing.assert_allclose(
        float(actual["fit"]["family_theta"]),
        float(expected["fit"]["family_theta"]),
        atol=2e-6,
        rtol=0.0,
    )


def test_gaulss_fixed_sp_ncv_jackknife_dd_matches_mgcv():
    data = _make_gaulss_data(seed=17, n=60)
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]
    sp = np.array([0.9], dtype=np.float64)
    nei_r = {
        "d": list(range(1, 61)),
        "md": [20, 40, 60],
        "a": list(range(1, 61)),
        "ma": [20, 40, 60],
        "jackknife": 10,
    }

    gam = _fit_nampy_model_fixed_sp(data, formula, "gaulss", sp)
    gam.nei = dict(nei_r, index_base=1)
    y = gam.family.validate_y(gam.y_)
    log_sp = np.log(sp)

    expected = _run_mgcv_fixed_sp_jackknife(
        data,
        formula,
        "gaulss",
        "NCV",
        sp,
        nei_r,
    )
    actual = float(criterion_value(gam, y, log_sp, method="ncv"))
    ncv_state = getattr(gam, "_ncv_result_", None) or {}

    np.testing.assert_allclose(
        actual,
        float(expected["criterion_value"]),
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(ncv_state["dd"], dtype=np.float64),
        np.asarray(expected["dd"], dtype=np.float64),
        atol=1e-8,
        rtol=0.0,
    )
