"""Shared mgcv/R snapshot helpers for parity tests (decoupled from ``test_gam_mgcv_parity``)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent

R_SCRIPT = shutil.which("Rscript")
MGCV_SNAPSHOT_SCRIPT = _TESTS_DIR / "parity" / "mgcv_snapshot.R"


def _make_gaussian_data(seed=123, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_binomial_data(seed=456, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) - 0.45 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_poisson_data(seed=789, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_gamma_data(seed=1701, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.15 + 0.6 * np.sin(x0) - 0.2 * x1
    mu = np.exp(eta)
    shape = 3.5
    y = rng.gamma(shape=shape, scale=mu / shape)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_negbin_data(seed=2024, n=240, theta=1.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.2 + 0.55 * np.sin(x0) - 0.25 * x1
    mu = np.exp(eta)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_random_effect_data():
    f = np.array(["b", "a", "c", "a", "b", "c", "a", "c"], dtype=object)
    effects = {"a": 1.5, "b": -0.25, "c": 0.75}
    y = np.array([effects[v] for v in f], dtype=np.float64)
    return pd.DataFrame({"y": y, "f": f})


def _make_random_effect_data_noisy(*, seed=21, n_draws=36, sigma=0.35):
    """Larger noisy sample so REML \\lambda has an interior optimum (tight ``log\\lambda`` vs mgcv)."""
    rng = np.random.default_rng(seed)
    levels = np.array(["a", "b", "c"])
    u = {"a": 1.1, "b": -0.4, "c": 0.55}
    f = rng.choice(levels, size=n_draws)
    signal = np.array([u[str(v)] for v in f], dtype=np.float64)
    y = signal + rng.normal(scale=sigma, size=n_draws)
    return pd.DataFrame({"y": y, "f": f})


def _make_fs_data():
    rng = np.random.default_rng(27)
    levels = ["a", "b", "c"]
    n = 18
    f = np.array([levels[i % len(levels)] for i in range(n)], dtype=object)
    x = np.linspace(0.1, 1.7, n)
    offsets = {"a": 1.0, "b": -0.5, "c": 0.9}
    y = 0.4 * np.sin(2 * x) + np.array([offsets[v] for v in f])
    return pd.DataFrame({"y": y, "f": f, "x": x})


def _make_sz_data():
    rng = np.random.default_rng(41)
    f1 = np.array(["a", "b", "b", "c", "c", "a", "b", "c"], dtype=object)
    f2 = np.array(["x", "x", "y", "x", "y", "y", "x", "y"], dtype=object)
    x = rng.uniform(0.0, 2.0, size=len(f1))
    base = np.sin(x) + 0.2 * x
    offsets = {
        ("a", "x"): 0.2,
        ("a", "y"): -0.1,
        ("b", "x"): -0.6,
        ("b", "y"): 0.4,
        ("c", "x"): 0.1,
        ("c", "y"): -0.3,
    }
    y = np.array([base[i] + offsets[(f1[i], f2[i])] for i in range(len(f1))])
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2, "x": x})


def _make_mrf_data():
    regions = np.array(["A", "B", "C", "A", "B", "C", "A", "B"], dtype=object)
    vals = {"A": 1.0, "B": -0.5, "C": 0.3}
    y = np.array([vals[r] for r in regions], dtype=np.float64)
    return pd.DataFrame({"y": y, "region": regions})


def _family_specs(family):
    if isinstance(family, dict):
        key = str(family.get("name", "")).lower()
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            theta = float(family.get("theta", 1.0))
            return family, f"negbin:{theta:.12g}"
        return family, key
    key = str(family).lower()
    return family, key


def _run_mgcv_snapshot(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    weights_column: str | None = None,
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    _family_nampy, family_token = _family_specs(family)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "snapshot.json"
        data.to_csv(csv_path, index=False)

        cmd = [
            R_SCRIPT,
            str(MGCV_SNAPSHOT_SCRIPT),
            str(csv_path),
            str(json_path),
            formula,
            family_token,
            method,
            "true" if select else "false",
        ]
        if weights_column is not None:
            cmd.append(str(weights_column))

        subprocess.run(
            cmd,
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )

        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_model(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    sample_weight=None,
):
    family_nampy, _family_token = _family_specs(family)
    method_key = str(method).lower()
    gam = GAM(
        family=family_nampy,
        formula=formula,
        select=select,
        optimize_smoothing=(method_key != "fixed"),
        smoothing_method=method,
    )
    gam.fit(data=data, sample_weight=sample_weight)
    return gam


def _fit_nampy_model_fixed_sp(
    data: pd.DataFrame,
    formula: str,
    family,
    smoothing_params,
    *,
    select: bool = False,
    sample_weight=None,
):
    """Fit at explicit linear smoothing parameters (no outer optimisation)."""
    family_nampy, _ = _family_specs(family)
    sp = np.asarray(smoothing_params, dtype=np.float64).ravel()
    gam = GAM(
        family=family_nampy,
        formula=formula,
        select=select,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=sp,
    )
    gam.fit(data=data, sample_weight=sample_weight)
    return gam


def _fit_nampy_snapshot(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    sample_weight=None,
):
    return _fit_nampy_model(
        data, formula, family, method, select=select, sample_weight=sample_weight
    ).parity_snapshot(X=data, include_covariances=False)


def _run_mgcv_smoothcon_matrix(data: pd.DataFrame, smooth_expr: str):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
sm <- smoothCon(eval(parse(text = args[[3]])), d, absorb.cons = TRUE)[[1]]
write_json(list(X = unname(sm$X)), out, auto_unbox = TRUE, digits = 17)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "matrix.json"
        script_path = tmpdir_path / "smoothcon_dump.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), smooth_expr],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_smoothcon_penalties(
    data: pd.DataFrame,
    smooth_expr: str,
    *,
    absorb_cons: bool,
    scale_penalty: bool,
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
sm <- smoothCon(
  eval(parse(text = args[[3]])),
  d,
  absorb.cons = tolower(args[[4]]) == "true",
  scale.penalty = tolower(args[[5]]) == "true"
)[[1]]
write_json(list(S = lapply(sm$S, unname)), out, auto_unbox = TRUE, digits = 17)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "penalties.json"
        script_path = tmpdir_path / "smoothcon_penalties.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [
                R_SCRIPT,
                str(script_path),
                str(csv_path),
                str(json_path),
                smooth_expr,
                "true" if absorb_cons else "false",
                "true" if scale_penalty else "false",
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_smoothcon_matrix_unscaled(data: pd.DataFrame, smooth_expr: str):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
sm <- smoothCon(
  eval(parse(text = args[[3]])),
  d,
  absorb.cons = FALSE,
  scale.penalty = FALSE
)[[1]]
write_json(list(X = unname(sm$X)), out, auto_unbox = TRUE, digits = 17)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "matrix.json"
        script_path = tmpdir_path / "smoothcon_dump_unscaled.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), smooth_expr],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_natparam_cr(data: pd.DataFrame, *, k: int):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
out <- args[[2]]
k <- as.integer(args[[3]])
sm <- smoothCon(s(x, bs = "cr", k = k), d, absorb.cons = FALSE)[[1]]
np <- mgcv:::nat.param(sm$X, sm$S[[1]], rank = sm$rank, type = 3, unit.fnorm = TRUE)
write_json(
  list(
    X = unname(np$X),
    P = unname(np$P),
    rank = unname(np$rank),
    rawX = unname(sm$X),
    rawS = unname(sm$S[[1]])
  ),
  out,
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "natparam.json"
        script_path = tmpdir_path / "natparam_dump.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), str(int(k))],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_predict_on_newdata(
    data: pd.DataFrame,
    newdata: pd.DataFrame,
    formula: str,
    *,
    family="gaussian",
    method="REML",
    type="link",
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    _family_nampy, family_token = _family_specs(family)
    del _family_nampy

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
train <- read.csv(args[[1]], stringsAsFactors = FALSE)
newd <- read.csv(args[[2]], stringsAsFactors = FALSE)
formula_text <- args[[3]]
family_name <- tolower(args[[4]])
method_name <- args[[5]]
pred_type <- args[[6]]
for (nm in names(train)) if (is.character(train[[nm]])) train[[nm]] <- factor(train[[nm]])
for (nm in names(newd)) {
  if (is.character(newd[[nm]]) && nm %in% names(train) && is.factor(train[[nm]])) {
    newd[[nm]] <- factor(newd[[nm]], levels = levels(train[[nm]]))
  } else if (is.character(newd[[nm]])) {
    newd[[nm]] <- factor(newd[[nm]])
  }
}
family_obj <- switch(
  family_name,
  gaussian = gaussian(),
  binomial = binomial(link = "logit"),
  poisson = poisson(link = "log"),
  gamma = Gamma(link = "log"),
  stop(sprintf("Unsupported family for newdata parity: %s", family_name))
)
fit <- gam(
  formula = as.formula(formula_text),
  data = train,
  family = family_obj,
  method = method_name
)
write_json(
  list(pred = unname(as.numeric(predict(fit, newdata = newd, type = pred_type)))),
  args[[7]],
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_path = tmpdir_path / "train.csv"
        new_path = tmpdir_path / "new.csv"
        json_path = tmpdir_path / "pred.json"
        script_path = tmpdir_path / "predict_newdata.R"
        data.to_csv(train_path, index=False)
        newdata.to_csv(new_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [
                R_SCRIPT,
                str(script_path),
                str(train_path),
                str(new_path),
                formula,
                family_token,
                method,
                type,
                str(json_path),
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _assert_basic_mgcv_parity(
    actual,
    expected,
    *,
    pred_atol,
    pred_rtol,
    sp_log_atol,
    check_sp=True,
    check_criterion=True,
    criterion_atol=0.5,
):
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    a_sp = np.atleast_1d(np.asarray(a_fit["smoothing_params"], dtype=np.float64))
    e_sp = np.atleast_1d(np.asarray(e_fit["smoothing_params"], dtype=np.float64))

    assert len(a_sp) == len(e_sp)
    if check_sp:
        np.testing.assert_allclose(
            np.log(a_sp),
            np.log(e_sp),
            atol=sp_log_atol,
            rtol=0.0,
        )

    np.testing.assert_allclose(
        np.asarray(a_fit["edf_total"], dtype=np.float64),
        np.asarray(e_fit["edf_total"], dtype=np.float64),
        atol=0.15,
        rtol=0.06,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_by_term"], dtype=np.float64),
        np.asarray(e_fit["edf_by_term"], dtype=np.float64),
        atol=0.15,
        rtol=0.08,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["deviance"], dtype=np.float64),
        np.asarray(e_fit["deviance"], dtype=np.float64),
        atol=0.3,
        rtol=0.06,
    )

    if (
        check_criterion
        and a_fit.get("criterion_value", None) is not None
        and e_fit.get("criterion_value", None) is not None
    ):
        np.testing.assert_allclose(
            np.asarray(a_fit["criterion_value"], dtype=np.float64),
            np.asarray(e_fit["criterion_value"], dtype=np.float64),
            atol=float(criterion_atol),
            rtol=0.05,
        )

    np.testing.assert_allclose(
        np.asarray(a_pred["response"], dtype=np.float64),
        np.asarray(e_pred["response"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_rtol,
    )
    np.testing.assert_allclose(
        np.asarray(a_pred["link"], dtype=np.float64),
        np.asarray(e_pred["link"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_rtol,
    )


def _assert_allclose_up_to_column_sign(actual, expected, *, atol, rtol):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    aligned = actual.copy()
    for j in range(actual.shape[1]):
        if np.linalg.norm(actual[:, j] - expected[:, j]) > np.linalg.norm(
            -actual[:, j] - expected[:, j]
        ):
            aligned[:, j] *= -1.0
    np.testing.assert_allclose(aligned, expected, atol=atol, rtol=rtol)


__all__ = [
    "MGCV_SNAPSHOT_SCRIPT",
    "R_SCRIPT",
    "_assert_allclose_up_to_column_sign",
    "_assert_basic_mgcv_parity",
    "_family_specs",
    "_fit_nampy_model",
    "_fit_nampy_model_fixed_sp",
    "_fit_nampy_snapshot",
    "_make_binomial_data",
    "_make_fs_data",
    "_make_gamma_data",
    "_make_gaussian_data",
    "_make_mrf_data",
    "_make_negbin_data",
    "_make_poisson_data",
    "_make_random_effect_data",
    "_make_random_effect_data_noisy",
    "_make_sz_data",
    "_run_mgcv_natparam_cr",
    "_run_mgcv_predict_on_newdata",
    "_run_mgcv_smoothcon_matrix",
    "_run_mgcv_smoothcon_matrix_unscaled",
    "_run_mgcv_smoothcon_penalties",
    "_run_mgcv_snapshot",
]
