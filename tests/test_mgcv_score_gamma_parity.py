from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import (
    _make_gamma_data,
    _make_gaussian_data,
    _make_poisson_data,
    _make_random_effect_data,
    _make_random_effect_data_noisy,
)

from nampy.gam import GAM
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)

R_SCRIPT = shutil.which("Rscript")
_REPO_ROOT = Path(__file__).resolve().parents[1]

_MGCV_FIXED_SP_GAMMA_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Usage: <csv> <json> <formula> <family> <method> <score_gamma> <sp_json> <select>")
}

csv_path <- args[[1]]
json_path <- args[[2]]
formula_text <- args[[3]]
family_name <- tolower(args[[4]])
method_name <- args[[5]]
score_gamma <- as.numeric(args[[6]])
sp <- as.numeric(jsonlite::fromJSON(args[[7]]))
select_flag <- tolower(args[[8]]) %in% c("true", "1", "yes")

if (!is.finite(score_gamma) || score_gamma <= 0) {
  stop("score_gamma must be finite and positive")
}

data <- read.csv(csv_path, stringsAsFactors = FALSE)
for (nm in names(data)) {
  if (is.character(data[[nm]])) data[[nm]] <- factor(data[[nm]])
}

family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL

family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  poisson = poisson(link = "log"),
  gamma = {
    link <- if (is.null(family_param) || family_param == "") "log" else family_param
    Gamma(link = link)
  },
  stop(sprintf("Unsupported family: %s", family_name))
)

formula_obj <- as.formula(formula_text)
gam_args <- list(
  formula = formula_obj,
  data = data,
  family = family_obj,
  method = method_name,
  sp = unname(sp),
  gamma = score_gamma,
  select = select_flag
)

eval_at_log_sp <- function(log_sp) {
  args_local <- gam_args
  args_local$sp <- unname(exp(log_sp))
  fit_local <- do.call(mgcv::gam, args_local)
  unname(as.numeric(fit_local$gcv.ubre))
}

fixed_sp_derivatives <- function(sp_ref, eps_grad = 1e-6, eps_hess = 1e-4) {
  if (length(sp_ref) == 0) {
    return(list(grad = numeric(0), hess = matrix(numeric(0), 0, 0)))
  }
  log_sp_ref <- log(pmax(as.numeric(sp_ref), 1e-300))
  grad <- rep(NA_real_, length(log_sp_ref))
  grad_steps <- pmax(eps_grad, 1e-5 * (1 + abs(log_sp_ref)))
  for (i in seq_along(log_sp_ref)) {
    plus1 <- log_sp_ref
    minus1 <- log_sp_ref
    plus2 <- log_sp_ref
    minus2 <- log_sp_ref
    plus1[i] <- plus1[i] + grad_steps[i]
    minus1[i] <- minus1[i] - grad_steps[i]
    plus2[i] <- plus2[i] + 2 * grad_steps[i]
    minus2[i] <- minus2[i] - 2 * grad_steps[i]
    grad[i] <- (
      -eval_at_log_sp(plus2) +
        8 * eval_at_log_sp(plus1) -
        8 * eval_at_log_sp(minus1) +
        eval_at_log_sp(minus2)
    ) / (12 * grad_steps[i])
  }

  hess <- matrix(0.0, length(log_sp_ref), length(log_sp_ref))
  hess_steps <- pmax(eps_hess, 1e-3 * (1 + abs(log_sp_ref)))
  f0 <- eval_at_log_sp(log_sp_ref)
  for (j in seq_along(log_sp_ref)) {
    for (k in j:length(log_sp_ref)) {
      if (j == k) {
        plus1 <- log_sp_ref
        minus1 <- log_sp_ref
        plus2 <- log_sp_ref
        minus2 <- log_sp_ref
        plus1[j] <- plus1[j] + hess_steps[j]
        minus1[j] <- minus1[j] - hess_steps[j]
        plus2[j] <- plus2[j] + 2 * hess_steps[j]
        minus2[j] <- minus2[j] - 2 * hess_steps[j]
        hess[j, j] <- (
          -eval_at_log_sp(plus2) +
            16 * eval_at_log_sp(plus1) -
            30 * f0 +
            16 * eval_at_log_sp(minus1) -
            eval_at_log_sp(minus2)
        ) / (12 * hess_steps[j] * hess_steps[j])
      } else {
        pp <- log_sp_ref
        pm <- log_sp_ref
        mp <- log_sp_ref
        mm <- log_sp_ref
        pp[j] <- pp[j] + hess_steps[j]
        pp[k] <- pp[k] + hess_steps[k]
        pm[j] <- pm[j] + hess_steps[j]
        pm[k] <- pm[k] - hess_steps[k]
        mp[j] <- mp[j] - hess_steps[j]
        mp[k] <- mp[k] + hess_steps[k]
        mm[j] <- mm[j] - hess_steps[j]
        mm[k] <- mm[k] - hess_steps[k]
        hess[j, k] <- (
          eval_at_log_sp(pp) -
            eval_at_log_sp(pm) -
            eval_at_log_sp(mp) +
            eval_at_log_sp(mm)
        ) / (4 * hess_steps[j] * hess_steps[k])
        hess[k, j] <- hess[j, k]
      }
    }
  }
  list(grad = unname(grad), hess = unname(hess))
}

fit <- do.call(mgcv::gam, gam_args)
fd <- fixed_sp_derivatives(sp)

jsonlite::write_json(
  list(
    criterion = unname(as.numeric(fit$gcv.ubre)),
    smoothing_params = unname(as.numeric(fit$sp)),
    scale = unname(as.numeric(fit$sig2)),
    grad = fd$grad,
    hess = fd$hess
  ),
  path = json_path,
  auto_unbox = TRUE
)
"""


def _run_mgcv_fixed_sp_score_gamma(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    score_gamma: float,
    sp: np.ndarray,
    select: bool = False,
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        script_path = tmpdir_path / "mgcv_score_gamma_fixed_sp.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(_MGCV_FIXED_SP_GAMMA_SCRIPT, encoding="utf-8")
        subprocess.run(
            [
                R_SCRIPT,
                str(script_path),
                str(csv_path),
                str(json_path),
                formula,
                family,
                method,
                str(float(score_gamma)),
                json.dumps(np.asarray(sp, dtype=np.float64).tolist()),
                "true" if select else "false",
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_fixed_sp_model(
    data: pd.DataFrame,
    formula: str,
    family: str,
    sp: np.ndarray,
    *,
    score_gamma: float,
):
    gam = GAM(
        formula=formula,
        family=family,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray(sp, dtype=np.float64),
        score_gamma=float(score_gamma),
    )
    gam.fit(data=data)
    y = gam.family.validate_y(np.asarray(data["y"], dtype=np.float64))
    log_sp = np.log(np.asarray(sp, dtype=np.float64))
    return gam, y, log_sp


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
class TestMgcvScoreGammaParity:
    def test_gaussian_exact_reml_fixed_sp_matches_mgcv(self):
        data = _make_gaussian_data(seed=1401, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        sp = np.array([0.45, 2.15], dtype=np.float64)
        score_gamma = 1.4

        expected = _run_mgcv_fixed_sp_score_gamma(
            data,
            formula,
            "gaussian",
            "REML",
            score_gamma=score_gamma,
            sp=sp,
        )
        gam, y, log_sp = _fit_fixed_sp_model(
            data,
            formula,
            "gaussian",
            sp,
            score_gamma=score_gamma,
        )

        assert gam._resolve_ml_reml_scoring_backend("reml") == "gaussian_exact"
        actual = float(criterion_value(gam, y, log_sp, method="reml"))

        np.testing.assert_allclose(
            actual,
            float(expected["criterion"]),
            rtol=0.0,
            atol=5e-5,
        )

    def test_gaussian_random_effect_reml_fixed_sp_value_gradient_hessian_match_mgcv(self):
        data = _make_random_effect_data()
        formula = 'y ~ s(f, bs="re")'
        sp = np.array([1.35], dtype=np.float64)
        score_gamma = 1.6

        expected = _run_mgcv_fixed_sp_score_gamma(
            data,
            formula,
            "gaussian",
            "REML",
            score_gamma=score_gamma,
            sp=sp,
        )
        gam, y, log_sp = _fit_fixed_sp_model(
            data,
            formula,
            "gaussian",
            sp,
            score_gamma=score_gamma,
        )

        assert gam._resolve_ml_reml_scoring_backend("reml") == "gaussian_exact"
        actual = float(criterion_value(gam, y, log_sp, method="reml"))
        actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="reml"))
        actual_hess = np.asarray(criterion_hessian(gam, y, log_sp, method="reml"))

        np.testing.assert_allclose(
            actual,
            float(expected["criterion"]),
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            actual_grad,
            np.asarray(expected["grad"], dtype=np.float64),
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            actual_hess,
            np.asarray(expected["hess"], dtype=np.float64),
            rtol=0.0,
            atol=5e-4,
        )

    def test_poisson_pirls_reml_fixed_sp_value_gradient_hessian_match_mgcv(self):
        data = _make_poisson_data(seed=1403, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        sp = np.array([0.55, 1.85], dtype=np.float64)
        score_gamma = 1.3

        expected = _run_mgcv_fixed_sp_score_gamma(
            data,
            formula,
            "poisson",
            "REML",
            score_gamma=score_gamma,
            sp=sp,
        )
        gam, y, log_sp = _fit_fixed_sp_model(
            data,
            formula,
            "poisson",
            sp,
            score_gamma=score_gamma,
        )

        assert gam._resolve_ml_reml_scoring_backend("reml") == "pirls_laplace"
        actual = float(criterion_value(gam, y, log_sp, method="reml"))
        actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="reml"))
        actual_hess = np.asarray(criterion_hessian(gam, y, log_sp, method="reml"))

        np.testing.assert_allclose(
            actual,
            float(expected["criterion"]),
            rtol=0.0,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            actual_grad,
            np.asarray(expected["grad"], dtype=np.float64),
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            actual_hess,
            np.asarray(expected["hess"], dtype=np.float64),
            rtol=0.0,
            atol=5e-4,
        )

    def test_gamma_pirls_reml_fixed_sp_value_gradient_hessian_match_mgcv(self):
        data = _make_gamma_data(seed=1404, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        sp = np.array([0.75, 1.45], dtype=np.float64)
        score_gamma = 1.25

        expected = _run_mgcv_fixed_sp_score_gamma(
            data,
            formula,
            "gamma",
            "REML",
            score_gamma=score_gamma,
            sp=sp,
        )
        gam, y, log_sp = _fit_fixed_sp_model(
            data,
            formula,
            "gamma",
            sp,
            score_gamma=score_gamma,
        )

        assert gam._resolve_ml_reml_scoring_backend("reml") == "pirls_laplace"
        actual = float(criterion_value(gam, y, log_sp, method="reml"))
        actual_grad = np.asarray(criterion_gradient(gam, y, log_sp, method="reml"))
        actual_hess = np.asarray(criterion_hessian(gam, y, log_sp, method="reml"))

        np.testing.assert_allclose(
            actual,
            float(expected["criterion"]),
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            actual_grad,
            np.asarray(expected["grad"], dtype=np.float64),
            rtol=0.0,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            actual_hess,
            np.asarray(expected["hess"], dtype=np.float64),
            rtol=0.0,
            atol=1e-3,
        )
