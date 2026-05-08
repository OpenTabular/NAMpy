from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam._model_state import _deviance, _edf_total, _fit_scale  # noqa: E402
from tests.mgcv_parity_utils import (  # noqa: E402
    _build_r_command,
    _family_specs,
    _normalize_python_formula_text,
    _run_mgcv_predict_on_newdata,
)
from tests.parity.test_mgcv_prediction_inference_diagnostics_parity import (  # noqa: E402
    CASE_BY_ID,
    _case_bundle,
    _newdata_for_case,
    _normalize_matrix,
)


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size or b.size else 0.0


def _run_mgcv_scale_probe(data, formula, family, method, select):
    _, family_token = _family_specs(family)
    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
train <- read.csv(args[[1]], stringsAsFactors = FALSE)
formula_text <- args[[2]]
family_name <- tolower(args[[3]])
method_name <- args[[4]]
select_flag <- identical(tolower(args[[5]]), "true")
out_path <- args[[6]]
for (nm in names(train)) if (is.character(train[[nm]])) train[[nm]] <- factor(train[[nm]])
coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}
num_or_null <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) == 0) return(NULL)
  unname(as.numeric(x))
}
family_obj <- switch(
  strsplit(family_name, ":", fixed = TRUE)[[1]][1],
  gaussian = gaussian(),
  binomial = binomial(link = "logit"),
  poisson = poisson(link = "log"),
  gamma = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    link <- if (length(family_parts) >= 2) family_parts[[2]] else "inverse"
    Gamma(link = link)
  },
  stop(sprintf("Unsupported family for scale probe: %s", family_name))
)
fit <- gam(
  formula = coerce_formula(formula_text),
  data = train,
  family = family_obj,
  method = method_name,
  select = select_flag
)
mu <- fit$fitted.values
w <- fit$prior.weights
pearson <- sum(w * (fit$y - mu)^2 / fit$family$variance(mu))
edf <- sum(fit$edf)
payload <- list(
  sig2 = num_or_null(fit$sig2),
  scale = num_or_null(fit$scale),
  scale_est = num_or_null(fit$scale.est),
  reml_scale = num_or_null(fit$reml.scale),
  pearson_over_n_minus_edf = pearson / (length(fit$y) - edf),
  deviance_over_n_minus_edf = fit$deviance / (length(fit$y) - edf),
  pearson = pearson,
  deviance = fit$deviance,
  edf = edf,
  sp = unname(as.numeric(fit$sp)),
  covariance_diag = unname(as.numeric(diag(fit$Vp)))
)
write_json(payload, out_path, auto_unbox = TRUE, digits = 17)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_path = tmpdir_path / "train.csv"
        script_path = tmpdir_path / "probe_scale.R"
        json_path = tmpdir_path / "probe_scale.json"
        data.to_csv(train_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(train_path),
                _normalize_python_formula_text(formula),
                family_token,
                method,
                "true" if select else "false",
                str(json_path),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    case_id = "gamma_log"
    case = CASE_BY_ID[case_id]
    data, expected, model = _case_bundle(case_id)
    newdata = _newdata_for_case(case_id)

    r_terms = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type="terms",
        return_se=True,
        select=case.select,
        weights_column=case.weights_column,
    )
    r_link = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type="link",
        return_se=True,
        select=case.select,
        weights_column=case.weights_column,
    )
    r_lpmatrix = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="REML",
        type="lpmatrix",
        return_se=False,
        select=case.select,
        weights_column=case.weights_column,
    )

    actual_terms, actual_terms_se = model.predict(
        X=newdata, type="terms", return_se=True
    )
    actual_link, actual_link_se = model.predict(X=newdata, type="link", return_se=True)
    actual_lpmatrix = model.predict(X=newdata, type="lpmatrix")

    expected_terms = _normalize_matrix(r_terms["pred"])
    expected_terms_se = _normalize_matrix(r_terms["se"])
    actual_terms = _normalize_matrix(actual_terms)
    actual_terms_se = _normalize_matrix(actual_terms_se)
    expected_link_se = np.asarray(r_link["se"], dtype=np.float64).reshape(-1)
    actual_link_se = np.asarray(actual_link_se, dtype=np.float64).reshape(-1)
    expected_lpmatrix = np.asarray(r_lpmatrix["pred"], dtype=np.float64)
    actual_lpmatrix = np.asarray(actual_lpmatrix, dtype=np.float64)

    fit = expected["fit"]
    cov = np.asarray(model._select_cov(None), dtype=np.float64)
    scale = float(_fit_scale(model))
    expected_cov = np.asarray(fit.get("cov_bayes", []), dtype=np.float64)
    cov_ratio = None
    if expected_cov.shape == cov.shape:
        diag_ok = np.abs(np.diag(expected_cov)) > 0.0
        cov_ratio = (np.diag(cov)[diag_ok] / np.diag(expected_cov)[diag_ok]).tolist()
    y_train = np.asarray(model.y_, dtype=np.float64).ravel()
    mu_train = np.asarray(model.predict(type="response"), dtype=np.float64).ravel()
    pearson = float(np.sum((y_train - mu_train) ** 2 / model.family.variance(mu_train)))

    term_se_ratio = actual_terms_se / expected_terms_se
    link_se_ratio = actual_link_se / expected_link_se
    report = {
        "case_id": case_id,
        "fit_keys": sorted(fit.keys()),
        "mgcv_direct_scale_probe": _run_mgcv_scale_probe(
            data, case.formula, case.family, "REML", case.select
        ),
        "model_scale": scale,
        "mgcv_scale": fit.get("scale", None),
        "model_trace_H": float(_edf_total(model)),
        "mgcv_trace_H": fit.get("trace_H", None),
        "model_deviance": float(_deviance(model)),
        "mgcv_deviance": fit.get("deviance", None),
        "model_pearson": float(
            pearson
        ),
        "pearson_over_n_minus_edf": float(
            pearson / (len(y_train) - float(_edf_total(model)))
        ),
        "pearson_over_n": float(
            pearson / len(y_train)
        ),
        "terms_pred_max_abs": _max_abs(actual_terms, expected_terms),
        "terms_se_max_abs": _max_abs(actual_terms_se, expected_terms_se),
        "terms_se_max_rel": float(
            np.max(np.abs(actual_terms_se / expected_terms_se - 1.0))
        ),
        "terms_se_ratio_min_max_mean": [
            float(np.min(term_se_ratio)),
            float(np.max(term_se_ratio)),
            float(np.mean(term_se_ratio)),
        ],
        "link_pred_max_abs": _max_abs(actual_link, r_link["pred"]),
        "link_se_max_abs": _max_abs(actual_link_se, expected_link_se),
        "link_se_max_rel": float(np.max(np.abs(link_se_ratio - 1.0))),
        "link_se_ratio_min_max_mean": [
            float(np.min(link_se_ratio)),
            float(np.max(link_se_ratio)),
            float(np.mean(link_se_ratio)),
        ],
        "lpmatrix_shape_actual": list(actual_lpmatrix.shape),
        "lpmatrix_shape_expected": list(expected_lpmatrix.shape),
        "lpmatrix_max_abs": _max_abs(actual_lpmatrix, expected_lpmatrix),
        "cov_diag_min_max": [
            float(np.min(np.diag(cov))),
            float(np.max(np.diag(cov))),
        ],
        "cov_diag_ratio_min_max_mean": None
        if cov_ratio is None
        else [
            float(np.min(cov_ratio)),
            float(np.max(cov_ratio)),
            float(np.mean(cov_ratio)),
        ],
        "scale_ratio": None if fit.get("scale", None) is None else scale / fit["scale"],
        "term_names": r_terms.get("term_names", None),
        "first_terms_se_pairs": np.column_stack(
            [actual_terms_se[:8, 0], expected_terms_se[:8, 0]]
        ).tolist(),
        "first_link_se_pairs": np.column_stack(
            [actual_link_se[:8], expected_link_se[:8]]
        ).tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
