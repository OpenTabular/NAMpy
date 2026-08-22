"""Small R/SCAM reference harness for layered parity tests."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import R_SCRIPT, _build_r_command
from tests.reference_fixtures import (
    REFRESH_ENV,
    load_reference,
    portable_dataframe_repr,
    reference_key,
    refresh_enabled,
    save_reference,
)


def scam_reference_available() -> bool:
    return True


def _load_scam_reference(operation: str, payload) -> tuple[str, dict | None]:
    key = reference_key(operation, payload)
    return key, load_reference("scam", key)


def _require_local_scam() -> str:
    library = os.environ.get("SCAM_LIB_PATH")
    if not refresh_enabled():
        raise RuntimeError("SCAM execution is restricted to fixture refresh mode.")
    if R_SCRIPT is None or not library:
        raise RuntimeError(
            f"Rscript and SCAM_LIB_PATH are required when {REFRESH_ENV}=1."
        )
    return library


def run_scam_raw_constructor(
    data: pd.DataFrame,
    smooth_expr: str,
    *,
    new_data: pd.DataFrame | None = None,
    smoothcon: bool = False,
) -> dict:
    prediction_data = data if new_data is None else new_data
    key, cached = _load_scam_reference(
        "raw_constructor",
        {
            "data": data.to_csv(index=False),
            "new_data": prediction_data.to_csv(index=False),
            "smooth_expr": smooth_expr,
            "smoothcon": smoothcon,
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
s <- mgcv::s
d <- read.csv(args[[2]], stringsAsFactors=FALSE)
nd <- read.csv(args[[3]], stringsAsFactors=FALSE)
spec <- eval(parse(text=args[[4]]))
sm <- if (tolower(args[[6]]) == "true") {
  mgcv::smoothCon(spec, d, NULL, absorb.cons=TRUE, scale.penalty=TRUE)[[1]]
} else {
  mgcv:::smooth.construct3(spec, d, NULL)
}
prediction <- mgcv::PredictMat(sm, nd)
if (inherits(sm, c("po.smooth", "ipo.smooth", "cpopspline.smooth"))) {
  prediction <- prediction[, 2:ncol(prediction), drop=FALSE]
} else if (inherits(sm, "dpo.smooth")) {
  prediction <- prediction[, 1:(ncol(prediction)-1), drop=FALSE]
} else if (inherits(sm, c("miso.smooth", "mifo.smooth"))) {
  prediction <- prediction[, -sm$n.zero, drop=FALSE]
} else if (inherits(sm, "lipl.smooth")) {
  prediction <- prediction[, 2:sm$q1, drop=FALSE]
}
pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list("__kind__"="matrix", dim=as.integer(dim(x)), data=as.numeric(t(x)))
}
out <- list(
  class_name=as.character(class(sm)[1]),
  X=pack_matrix(sm$X),
  S=lapply(sm$S, pack_matrix),
  P=lapply(sm$P, pack_matrix),
  Sigma=pack_matrix(sm$Sigma),
  cmX=as.numeric(sm$cmX),
  p_ident=as.logical(sm$p.ident),
  knots=if (is.list(sm$knots)) lapply(sm$knots, as.numeric) else as.numeric(sm$knots),
  Xdf1=pack_matrix(sm$Xdf1),
  Xdf2=pack_matrix(sm$Xdf2),
  C=pack_matrix(sm$C),
  rank=as.integer(sm$rank),
  null_space_dim=as.integer(sm$null.space.dim),
  prediction=pack_matrix(prediction)
)
write_json(out, args[[5]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "data.csv"
        new_path = root / "new.csv"
        script_path = root / "constructor.R"
        output_path = root / "constructor.json"
        data.to_csv(data_path, index=False)
        prediction_data.to_csv(new_path, index=False)
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                str(data_path),
                str(new_path),
                smooth_expr,
                str(output_path),
                "true" if smoothcon else "false",
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


def run_scam_linear_functional_constructor(
    locations,
    weights,
    *,
    basis_code: str,
    k: int,
    m: int,
    new_locations=None,
    new_weights=None,
) -> dict:
    """Construct and predict a matrix-argument SCAM linear-functional term."""
    locations = np.asarray(locations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    new_locations = (
        locations if new_locations is None else np.asarray(new_locations, dtype=np.float64)
    )
    new_weights = (
        weights if new_weights is None else np.asarray(new_weights, dtype=np.float64)
    )
    key, cached = _load_scam_reference(
        "linear_functional_constructor",
        {
            "locations": locations.tolist(),
            "weights": weights.tolist(),
            "new_locations": new_locations.tolist(),
            "new_weights": new_weights.tolist(),
            "basis_code": basis_code,
            "k": k,
            "m": m,
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
decode_matrix <- function(value) {
  rows <- fromJSON(value, simplifyVector=FALSE)
  matrix(as.numeric(unlist(rows)), nrow=length(rows), byrow=TRUE)
}
X <- decode_matrix(args[[2]])
L <- decode_matrix(args[[3]])
X_new <- decode_matrix(args[[4]])
L_new <- decode_matrix(args[[5]])
spec <- mgcv::s(X, by=L, bs=args[[6]], k=as.integer(args[[7]]), m=as.integer(args[[8]]))
sm <- mgcv::smoothCon(
  spec, list(X=X, L=L), NULL, absorb.cons=TRUE, scale.penalty=TRUE, n=nrow(X)
)[[1]]
prediction <- mgcv::PredictMat(
  sm, list(X=X_new, L=L_new), n=nrow(X_new)
)
pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list("__kind__"="matrix", dim=as.integer(dim(x)), data=as.numeric(t(x)))
}
out <- list(
  X=pack_matrix(sm$X),
  S=lapply(sm$S, pack_matrix),
  p_ident=as.logical(sm$p.ident),
  knots=as.numeric(sm$knots),
  prediction=pack_matrix(prediction)
)
write_json(out, args[[9]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script_path = root / "linear_functional.R"
        output_path = root / "linear_functional.json"
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                json.dumps(locations.tolist()),
                json.dumps(weights.tolist()),
                json.dumps(new_locations.tolist()),
                json.dumps(new_weights.tolist()),
                basis_code,
                str(k),
                str(m),
                str(output_path),
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


def run_scam_linear_functional_fixed_fit(
    locations,
    weights,
    y,
    *,
    basis_code: str,
    k: int,
    m: int,
    sp: float,
    start,
) -> dict:
    """Fit a fixed-SP Gaussian SCAM linear-functional term."""
    locations = np.asarray(locations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    start_values = np.asarray(start, dtype=np.float64)
    key, cached = _load_scam_reference(
        "linear_functional_fixed_fit",
        {
            "locations": locations.tolist(),
            "weights": weights.tolist(),
            "y": response.tolist(),
            "basis_code": basis_code,
            "k": k,
            "m": m,
            "sp": sp,
            "start": start_values.tolist(),
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
decode_matrix <- function(value) {
  rows <- fromJSON(value, simplifyVector=FALSE)
  matrix(as.numeric(unlist(rows)), nrow=length(rows), byrow=TRUE)
}
X <- decode_matrix(args[[2]])
L <- decode_matrix(args[[3]])
y <- as.numeric(fromJSON(args[[4]]))
start <- as.numeric(fromJSON(args[[5]]))
fit <- scam(
  y ~ s(X, by=L, bs=args[[6]], k=as.integer(args[[7]]), m=as.integer(args[[8]])),
  sp=as.numeric(args[[9]]), start=start,
  optimizer=c("bfgs", "newton"),
  control=list(maxit=200, devtol.fit=1e-7, steptol.fit=1e-7)
)
out <- list(
  coefficients=as.numeric(fit$coefficients),
  coefficients_t=as.numeric(fit$coefficients.t),
  eta=as.numeric(fit$linear.predictors),
  mu=as.numeric(fit$fitted.values),
  deviance=as.numeric(fit$deviance),
  trA=as.numeric(fit$trA)
)
write_json(out, args[[10]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script_path = root / "linear_functional_fit.R"
        output_path = root / "linear_functional_fit.json"
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                json.dumps(locations.tolist()),
                json.dumps(weights.tolist()),
                json.dumps(response.tolist()),
                json.dumps(start_values.tolist()),
                basis_code,
                str(k),
                str(m),
                str(sp),
                str(output_path),
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


def _decode_matrices(value):
    if isinstance(value, dict):
        if value.get("__kind__") == "matrix":
            import numpy as np

            return np.asarray(value["data"], dtype=np.float64).reshape(value["dim"])
        return {key: _decode_matrices(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_matrices(item) for item in value]
    return value


def run_scam_fixed_sp_fit(
    data: pd.DataFrame,
    formula: str,
    *,
    family: str,
    sp,
    start,
    positive_transform: str = "exp",
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    include_behavior: bool = False,
) -> dict:
    """Run vendored ``scam`` with fixed smoothing and a shared start vector."""
    sp_values = np.asarray(sp, dtype=float).reshape(-1).tolist()
    start_values = np.asarray(start, dtype=float).reshape(-1).tolist()
    key, cached = _load_scam_reference(
        "fixed_sp_fit",
        {
            "data": portable_dataframe_repr(data),
            "formula": formula,
            "family": family,
            "sp": sp_values,
            "start": start_values,
            "positive_transform": positive_transform,
            "softplus_beta": softplus_beta,
            "softplus_threshold": softplus_threshold,
            "include_behavior": include_behavior,
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
d <- read.csv(args[[2]], stringsAsFactors=FALSE)
formula <- as.formula(args[[3]])
family_spec <- eval(parse(text=args[[4]]))
family <- if (is.function(family_spec)) family_spec() else family_spec
sp <- as.numeric(fromJSON(args[[5]]))
start <- as.numeric(fromJSON(args[[6]]))
not_exp <- tolower(args[[7]]) == "softplus"
control <- list(
  maxit=200,
  devtol.fit=1e-7,
  steptol.fit=1e-7,
  b.notexp=as.numeric(args[[8]]),
  threshold.notexp=as.numeric(args[[9]])
)
include_behavior <- tolower(args[[10]]) == "true"
fit <- scam(
  formula, data=d, family=family, sp=sp, start=start,
  optimizer=c("bfgs", "newton"), not.exp=not_exp, control=control
)
pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list("__kind__"="matrix", dim=as.integer(dim(x)), data=as.numeric(t(x)))
}
out <- list(
  coefficients=as.numeric(fit$coefficients),
  coefficients_t=as.numeric(fit$coefficients.t),
  p_ident=as.logical(fit$p.ident),
  eta=as.numeric(fit$linear.predictors),
  mu=as.numeric(fit$fitted.values),
  deviance=as.numeric(fit$deviance),
  edf=as.numeric(fit$edf),
  edf1=as.numeric(fit$edf1),
  trA=as.numeric(fit$trA),
  scale=as.numeric(fit$sig2),
  Vp=pack_matrix(fit$Vp),
  Vp_t=pack_matrix(fit$Vp.t),
  Ve=pack_matrix(fit$Ve),
  Ve_t=pack_matrix(fit$Ve.t),
  iter=as.integer(fit$iter),
  pdev_hist=as.numeric(fit$pdev.hist)
)
if (include_behavior) {
  if (length(fit$smooth) > 0 && length(fit$smooth[[1]]$term) == 1) {
    deriv1 <- derivative.scam(fit, smooth.number=1, deriv=1)
    deriv2 <- derivative.scam(fit, smooth.number=1, deriv=2)
    out$derivative1 <- as.numeric(deriv1$d)
    out$derivative1_se <- as.numeric(deriv1$se.d)
    out$derivative2 <- as.numeric(deriv2$d)
    out$derivative2_se <- as.numeric(deriv2$se.d)
  }
  pred_link <- predict(fit, type="link", se.fit=TRUE)
  pred_response <- predict(fit, type="response", se.fit=TRUE)
  pred_terms <- predict(fit, type="terms", se.fit=TRUE)
  out$predict_link <- as.numeric(pred_link$fit)
  out$predict_link_se <- as.numeric(pred_link$se.fit)
  out$predict_response <- as.numeric(pred_response$fit)
  out$predict_response_se <- as.numeric(pred_response$se.fit)
  out$predict_terms <- pack_matrix(pred_terms$fit)
  out$predict_terms_se <- pack_matrix(pred_terms$se.fit)
  out$residual_deviance <- as.numeric(residuals(fit, type="deviance"))
  out$residual_pearson <- as.numeric(residuals(fit, type="pearson"))
  out$residual_scaled_pearson <- as.numeric(residuals(fit, type="scaled.pearson"))
  out$residual_working <- as.numeric(residuals(fit, type="working"))
  out$residual_response <- as.numeric(residuals(fit, type="response"))
  out$residual_rquantile <- as.numeric(residuals(fit, type="rquantile", setseed=314))
  summary_fit <- summary(fit)
  out$summary <- list(
    p_table=pack_matrix(summary_fit$p.table),
    pterms_table=pack_matrix(summary_fit$pTerms.table),
    s_table=pack_matrix(summary_fit$s.table),
    residual_df=as.numeric(summary_fit$residual.df),
    scale=as.numeric(summary_fit$scale),
    r_sq=as.numeric(summary_fit$r.sq),
    dev_expl=as.numeric(summary_fit$dev.expl),
    n=as.integer(summary_fit$n),
    np=as.integer(summary_fit$np),
    rank=as.integer(summary_fit$rank)
  )
}
out$gcv_score <- as.numeric(nrow(d) * fit$deviance / (nrow(d) - fit$trA)^2)
if (!is.null(fit$family$family) && fit$family$family %in% c("poisson", "binomial")) {
  out$ubre_score <- as.numeric(fit$deviance / nrow(d) - 1 + 2 * fit$trA / nrow(d))
} else {
  out$ubre_score <- NULL
}
write_json(out, args[[11]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "data.csv"
        script_path = root / "fit.R"
        output_path = root / "fit.json"
        data.to_csv(data_path, index=False)
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                str(data_path),
                formula,
                family,
                json.dumps(sp_values),
                json.dumps(start_values),
                positive_transform,
                str(softplus_beta),
                str(softplus_threshold),
                str(bool(include_behavior)).lower(),
                str(output_path),
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


def run_scam_selected_sp_fit(
    data: pd.DataFrame,
    formula: str,
    *,
    family: str,
    start,
    positive_transform: str = "exp",
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> dict:
    """Run the released ``scam`` BFGS GCV/UBRE smoothing search."""
    start_values = np.asarray(start, dtype=float).reshape(-1).tolist()
    key, cached = _load_scam_reference(
        "selected_sp_fit",
        {
            "data": data.to_csv(index=False),
            "formula": formula,
            "family": family,
            "start": start_values,
            "positive_transform": positive_transform,
            "softplus_beta": softplus_beta,
            "softplus_threshold": softplus_threshold,
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
d <- read.csv(args[[2]], stringsAsFactors=FALSE)
formula <- as.formula(args[[3]])
family_spec <- eval(parse(text=args[[4]]))
family <- if (is.function(family_spec)) family_spec() else family_spec
start <- as.numeric(fromJSON(args[[5]]))
not_exp <- tolower(args[[6]]) == "softplus"
control <- list(
  maxit=200,
  devtol.fit=1e-7,
  steptol.fit=1e-7,
  b.notexp=as.numeric(args[[7]]),
  threshold.notexp=as.numeric(args[[8]])
)
fit <- scam(
  formula, data=d, family=family, start=start,
  optimizer=c("bfgs", "newton"), not.exp=not_exp, control=control
)
out <- list(
  sp=as.numeric(fit$sp),
  coefficients=as.numeric(fit$coefficients),
  coefficients_t=as.numeric(fit$coefficients.t),
  eta=as.numeric(fit$linear.predictors),
  mu=as.numeric(fit$fitted.values),
  deviance=as.numeric(fit$deviance),
  trA=as.numeric(fit$trA),
  score=as.numeric(fit$gcv.ubre),
  gradient=as.numeric(fit$dgcv.ubre),
  iterations=as.integer(fit$iterations),
  termcode=as.integer(fit$termcode),
  score_hist=as.numeric(fit$score.hist)
)
write_json(out, args[[9]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "data.csv"
        script_path = root / "fit.R"
        output_path = root / "fit.json"
        data.to_csv(data_path, index=False)
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                str(data_path),
                formula,
                family,
                json.dumps(start_values),
                positive_transform,
                str(softplus_beta),
                str(softplus_threshold),
                str(output_path),
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


def run_scam_ar1_fixed_fit(
    data: pd.DataFrame,
    formula: str,
    *,
    sp,
    start,
    ar1_rho: float,
    ar_start,
) -> dict:
    """Run fixed-SP Gaussian-identity SCAM with its AR(1) root transform."""
    sp_values = np.asarray(sp, dtype=float).reshape(-1).tolist()
    start_values = np.asarray(start, dtype=float).reshape(-1).tolist()
    ar_start_values = np.asarray(ar_start, dtype=bool).reshape(-1).tolist()
    key, cached = _load_scam_reference(
        "ar1_fixed_fit",
        {
            "data": data.to_csv(index=False),
            "formula": formula,
            "sp": sp_values,
            "start": start_values,
            "ar1_rho": ar1_rho,
            "ar_start": ar_start_values,
        },
    )
    if cached is not None:
        return _decode_matrices(cached)
    library = _require_local_scam()
    code = r'''
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(scam))
suppressPackageStartupMessages(library(jsonlite))
d <- read.csv(args[[2]], stringsAsFactors=FALSE)
sp <- as.numeric(fromJSON(args[[4]]))
start <- as.numeric(fromJSON(args[[5]]))
ar_start <- as.logical(fromJSON(args[[7]]))
fit <- scam(
  as.formula(args[[3]]), data=d, family=gaussian(), sp=sp, start=start,
  AR1.rho=as.numeric(args[[6]]), AR.start=ar_start,
  optimizer=c("bfgs", "newton"),
  control=list(maxit=200, devtol.fit=1e-7, steptol.fit=1e-7)
)
pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list("__kind__"="matrix", dim=as.integer(dim(x)), data=as.numeric(t(x)))
}
out <- list(
  coefficients=as.numeric(fit$coefficients),
  coefficients_t=as.numeric(fit$coefficients.t),
  eta=as.numeric(fit$linear.predictors),
  mu=as.numeric(fit$fitted.values),
  deviance=as.numeric(fit$deviance),
  trA=as.numeric(fit$trA),
  scale=as.numeric(fit$sig2),
  Vp_t=pack_matrix(fit$Vp.t),
  std_rsd=as.numeric(fit$std.rsd)
)
write_json(out, args[[8]], digits=17, auto_unbox=TRUE, null="null")
'''
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "data.csv"
        script_path = root / "ar1_fit.R"
        output_path = root / "ar1_fit.json"
        data.to_csv(data_path, index=False)
        script_path.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                library,
                str(data_path),
                formula,
                json.dumps(sp_values),
                json.dumps(start_values),
                str(float(ar1_rho)),
                json.dumps(ar_start_values),
                str(output_path),
            ),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("scam", key, result)
        return _decode_matrices(result)


__all__ = [
    "run_scam_fixed_sp_fit",
    "run_scam_ar1_fixed_fit",
    "run_scam_linear_functional_constructor",
    "run_scam_linear_functional_fixed_fit",
    "run_scam_raw_constructor",
    "run_scam_selected_sp_fit",
    "scam_reference_available",
]
