"""Probe Gaussian GCV gam.vcomp pieces against the cached mgcv payload."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nampy.gam._model_state import _fit_scale
from nampy.gam.smoothing_selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.smoothing_selection.postfit import _mgcv_penalty_rescale_factors
from tests.mgcv_parity_utils import (
    _REPO_ROOT,
    _build_r_command,
    _fit_nampy_model,
    _make_gaussian_data,
    _run_mgcv_gam_vcomp,
)


def _mgcv_raw(data, formula: str) -> dict:
    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
fit <- gam(as.formula(args[[3]]), data = d, family = gaussian(), method = "GCV.Cp")
out <- list(
  sig2 = unname(fit$sig2),
  reml_scale = if (is.null(fit$reml.scale)) NULL else unname(fit$reml.scale),
  sp = unname(fit$sp),
  full_sp = if (is.null(fit$full.sp)) NULL else unname(fit$full.sp),
  s_scale = unname(fit$smooth[[1]]$S.scale),
  vc_false = unname(gam.vcomp(fit, rescale = FALSE)),
  vc_true = unname(gam.vcomp(fit, rescale = TRUE))
)
write_json(out, args[[2]], auto_unbox = TRUE, digits = 17)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "raw.json"
        script_path = tmpdir_path / "probe.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(script_path, str(csv_path), str(json_path), formula),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    data = _make_gaussian_data(seed=41, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8)'
    gam = _fit_nampy_model(data, formula, "gaussian", "GCV")
    expected_false = _run_mgcv_gam_vcomp(
        data, formula, "gaussian", "GCV", rescale=False
    )
    expected_true = _run_mgcv_gam_vcomp(data, formula, "gaussian", "GCV", rescale=True)
    raw = _mgcv_raw(data, formula)

    sp = np.asarray(gam.smoothing_params, dtype=np.float64)
    scale = float(_fit_scale(gam))
    s_scale = _mgcv_penalty_rescale_factors(gam)
    actual_false = gam.gam_vcomp(rescale=False)
    actual_true = gam.gam_vcomp(rescale=True)

    print("nampy scale", repr(scale))
    print("nampy sp", repr(sp))
    print("nampy s.scale", repr(s_scale))
    print("nampy optim x", repr(getattr(gam._optim_result, "x", None)))
    print("nampy optim message", repr(getattr(gam._optim_result, "message", None)))
    for label, log_sp in (
        ("nampy", np.log(sp)),
        ("mgcv", np.log(np.asarray([raw["sp"]], dtype=np.float64))),
    ):
        print(
            label,
            "criterion",
            repr(float(criterion_value(gam, gam.y_, log_sp, method="gcv"))),
            "grad",
            repr(criterion_gradient(gam, gam.y_, log_sp, method="gcv")),
            "hess",
            repr(criterion_hessian(gam, gam.y_, log_sp, method="gcv")),
        )
    print("mgcv raw", repr(raw))
    print("nampy false", repr(actual_false))
    print("mgcv false", repr(expected_false))
    print("nampy true", repr(actual_true))
    print("mgcv true", repr(expected_true))


if __name__ == "__main__":
    main()
