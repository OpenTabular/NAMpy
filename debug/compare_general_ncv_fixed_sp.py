from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam.smoothing_selection.criteria.dispatch import (  # noqa: E402
    criterion_gradient,
    criterion_value,
)
from nampy.gam.parity.trace import build_optimizer_trace  # noqa: E402
from nampy.gam.smoothing_selection.optimize.basics import (  # noqa: E402
    _initial_smoothing_params_mgcv_style,
)
from nampy.gam.smoothing_selection.optimize.basics import r_matrix_norm_max_abs  # noqa: E402
from tests.families.test_general_family_mgcv_parity import (  # noqa: E402
    _gevlss_data,
    _gammals_data,
    _shashlss_data,
    _ziplss_data,
)
from tests.optimization.test_mgcv_outer_optimization_parity import (  # noqa: E402
    _compile_optimization_state,
    _gaulss_two_smooth_data,
    _run_mgcv_initial_spg,
)
from nampy.gam.fit.solvers.general_family_solver import (  # noqa: E402
    build_general_family_setup_state,
    sl_initial_repara,
)
from nampy.gam.smoothing_selection.reparam import (  # noqa: E402
    build_estimate_gam_setup_state,
)
from tests.mgcv_parity_utils import (  # noqa: E402
    _REPO_ROOT,
    _build_r_command,
    _family_specs,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _normalize_python_formula_text,
    _run_mgcv_snapshot,
)
from tests.optimization.test_mgcv_ncv_qncv_parity import (  # noqa: E402
    _make_gaulss_data,
)

R_SCRIPT = shutil.which("Rscript")


CASES = {
    "gaulss_ncv": (
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        lambda: _make_gaulss_data(seed=11, n=90),
        "NCV",
    ),
    "gaulss_qncv": (
        "gaulss",
        ['y ~ s(x, bs="cr", k=6)', '~ s(x, bs="cr", k=5)'],
        lambda: _make_gaulss_data(seed=13, n=90),
        "QNCV",
    ),
    "gaulss_ml_two_smooth": (
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        lambda: _gaulss_two_smooth_data(seed=33, n=140),
        "ML",
    ),
    "gammals_ncv": (
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "NCV",
    ),
    "gammals_qncv": (
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "QNCV",
    ),
    "gevlss_ncv": (
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "NCV",
    ),
    "gevlss_qncv": (
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "QNCV",
    ),
    "shashlss_ncv": (
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "NCV",
    ),
    "shashlss_qncv": (
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "QNCV",
    ),
    "ziplss_ncv": (
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "NCV",
    ),
    "ziplss_qncv": (
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "QNCV",
    ),
}


def _run_mgcv_general_fixed_sp(data, formula, family, method, smoothing_params):
    if R_SCRIPT is None:
        raise RuntimeError("Rscript required")
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
out <- args[[6]]
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
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  gevlss = mgcv::gevlss(),
  shashlss = mgcv::shashlss(),
  ziplss = mgcv::ziplss(),
  stop(sprintf("Unsupported family for fixed-sp NCV debug: %s", family_name))
)
formula_obj <- coerce_formula(formula_text)
fit <- gam(
  formula = formula_obj,
  data = d,
  family = family_obj,
  method = method_name,
  sp = sp
)
write_json(
  list(
    criterion_value = unname(as.numeric(fit$gcv.ubre)),
    outer_grad = if (is.null(fit$outer.info$grad)) NULL else unname(as.numeric(fit$outer.info$grad)),
    outer_hess = if (is.null(fit$outer.info$hess)) NULL else unname(fit$outer.info$hess)
  ),
  out,
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "general_fixed_sp.json"
        script_path = tmpdir_path / "general_fixed_sp_ncv.R"
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
                str(json_path),
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _array_or_none(x):
    if x is None:
        return None
    if isinstance(x, dict) and len(x) == 0:
        return None
    return np.asarray(x, dtype=np.float64)


def _column_sign_alignment_stats(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or a.ndim != 2:
        return None
    signs = []
    max_diffs = []
    for j in range(a.shape[1]):
        diff_pos = float(np.max(np.abs(a[:, j] - e[:, j])))
        diff_neg = float(np.max(np.abs(-a[:, j] - e[:, j])))
        if diff_neg < diff_pos:
            signs.append(-1.0)
            max_diffs.append(diff_neg)
        else:
            signs.append(1.0)
            max_diffs.append(diff_pos)
    return {
        "signs": signs,
        "max_abs_diff_after_sign_align": max_diffs,
        "worst_after_sign_align": float(max(max_diffs) if max_diffs else 0.0),
    }


def _column_signature_stats(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        return None
    out = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        idx = int(np.argmax(np.abs(col))) if col.size else 0
        out.append(
            {
                "sum": float(np.sum(col)),
                "max_abs_value": float(np.max(np.abs(col))) if col.size else 0.0,
                "max_abs_index": idx,
                "max_abs_sign": float(np.sign(col[idx])) if col.size else 0.0,
                "first_value": float(col[0]) if col.size else 0.0,
            }
        )
    return out


def _alt_singleton_repara_x(model, eig_mode: str):
    n_sp = int(np.asarray(model.smoothing_params, dtype=np.float64).size)
    setup = build_general_family_setup_state(
        model,
        np.ones(n_sp, dtype=np.float64),
        score_type="REML",
    )
    blocks = []
    for block in setup.Sl:
        block_copy = type(block)(**vars(block))
        if len(block_copy.S) == 1:
            S0 = np.asarray(block_copy.S[0], dtype=np.float64)
            ut = np.triu_indices_from(S0, k=1)
            if float(np.sum(np.abs(S0[ut]))) != 0.0:
                if eig_mode == "numpy":
                    vals, vecs = np.linalg.eigh(S0)
                elif eig_mode == "scipy_evr":
                    vals, vecs = scipy_eigh(
                        S0,
                        check_finite=False,
                        driver="evr",
                    )
                elif eig_mode == "scipy_ev":
                    vals, vecs = scipy_eigh(
                        S0,
                        check_finite=False,
                        driver="ev",
                    )
                else:
                    raise ValueError(f"Unsupported eig_mode: {eig_mode}")
                order = np.argsort(vals)[::-1]
                vals = np.asarray(vals[order], dtype=np.float64)
                vecs = np.asarray(vecs[:, order], dtype=np.float64)
                ind = np.asarray(block_copy.ind, dtype=bool)
                dvals = vals.copy()
                dvals[ind] = 1.0 / np.sqrt(dvals[ind])
                dvals[~ind] = 1.0
                block_copy.D = np.asarray(
                    vecs * dvals[np.newaxis, :],
                    dtype=np.float64,
                )
                block_copy.Di = np.asarray(
                    vecs.T / dvals[:, np.newaxis],
                    dtype=np.float64,
                )
        blocks.append(block_copy)

    alt_sl = type(setup.Sl)(
        blocks=blocks,
        E=np.asarray(setup.Sl.E, dtype=np.float64),
        S=np.asarray(setup.Sl.S, dtype=np.float64),
        lambda_=np.asarray(setup.Sl.lambda_, dtype=np.float64),
        cholesky=bool(setup.Sl.cholesky),
    )
    return np.asarray(
        sl_initial_repara(alt_sl, np.asarray(setup.X_full, dtype=np.float64), both_sides=False),
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", choices=sorted(CASES))
    args = parser.parse_args()

    family, formula, data_factory, method = CASES[args.case_id]
    data = data_factory()
    gam_init = _compile_optimization_state(data, formula, family, method)
    init_actual = _initial_smoothing_params_mgcv_style(
        gam_init, np.asarray(gam_init.y_, dtype=np.float64)
    )
    init_expected = _run_mgcv_initial_spg(data, formula, family, method)
    n_sp = int(np.asarray(gam_init.smoothing_params, dtype=np.float64).size)
    fit5_setup = build_general_family_setup_state(
        gam_init,
        np.ones(n_sp, dtype=np.float64),
        score_type="REML",
    )
    exact_setup = build_estimate_gam_setup_state(gam_init)
    weights_init = (
        np.ones_like(np.asarray(gam_init.y_, dtype=np.float64), dtype=np.float64)
        if gam_init.prior_weights_ is None
        else np.asarray(gam_init.prior_weights_, dtype=np.float64)
    )
    start_actual = np.asarray(
        gam_init.family.initialize(
            np.asarray(gam_init.y_, dtype=np.float64),
            np.asarray(fit5_setup.X_initial, dtype=np.float64),
            fit5_setup.jj,
            offset=fit5_setup.offset_list,
            weights=weights_init,
            E=np.asarray(exact_setup.Eb, dtype=np.float64),
        ),
        dtype=np.float64,
    )
    lbb_actual = np.asarray(
        gam_init.family.ll(
            np.asarray(gam_init.y_, dtype=np.float64),
            np.asarray(fit5_setup.X_initial, dtype=np.float64),
            fit5_setup.jj,
            start_actual,
            weights_init,
            offset=fit5_setup.offset_list,
            deriv=1,
        )["lbb"],
        dtype=np.float64,
    )
    x_sign_stats = _column_sign_alignment_stats(
        np.asarray(fit5_setup.X_initial, dtype=np.float64),
        np.asarray(init_expected["X_initial"], dtype=np.float64),
    )
    x_alt_numpy = _alt_singleton_repara_x(gam_init, "numpy")
    x_alt_scipy_evr = _alt_singleton_repara_x(gam_init, "scipy_evr")
    x_alt_scipy_ev = _alt_singleton_repara_x(gam_init, "scipy_ev")
    x_sig_actual = _column_signature_stats(np.asarray(fit5_setup.X_initial, dtype=np.float64))
    x_sig_expected = _column_signature_stats(np.asarray(init_expected["X_initial"], dtype=np.float64))
    gam_outer = _fit_nampy_model(data, formula, family, method)
    trace = build_optimizer_trace(gam_outer)
    actual = _fit_nampy_snapshot(data, formula, family, method)
    expected = _run_mgcv_snapshot(data, formula, family, method)

    actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
    expected_sp = np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)

    gam_expected = _fit_nampy_model_fixed_sp(data, formula, family, expected_sp)
    y_expected = gam_expected.family.validate_y(gam_expected.y_)
    log_sp_expected = np.log(expected_sp)
    nampy_value_expected = float(
        criterion_value(gam_expected, y_expected, log_sp_expected, method=method.lower())
    )
    nampy_grad_expected = np.asarray(
        criterion_gradient(gam_expected, y_expected, log_sp_expected, method=method.lower())
    )
    mgcv_fixed_expected = _run_mgcv_general_fixed_sp(
        data, formula, family, method, expected_sp
    )

    gam_actual = _fit_nampy_model_fixed_sp(data, formula, family, actual_sp)
    y_actual = gam_actual.family.validate_y(gam_actual.y_)
    log_sp_actual = np.log(actual_sp)
    nampy_value_actual = float(
        criterion_value(gam_actual, y_actual, log_sp_actual, method=method.lower())
    )
    nampy_grad_actual = np.asarray(
        criterion_gradient(gam_actual, y_actual, log_sp_actual, method=method.lower())
    )
    mgcv_fixed_actual = _run_mgcv_general_fixed_sp(data, formula, family, method, actual_sp)

    report = {
        "case_id": args.case_id,
        "family": family,
        "method": method,
        "initial_sp_actual": (
            None if init_actual is None else np.asarray(init_actual, dtype=np.float64).tolist()
        ),
        "initial_sp_expected": init_expected.get("initial_sp"),
        "initial_sp_X_max_abs_diff": float(
            np.max(
                np.abs(
                    np.asarray(fit5_setup.X_initial, dtype=np.float64)
                    - np.asarray(init_expected["X_initial"], dtype=np.float64)
                )
            )
        ),
        "initial_sp_X_sign_alignment": x_sign_stats,
        "alt_singleton_x_sign_alignment": {
            "numpy": _column_sign_alignment_stats(
                x_alt_numpy,
                np.asarray(init_expected["X_initial"], dtype=np.float64),
            ),
            "scipy_evr": _column_sign_alignment_stats(
                x_alt_scipy_evr,
                np.asarray(init_expected["X_initial"], dtype=np.float64),
            ),
            "scipy_ev": _column_sign_alignment_stats(
                x_alt_scipy_ev,
                np.asarray(init_expected["X_initial"], dtype=np.float64),
            ),
        },
        "initial_sp_X_signatures_actual": x_sig_actual,
        "initial_sp_X_signatures_expected": x_sig_expected,
        "initial_sp_Eb_norm_diff": float(
            abs(
                r_matrix_norm_max_abs(np.asarray(exact_setup.Eb, dtype=np.float64))
                - r_matrix_norm_max_abs(np.asarray(init_expected["Eb"], dtype=np.float64))
            )
        ),
        "initial_start_max_abs_diff": float(
            np.max(
                np.abs(
                    np.asarray(start_actual, dtype=np.float64)
                    - np.asarray(init_expected["start"], dtype=np.float64)
                )
            )
        ),
        "initial_lbb_max_abs_diff": float(
            np.max(
                np.abs(
                    np.asarray(lbb_actual, dtype=np.float64)
                    - np.asarray(init_expected["lbb"], dtype=np.float64)
                )
            )
        ),
        "actual_snapshot_sp": actual_sp.tolist(),
        "expected_snapshot_sp": expected_sp.tolist(),
        "sp_max_abs_diff": float(np.max(np.abs(actual_sp - expected_sp))),
        "actual_snapshot_criterion": float(actual["fit"]["criterion_value"]),
        "expected_snapshot_criterion": float(expected["fit"]["criterion_value"]),
        "nampy_outer_trace": trace,
        "nampy_at_expected_sp": {
            "criterion_value": nampy_value_expected,
            "gradient": nampy_grad_expected.tolist(),
        },
        "mgcv_at_expected_sp": mgcv_fixed_expected,
        "nampy_at_actual_sp": {
            "criterion_value": nampy_value_actual,
            "gradient": nampy_grad_actual.tolist(),
        },
        "mgcv_at_actual_sp": mgcv_fixed_actual,
        "criterion_diff_at_expected_sp": float(
            nampy_value_expected - float(mgcv_fixed_expected["criterion_value"])
        ),
        "criterion_diff_at_actual_sp": float(
            nampy_value_actual - float(mgcv_fixed_actual["criterion_value"])
        ),
        "mgcv_outer_grad_at_expected_sp": (
            None
            if _array_or_none(mgcv_fixed_expected.get("outer_grad")) is None
            else _array_or_none(mgcv_fixed_expected["outer_grad"]).tolist()
        ),
        "mgcv_outer_grad_at_actual_sp": (
            None
            if _array_or_none(mgcv_fixed_actual.get("outer_grad")) is None
            else _array_or_none(mgcv_fixed_actual["outer_grad"]).tolist()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
