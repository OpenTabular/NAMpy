from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nampy.gam._mgcv_constants import EIG_TOL_POWER
from nampy.gam.linalg import symmetric_eigh
from nampy.gam.smooths.tensor.marginals import tensor_marginal_fit_matrices
from nampy.gam.smooths.tensor.t2 import TensorANOVASplineTerm
from tests.families.test_general_family_mgcv_parity import (
    GENERAL_SE_CASES,
    _gaulss_tensor_data,
    _gammals_tensor_data,
    _gevlss_tensor_data,
    _shashlss_tensor_data,
    _ziplss_tensor_data,
)
from tests.mgcv_parity_utils import _build_r_command


@dataclass(frozen=True)
class _ProbeCase:
    case_id: str
    family: str
    full: bool
    data: object


def _general_t2_cases():
    return [case for case in GENERAL_SE_CASES if "_t2_" in case[0]]


def _synthetic_seed_sweep_cases(seed_count: int) -> list[_ProbeCase]:
    families = [
        ("gaulss", _gaulss_tensor_data),
        ("gammals", _gammals_tensor_data),
        ("gevlss", _gevlss_tensor_data),
        ("shashlss", _shashlss_tensor_data),
        ("ziplss", _ziplss_tensor_data),
    ]
    cases: list[_ProbeCase] = []
    for family, factory in families:
        for full in (False, True):
            for seed_idx in range(int(seed_count)):
                seed = 1000 + (200 * len(cases)) + seed_idx
                cases.append(
                    _ProbeCase(
                        case_id=(
                            f"{family}_t2_full_{str(full).lower()}_seed_{seed_idx:02d}"
                        ),
                        family=family,
                        full=bool(full),
                        data=factory(seed=seed),
                    )
                )
    return cases


def _max_abs_diff(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Shape mismatch: {aa.shape} != {bb.shape}")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _column_sign_alignment(actual, expected) -> dict[str, object]:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {e.shape}")
    if a.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {a.ndim}D.")

    signs = np.ones(a.shape[1], dtype=np.float64)
    for j in range(a.shape[1]):
        if np.linalg.norm(a[:, j] - e[:, j]) > np.linalg.norm(-a[:, j] - e[:, j]):
            signs[j] = -1.0
    aligned = a * signs[np.newaxis, :]
    return {
        "best_signs": signs.tolist(),
        "signed_max_abs_diff": _max_abs_diff(aligned, e),
    }


def _status_from_diffs(exact_diff, signed_diff, tol=1e-10) -> str:
    if float(exact_diff) <= float(tol):
        return "exact"
    if float(signed_diff) <= float(tol):
        return "sign_only"
    return "other"


def _apply_current_sign_rule(X, P, basis_name):
    basis_key = None if basis_name is None else str(basis_name).lower()
    if basis_key is None:
        return np.asarray(X, dtype=np.float64), np.asarray(P, dtype=np.float64)

    X_out = np.asarray(X, dtype=np.float64).copy()
    P_out = np.asarray(P, dtype=np.float64).copy()
    n_cols = int(X_out.shape[1])
    sign_idx = []

    if basis_key in {"cr", "cs"} and n_cols > 0:
        sign_idx.append(n_cols - 1)
    elif basis_key == "ps":
        if n_cols > 0:
            sign_idx.append(0)
        if n_cols > 1:
            sign_idx.append(1)
    elif basis_key == "cc" and n_cols > 0:
        sign_idx.append(n_cols - 1)

    for j in sorted({int(idx) for idx in sign_idx if 0 <= int(idx) < n_cols}):
        X_out[:, j] *= -1.0
        P_out[:, j] *= -1.0

    return X_out, P_out


def _python_natparam_type3_debug(raw_basis, raw_penalty, *, tol=None, basis_name=None):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)
    p = int(X.shape[1])

    if tol is None:
        tol = float(np.finfo(np.float64).eps ** EIG_TOL_POWER)

    evals, U = symmetric_eigh(S, descending=True, use_scipy=True)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    tol_eff = float(max_eval * tol)
    rank = int(np.sum(evals > tol_eff))
    null_exists = bool(rank < p)

    E = np.ones(p, dtype=np.float64)
    if rank > 0:
        E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))

    X_eig = X @ U
    col_norm = np.sum(X_eig**2, axis=0) / (E**2)
    av_norm = float(np.mean(col_norm[:rank])) if rank > 0 else 1.0

    if null_exists:
        for i in range(rank, p):
            if av_norm > 0.0 and col_norm[i] > 0.0:
                E[i] = np.sqrt(col_norm[i] / av_norm)

    P_pre = U / E[np.newaxis, :]
    X_pre = X_eig / E[np.newaxis, :]

    null_idx = list(range(rank, p))
    rind = list(range(p - 1, rank - 1, -1))
    Xn_centered = None
    null_gram = None
    null_evals = np.array([], dtype=np.float64)
    null_vecs = np.empty((0, 0), dtype=np.float64)

    X_rot = np.asarray(X_pre, dtype=np.float64).copy()
    P_rot = np.asarray(P_pre, dtype=np.float64).copy()
    if null_exists and rank < p - 1:
        Xn = np.asarray(X_pre[:, null_idx], dtype=np.float64)
        n = int(Xn.shape[0])
        one = np.ones(n, dtype=np.float64)
        Xn_centered = Xn - (one[:, None] * (one[None, :] @ Xn)) / n
        null_gram = np.asarray(Xn_centered.T @ Xn_centered, dtype=np.float64)
        null_evals, null_vecs = symmetric_eigh(
            null_gram,
            descending=True,
            use_scipy=True,
        )
        X_rot[:, rind] = X_pre[:, null_idx] @ null_vecs
        P_rot[:, rind] = P_pre[:, null_idx] @ null_vecs

    X_core = np.asarray(X_rot, dtype=np.float64).copy()
    P_core = np.asarray(P_rot, dtype=np.float64).copy()
    if rank > 0:
        pen_scale = 1.0 / np.sqrt(float(np.mean(X_core[:, :rank] ** 2)))
        X_core[:, :rank] *= pen_scale
        P_core[:rank, :] *= pen_scale
    if null_exists:
        null_scale = 1.0 / np.sqrt(float(np.mean(X_core[:, rank:] ** 2)))
        X_core[:, rank:] *= null_scale
        P_core[rank:, :] *= null_scale

    X_current, P_current = _apply_current_sign_rule(X_core, P_core, basis_name)

    return {
        "evals": np.asarray(evals, dtype=np.float64),
        "vectors": np.asarray(U, dtype=np.float64),
        "rank": int(rank),
        "tol_eff": float(tol_eff),
        "E": np.asarray(E, dtype=np.float64),
        "X_pre": np.asarray(X_pre, dtype=np.float64),
        "P_pre": np.asarray(P_pre, dtype=np.float64),
        "Xn_centered": None if Xn_centered is None else np.asarray(Xn_centered, dtype=np.float64),
        "null_gram": None if null_gram is None else np.asarray(null_gram, dtype=np.float64),
        "null_evals": np.asarray(null_evals, dtype=np.float64),
        "null_vectors": np.asarray(null_vecs, dtype=np.float64),
        "X_post_null": np.asarray(X_rot, dtype=np.float64),
        "P_post_null": np.asarray(P_rot, dtype=np.float64),
        "X_core": np.asarray(X_core, dtype=np.float64),
        "P_core": np.asarray(P_core, dtype=np.float64),
        "X_current": np.asarray(X_current, dtype=np.float64),
        "P_current": np.asarray(P_current, dtype=np.float64),
    }


def _run_r_natparam_type3_debug(raw_basis, raw_penalty, *, tol=None):
    X = np.asarray(raw_basis, dtype=np.float64)
    S = np.asarray(raw_penalty, dtype=np.float64)
    tol_value = float(np.finfo(np.float64).eps ** EIG_TOL_POWER) if tol is None else float(tol)

    r_code = """
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[[1]], header = FALSE))
S <- as.matrix(read.csv(args[[2]], header = FALSE))
tol <- as.numeric(args[[3]])
out <- args[[4]]

pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list(
    "__kind__" = "matrix",
    dim = as.integer(dim(x)),
    data = unname(as.numeric(t(x)))
  )
}

er <- eigen(S, symmetric = TRUE)
rank <- sum(er$values > max(er$values) * tol)
null.exists <- rank < ncol(X)
E <- rep(1, ncol(X))
if (rank > 0) E[1:rank] <- sqrt(er$values[1:rank])

X_eig <- X %*% er$vectors
col.norm <- colSums(X_eig^2)
col.norm <- col.norm / E^2
av.norm <- if (rank > 0) mean(col.norm[1:rank]) else 1
if (null.exists) {
  for (i in (rank + 1):ncol(X)) {
    E[i] <- sqrt(col.norm[i] / av.norm)
  }
}

P_pre <- t(t(er$vectors) / E)
X_pre <- t(t(X_eig) / E)

Xn_centered <- NULL
null_gram <- NULL
null_evals <- numeric(0)
null_vectors <- NULL
X_post_null <- X_pre
P_post_null <- P_pre
if (null.exists && rank < ncol(X) - 1) {
  ind <- (rank + 1):ncol(X)
  rind <- ncol(X):(rank + 1)
  Xn <- X_pre[, ind, drop = FALSE]
  n <- nrow(Xn)
  one <- rep(1, n)
  Xn_centered <- Xn - one %*% (t(one) %*% Xn) / n
  null_gram <- t(Xn_centered) %*% Xn_centered
  um <- eigen(null_gram, symmetric = TRUE)
  null_evals <- um$values
  null_vectors <- um$vectors
  X_post_null[, rind] <- X_pre[, ind, drop = FALSE] %*% um$vectors
  P_post_null[, rind] <- P_pre[, ind, drop = FALSE] %*% um$vectors
}

X_core <- X_post_null
P_core <- P_post_null
if (rank > 0) {
  ind <- 1:rank
  scale <- 1 / sqrt(mean(X_core[, ind, drop = FALSE]^2))
  X_core[, ind] <- X_core[, ind] * scale
  P_core[ind, ] <- P_core[ind, ] * scale
}
if (null.exists) {
  ind <- (rank + 1):ncol(X)
  scalef <- 1 / sqrt(mean(X_core[, ind, drop = FALSE]^2))
  X_core[, ind] <- X_core[, ind] * scalef
  P_core[ind, ] <- P_core[ind, ] * scalef
}

write_json(
  list(
    evals = unname(as.numeric(er$values)),
    vectors = pack_matrix(er$vectors),
    rank = as.integer(rank),
    tol_eff = max(er$values) * tol,
    E = unname(as.numeric(E)),
    X_pre = pack_matrix(X_pre),
    P_pre = pack_matrix(P_pre),
    Xn_centered = pack_matrix(Xn_centered),
    null_gram = pack_matrix(null_gram),
    null_evals = unname(as.numeric(null_evals)),
    null_vectors = pack_matrix(null_vectors),
    X_post_null = pack_matrix(X_post_null),
    P_post_null = pack_matrix(P_post_null),
    X_core = pack_matrix(X_core),
    P_core = pack_matrix(P_core)
  ),
  out,
  auto_unbox = TRUE,
  digits = 17,
  null = "null"
)
"""

    def _decode_packed(obj):
        if isinstance(obj, dict) and obj.get("__kind__") == "matrix":
            dim = tuple(int(v) for v in obj["dim"])
            data = np.asarray(obj["data"], dtype=np.float64)
            return data.reshape(dim, order="C")
        if obj is None:
            return None
        return obj

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        x_path = tmp / "X.csv"
        s_path = tmp / "S.csv"
        out_path = tmp / "out.json"
        script_path = tmp / "natparam_type3_debug.R"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(s_path, S, delimiter=",")
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(x_path),
                str(s_path),
                str(tol_value),
                str(out_path),
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(out_path.read_text(encoding="utf-8"))

    return {
        "evals": np.asarray(result["evals"], dtype=np.float64),
        "vectors": np.asarray(_decode_packed(result["vectors"]), dtype=np.float64),
        "rank": int(result["rank"]),
        "tol_eff": float(result["tol_eff"]),
        "E": np.asarray(result["E"], dtype=np.float64),
        "X_pre": np.asarray(_decode_packed(result["X_pre"]), dtype=np.float64),
        "P_pre": np.asarray(_decode_packed(result["P_pre"]), dtype=np.float64),
        "Xn_centered": (
            None
            if _decode_packed(result["Xn_centered"]) is None
            else np.asarray(_decode_packed(result["Xn_centered"]), dtype=np.float64)
        ),
        "null_gram": (
            None
            if _decode_packed(result["null_gram"]) is None
            else np.asarray(_decode_packed(result["null_gram"]), dtype=np.float64)
        ),
        "null_evals": np.asarray(result["null_evals"], dtype=np.float64),
        "null_vectors": (
            np.empty((0, 0), dtype=np.float64)
            if _decode_packed(result["null_vectors"]) is None
            else np.asarray(_decode_packed(result["null_vectors"]), dtype=np.float64)
        ),
        "X_post_null": np.asarray(_decode_packed(result["X_post_null"]), dtype=np.float64),
        "P_post_null": np.asarray(_decode_packed(result["P_post_null"]), dtype=np.float64),
        "X_core": np.asarray(_decode_packed(result["X_core"]), dtype=np.float64),
        "P_core": np.asarray(_decode_packed(result["P_core"]), dtype=np.float64),
    }


def _fit_runtime_term(case):
    if isinstance(case, _ProbeCase):
        case_id = case.case_id
        data = case.data
        full = bool(case.full)
    else:
        case_id, _family, formula, data_factory, _method, *_rest = case
        data = data_factory()
        full = "full=True" in " ".join(str(f) for f in formula)
    term = TensorANOVASplineTerm(
        feature=["x0", "x1"],
        k=[6, 6],
        basis=["tp", "cr"],
        full=full,
    )
    X = data[["x0", "x1"]].to_numpy(dtype=np.float64)
    term.fit(X, ["x0", "x1"])
    return case_id, data, term


def _stage_metrics(py_steps, r_steps, *, key, sign_compare=False):
    py_val = py_steps[key]
    r_val = r_steps[key]
    if py_val is None or r_val is None:
        return {"exact_max_abs_diff": None, "signed_max_abs_diff": None}
    exact = _max_abs_diff(py_val, r_val)
    if sign_compare and np.asarray(py_val).ndim == 2:
        signed = _column_sign_alignment(py_val, r_val)["signed_max_abs_diff"]
    else:
        signed = exact
    return {
        "exact_max_abs_diff": float(exact),
        "signed_max_abs_diff": float(signed),
    }


def _record_for_marginal(case_id, basis_name, feature_name, raw_x, raw_s):
    py_steps = _python_natparam_type3_debug(
        raw_x,
        raw_s,
        basis_name=basis_name,
    )
    r_steps = _run_r_natparam_type3_debug(raw_x, raw_s)

    stage_report = {
        "evals": _stage_metrics(py_steps, r_steps, key="evals", sign_compare=False),
        "vectors": _stage_metrics(py_steps, r_steps, key="vectors", sign_compare=True),
        "E": _stage_metrics(py_steps, r_steps, key="E", sign_compare=False),
        "X_pre": _stage_metrics(py_steps, r_steps, key="X_pre", sign_compare=True),
        "P_pre": _stage_metrics(py_steps, r_steps, key="P_pre", sign_compare=True),
        "null_gram": _stage_metrics(py_steps, r_steps, key="null_gram", sign_compare=False),
        "null_evals": _stage_metrics(py_steps, r_steps, key="null_evals", sign_compare=False),
        "null_vectors": _stage_metrics(
            py_steps, r_steps, key="null_vectors", sign_compare=True
        ),
        "X_post_null": _stage_metrics(
            py_steps, r_steps, key="X_post_null", sign_compare=True
        ),
        "P_post_null": _stage_metrics(
            py_steps, r_steps, key="P_post_null", sign_compare=True
        ),
        "X_core": _stage_metrics(py_steps, r_steps, key="X_core", sign_compare=True),
        "P_core": _stage_metrics(py_steps, r_steps, key="P_core", sign_compare=True),
    }

    current_x_exact = _max_abs_diff(py_steps["X_current"], r_steps["X_core"])
    current_p_exact = _max_abs_diff(py_steps["P_current"], r_steps["P_core"])
    current_x_sign = _column_sign_alignment(
        py_steps["X_current"], r_steps["X_core"]
    )["signed_max_abs_diff"]
    current_p_sign = _column_sign_alignment(
        py_steps["P_current"], r_steps["P_core"]
    )["signed_max_abs_diff"]
    core_x_exact = stage_report["X_core"]["exact_max_abs_diff"]
    core_p_exact = stage_report["P_core"]["exact_max_abs_diff"]
    core_x_sign = stage_report["X_core"]["signed_max_abs_diff"]
    core_p_sign = stage_report["P_core"]["signed_max_abs_diff"]

    return {
        "case_id": case_id,
        "basis_name": str(basis_name),
        "feature": str(feature_name),
        "rank": int(r_steps["rank"]),
        "core_x_status": _status_from_diffs(core_x_exact, core_x_sign),
        "core_p_status": _status_from_diffs(core_p_exact, core_p_sign),
        "current_x_status": _status_from_diffs(current_x_exact, current_x_sign),
        "current_p_status": _status_from_diffs(current_p_exact, current_p_sign),
        "core_x_exact_max_abs_diff": float(core_x_exact),
        "core_p_exact_max_abs_diff": float(core_p_exact),
        "current_x_exact_max_abs_diff": float(current_x_exact),
        "current_p_exact_max_abs_diff": float(current_p_exact),
        "current_x_signed_max_abs_diff": float(current_x_sign),
        "current_p_signed_max_abs_diff": float(current_p_sign),
        "current_x_best_signs": _column_sign_alignment(
            py_steps["X_current"], r_steps["X_core"]
        )["best_signs"],
        "current_p_best_signs": _column_sign_alignment(
            py_steps["P_current"], r_steps["P_core"]
        )["best_signs"],
        "stage_report": stage_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Restrict to one or more GENERAL_SE_CASES ids. Default: all t2 cases.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only compact per-record summary plus aggregate counts.",
    )
    parser.add_argument(
        "--synthetic-seed-sweep",
        type=int,
        default=0,
        help=(
            "Generate seed-swept synthetic cases instead of GENERAL_SE_CASES. "
            "Example: 10 => 5 families * 2 full flags * 10 seeds = 100 cases."
        ),
    )
    args = parser.parse_args()

    if args.synthetic_seed_sweep > 0:
        selected_cases = _synthetic_seed_sweep_cases(args.synthetic_seed_sweep)
    else:
        case_table = {case[0]: case for case in _general_t2_cases()}
        selected_ids = (
            list(case_table)
            if not args.case_ids
            else [case_id for case_id in args.case_ids if case_id in case_table]
        )
        if not selected_ids:
            raise SystemExit("No matching t2 case ids selected.")
        selected_cases = [case_table[case_id] for case_id in selected_ids]

    records = []
    for case in selected_cases:
        _, data, term = _fit_runtime_term(case)
        case_id = case.case_id if isinstance(case, _ProbeCase) else case[0]
        for basis_name, marginal, feature_name in zip(
            term.basis,
            term._marginals,
            term.feature,
        ):
            raw_x, raw_s, _ = tensor_marginal_fit_matrices(marginal, centered=False)
            records.append(
                _record_for_marginal(
                    case_id,
                    basis_name,
                    feature_name,
                    raw_x,
                    raw_s,
                )
            )

    aggregate = {
        "n_case_ids": len(selected_cases),
        "n_records": len(records),
        "core_x_status_counts": {},
        "current_x_status_counts": {},
        "core_p_status_counts": {},
        "current_p_status_counts": {},
    }
    for key in (
        "core_x_status",
        "current_x_status",
        "core_p_status",
        "current_p_status",
    ):
        counts_key = f"{key}_counts"
        counts = {}
        for record in records:
            counts[record[key]] = counts.get(record[key], 0) + 1
        aggregate[counts_key] = counts

    for key in ("current_x_best_signs", "current_p_best_signs"):
        counts = {}
        for record in records:
            if record["basis_name"] != "cr":
                continue
            pattern = tuple(int(v) for v in record[key])
            counts[str(pattern)] = counts.get(str(pattern), 0) + 1
        aggregate[f"cr_{key}_pattern_counts"] = counts

    if args.summary_only:
        compact = [
            {
                "case_id": record["case_id"],
                "basis_name": record["basis_name"],
                "feature": record["feature"],
                "core_x_status": record["core_x_status"],
                "current_x_status": record["current_x_status"],
                "core_x_exact_max_abs_diff": record["core_x_exact_max_abs_diff"],
                "current_x_exact_max_abs_diff": record["current_x_exact_max_abs_diff"],
                "current_x_best_signs": record["current_x_best_signs"],
            }
            for record in records
        ]
        print(json.dumps({"aggregate": aggregate, "records": compact}, indent=2, sort_keys=True))
        return

    print(json.dumps({"aggregate": aggregate, "records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
