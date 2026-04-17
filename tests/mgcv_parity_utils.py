"""Shared mgcv/R snapshot helpers for parity tests (decoupled from ``test_mgcv_snapshot_parity``)."""

from __future__ import annotations

import atexit
import hashlib
import io
import itertools
import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam import GAM
from tests._paths import PARITY_DIR, REPO_ROOT, TESTS_DIR

_REPO_ROOT = REPO_ROOT
_TESTS_DIR = TESTS_DIR

R_SCRIPT = shutil.which("Rscript")
MGCV_SNAPSHOT_SCRIPT = PARITY_DIR / "mgcv_snapshot.R"
MGCV_SNAPSHOT_SERVER_SCRIPT = PARITY_DIR / "mgcv_snapshot_server.R"
MGCV_ANOVA_SCRIPT = PARITY_DIR / "mgcv_anova.R"

# ---------------------------------------------------------------------------
# mgcv result cache
# ---------------------------------------------------------------------------
# R mgcv results are deterministic given the same inputs — cache them so
# tests only call R once per unique input combination.  Only mgcv outputs are
# cached; nampy results are never cached.
_MGCV_CACHE_DIR = _TESTS_DIR / "mgcv_r_cache"
_MGCV_SNAPSHOT_SERVER_PROCESS = None
_MGCV_SNAPSHOT_SERVER_LOCK = threading.Lock()
_MGCV_SNAPSHOT_SERVER_REQUEST_IDS = itertools.count(1)
_RAW_CONSTRUCTOR_CACHE_VERSION = 4


def _start_mgcv_snapshot_server() -> subprocess.Popen:
    """Start the persistent mgcv snapshot worker process."""
    cmd = [
        R_SCRIPT,
        str(MGCV_SNAPSHOT_SERVER_SCRIPT),
        str(MGCV_SNAPSHOT_SCRIPT),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=_REPO_ROOT,
        text=True,
        bufsize=1,
    )
    return proc


def _stop_mgcv_snapshot_server() -> None:
    """Stop the long-lived snapshot worker if it is running."""
    global _MGCV_SNAPSHOT_SERVER_PROCESS
    proc = _MGCV_SNAPSHOT_SERVER_PROCESS
    if proc is None:
        return

    try:
        if proc.stdin is not None:
            proc.stdin.write(
                json.dumps({"action": "shutdown", "id": "shutdown"}) + "\n"
            )
            proc.stdin.flush()
    except Exception:
        pass

    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _MGCV_SNAPSHOT_SERVER_PROCESS = None


atexit.register(_stop_mgcv_snapshot_server)


def _run_mgcv_snapshot_single(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    weights_column: str | None = None,
):
    """Run a single snapshot via subprocess call (legacy behavior)."""
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
            str(formula),
            family,
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

        result = json.loads(json_path.read_text(encoding="utf-8"))

    return result


def _run_mgcv_snapshot_batched(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    weights_column: str | None = None,
):
    """Run a snapshot request through the long-lived R worker."""
    global _MGCV_SNAPSHOT_SERVER_PROCESS
    family_token = _family_specs(family)[1]

    if MGCV_SNAPSHOT_SERVER_SCRIPT is None or not MGCV_SNAPSHOT_SERVER_SCRIPT.exists():
        return _run_mgcv_snapshot_single(
            data,
            formula,
            family_token,
            method,
            select=select,
            weights_column=weights_column,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        data.to_csv(csv_path, index=False)

        request = {
            "action": "snapshot",
            "id": str(next(_MGCV_SNAPSHOT_SERVER_REQUEST_IDS)),
            "csv_path": str(csv_path),
            "formula": str(formula),
            "family": family_token,
            "method": str(method),
            "select": bool(select),
            "weights_column": "" if weights_column is None else str(weights_column),
        }

        with _MGCV_SNAPSHOT_SERVER_LOCK:
            if _MGCV_SNAPSHOT_SERVER_PROCESS is None:
                _MGCV_SNAPSHOT_SERVER_PROCESS = _start_mgcv_snapshot_server()

            proc = _MGCV_SNAPSHOT_SERVER_PROCESS
            if proc is None or proc.poll() is not None:
                _MGCV_SNAPSHOT_SERVER_PROCESS = _start_mgcv_snapshot_server()
                proc = _MGCV_SNAPSHOT_SERVER_PROCESS

            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
            except Exception:
                _stop_mgcv_snapshot_server()
                return _run_mgcv_snapshot_single(
                    data,
                    formula,
                    family_token,
                    method,
                    select=select,
                    weights_column=weights_column,
                )

            try:
                response_line = ""
                for _ in range(200):
                    candidate = proc.stdout.readline()
                    if not candidate:
                        response_line = ""
                        break
                    if candidate.strip():
                        response_line = candidate
                        break
            except Exception:
                _stop_mgcv_snapshot_server()
                return _run_mgcv_snapshot_single(
                    data,
                    formula,
                    family_token,
                    method,
                    select=select,
                    weights_column=weights_column,
                )

            if not response_line:
                _stop_mgcv_snapshot_server()
                return _run_mgcv_snapshot_single(
                    data,
                    formula,
                    family_token,
                    method,
                    select=select,
                    weights_column=weights_column,
                )

            try:
                response = json.loads(response_line)
            except Exception:
                _stop_mgcv_snapshot_server()
                return _run_mgcv_snapshot_single(
                    data,
                    formula,
                    family_token,
                    method,
                    select=select,
                    weights_column=weights_column,
                )
            if response.get("id") != request["id"]:
                raise RuntimeError(
                    "mgcv snapshot server returned mismatched response id."
                )

            if response.get("status") != "ok":
                raise RuntimeError(
                    response.get("message", "mgcv snapshot server error.")
                )

            return response["result"]


def _mgcv_cache_key(fn_name: str, key_parts: dict) -> str:
    """Return a stable hex digest that identifies a unique mgcv call."""
    buf = io.StringIO()
    buf.write(fn_name)
    buf.write("|")
    # Sort keys for stability; values are already strings/primitives.
    buf.write(json.dumps(key_parts, sort_keys=True, default=str))
    digest = hashlib.sha256(buf.getvalue().encode()).hexdigest()
    return digest


def _mgcv_cache_load(key: str):
    """Return cached JSON result or None if not cached."""
    path = _MGCV_CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _mgcv_cache_save(key: str, result) -> None:
    """Persist mgcv JSON result to cache."""
    _MGCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _MGCV_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(result), encoding="utf-8")


def _df_cache_repr(df: pd.DataFrame) -> str:
    """Deterministic string representation of a DataFrame for cache keying."""
    return df.to_csv(index=False)


def _make_gaussian_data(seed=123, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_gaussian_data_3col(seed=51, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    y = (
        np.sin(1.2 * x0)
        + 0.4 * x1**2
        - 0.3 * np.cos(1.5 * x2)
        + rng.normal(scale=0.15, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "x2": x2})


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


def _make_fs_data_4levels(seed=77, n=24):
    """FS smooth data with 4 factor levels — exercises the ≥3 level code path."""
    rng = np.random.default_rng(seed)
    levels = ["a", "b", "c", "d"]
    f = np.array([levels[i % len(levels)] for i in range(n)], dtype=object)
    x = np.linspace(0.1, 1.9, n)
    offsets = {"a": 1.2, "b": -0.7, "c": 0.5, "d": -0.3}
    y = (
        0.4 * np.sin(2 * x)
        + np.array([offsets[v] for v in f])
        + rng.normal(scale=0.05, size=n)
    )
    return pd.DataFrame({"y": y, "f": f, "x": x})


def _make_sz_data_3x3(seed=83):
    """SZ smooth data with f1 (3 levels) × f2 (3 levels) — full 3×3 factor grid."""
    rng = np.random.default_rng(seed)
    f1_levels = ["a", "b", "c"]
    f2_levels = ["u", "v", "w"]
    # All 9 combinations appear; each repeated twice for a 18-row frame.
    f1_base = [a for a in f1_levels for b in f2_levels] * 2
    f2_base = [b for a in f1_levels for b in f2_levels] * 2
    f1 = np.array(f1_base, dtype=object)
    f2 = np.array(f2_base, dtype=object)
    n = len(f1)
    x = rng.uniform(0.0, 2.0, size=n)
    base = np.sin(x) + 0.2 * x
    offsets = {
        ("a", "u"): 0.3,
        ("a", "v"): -0.1,
        ("a", "w"): 0.5,
        ("b", "u"): -0.6,
        ("b", "v"): 0.2,
        ("b", "w"): -0.4,
        ("c", "u"): 0.1,
        ("c", "v"): 0.4,
        ("c", "w"): -0.2,
    }
    y = np.array([base[i] + offsets[(f1[i], f2[i])] for i in range(n)])
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2, "x": x})


def _make_distributional_gaussian_data(seed=7, n=60):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(x0) + 0.3 * x1**2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_mrf_data():
    regions = np.array(["A", "B", "C", "A", "B", "C", "A", "B"], dtype=object)
    vals = {"A": 1.0, "B": -0.5, "C": 0.3}
    y = np.array([vals[r] for r in regions], dtype=np.float64)
    return pd.DataFrame({"y": y, "region": regions})


def _make_mrf_low_rank_data():
    regions = np.array(
        ["A"] * 8 + ["B"] * 7 + ["C"] * 9 + ["D"] * 6,
        dtype=object,
    )
    vals = {"A": -1.0, "B": 0.5, "C": 1.25, "D": -0.25}
    rng = np.random.default_rng(123)
    y = np.array([vals[r] for r in regions], dtype=np.float64)
    y = y + rng.normal(scale=0.05, size=regions.size)
    return pd.DataFrame({"y": y, "region": regions})


@dataclass(frozen=True)
class ParityCaseSpec:
    case_id: str
    data_factory: str
    formula: str
    family: str | dict
    method: str


PARITY_CASES: dict[str, ParityCaseSpec] = {
    "gaussian_cr_uni_reml": ParityCaseSpec(
        case_id="gaussian_cr_uni_reml",
        data_factory="gaussian",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="gaussian",
        method="REML",
    ),
    "gaussian_cr_uni_fixed": ParityCaseSpec(
        case_id="gaussian_cr_uni_fixed",
        data_factory="gaussian",
        formula='y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)',
        family="gaussian",
        method="fixed",
    ),
    "poisson_cr_uni_reml": ParityCaseSpec(
        case_id="poisson_cr_uni_reml",
        data_factory="poisson",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="poisson",
        method="REML",
    ),
    "gamma_cr_uni_reml": ParityCaseSpec(
        case_id="gamma_cr_uni_reml",
        data_factory="gamma",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="gamma",
        method="REML",
    ),
    "binomial_cr_uni_reml": ParityCaseSpec(
        case_id="binomial_cr_uni_reml",
        data_factory="binomial",
        formula='y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        family="binomial",
        method="REML",
    ),
    "poisson_te_reml": ParityCaseSpec(
        case_id="poisson_te_reml",
        data_factory="poisson",
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])',
        family="poisson",
        method="REML",
    ),
    "negbin_t2_reml": ParityCaseSpec(
        case_id="negbin_t2_reml",
        data_factory="negbin_theta_1",
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])',
        family={"name": "negbin", "theta": 1.0},
        method="REML",
    ),
    "gaussian_re_reml": ParityCaseSpec(
        case_id="gaussian_re_reml",
        data_factory="random_effect_noisy",
        formula='y ~ s(f, bs="re")',
        family="gaussian",
        method="REML",
    ),
}


def parity_case_ids() -> list[str]:
    return sorted(PARITY_CASES.keys())


def get_parity_case(case_id: str) -> ParityCaseSpec:
    try:
        return PARITY_CASES[case_id]
    except KeyError as exc:
        known = ", ".join(parity_case_ids())
        raise KeyError(f"Unknown parity case '{case_id}'. Known: {known}") from exc


def make_parity_case_data(case_id: str) -> pd.DataFrame:
    spec = get_parity_case(case_id)
    if spec.data_factory == "gaussian":
        return _make_gaussian_data()
    if spec.data_factory == "binomial":
        return _make_binomial_data()
    if spec.data_factory == "poisson":
        return _make_poisson_data()
    if spec.data_factory == "gamma":
        return _make_gamma_data()
    if spec.data_factory == "negbin_theta_1":
        return _make_negbin_data(theta=1.0)
    if spec.data_factory == "random_effect_noisy":
        return _make_random_effect_data_noisy()
    if spec.data_factory == "mrf":
        return _make_mrf_data()
    if spec.data_factory == "fs":
        return _make_fs_data()
    raise ValueError(
        f"Unsupported data factory '{spec.data_factory}' for case '{case_id}'."
    )


def _family_specs(family):
    if isinstance(family, dict):
        key = str(family.get("name", "")).lower()
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            theta = float(family.get("theta", 1.0))
            if bool(family.get("estimate_theta", False)):
                return family, f"negbin_est:{theta:.12g}"
            return family, f"negbin:{theta:.12g}"
        link = str(family.get("link", "")).lower()
        if link and link not in {"logit", "log"}:
            # Non-default link: encode as "family:link" token for the R script.
            return family, f"{key}:{link}"
        return family, key
    key = str(family).lower()
    if key == "shashlss":
        return family, "shash"
    return family, key


def _normalize_python_formula_text(formula) -> str:
    """Translate Python-style list/bool/null formula syntax into R syntax."""
    out = str(formula)
    out = out.replace("[", "c(")
    out = out.replace("]", ")")
    out = out.replace("True", "TRUE")
    out = out.replace("False", "FALSE")
    out = out.replace("None", "NULL")
    return out


def _run_mgcv_snapshot(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    weights_column: str | None = None,
):
    _family_nampy, family_token = _family_specs(family)

    _cache_key = _mgcv_cache_key(
        "snapshot",
        {
            "data": _df_cache_repr(data),
            "formula": str(formula),
            "family_token": family_token,
            "method": method,
            "select": select,
            "weights_column": weights_column,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

    try:
        result = _run_mgcv_snapshot_batched(
            data,
            formula,
            family_token,
            method,
            select=select,
            weights_column=weights_column,
        )
    except Exception:
        result = _run_mgcv_snapshot_single(
            data,
            formula,
            family_token,
            method,
            select=select,
            weights_column=weights_column,
        )

    _mgcv_cache_save(_cache_key, result)
    return result


def _run_mgcv_anova(
    data: pd.DataFrame,
    formulas: list[str],
    family,
    method: str,
    *,
    select: bool = False,
    test: str | None = None,
):
    _family_nampy, family_token = _family_specs(family)

    _cache_key = _mgcv_cache_key(
        "anova",
        {
            "data": _df_cache_repr(data),
            "formulas": list(formulas),
            "family_token": family_token,
            "method": method,
            "select": select,
            "test": test,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "anova.json"
        data.to_csv(csv_path, index=False)
        cmd = [
            R_SCRIPT,
            str(MGCV_ANOVA_SCRIPT),
            str(csv_path),
            str(json_path),
            json.dumps(list(formulas)),
            family_token,
            method,
            "true" if select else "false",
            "NULL" if test is None else str(test),
        ]
        subprocess.run(
            cmd,
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _fit_nampy_model(
    data: pd.DataFrame,
    formula,
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
    formula,
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
    ).parity_snapshot(X=data, include_covariances=True)


def _run_mgcv_smoothcon_matrix(data: pd.DataFrame, smooth_expr: str):
    _cache_key = _mgcv_cache_key(
        "smoothcon_matrix",
        {"data": _df_cache_repr(data), "smooth_expr": smooth_expr},
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

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
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _run_mgcv_smoothcon_penalties(
    data: pd.DataFrame,
    smooth_expr: str,
    *,
    absorb_cons: bool,
    scale_penalty: bool,
):
    _cache_key = _mgcv_cache_key(
        "smoothcon_penalties",
        {
            "data": _df_cache_repr(data),
            "smooth_expr": smooth_expr,
            "absorb_cons": absorb_cons,
            "scale_penalty": scale_penalty,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

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
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _run_mgcv_smoothcon_matrix_unscaled(data: pd.DataFrame, smooth_expr: str):
    _cache_key = _mgcv_cache_key(
        "smoothcon_matrix_unscaled",
        {"data": _df_cache_repr(data), "smooth_expr": smooth_expr},
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

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
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _decode_packed_matrix_payload(value):
    if isinstance(value, dict):
        if value.get("__kind__") == "matrix":
            dim = tuple(int(v) for v in value.get("dim", []))
            data = np.asarray(value.get("data", []), dtype=np.float64)
            return data.reshape(dim)
        return {k: _decode_packed_matrix_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_packed_matrix_payload(v) for v in value]
    return value


def _run_mgcv_raw_constructor(data: pd.DataFrame, smooth_expr: str):
    smooth_expr_r = _normalize_python_formula_text(smooth_expr)
    _cache_key = _mgcv_cache_key(
        "raw_constructor",
        {
            "version": _RAW_CONSTRUCTOR_CACHE_VERSION,
            "data": _df_cache_repr(data),
            "smooth_expr": smooth_expr_r,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return _decode_packed_matrix_payload(cached)

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]

pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list(
    "__kind__" = "matrix",
    dim = as.integer(dim(x)),
    data = unname(as.numeric(t(x)))
  )
}

pack_vector <- function(x, mode = c("numeric", "integer", "logical", "character")) {
  mode <- match.arg(mode)
  if (is.null(x)) return(NULL)
  if (mode == "numeric") return(unname(as.numeric(x)))
  if (mode == "integer") return(unname(as.integer(x)))
  if (mode == "logical") return(unname(as.logical(x)))
  unname(as.character(x))
}

pack_constraint <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) return(pack_matrix(x))
  if (is.logical(x)) return(pack_vector(x, "logical"))
  pack_vector(x, "numeric")
}

serialize_base_summary <- function(base) {
  if (is.null(base)) return(NULL)
  out <- list(
    class_name = if (is.null(base$bs)) NULL else as.character(base$bs[[1]]),
    bs_dim = if (is.null(base$bs.dim)) NULL else as.integer(base$bs.dim),
    rank = if (is.null(base$rank)) NULL else pack_vector(base$rank, "integer"),
    null_space_dim = if (is.null(base$null.space.dim)) NULL else as.integer(base$null.space.dim),
    term = if (is.null(base$term)) NULL else pack_vector(base$term, "character")
  )
  if (!is.null(base$dim)) out$dim <- as.integer(base$dim)
  out
}

serialize_gp_defn <- function(defn) {
  if (is.null(defn)) return(NULL)
  vals <- as.numeric(defn)
  list(
    type = as.integer(abs(vals[1])),
    stationary = isTRUE(vals[1] < 0),
    rho = if (length(vals) >= 2) as.numeric(vals[2]) else NULL,
    power = if (length(vals) >= 3) as.numeric(vals[3]) else 1.0
  )
}

pack_numeric_object <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.null(dim(x))) return(pack_vector(x, "numeric"))
  pack_matrix(x)
}

serialize_smooth <- function(sm) {
  cls <- as.character(class(sm)[1])
  out <- list(
    class_name = cls,
    X = pack_matrix(sm$X),
    S = if (is.null(sm$S)) list() else lapply(sm$S, pack_matrix),
    rank = if (is.null(sm$rank)) NULL else pack_vector(sm$rank, "integer"),
    null_space_dim = if (is.null(sm$null.space.dim)) NULL else as.integer(sm$null.space.dim)
  )

  extra <- switch(
    cls,
    "cr.smooth" = list(
      xp = pack_vector(sm$xp, "numeric"),
      F = pack_numeric_object(sm$F),
      noterp = isTRUE(sm$noterp)
    ),
    "cs.smooth" = list(
      xp = pack_vector(sm$xp, "numeric"),
      F = pack_numeric_object(sm$F),
      noterp = isTRUE(sm$noterp)
    ),
    "cyclic.smooth" = list(
      xp = pack_vector(sm$xp, "numeric"),
      BD = pack_matrix(sm$BD),
      noterp = isTRUE(sm$noterp)
    ),
    "pspline.smooth" = list(
      knots = pack_vector(sm$knots, "numeric"),
      m = pack_vector(sm$m, "integer")
    ),
    "tprs.smooth" = list(
      Xu = pack_matrix(sm$Xu),
      UZ = pack_matrix(sm$UZ),
      shift = pack_vector(sm$shift, "numeric"),
      drop_null = isTRUE(sm$drop.null != 0)
    ),
    "ts.smooth" = list(
      Xu = pack_matrix(sm$Xu),
      UZ = pack_matrix(sm$UZ),
      shift = pack_vector(sm$shift, "numeric"),
      drop_null = isTRUE(sm$drop.null != 0)
    ),
    "gp.smooth" = list(
      shift = pack_vector(sm$shift, "numeric"),
      gp_defn = serialize_gp_defn(sm$gp.defn),
      UZ = pack_matrix(sm$UZ),
      knt = pack_matrix(sm$knt)
    ),
    "mrf.smooth" = list(
      knots = if (is.null(sm$knots)) NULL else pack_vector(as.character(sm$knots), "character"),
      P = pack_matrix(sm$P),
      plot_me = isTRUE(sm$plot.me),
      te_ok = if (is.null(sm$te.ok)) NULL else as.integer(sm$te.ok),
      noterp = isTRUE(sm$noterp)
    ),
    "random.effect" = list(
      C = pack_constraint(sm$C),
      random = isTRUE(sm$random),
      noterp = isTRUE(sm$noterp)
    ),
    "fs.interaction" = list(
      base = serialize_base_summary(sm$base),
      P = pack_matrix(sm$P),
      fterm = pack_vector(sm$fterm, "character"),
      flev = pack_vector(sm$flev, "character"),
      Xb = pack_matrix(sm$Xb),
      C = pack_constraint(sm$C),
      te_ok = if (is.null(sm$te.ok)) NULL else as.integer(sm$te.ok),
      side_constrain = isTRUE(sm$side.constrain)
    ),
    "sz.interaction" = list(
      base = serialize_base_summary(sm$base),
      fterm = pack_vector(sm$fterm, "character"),
      flev = if (is.null(sm$flev)) NULL else lapply(sm$flev, function(v) pack_vector(v, "character")),
      Xb = pack_matrix(sm$Xb),
      C = pack_constraint(sm$C),
      te_ok = if (is.null(sm$te.ok)) NULL else as.integer(sm$te.ok),
      side_constrain = isTRUE(sm$side.constrain)
    ),
    "tensor.smooth" = list(
      mc = if (is.null(sm$mc)) NULL else pack_vector(sm$mc, "logical"),
      XP = if (is.null(sm$XP)) NULL else lapply(sm$XP, pack_matrix),
      C = pack_constraint(sm$C)
    ),
    "t2.smooth" = list(
      full = isTRUE(sm$full),
      ord = if (is.null(sm$ord)) NULL else pack_vector(sm$ord, "integer"),
      C = pack_constraint(sm$C),
      Cp = pack_constraint(sm$Cp),
      P = if (is.null(sm$P)) NULL else lapply(sm$P, pack_matrix),
      penalty_labels = if (is.null(names(sm$S))) NULL else pack_vector(names(sm$S), "character")
    ),
    list()
  )
  out$extra <- extra
  out
}

sm <- mgcv:::smooth.construct3(eval(parse(text = args[[3]])), d, NULL)
write_json(serialize_smooth(sm), out, auto_unbox = TRUE, digits = 17, null = "null")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "raw_constructor.json"
        script_path = tmpdir_path / "raw_constructor.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path), smooth_expr_r],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return _decode_packed_matrix_payload(result)


def _run_mgcv_natparam_cr(data: pd.DataFrame, *, k: int):
    _cache_key = _mgcv_cache_key(
        "natparam_cr",
        {"data": _df_cache_repr(data), "k": k},
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

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
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _run_mgcv_predict_on_newdata(
    data: pd.DataFrame,
    newdata: pd.DataFrame,
    formula: str,
    *,
    family="gaussian",
    method="REML",
    type="link",
    return_se=False,
):
    _family_nampy_unused, family_token = _family_specs(family)
    fit_method = "REML" if str(method).lower() == "fixed" else method
    formula_r = _normalize_python_formula_text(formula)

    _cache_key = _mgcv_cache_key(
        "predict_on_newdata",
        {
            "data": _df_cache_repr(data),
            "newdata": _df_cache_repr(newdata),
            "formula_r": formula_r,
            "family_token": family_token,
            "fit_method": fit_method,
            "type": type,
            "return_se": return_se,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

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
want_se <- identical(tolower(args[[7]]), "true")
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
pred <- predict(fit, newdata = newd, type = pred_type, se.fit = want_se)
out <- list()
if (pred_type == "terms") {
  if (want_se) {
    out$pred <- unname(as.matrix(pred$fit))
    out$se <- unname(as.matrix(pred$se.fit))
    out$term_names <- colnames(pred$fit)
  } else {
    out$pred <- unname(as.matrix(pred))
    out$term_names <- colnames(pred)
  }
} else if (pred_type == "lpmatrix") {
  if (want_se) {
    stop("se.fit is not supported for type='lpmatrix'")
  }
  out$pred <- unname(as.matrix(pred))
} else if (want_se) {
  out$pred <- unname(as.numeric(pred$fit))
  out$se <- unname(as.numeric(pred$se.fit))
} else {
  out$pred <- unname(as.numeric(pred))
}
write_json(
  out,
  args[[8]],
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
                formula_r,
                family_token,
                fit_method,
                type,
                "true" if return_se else "false",
                str(json_path),
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


def _run_mgcv_fixed_sp_score(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    smoothing_params,
    *,
    select: bool = False,
):
    _family_nampy, family_token = _family_specs(family)
    del _family_nampy
    sp_list = np.asarray(smoothing_params, dtype=np.float64).tolist()

    _cache_key = _mgcv_cache_key(
        "fixed_sp_score",
        {
            "data": _df_cache_repr(data),
            "formula": formula,
            "family_token": family_token,
            "method": method,
            "select": select,
            "smoothing_params": sp_list,
        },
    )
    cached = _mgcv_cache_load(_cache_key)
    if cached is not None:
        return cached

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
formula_text <- args[[2]]
family_name <- tolower(args[[3]])
method_name <- args[[4]]
select_flag <- tolower(args[[5]]) == "true"
sp <- as.numeric(fromJSON(args[[6]]))
family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL
family_obj <- switch(
  family_key,
  gaussian = gaussian(),
  binomial = {
    link <- if (is.null(family_param) || family_param == "") "logit" else family_param
    binomial(link = link)
  },
  poisson = poisson(link = "log"),
  gamma = {
    link <- if (is.null(family_param) || family_param == "") "log" else family_param
    Gamma(link = link)
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    mgcv::nb(theta = theta, link = "log")
  },
  stop(sprintf("Unsupported family for fixed-sp score: %s", family_name))
)
fit <- gam(
  formula = as.formula(formula_text),
  data = d,
  family = family_obj,
  method = method_name,
  select = select_flag,
  sp = sp
)
log_sp_ref <- log(pmax(sp, 1e-300))
eval_at_log_sp <- function(log_sp) {
  fixed_fit <- gam(
    formula = as.formula(formula_text),
    data = d,
    family = family_obj,
    method = method_name,
    select = select_flag,
    sp = unname(exp(log_sp))
  )
  unname(as.numeric(fixed_fit$gcv.ubre))
}
steps <- pmax(1e-6, 1e-5 * (1 + abs(log_sp_ref)))
g <- rep(NA_real_, length(log_sp_ref))
for (i in seq_along(log_sp_ref)) {
  plus1 <- log_sp_ref
  minus1 <- log_sp_ref
  plus2 <- log_sp_ref
  minus2 <- log_sp_ref
  plus1[i] <- plus1[i] + steps[i]
  minus1[i] <- minus1[i] - steps[i]
  plus2[i] <- plus2[i] + 2 * steps[i]
  minus2[i] <- minus2[i] - 2 * steps[i]
  g[i] <- (
    -eval_at_log_sp(plus2) +
      8 * eval_at_log_sp(plus1) -
      8 * eval_at_log_sp(minus1) +
      eval_at_log_sp(minus2)
  ) / (12 * steps[i])
}
write_json(
  list(
    criterion_value = unname(as.numeric(fit$gcv.ubre)),
    gradient = unname(as.numeric(g))
  ),
  args[[7]],
  auto_unbox = TRUE,
  digits = 17
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "fixed_sp_score.json"
        script_path = tmpdir_path / "fixed_sp_score.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            [
                R_SCRIPT,
                str(script_path),
                str(csv_path),
                formula,
                family_token,
                method,
                "true" if select else "false",
                json.dumps(sp_list),
                str(json_path),
            ],
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_cache_save(_cache_key, result)
    return result


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


def _assert_exact_mgcv_snapshot_parity(
    actual,
    expected,
    *,
    pred_atol=1e-10,
    pred_rtol=1e-10,
    edf_atol=1e-10,
    criterion_atol=1e-10,
    criterion_rtol=1e-10,
    sp_atol=1e-10,
    sp_rtol=1e-10,
    log_sp_atol=1e-10,
):
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    np.testing.assert_allclose(
        np.asarray(a_fit["smoothing_params"], dtype=np.float64),
        np.asarray(e_fit["smoothing_params"], dtype=np.float64),
        atol=sp_atol,
        rtol=sp_rtol,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["log_smoothing_params"], dtype=np.float64),
        np.asarray(e_fit["log_smoothing_params"], dtype=np.float64),
        atol=log_sp_atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_total"], dtype=np.float64),
        np.asarray(e_fit["edf_total"], dtype=np.float64),
        atol=edf_atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_by_term"], dtype=np.float64),
        np.asarray(e_fit["edf_by_term"], dtype=np.float64),
        atol=edf_atol,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["criterion_value"], dtype=np.float64),
        np.asarray(e_fit["criterion_value"], dtype=np.float64),
        atol=criterion_atol,
        rtol=criterion_rtol,
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
    """Compare two matrices up to per-column sign flips or 2D subspace rotations.

    For columns that don't match under a sign flip (e.g. degenerate null-space
    subspaces where any 2D rotation is equally valid), falls back to a Procrustes
    alignment of the column pair before asserting.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    aligned = actual.copy()
    j = 0
    while j < actual.shape[1]:
        # Try sign flip for column j.
        if np.linalg.norm(actual[:, j] - expected[:, j]) > np.linalg.norm(
            -actual[:, j] - expected[:, j]
        ):
            aligned[:, j] *= -1.0

        # If still doesn't match and a next column exists, try a 2D Procrustes
        # rotation for the pair (j, j+1).  This handles degenerate null-space
        # subspaces where different LAPACK implementations produce different (but
        # equally valid) orthonormal bases for the same 2D subspace.
        if (
            np.max(np.abs(aligned[:, j] - expected[:, j])) > atol
            and j + 1 < actual.shape[1]
        ):
            A2 = actual[:, j : j + 2]
            B2 = expected[:, j : j + 2]
            U_svd, _, Vt = np.linalg.svd(A2.T @ B2)
            M = U_svd @ Vt
            rotated = A2 @ M
            if np.max(np.abs(rotated - B2)) <= atol:
                aligned[:, j : j + 2] = rotated
                j += 2
                continue
        j += 1
    np.testing.assert_allclose(aligned, expected, atol=atol, rtol=rtol)


__all__ = [
    "MGCV_ANOVA_SCRIPT",
    "MGCV_SNAPSHOT_SCRIPT",
    "R_SCRIPT",
    "_assert_allclose_up_to_column_sign",
    "_assert_basic_mgcv_parity",
    "_assert_exact_mgcv_snapshot_parity",
    "_family_specs",
    "_fit_nampy_model",
    "_fit_nampy_model_fixed_sp",
    "_fit_nampy_snapshot",
    "_make_binomial_data",
    "_make_distributional_gaussian_data",
    "_make_fs_data",
    "_make_fs_data_4levels",
    "_make_gamma_data",
    "_make_gaussian_data",
    "_make_gaussian_data_3col",
    "_make_mrf_data",
    "_make_mrf_low_rank_data",
    "_make_negbin_data",
    "_make_poisson_data",
    "_make_random_effect_data",
    "_make_random_effect_data_noisy",
    "_make_sz_data",
    "_make_sz_data_3x3",
    "_run_mgcv_anova",
    "_run_mgcv_natparam_cr",
    "_run_mgcv_fixed_sp_score",
    "_run_mgcv_predict_on_newdata",
    "_run_mgcv_smoothcon_matrix",
    "_run_mgcv_smoothcon_matrix_unscaled",
    "_run_mgcv_smoothcon_penalties",
    "_run_mgcv_snapshot",
]
