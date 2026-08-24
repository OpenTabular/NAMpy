"""Shared mgcv/R snapshot helpers for parity tests (decoupled from ``test_mgcv_snapshot_parity``)."""

from __future__ import annotations

import ast
import atexit
import hashlib
import io
import itertools
import json
import os
import re
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
from tests.reference_fixtures import (
    REFRESH_ENV,
    AliasedReferenceKey,
    MissingReferenceFixture,
    fixture_payload_variants,
    load_aliased_reference,
    portable_dataframe_identity,
    save_reference,
)

_REPO_ROOT = REPO_ROOT
_TESTS_DIR = TESTS_DIR

MGCV_SNAPSHOT_SCRIPT = PARITY_DIR / "mgcv_snapshot.R"
MGCV_SNAPSHOT_SERVER_SCRIPT = PARITY_DIR / "mgcv_snapshot_server.R"
MGCV_ANOVA_SCRIPT = PARITY_DIR / "mgcv_anova.R"
MGCV_GAM_SETUP_ASSEMBLY_SCRIPT = PARITY_DIR / "mgcv_gam_setup_assembly.R"


def _resolve_r_executable() -> str | None:
    """Resolve a working R invocation, preferring Rscript and then R."""
    explicit = os.environ.get("MGCV_RSCRIPT")
    if explicit:
        return explicit

    rscript = shutil.which("Rscript")
    if rscript:
        return rscript

    r_binary = shutil.which("R")
    if r_binary:
        return r_binary

    r_home = os.environ.get("R_HOME", "")
    if r_home:
        for rel in ("bin/Rscript", "bin/R"):
            candidate = Path(r_home) / rel
            if candidate.exists():
                return str(candidate)

    return None


def _build_r_command(script_path: Path, *args: str) -> list[str]:
    """Build a command list that can run `script_path` with `args`."""
    if R_SCRIPT is None:
        raise RuntimeError(
            "R not available: install R (including Rscript) or set MGCV_RSCRIPT."
        )

    exe = Path(R_SCRIPT).name.lower()
    if exe == "r" or exe == "r.exe":
        return [
            R_SCRIPT,
            "--quiet",
            "--no-restore",
            "--no-save",
            "--slave",
            "-f",
            str(script_path),
            "--args",
            *[str(arg) for arg in args],
        ]
    return [R_SCRIPT, str(script_path), *[str(arg) for arg in args]]


R_SCRIPT = _resolve_r_executable()

# ---------------------------------------------------------------------------
# Static mgcv reference fixtures
# ---------------------------------------------------------------------------
# Normal tests only read committed upstream outputs. R is invoked solely when
# a developer explicitly opts into fixture generation with REFRESH_ENV.
_MGCV_FIXTURE_DIR = _TESTS_DIR / "reference_fixtures" / "mgcv"
_MGCV_FIXTURE_ALIASES = _MGCV_FIXTURE_DIR / "aliases.json"
_MGCV_SNAPSHOT_SERVER_PROCESS = None
_MGCV_SNAPSHOT_SERVER_LOCK = threading.Lock()
_MGCV_SNAPSHOT_SERVER_REQUEST_IDS = itertools.count(1)
# Version 7: mgcv_snapshot.R gained the summary block (p.table, r.sq,
# dev.expl, residual.df, method, ...) plus fit-level null_deviance / rank /
# scale_estimated for the summary.gam port.
_SNAPSHOT_FIXTURE_VERSION = 7
_RAW_CONSTRUCTOR_FIXTURE_VERSION = 5
_GAM_SETUP_ASSEMBLY_FIXTURE_VERSION = 5
_GAM_VCOMP_FIXTURE_VERSION = 4
_NATPARAM_TYPE3_FIXTURE_VERSION = 1
_SMOOTHCON_PREDICT_MATRIX_FIXTURE_VERSION = 1
_FIXED_SP_SCORE_FIXTURE_VERSION = 2
_PREDICT_ON_NEWDATA_FIXTURE_VERSION = 3


def _start_mgcv_snapshot_server() -> subprocess.Popen:
    """Start the persistent mgcv snapshot worker process."""
    cmd = _build_r_command(
        MGCV_SNAPSHOT_SERVER_SCRIPT,
        str(MGCV_SNAPSHOT_SCRIPT),
    )
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
    optimizer: str | None = None,
):
    """Run a single snapshot via subprocess call (legacy behavior)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "snapshot.json"
        data.to_csv(csv_path, index=False)

        cmd = _build_r_command(
            MGCV_SNAPSHOT_SCRIPT,
            str(csv_path),
            str(json_path),
            str(formula),
            family,
            method,
            "true" if select else "false",
        )
        if weights_column is not None:
            cmd.append(str(weights_column))
        elif optimizer is not None:
            cmd.append("-")
        if optimizer is not None:
            cmd.append(str(optimizer))

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


def _mgcv_fixture_key(fn_name: str, key_parts: dict) -> AliasedReferenceKey:
    """Return a stable hex digest that identifies a unique mgcv call."""
    canonical_parts, legacy_parts = fixture_payload_variants(
        key_parts, normalize_floats=True
    )

    def digest(parts) -> str:
        buf = io.StringIO()
        buf.write(fn_name)
        buf.write("|")
        # Sort keys for stability; values are already strings/primitives.
        buf.write(json.dumps(parts, sort_keys=True, default=str))
        return hashlib.sha256(buf.getvalue().encode()).hexdigest()

    canonical = digest(canonical_parts)
    legacy = digest(legacy_parts)
    return AliasedReferenceKey(canonical, None if legacy == canonical else legacy)


def _mgcv_fixture_load(key: str):
    """Return a committed result, or ``None`` only during explicit refresh."""
    try:
        return load_aliased_reference(
            "",
            key,
            aliases_path=_MGCV_FIXTURE_ALIASES,
            root=_MGCV_FIXTURE_DIR,
        )
    except MissingReferenceFixture as exc:
        raise RuntimeError(
            "This test requires a committed static mgcv reference fixture. "
            f"Key '{key}' is not present in {_MGCV_FIXTURE_DIR}. Generate it locally "
            f"with {REFRESH_ENV}=1 and commit the result."
        ) from exc


def _mgcv_fixture_save(key: str, result) -> None:
    """Persist deterministic compressed JSON during explicit refresh mode."""
    save_reference("", str(key), result, root=_MGCV_FIXTURE_DIR)


def _df_fixture_repr(df: pd.DataFrame) -> str:
    """Deterministic string representation of a DataFrame for cache keying."""
    return portable_dataframe_identity(df)


def _portable_df_fixture_repr(df: pd.DataFrame) -> str:
    """Return a platform-stable DataFrame identity for static fixture keys.

    Transcendental NumPy operations can differ by a final binary digit across
    system math libraries. Those differences are immaterial to constructor
    parity but pandas' default full-precision CSV output turns them into
    different hashes. Twelve significant decimal digits retain more precision
    than the parity data scales while removing that platform noise.
    """
    return portable_dataframe_identity(df, legacy_float_format="%.15g")


def _raw_constructor_fixture_frame(
    data: pd.DataFrame, smooth_expr: str
) -> pd.DataFrame:
    """Select only columns that can affect the requested smooth constructor."""
    identifier_char = r"A-Za-z0-9_."
    referenced = [
        column
        for column in data.columns
        if re.search(
            rf"(?<![{identifier_char}]){re.escape(str(column))}"
            rf"(?![{identifier_char}])",
            smooth_expr,
        )
    ]
    # Preserve the complete input for unusual expressions whose referenced
    # columns cannot be identified lexically.
    return data.loc[:, referenced] if referenced else data


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


def _make_gaussian_link_data(seed=7, n=150):
    """Positive-response gaussian data for noncanonical (log/inverse) links."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(size=n)
    mu = np.exp(0.4 + np.sin(2.0 * np.pi * x0))
    y = mu + 0.15 * mu * rng.standard_normal(n)
    y = np.maximum(y, 1e-3)
    return pd.DataFrame({"y": y, "x0": x0})


def _make_negbin_data(seed=2024, n=240, theta=1.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.2 + 0.55 * np.sin(x0) - 0.25 * x1
    mu = np.exp(eta)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p, size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": w})


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
    if spec.data_factory == "fs":
        return _make_fs_data()
    raise ValueError(
        f"Unsupported data factory '{spec.data_factory}' for case '{case_id}'."
    )


def _family_specs(family):
    if isinstance(family, dict):
        key = str(family.get("name", "")).lower()
        key = {
            "bernoulli": "binomial",
            "logistic": "binomial",
            "normal": "gaussian",
        }.get(key, key)
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            theta = float(family.get("theta", 1.0))
            link = str(family.get("link", "log") or "log").lower()
            suffix = f":{link}" if link != "log" else ""
            if bool(family.get("estimate_theta", False)):
                return family, f"negbin_est:{theta:.12g}{suffix}"
            return family, f"negbin:{theta:.12g}{suffix}"
        link = str(family.get("link", "")).lower()
        default_links = {
            "binomial": "logit",
            "gamma": "inverse",
            "gaussian": "identity",
            "poisson": "log",
        }
        if link and link != default_links.get(key, link):
            # Non-default link: encode as "family:link" token for the R script.
            return family, f"{key}:{link}"
        return family, key
    key = str(family).lower()
    return family, key


_R_NAME_RE = re.compile(r"^[A-Za-z.][A-Za-z0-9._]*$")


def _r_name(name: str) -> str:
    text = str(name)
    return text if _R_NAME_RE.fullmatch(text) else f"`{text}`"


def _is_scalar_literal_node(node) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, (bool, int, float, str))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    return False


def _is_matrix_literal_node(node) -> bool:
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) == 0:
        return False
    row_lengths = []
    for elt in node.elts:
        if not isinstance(elt, (ast.List, ast.Tuple)) or len(elt.elts) == 0:
            return False
        if not all(_is_scalar_literal_node(item) for item in elt.elts):
            return False
        row_lengths.append(len(elt.elts))
    return len(set(row_lengths)) == 1


def _emit_r_expr(
    node, *, dict_key: str | None = None, kw_name: str | None = None
) -> str:
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, str):
            return json.dumps(value)
        return repr(value)

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return "-" + _emit_r_expr(node.operand, dict_key=dict_key, kw_name=kw_name)

    if isinstance(node, (ast.List, ast.Tuple)):
        if _is_matrix_literal_node(node):
            if (kw_name or dict_key) == "m":
                rows = []
                for row in node.elts:
                    row_values = ", ".join(_emit_r_expr(item) for item in row.elts)
                    rows.append(f"c({row_values})")
                return f"list({', '.join(rows)})"

            rows = len(node.elts)
            flat = []
            for row in node.elts:
                flat.extend(_emit_r_expr(item) for item in row.elts)
            return f"matrix(c({', '.join(flat)}), nrow={rows}, byrow=TRUE)"

        values = [_emit_r_expr(elt) for elt in node.elts]
        if all(_is_scalar_literal_node(elt) for elt in node.elts):
            return f"c({', '.join(values)})"
        return f"list({', '.join(values)})"

    if isinstance(node, ast.Dict):
        parts = []
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                parts.append(_emit_r_expr(value_node))
                continue
            if not isinstance(key_node, ast.Constant) or not isinstance(
                key_node.value, str
            ):
                raise TypeError("Only string dict keys are supported in formula text.")
            parts.append(
                f"{_r_name(key_node.value)}="
                f"{_emit_r_expr(value_node, dict_key=str(key_node.value))}"
            )
        return f"list({', '.join(parts)})"

    if isinstance(node, ast.Call):
        func = _emit_r_expr(node.func)
        args = [_emit_r_expr(arg) for arg in node.args]
        kwargs = []
        for kw in node.keywords:
            if kw.arg is None:
                if not isinstance(kw.value, ast.Dict):
                    raise TypeError(
                        "**kwargs in formula text must expand a dictionary literal."
                    )
                for key_node, value_node in zip(
                    kw.value.keys, kw.value.values, strict=True
                ):
                    if key_node is None:
                        raise TypeError(
                            "Nested **kwargs expansion is not supported in formula text."
                        )
                    if not isinstance(key_node, ast.Constant) or not isinstance(
                        key_node.value, str
                    ):
                        raise TypeError(
                            "Only string dict keys are supported in formula text."
                        )
                    kwargs.append(
                        f"{_r_name(str(key_node.value))}="
                        f"{_emit_r_expr(value_node, kw_name=str(key_node.value))}"
                    )
                continue
            kwargs.append(f"{_r_name(kw.arg)}={_emit_r_expr(kw.value, kw_name=kw.arg)}")
        return f"{func}({', '.join([*args, *kwargs])})"

    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "^",
        }
        op = op_map.get(type(node.op), None)
        if op is None:
            raise TypeError(f"Unsupported operator in formula text: {type(node.op)}")
        return (
            f"{_emit_r_expr(node.left, dict_key=dict_key, kw_name=kw_name)} "
            f"{op} "
            f"{_emit_r_expr(node.right, dict_key=dict_key, kw_name=kw_name)}"
        )

    raise TypeError(f"Unsupported formula node: {type(node).__name__}")


def _normalize_python_formula_text(formula) -> str:
    """Translate Python-style formula literals into R syntax."""
    text = str(formula).strip()
    if "~" in text:
        lhs, rhs = text.split("~", 1)
        try:
            rhs_r = _emit_r_expr(ast.parse(rhs.strip(), mode="eval").body)
            return f"{lhs.strip()} ~ {rhs_r}"
        except Exception:
            pass
    else:
        try:
            return _emit_r_expr(ast.parse(text, mode="eval").body)
        except Exception:
            pass

    out = text
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
    allow_live_run: bool = False,
    optimizer: str | None = None,
    portable_fixture_identity: bool = False,
):
    def _normalize_snapshot_payload(payload):
        if not isinstance(payload, dict):
            return payload
        parity = payload.get("parity", None)
        if not isinstance(parity, dict):
            return payload
        diagnostics = parity.get("diagnostics", None)
        if not isinstance(diagnostics, dict):
            return payload
        residuals = diagnostics.get("residuals", None)
        if not isinstance(residuals, dict):
            return payload

        residuals = dict(residuals)
        changed = False
        for key in (
            "response",
            "working",
            "pearson",
            "scaled_pearson",
            "deviance",
        ):
            if isinstance(residuals.get(key, None), dict) and len(residuals[key]) == 0:
                residuals[key] = None
                changed = True
        if not changed:
            return payload

        diagnostics = dict(diagnostics)
        diagnostics["residuals"] = residuals
        parity = dict(parity)
        parity["diagnostics"] = diagnostics
        payload = dict(payload)
        payload["parity"] = parity
        return payload

    _family_nampy, family_token = _family_specs(family)
    cache_parts = {
        "version": _SNAPSHOT_FIXTURE_VERSION,
        "data": (
            _portable_df_fixture_repr(data)
            if portable_fixture_identity
            else _df_fixture_repr(data)
        ),
        "formula": str(formula),
        "family_token": family_token,
        "method": method,
        "select": select,
        "weights_column": weights_column,
    }
    if optimizer is not None:
        cache_parts["optimizer"] = optimizer

    _cache_key = _mgcv_fixture_key(
        "snapshot",
        cache_parts,
    )
    try:
        cached = _mgcv_fixture_load(_cache_key)
    except RuntimeError:
        if not allow_live_run:
            raise
        cached = None
    if cached is not None:
        return _normalize_snapshot_payload(cached)

    formula_for_r = _normalize_python_formula_text(formula)
    if optimizer is not None:
        result = _run_mgcv_snapshot_single(
            data,
            formula_for_r,
            family_token,
            method,
            select=select,
            weights_column=weights_column,
            optimizer=optimizer,
        )
    else:
        try:
            result = _run_mgcv_snapshot_batched(
                data,
                formula_for_r,
                family_token,
                method,
                select=select,
                weights_column=weights_column,
            )
        except Exception:
            result = _run_mgcv_snapshot_single(
                data,
                formula_for_r,
                family_token,
                method,
                select=select,
                weights_column=weights_column,
            )

    result = _normalize_snapshot_payload(result)
    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_anova(
    data: pd.DataFrame,
    formulas,
    family,
    method: str,
    *,
    select: bool = False,
    test: str | None = None,
):
    _family_nampy, family_token = _family_specs(family)
    formula_texts = [str(formula) for formula in list(formulas)]

    _cache_key = _mgcv_fixture_key(
        "anova",
        {
            "data": _df_fixture_repr(data),
            "formulas": formula_texts,
            "family_token": family_token,
            "method": method,
            "select": select,
            "test": test,
        },
    )
    cached = _mgcv_fixture_load(_cache_key)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "anova.json"
        data.to_csv(csv_path, index=False)
        cmd = _build_r_command(
            str(MGCV_ANOVA_SCRIPT),
            str(csv_path),
            str(json_path),
            json.dumps(formula_texts),
            family_token,
            method,
            "true" if select else "false",
            "NULL" if test is None else str(test),
        )
        subprocess.run(
            cmd,
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_gam_vcomp(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    rescale: bool = True,
    weights_column: str | None = None,
    conf_lev: float = 0.95,
):
    _family_nampy, family_token = _family_specs(family)

    _cache_key = _mgcv_fixture_key(
        "gam_vcomp",
        {
            "version": _GAM_VCOMP_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "formula": str(formula),
            "family_token": family_token,
            "method": str(method),
            "select": bool(select),
            "rescale": bool(rescale),
            "weights_column": weights_column,
            "conf_lev": float(conf_lev),
        },
    )

    def _normalize_gam_vcomp_payload(payload):
        if payload is None:
            return None
        if not isinstance(payload, dict):
            return payload
        out = dict(payload)
        for key in ("all_names", "rank", "rank_hess", "conf_lev"):
            if isinstance(out.get(key, None), dict) and len(out[key]) == 0:
                out[key] = None
        return out

    cached = _mgcv_fixture_load(_cache_key)
    if cached is not None:
        return _normalize_gam_vcomp_payload(cached)

    r_code = """
normalize_formula_text <- function(x) {
  x <- gsub("\\\\[", "c(", x)
  x <- gsub("\\\\]", ")", x)
  x <- gsub("\\\\bTrue\\\\b", "TRUE", x)
  x <- gsub("\\\\bFalse\\\\b", "FALSE", x)
  x <- gsub("\\\\bNone\\\\b", "NULL", x)
  x
}

serialize_gam_vcomp <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.list(v)) {
    vc_part <- v$vc
    vc_names <- if (is.null(vc_part)) NULL else if (!is.null(dim(vc_part))) rownames(vc_part) else names(vc_part)
    all_part <- v$all
    all_names <- if (is.null(all_part)) NULL else names(all_part)
    return(list(
      vc = if (is.null(vc_part)) NULL else unname(vc_part),
      names = vc_names,
      all = if (is.null(all_part)) NULL else unname(all_part),
      all_names = all_names,
      rank = if (is.null(v$rank)) NULL else as.integer(v$rank),
      rank_hess = if (is.null(v$rank.hess)) NULL else as.integer(v$rank.hess),
      conf_lev = if (is.null(v$conf.lev)) NULL else as.numeric(v$conf.lev)
    ))
  }
  list(
    vc = unname(v),
    names = names(v),
    all = NULL,
    all_names = NULL,
    rank = NULL,
    rank_hess = NULL,
    conf_lev = NULL
  )
}

suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
formula_text <- normalize_formula_text(args[[3]])
coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}
family_name <- tolower(args[[4]])
method_name <- args[[5]]
select_flag <- tolower(args[[6]]) %in% c("true", "1", "yes")
rescale_flag <- tolower(args[[7]]) %in% c("true", "1", "yes")
weights_column <- if (length(args) >= 8 && nzchar(args[[8]]) && args[[8]] != "NULL") args[[8]] else NULL
conf_lev <- if (length(args) >= 9) as.numeric(args[[9]]) else 0.95
fit_method <- if (tolower(method_name) == "fixed") "REML" else method_name
if (tolower(method_name) == "gcv") fit_method <- "GCV.Cp"

family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL
family_link <- function(default, index = 2L) {
  if (length(family_parts) >= index && nzchar(family_parts[[index]])) {
    family_parts[[index]]
  } else {
    default
  }
}
family_obj <- switch(
  family_key,
  gaussian = {
    link <- family_link("identity")
    gaussian(link = link)
  },
  binomial = {
    link <- family_link("logit")
    binomial(link = link)
  },
  poisson = {
    link <- family_link("log")
    poisson(link = link)
  },
  gamma = {
    link <- family_link("inverse")
    Gamma(link = link)
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    link <- family_link("log", 3L)
    do.call(mgcv::nb, list(theta = theta, link = link))
  },
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    link <- family_link("log", 3L)
    do.call(mgcv::nb, list(theta = -abs(theta), link = link))
  },
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family for gam_vcomp parity: %s", family_name))
)

gam_args <- list(
  formula = coerce_formula(formula_text),
  data = d,
  family = family_obj,
  method = fit_method,
  select = select_flag
)
if (!is.null(weights_column)) gam_args$weights <- d[[weights_column]]
fit <- do.call(gam, gam_args)
vc <- gam.vcomp(fit, rescale = rescale_flag, conf.lev = conf_lev)
write_json(serialize_gam_vcomp(vc), out, auto_unbox = TRUE, digits = 17)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "gam_vcomp.json"
        script_path = tmpdir_path / "gam_vcomp.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                str(formula),
                family_token,
                str(method),
                "true" if select else "false",
                "true" if rescale else "false",
                "NULL" if weights_column is None else str(weights_column),
                f"{float(conf_lev):.12g}",
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = _normalize_gam_vcomp_payload(
            json.loads(json_path.read_text(encoding="utf-8"))
        )

    _mgcv_fixture_save(_cache_key, result)
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
    _cache_key = _mgcv_fixture_key(
        "smoothcon_matrix",
        {"data": _df_fixture_repr(data), "smooth_expr": smooth_expr},
    )
    cached = _mgcv_fixture_load(_cache_key)
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
            _build_r_command(script_path, str(csv_path), str(json_path), smooth_expr),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_smoothcon_penalties(
    data: pd.DataFrame,
    smooth_expr: str,
    *,
    absorb_cons: bool,
    scale_penalty: bool,
):
    _cache_key = _mgcv_fixture_key(
        "smoothcon_penalties",
        {
            "data": _df_fixture_repr(data),
            "smooth_expr": smooth_expr,
            "absorb_cons": absorb_cons,
            "scale_penalty": scale_penalty,
        },
    )
    cached = _mgcv_fixture_load(_cache_key)
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
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                smooth_expr,
                "true" if absorb_cons else "false",
                "true" if scale_penalty else "false",
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_smoothcon_matrix_unscaled(data: pd.DataFrame, smooth_expr: str):
    _cache_key = _mgcv_fixture_key(
        "smoothcon_matrix_unscaled",
        {"data": _df_fixture_repr(data), "smooth_expr": smooth_expr},
    )
    cached = _mgcv_fixture_load(_cache_key)
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
            _build_r_command(script_path, str(csv_path), str(json_path), smooth_expr),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
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


def _normalize_raw_constructor_knots(knots):
    if knots is None:
        return None
    if not isinstance(knots, dict):
        raise TypeError("raw constructor knots must be a dict keyed by feature name.")
    out = {}
    for key, value in knots.items():
        if value is None:
            out[str(key)] = None
            continue
        arr = np.asarray(value, dtype=object).ravel()
        out[str(key)] = arr.tolist()
    return out


def _run_mgcv_raw_constructor(
    data: pd.DataFrame, smooth_expr: str, knots: dict | None = None
):
    smooth_expr_r = _normalize_python_formula_text(smooth_expr)
    knots_payload = _normalize_raw_constructor_knots(knots)
    key_parts = {
        "version": _RAW_CONSTRUCTOR_FIXTURE_VERSION,
        "data": _portable_df_fixture_repr(
            _raw_constructor_fixture_frame(data, smooth_expr_r)
        ),
        "smooth_expr": smooth_expr_r,
        "knots": json.dumps(knots_payload, sort_keys=True, default=str),
    }
    _cache_key = _mgcv_fixture_key("raw_constructor", key_parts)
    try:
        cached = _mgcv_fixture_load(_cache_key)
    except RuntimeError as portable_error:
        # Preserve access to the committed v5 fixtures while they are migrated
        # first from full-frame portable identities, then exact pandas CSV.
        legacy_keys = [
            _mgcv_fixture_key(
                "raw_constructor",
                {
                    **key_parts,
                    "data": representation,
                },
            )
            for representation in (
                _portable_df_fixture_repr(data),
                _df_fixture_repr(data),
            )
        ]
        cached = None
        seen_candidates: set[tuple[str, str | None]] = set()
        for legacy_key in legacy_keys:
            candidate = (str(legacy_key), getattr(legacy_key, "legacy", None))
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            try:
                cached = _mgcv_fixture_load(legacy_key)
            except RuntimeError:
                continue
            break
        if cached is None:
            raise portable_error from None
    if cached is not None:
        return _decode_packed_matrix_payload(cached)

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
kn <- NULL
if (length(args) >= 4 && nzchar(args[[4]])) {
  kraw <- fromJSON(args[[4]], simplifyVector = FALSE)
  kn <- lapply(kraw, function(v) {
    if (is.null(v)) return(NULL)
    vals <- unlist(v, recursive = TRUE, use.names = FALSE)
    if (is.numeric(vals)) return(unname(as.numeric(vals)))
    if (is.integer(vals)) return(unname(as.integer(vals)))
    if (is.logical(vals)) return(unname(as.logical(vals)))
    unname(as.character(vals))
  })
}

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
    "cpspline.smooth" = list(
      knots = pack_vector(sm$knots, "numeric"),
      m = pack_vector(sm$m, "integer")
    ),
    "Bspline.smooth" = list(
      knots = pack_vector(sm$knots, "numeric"),
      m = pack_vector(sm$m, "numeric"),
      D = if (is.null(sm$D)) list() else lapply(sm$D, pack_matrix)
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
    "duchon.spline" = list(
      knt = pack_matrix(sm$knt),
      UZ = pack_matrix(sm$UZ),
      shift = pack_vector(sm$shift, "numeric"),
      p_order = pack_vector(sm$p.order, "numeric")
    ),
    "gp.smooth" = list(
      knt = pack_matrix(sm$knt),
      UZ = pack_matrix(sm$UZ),
      shift = pack_vector(sm$shift, "numeric"),
      gp_defn = pack_vector(sm$gp.defn, "numeric")
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
    list()
  )
  out$extra <- extra
  out
}

sm <- mgcv:::smooth.construct3(eval(parse(text = args[[3]])), d, kn)
write_json(serialize_smooth(sm), out, auto_unbox = TRUE, digits = 17, null = "null")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "raw_constructor.json"
        script_path = tmpdir_path / "raw_constructor.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        knots_json = (
            ""
            if knots_payload is None
            else json.dumps(knots_payload, sort_keys=True, default=str)
        )
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                smooth_expr_r,
                knots_json,
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return _decode_packed_matrix_payload(result)


def _run_mgcv_natparam_cr(data: pd.DataFrame, *, k: int):
    _cache_key = _mgcv_fixture_key(
        "natparam_cr",
        {"data": _df_fixture_repr(data), "k": k},
    )
    cached = _mgcv_fixture_load(_cache_key)
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
            _build_r_command(script_path, str(csv_path), str(json_path), str(int(k))),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_natparam_type3(
    data: pd.DataFrame,
    smooth_expr: str,
    *,
    knots: dict | None = None,
):
    smooth_expr_r = _normalize_python_formula_text(smooth_expr)
    knots_payload = _normalize_raw_constructor_knots(knots)
    _cache_key = _mgcv_fixture_key(
        "natparam_type3",
        {
            "version": _NATPARAM_TYPE3_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "smooth_expr": smooth_expr_r,
            "knots": json.dumps(knots_payload, sort_keys=True, default=str),
        },
    )
    cached = _mgcv_fixture_load(_cache_key)
    if cached is not None:
        return _decode_packed_matrix_payload(cached)

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
kn <- NULL
if (length(args) >= 4 && nzchar(args[[4]])) {
  kraw <- fromJSON(args[[4]], simplifyVector = FALSE)
  kn <- lapply(kraw, function(v) {
    if (is.null(v)) return(NULL)
    vals <- unlist(v, recursive = TRUE, use.names = FALSE)
    if (is.numeric(vals)) return(unname(as.numeric(vals)))
    if (is.integer(vals)) return(unname(as.integer(vals)))
    if (is.logical(vals)) return(unname(as.logical(vals)))
    unname(as.character(vals))
  })
}

pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list(
    "__kind__" = "matrix",
    dim = as.integer(dim(x)),
    data = unname(as.numeric(t(x)))
  )
}

sm <- mgcv:::smooth.construct3(eval(parse(text = args[[3]])), d, kn)
if (length(sm$S) != 1) stop("nat.param(type=3) helper requires exactly one penalty.")
rank_val <- if (length(sm$rank) >= 1) as.integer(sm$rank[[1]]) else NULL
np <- mgcv:::nat.param(
  sm$X,
  sm$S[[1]],
  rank = rank_val,
  type = 3,
  unit.fnorm = TRUE
)
write_json(
  list(
    X = pack_matrix(np$X),
    P = pack_matrix(np$P),
    rank = if (is.null(np$rank)) NULL else as.integer(np$rank),
    rawX = pack_matrix(sm$X),
    rawS = pack_matrix(sm$S[[1]])
  ),
  out,
  auto_unbox = TRUE,
  digits = 17,
  null = "null"
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "natparam_type3.json"
        script_path = tmpdir_path / "natparam_type3.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        knots_json = (
            ""
            if knots_payload is None
            else json.dumps(knots_payload, sort_keys=True, default=str)
        )
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                smooth_expr_r,
                knots_json,
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return _decode_packed_matrix_payload(result)


def _run_mgcv_smoothcon_predict_matrix(
    data: pd.DataFrame,
    newdata: pd.DataFrame,
    smooth_expr: str,
    *,
    knots: dict | None = None,
    absorb_cons: bool = True,
    scale_penalty: bool = True,
    deriv: int | None = None,
):
    smooth_expr_r = _normalize_python_formula_text(smooth_expr)
    knots_payload = _normalize_raw_constructor_knots(knots)
    cache_parts = {
            "version": _SMOOTHCON_PREDICT_MATRIX_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "newdata": _df_fixture_repr(newdata),
            "smooth_expr": smooth_expr_r,
            "knots": json.dumps(knots_payload, sort_keys=True, default=str),
            "absorb_cons": bool(absorb_cons),
            "scale_penalty": bool(scale_penalty),
    }
    if deriv is not None:
        cache_parts["deriv"] = int(deriv)
    _cache_key = _mgcv_fixture_key("smoothcon_predict_matrix", cache_parts)
    cached = _mgcv_fixture_load(_cache_key)
    if cached is not None:
        return _decode_packed_matrix_payload(cached)

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
train <- read.csv(args[[1]], stringsAsFactors = FALSE)
newd <- read.csv(args[[2]], stringsAsFactors = FALSE)
for (nm in names(train)) if (is.character(train[[nm]])) train[[nm]] <- factor(train[[nm]])
for (nm in names(newd)) {
  if (is.character(newd[[nm]]) && nm %in% names(train) && is.factor(train[[nm]])) {
    newd[[nm]] <- factor(newd[[nm]], levels = levels(train[[nm]]))
  } else if (is.character(newd[[nm]])) {
    newd[[nm]] <- factor(newd[[nm]])
  }
}
out <- args[[3]]
kn <- NULL
if (length(args) >= 7 && nzchar(args[[7]])) {
  kraw <- fromJSON(args[[7]], simplifyVector = FALSE)
  kn <- lapply(kraw, function(v) {
    if (is.null(v)) return(NULL)
    vals <- unlist(v, recursive = TRUE, use.names = FALSE)
    if (is.numeric(vals)) return(unname(as.numeric(vals)))
    if (is.integer(vals)) return(unname(as.integer(vals)))
    if (is.logical(vals)) return(unname(as.logical(vals)))
    unname(as.character(vals))
  })
}

pack_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  x <- as.matrix(x)
  list(
    "__kind__" = "matrix",
    dim = as.integer(dim(x)),
    data = unname(as.numeric(t(x)))
  )
}

absorb_cons <- tolower(args[[5]]) %in% c("true", "1", "yes")
scale_penalty <- tolower(args[[6]]) %in% c("true", "1", "yes")
sm <- smoothCon(
  eval(parse(text = args[[4]])),
  train,
  knots = kn,
  absorb.cons = absorb_cons,
  scale.penalty = scale_penalty
)[[1]]
if (length(args) >= 8 && nzchar(args[[8]])) sm$deriv <- as.integer(args[[8]])
pm <- PredictMat(sm, newd)
write_json(
  list(X = pack_matrix(pm)),
  out,
  auto_unbox = TRUE,
  digits = 17,
  null = "null"
)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_path = tmpdir_path / "train.csv"
        new_path = tmpdir_path / "new.csv"
        json_path = tmpdir_path / "predict_matrix.json"
        script_path = tmpdir_path / "predict_matrix.R"
        data.to_csv(train_path, index=False)
        newdata.to_csv(new_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        knots_json = (
            ""
            if knots_payload is None
            else json.dumps(knots_payload, sort_keys=True, default=str)
        )
        subprocess.run(
            _build_r_command(
                script_path,
                str(train_path),
                str(new_path),
                str(json_path),
                smooth_expr_r,
                "true" if absorb_cons else "false",
                "true" if scale_penalty else "false",
                knots_json,
                "" if deriv is None else str(int(deriv)),
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return _decode_packed_matrix_payload(result)


def _run_mgcv_predict_on_newdata(
    data: pd.DataFrame,
    newdata: pd.DataFrame,
    formula: str,
    *,
    family="gaussian",
    method="REML",
    type="link",
    return_se=False,
    unconditional=False,
    select: bool = False,
    weights_column: str | None = None,
    optimizer: str | None = None,
    terms=None,
    exclude=None,
    block_size=None,
    newdata_guaranteed=False,
    na_action=None,
    iterms_type=None,
    allow_live_run: bool = False,
):
    _family_nampy_unused, family_token = _family_specs(family)
    fit_method = "REML" if str(method).lower() == "fixed" else method
    formula_r = _normalize_python_formula_text(formula)

    cache_payload = {
            "version": _PREDICT_ON_NEWDATA_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "newdata": _df_fixture_repr(newdata),
            "formula_r": formula_r,
            "family_token": family_token,
            "fit_method": fit_method,
            "type": type,
            "return_se": return_se,
            "unconditional": unconditional,
            "select": select,
            "weights_column": weights_column,
            "optimizer": optimizer,
            "terms": terms,
            "exclude": exclude,
        }
    if (
        block_size is not None
        or bool(newdata_guaranteed)
        or na_action is not None
        or iterms_type is not None
    ):
        cache_payload["prediction_args"] = {
            "block_size": block_size,
            "newdata_guaranteed": bool(newdata_guaranteed),
            "na_action": na_action,
            "iterms_type": iterms_type,
        }
    _cache_key = _mgcv_fixture_key("predict_on_newdata", cache_payload)
    try:
        cached = _mgcv_fixture_load(_cache_key)
    except RuntimeError:
        if not allow_live_run:
            raise
        cached = None
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
want_unconditional <- identical(tolower(args[[8]]), "true")
select_flag <- identical(tolower(args[[9]]), "true")
weights_column <- args[[10]]
optimizer_name <- tolower(args[[11]])
terms_filter <- fromJSON(args[[12]])
exclude_filter <- fromJSON(args[[13]])
block_size <- if (tolower(args[[14]]) %in% c("none", "null", "")) NULL else as.integer(args[[14]])
newdata_guaranteed <- identical(tolower(args[[15]]), "true")
na_action_name <- tolower(args[[16]])
iterms_type <- if (tolower(args[[17]]) %in% c("none", "null", "")) NULL else as.integer(args[[17]])
coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}
serialize_numeric_object <- function(x) {
  if (is.null(x)) return(NULL)
  dims <- dim(x)
  if (is.null(dims) || length(dims) == 0) return(unname(as.numeric(x)))
  unname(as.matrix(x))
}
for (nm in names(train)) if (is.character(train[[nm]])) train[[nm]] <- factor(train[[nm]])
for (nm in names(newd)) {
  # Leave predictors that were factors at fit time as character here.
  # predict.gam performs its own level check and annotates genuinely new
  # levels before coercing to the training levels. Pre-coercing here turns
  # them into ordinary NA values and bypasses Predict.matrix.* xlev handling.
  if (is.character(newd[[nm]]) &&
      !(nm %in% names(train) && is.factor(train[[nm]]))) {
    newd[[nm]] <- factor(newd[[nm]])
  }
}
family_obj <- switch(
  strsplit(family_name, ":", fixed = TRUE)[[1]][1],
  gaussian = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    link <- if (length(family_parts) >= 2 && nzchar(family_parts[[2]])) family_parts[[2]] else "identity"
    gaussian(link = link)
  },
  binomial = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    link <- if (length(family_parts) >= 2) family_parts[[2]] else "logit"
    binomial(link = link)
  },
  poisson = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    link <- if (length(family_parts) >= 2 && nzchar(family_parts[[2]])) family_parts[[2]] else "log"
    poisson(link = link)
  },
  gamma = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    link <- if (length(family_parts) >= 2) family_parts[[2]] else "inverse"
    Gamma(link = link)
  },
  negbin = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    theta <- if (length(family_parts) >= 2) as.numeric(family_parts[[2]]) else 1.0
    link <- if (length(family_parts) >= 3 && nzchar(family_parts[[3]])) family_parts[[3]] else "log"
    do.call(mgcv::nb, list(theta = theta, link = link))
  },
  negbin_est = {
    family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
    theta <- if (length(family_parts) >= 2) as.numeric(family_parts[[2]]) else 1.0
    link <- if (length(family_parts) >= 3 && nzchar(family_parts[[3]])) family_parts[[3]] else "log"
    do.call(mgcv::nb, list(theta = -abs(theta), link = link))
  },
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family for newdata parity: %s", family_name))
)
fit_method <- if (tolower(method_name) == "gcv") "GCV.Cp" else method_name
gam_args <- list(
  formula = coerce_formula(formula_text),
  data = train,
  family = family_obj,
  method = fit_method,
  select = select_flag
)
if (!(tolower(weights_column) %in% c("none", "null", ""))) {
  gam_args$weights <- train[[weights_column]]
}
if (!(optimizer_name %in% c("none", "null", ""))) {
  if (optimizer_name == "efs") {
    gam_args$optimizer <- "efs"
  } else if (optimizer_name %in% c("outer_newton", "newton")) {
    gam_args$optimizer <- c("outer", "newton")
  } else {
    gam_args$optimizer <- c("outer", optimizer_name)
  }
}
fit <- do.call(gam, gam_args)
pred_args <- list(
    object = fit,
    newdata = newd,
    type = pred_type,
    se.fit = want_se,
    unconditional = want_unconditional,
    terms = terms_filter,
    exclude = exclude_filter
)
if (!is.null(block_size)) pred_args$block.size <- block_size
if (newdata_guaranteed) pred_args$newdata.guaranteed <- TRUE
if (!(na_action_name %in% c("none", "null", ""))) {
  pred_args$na.action <- switch(
    na_action_name,
    pass = na.pass,
    omit = na.omit,
    exclude = na.exclude,
    fail = na.fail,
    stop(sprintf("unsupported na.action: %s", na_action_name))
  )
}
if (!is.null(iterms_type)) pred_args$iterms.type <- iterms_type
pred <- do.call(predict, pred_args)
out <- list()
if (pred_type %in% c("terms", "iterms")) {
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
  out$pred <- serialize_numeric_object(pred$fit)
  out$se <- serialize_numeric_object(pred$se.fit)
} else {
  out$pred <- serialize_numeric_object(pred)
}
write_json(
  out,
  args[[18]],
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
            _build_r_command(
                script_path,
                str(train_path),
                str(new_path),
                formula_r,
                family_token,
                fit_method,
                type,
                "true" if return_se else "false",
                "true" if unconditional else "false",
                "true" if select else "false",
                "NULL" if weights_column is None else str(weights_column),
                "NULL" if optimizer is None else str(optimizer),
                json.dumps(terms),
                json.dumps(exclude),
                "NULL" if block_size is None else str(int(block_size)),
                "true" if newdata_guaranteed else "false",
                "NULL" if na_action is None else str(na_action),
                "NULL" if iterms_type is None else str(int(iterms_type)),
                str(json_path),
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
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
    formula_r = _normalize_python_formula_text(formula)

    _cache_key = _mgcv_fixture_key(
        "fixed_sp_score",
        {
            "version": _FIXED_SP_SCORE_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "formula": formula_r,
            "family_token": family_token,
            "method": method,
            "select": select,
            "smoothing_params": sp_list,
        },
    )
    cached = _mgcv_fixture_load(_cache_key)
    if cached is not None:
        return cached

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
formula_text <- args[[2]]
coerce_formula <- function(x) {
  obj <- eval(parse(text = x))
  if (is.character(obj)) {
    if (length(obj) == 1) return(as.formula(obj))
    return(lapply(obj, as.formula))
  }
  obj
}
family_name <- tolower(args[[3]])
method_name <- args[[4]]
select_flag <- tolower(args[[5]]) == "true"
sp <- as.numeric(fromJSON(args[[6]]))
family_parts <- strsplit(family_name, ":", fixed = TRUE)[[1]]
family_key <- family_parts[[1]]
family_param <- if (length(family_parts) >= 2) family_parts[[2]] else NULL
family_link <- function(default, index = 2L) {
  if (length(family_parts) >= index && nzchar(family_parts[[index]])) {
    family_parts[[index]]
  } else {
    default
  }
}
family_obj <- switch(
  family_key,
  gaussian = {
    link <- family_link("identity")
    gaussian(link = link)
  },
  binomial = {
    link <- family_link("logit")
    binomial(link = link)
  },
  poisson = {
    link <- family_link("log")
    poisson(link = link)
  },
  gamma = {
    link <- family_link("inverse")
    Gamma(link = link)
  },
  negbin = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    link <- family_link("log", 3L)
    do.call(mgcv::nb, list(theta = theta, link = link))
  },
  negbin_est = {
    theta <- if (is.null(family_param)) 1.0 else as.numeric(family_param)
    link <- family_link("log", 3L)
    do.call(mgcv::nb, list(theta = -abs(theta), link = link))
  },
  gaulss = mgcv::gaulss(),
  gammals = mgcv::gammals(),
  stop(sprintf("Unsupported family for fixed-sp score: %s", family_name))
)
fit <- gam(
  formula = coerce_formula(formula_text),
  data = d,
  family = family_obj,
  method = method_name,
  select = select_flag,
  sp = sp
)
log_sp_ref <- log(pmax(sp, 1e-300))
eval_at_log_sp <- function(log_sp) {
  fixed_fit <- gam(
    formula = coerce_formula(formula_text),
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
            _build_r_command(
                script_path,
                str(csv_path),
                formula_r,
                family_token,
                method,
                "true" if select else "false",
                json.dumps(sp_list),
                str(json_path),
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _run_mgcv_gam_setup_assembly(
    data: pd.DataFrame,
    formula: str,
    family,
    method: str,
    *,
    select: bool = False,
    allow_live_run: bool = False,
):
    _family_nampy, family_token = _family_specs(family)
    del _family_nampy
    formula_r = _normalize_python_formula_text(formula)

    _cache_key = _mgcv_fixture_key(
        "gam_setup_assembly",
        {
            "version": _GAM_SETUP_ASSEMBLY_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "formula": formula_r,
            "family_token": family_token,
            "method": method,
            "select": select,
        },
    )
    try:
        cached = _mgcv_fixture_load(_cache_key)
    except RuntimeError:
        if not allow_live_run:
            raise
        cached = None
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "gam_setup_assembly.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            _build_r_command(
                MGCV_GAM_SETUP_ASSEMBLY_SCRIPT,
                str(csv_path),
                str(json_path),
                formula_r,
                family_token,
                method,
                "true" if select else "false",
            ),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(_cache_key, result)
    return result


def _assert_basic_mgcv_parity(
    actual,
    expected,
    *,
    pred_atol,
    pred_rtol,
    sp_log_atol,
    check_sp=True,
    check_sp_count=True,
    check_edf_by_term=True,
    check_criterion=True,
    criterion_atol=0.5,
):
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    a_sp = np.atleast_1d(np.asarray(a_fit["smoothing_params"], dtype=np.float64))
    e_sp = np.atleast_1d(np.asarray(e_fit["smoothing_params"], dtype=np.float64))

    if check_sp_count:
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
    if check_edf_by_term:
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
    """Compare matrices up to independent, explicitly identified column signs."""
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
    "_make_negbin_data",
    "_make_poisson_data",
    "_make_random_effect_data",
    "_make_random_effect_data_noisy",
    "_make_sz_data",
    "_make_sz_data_3x3",
    "_run_mgcv_anova",
    "_run_mgcv_natparam_cr",
    "_run_mgcv_natparam_type3",
    "_run_mgcv_fixed_sp_score",
    "_run_mgcv_gam_setup_assembly",
    "_run_mgcv_gam_vcomp",
    "_run_mgcv_predict_on_newdata",
    "_run_mgcv_smoothcon_predict_matrix",
    "_run_mgcv_smoothcon_matrix",
    "_run_mgcv_smoothcon_matrix_unscaled",
    "_run_mgcv_smoothcon_penalties",
    "_run_mgcv_snapshot",
]
