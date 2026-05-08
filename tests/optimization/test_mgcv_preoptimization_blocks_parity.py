import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.mgcv_invariant_policy import (
    gam_setup_compares_dominant_penalty_spectrum,
    penalty_spectrum,
    preoptimization_blocks_align_basis_columns,
    preoptimization_blocks_compare_range_root_representation,
)
from tests.mgcv_parity_utils import (
    _family_specs,
    _make_binomial_data,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data_noisy,
    _make_sz_data,
)

R_SCRIPT = shutil.which("Rscript")
MGCV_PREOPT_SCRIPT = PARITY_DIR / "mgcv_preoptimization_blocks.R"


def _make_gaussian_offset_data(seed=321, n=180):
    data = _make_gaussian_data(seed=seed, n=n).copy()
    rng = np.random.default_rng(seed + 7000)
    off = rng.normal(scale=0.3, size=n)
    data["off"] = off
    data["y"] = np.asarray(data["y"], dtype=np.float64) + off
    return data


def _make_gaussian_univariate_data(seed=301, n=180):
    data = _make_gaussian_data(seed=seed, n=n)[["y", "x0"]].copy()
    return data.rename(columns={"x0": "x"})


def _make_cyclic_data(seed=77, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2 * np.pi, size=n)
    y = np.sin(x) + 0.3 * np.cos(2 * x) + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_ps_data(seed=81, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(1.3 * x) + 0.2 * x**2 + rng.normal(scale=0.14, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_tp_ts_data(seed=111, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(0.8 * x0) + 0.35 * x0 * x1 + 0.2 * x1**2 + rng.normal(scale=0.12, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_numeric_by_data(seed=101, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(x) * z + 0.2 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _make_poisson_by_data(seed=105, n=220):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    eta = 0.1 + 0.6 * np.sin(x) * z
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _make_negbin_tp_data(seed=116, n=220, theta=2.0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    eta = 0.2 + 0.45 * np.sin(0.8 * x0) + 0.25 * x0 * x1
    mu = np.exp(eta)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_factor_by_data(seed=107, n=240):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    shifts = {"a": 0.6, "b": -0.35, "c": 0.1}
    slopes = {"a": 1.0, "b": -0.7, "c": 0.4}
    y = (
        np.array([shifts[str(v)] for v in f], dtype=np.float64)
        + np.sin(x) * np.array([slopes[str(v)] for v in f], dtype=np.float64)
        + rng.normal(0.0, 0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x": x, "f": f})


def _make_fs_numeric_by_data(seed=381, n=120):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(0.5, 1.5, size=n)
    f = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    shifts = {"a": 0.35, "b": -0.25, "c": 0.15}
    y = z * (np.sin(1.4 * x) + np.array([shifts[str(v)] for v in f]))
    y = y + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "f": f})


def _make_random_slope_data(seed=109, n_levels=8, n_rep=18):
    rng = np.random.default_rng(seed)
    f = np.repeat([f"g{i}" for i in range(n_levels)], n_rep)
    x = rng.uniform(-1.2, 1.2, size=f.size)
    intercepts = {f"g{i}": rng.normal(scale=0.35) for i in range(n_levels)}
    slopes = {f"g{i}": rng.normal(scale=0.55) for i in range(n_levels)}
    y = np.array([intercepts[str(level)] for level in f], dtype=np.float64)
    y += x * np.array([slopes[str(level)] for level in f], dtype=np.float64)
    y += rng.normal(scale=0.08, size=f.size)
    return pd.DataFrame({"y": y, "x": x, "f": pd.Categorical(f)})


PREOPT_CASES = [
    (
        "gaussian_two_cr",
        _make_gaussian_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_offset_two_cr",
        _make_gaussian_offset_data,
        'y ~ offset(off) + s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_linked_id_two_cr",
        _make_gaussian_data,
        'y ~ s(x0, bs="cr", k=8, id="g") + s(x1, bs="cr", k=8, id="g")',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_select_true_two_cr",
        _make_gaussian_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gaussian",
        "REML",
        True,
        False,
    ),
    (
        "gaussian_cs_uni",
        _make_gaussian_univariate_data,
        'y ~ s(x, bs="cs", k=8, sp=1.1)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "binomial_two_cr",
        _make_binomial_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "binomial",
        "REML",
        False,
        False,
    ),
    (
        "binomial_select_true_two_cr",
        _make_binomial_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "binomial",
        "REML",
        True,
        False,
    ),
    (
        "poisson_two_cr",
        _make_poisson_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "poisson",
        "REML",
        False,
        False,
    ),
    (
        "poisson_numeric_by_cr",
        _make_poisson_by_data,
        'y ~ s(x, by=z, bs="cr", k=8)',
        "poisson",
        "REML",
        False,
        False,
    ),
    (
        "poisson_select_true_two_cr",
        _make_poisson_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "poisson",
        "REML",
        True,
        False,
    ),
    (
        "gamma_two_cr",
        _make_gamma_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
        "REML",
        False,
        False,
    ),
    (
        "gamma_ps_uni",
        lambda: _make_gamma_data(seed=1705, n=220)[["y", "x0"]].rename(
            columns={"x0": "x"}
        ),
        'y ~ s(x, bs="ps", k=10, sp=0.5)',
        "gamma",
        "REML",
        False,
        False,
    ),
    (
        "gamma_select_true_two_cr",
        _make_gamma_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
        "REML",
        True,
        False,
    ),
    (
        "negbin_two_cr",
        _make_negbin_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        {"name": "negbin", "theta": 2.0},
        "REML",
        False,
        False,
    ),
    (
        "negbin_est_fixed_sp",
        _make_negbin_data,
        'y ~ s(x0, bs="cr", k=8, sp=1.0)',
        {"name": "negbin", "theta": 2.0, "estimate_theta": True},
        "REML",
        False,
        False,
    ),
    (
        "negbin_select_true_two_cr",
        _make_negbin_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        {"name": "negbin", "theta": 2.0},
        "REML",
        True,
        False,
    ),
    (
        "gaussian_cc_uni",
        _make_cyclic_data,
        'y ~ s(x, bs="cc", k=9, sp=0.8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_ps_uni",
        _make_ps_data,
        'y ~ s(x, bs="ps", k=12, sp=0.5)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_tp_two_dim",
        _make_tp_ts_data,
        'y ~ s(x0, x1, bs="tp", k=15)',
        "gaussian",
        "REML",
        False,
        True,
    ),
    (
        "gaussian_ts_two_dim",
        lambda: _make_tp_ts_data(seed=112, n=180),
        'y ~ s(x0, x1, bs="ts", k=15)',
        "gaussian",
        "REML",
        False,
        True,
    ),
    (
        "gaussian_te_two_dim",
        _make_gaussian_data,
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_ti_two_dim",
        _make_gaussian_data,
        'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3])',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_fs",
        _make_fs_data,
        'y ~ s(f, x, bs="fs", k=6)',
        "gaussian",
        "REML",
        False,
        True,
    ),
    (
        "gaussian_fs_numeric_by",
        _make_fs_numeric_by_data,
        'y ~ s(f, x, bs="fs", by=z, k=5, xt="cr", sp=[0.9, 0.7, 0.5])',
        "gaussian",
        "REML",
        False,
        True,
    ),
    (
        "gaussian_sz",
        _make_sz_data,
        'y ~ s(f1, f2, x, bs="sz", k=6)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_random_effect",
        _make_random_effect_data_noisy,
        'y ~ s(f, bs="re")',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_random_slope_re",
        _make_random_slope_data,
        'y ~ s(x, f, bs="re")',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_numeric_by_cr",
        _make_numeric_by_data,
        'y ~ s(x, by=z, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
    (
        "gaussian_factor_by_cr",
        _make_factor_by_data,
        'y ~ f + s(x, by=f, bs="cr", k=8)',
        "gaussian",
        "REML",
        False,
        False,
    ),
]


_REQUIRED_PREOPT_FAMILIES = {"gaussian", "binomial", "poisson", "gamma", "negbin"}
_REQUIRED_PREOPT_SMOOTHS = {
    "cr",
    "cs",
    "cc",
    "ps",
    "tp",
    "ts",
    "te",
    "ti",
    "fs",
    "sz",
    "re",
}


def _formula_fragments(formula):
    if isinstance(formula, (tuple, list)):
        return [str(part) for part in formula]
    return [str(formula)]


def _smooths_in_formula(formula):
    text = " ".join(_formula_fragments(formula))
    smooths = set()
    for smooth in ("cr", "cs", "cc", "ps", "tp", "ts", "fs", "sz", "re"):
        if f'bs="{smooth}"' in text:
            smooths.add(smooth)
    if "te(" in text:
        smooths.add("te")
    if "ti(" in text:
        smooths.add("ti")
    return smooths


def _normalize_family_name(family):
    if isinstance(family, dict):
        return str(family.get("name", "")).lower()
    return str(family).lower()


def test_preoptimization_case_matrix_covers_supported_non_general_surface():
    """Verify that preoptimization case matrix covers supported non general surface."""
    families = {_normalize_family_name(case[3]) for case in PREOPT_CASES}
    assert families >= _REQUIRED_PREOPT_FAMILIES

    smooths = set()
    for case in PREOPT_CASES:
        smooths.update(_smooths_in_formula(case[2]))
    assert smooths >= _REQUIRED_PREOPT_SMOOTHS

    assert any(case[5] for case in PREOPT_CASES), "Missing select=True preopt coverage."
    assert any(
        not case[5] for case in PREOPT_CASES
    ), "Missing select=False preopt coverage."
    assert any(
        "offset(" in " ".join(_formula_fragments(case[2])) for case in PREOPT_CASES
    )
    assert any('id="' in " ".join(_formula_fragments(case[2])) for case in PREOPT_CASES)
    assert any("by=z" in " ".join(_formula_fragments(case[2])) for case in PREOPT_CASES)
    assert any("by=f" in " ".join(_formula_fragments(case[2])) for case in PREOPT_CASES)
    assert any(
        _normalize_family_name(case[3]) == "negbin"
        and not bool(case[3].get("estimate_theta", False))
        for case in PREOPT_CASES
        if isinstance(case[3], dict)
    )
    assert any(
        _normalize_family_name(case[3]) == "negbin"
        and bool(case[3].get("estimate_theta", False))
        for case in PREOPT_CASES
        if isinstance(case[3], dict)
    )
    assert any(
        's(x, f, bs="re")' in " ".join(_formula_fragments(case[2]))
        for case in PREOPT_CASES
    )


def _run_mgcv_preoptimization(data, formula, family, method, *, select=False):
    family_nampy, family_token = _family_specs(family)
    del family_nampy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "preopt.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_PREOPT_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family_token,
                method,
                "true" if select else "false",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_preoptimization(data, formula, family, method, *, select=False):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method=method,
        select=select,
    )
    gam.fit(data=data)
    return build_estimate_gam_setup_state(gam)


def _as_matrix_list(value):
    return [np.asarray(item, dtype=np.float64) for item in (value or [])]


def _normalize_optional_matrix(value):
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 0:
        return None
    if isinstance(value, list) and len(value) == 0:
        return None
    return np.asarray(value, dtype=np.float64)


def _assert_projector_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        rtol=0.0,
        atol=atol,
    )


def _assert_root_gram_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual.T @ actual, expected.T @ expected, rtol=0.0, atol=atol
    )


def _assert_root_crossprod_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual @ actual.T, expected @ expected.T, rtol=0.0, atol=atol
    )


def _positive_spectrum_values(matrix):
    spectrum = penalty_spectrum(matrix)
    scale = max(float(np.max(np.abs(spectrum))) if spectrum.size else 0.0, 1.0)
    return spectrum[spectrum > np.finfo(np.float64).eps**0.8 * scale]


def _assert_dominant_penalty_spectrum_close(actual, expected, *, atol):
    actual_pos = _positive_spectrum_values(actual)
    expected_pos = _positive_spectrum_values(expected)
    assert actual_pos.size == expected_pos.size
    assert actual_pos[0] > 0.0
    assert expected_pos[0] > 0.0
    assert actual_pos[0] < 0.1 * actual_pos[1]
    assert expected_pos[0] < 0.1 * expected_pos[1]
    np.testing.assert_allclose(
        actual_pos[1:],
        expected_pos[1:],
        rtol=0.0,
        atol=max(atol, 2e-4),
    )


def _assert_u1_subspaces_equal(actual, expected, *, q_range, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape

    q_range = int(q_range)
    if q_range > 0:
        _assert_projector_equal(actual[:, :q_range], expected[:, :q_range], atol=atol)

    q_null = int(actual.shape[1]) - q_range
    if q_null > 0:
        _assert_projector_equal(actual[:, q_range:], expected[:, q_range:], atol=atol)


def _align_columns_with_transform(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape

    aligned = actual.copy()
    transform = np.eye(actual.shape[1], dtype=np.float64)
    j = 0
    while j < actual.shape[1]:
        if np.linalg.norm(actual[:, j] - expected[:, j]) > np.linalg.norm(
            -actual[:, j] - expected[:, j]
        ):
            aligned[:, j] *= -1.0
            transform[j, j] = -1.0

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
                transform[j : j + 2, j : j + 2] = M
                j += 2
                continue
        j += 1

    np.testing.assert_allclose(aligned, expected, atol=atol, rtol=0.0)
    return aligned, transform


def _assert_preoptimization_parity(
    actual,
    expected,
    *,
    compare_dominant_penalty_spectrum: bool = False,
    compare_design_space_only=False,
    align_basis_columns=False,
    compare_range_root_repr=True,
    penalty_atol=1e-12,
    projector_atol=1e-10,
):
    actual_X = np.asarray(actual.X, dtype=np.float64)
    expected_X = np.asarray(expected["X"], dtype=np.float64)
    assert actual_X.shape == expected_X.shape
    basis_transform = None
    if align_basis_columns:
        _, basis_transform = _align_columns_with_transform(
            actual_X,
            expected_X,
            atol=1e-10,
        )
    elif compare_design_space_only:
        _assert_projector_equal(actual_X, expected_X, atol=projector_atol)
    else:
        np.testing.assert_allclose(
            actual_X,
            expected_X,
            rtol=0.0,
            atol=1e-12,
        )

    expected_offset = expected.get("offset", None)
    if expected_offset is None:
        assert actual.offset is None
    else:
        np.testing.assert_allclose(
            np.asarray(actual.offset, dtype=np.float64),
            np.asarray(expected_offset, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    np.testing.assert_array_equal(
        np.asarray(actual.off, dtype=np.int64),
        np.asarray(expected["off"], dtype=np.int64),
    )
    expected_S = _as_matrix_list(expected.get("S", []))
    assert len(actual.S) == len(expected_S)
    for off_i, a_S, e_S in zip(
        np.asarray(actual.off, dtype=np.int64), actual.S, expected_S, strict=True
    ):
        if basis_transform is not None:
            start = int(off_i) - 1
            stop = start + int(np.asarray(a_S, dtype=np.float64).shape[0])
            local_transform = basis_transform[start:stop, start:stop]
            a_S = (
                local_transform.T @ np.asarray(a_S, dtype=np.float64) @ local_transform
            )
        if compare_dominant_penalty_spectrum:
            _assert_dominant_penalty_spectrum_close(
                a_S,
                e_S,
                atol=penalty_atol,
            )
        else:
            np.testing.assert_allclose(a_S, e_S, rtol=0.0, atol=penalty_atol)
    np.testing.assert_array_equal(
        np.asarray(actual.rank, dtype=np.int64),
        np.asarray(expected["rank"], dtype=np.int64),
    )

    expected_L = _normalize_optional_matrix(expected.get("L", None))
    if expected_L is None:
        assert actual.L is None
    else:
        np.testing.assert_allclose(
            np.asarray(actual.L, dtype=np.float64),
            expected_L,
            rtol=0.0,
            atol=1e-12,
        )

    np.testing.assert_allclose(
        np.asarray(actual.lsp0, dtype=np.float64),
        np.asarray(expected["lsp0"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.sp, dtype=np.float64),
        np.asarray(expected["sp"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.log_sp_full, dtype=np.float64),
        np.asarray(expected["log_sp_full"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )

    expected_rS = _as_matrix_list(expected.get("rS", []))
    assert len(actual.rS) == len(expected_rS)
    for a_rS, e_rS in zip(actual.rS, expected_rS, strict=True):
        if basis_transform is not None:
            a_rS = basis_transform.T @ np.asarray(a_rS, dtype=np.float64)
        if compare_dominant_penalty_spectrum:
            _assert_dominant_penalty_spectrum_close(
                np.asarray(a_rS, dtype=np.float64) @ np.asarray(a_rS, dtype=np.float64).T,
                np.asarray(e_rS, dtype=np.float64) @ np.asarray(e_rS, dtype=np.float64).T,
                atol=penalty_atol,
            )
        else:
            _assert_root_crossprod_equal(a_rS, e_rS, atol=penalty_atol)

    expected_Y = np.asarray(expected["Y"], dtype=np.float64)
    expected_Z = np.asarray(expected["Z"], dtype=np.float64)
    expected_E = np.asarray(expected["E"], dtype=np.float64)
    expected_Eb = np.asarray(expected["Eb"], dtype=np.float64)
    expected_U1 = np.asarray(expected["U1"], dtype=np.float64)

    assert actual.Y.shape == expected_Y.shape
    assert actual.Z.shape == expected_Z.shape
    assert actual.E.shape == expected_E.shape
    assert actual.Eb.shape == expected_Eb.shape
    assert actual.U1.shape == expected_U1.shape
    assert actual.Mp == int(expected["Mp"])

    _assert_u1_subspaces_equal(
        actual.U1,
        expected_U1,
        q_range=expected_E.shape[0],
        atol=projector_atol,
    )
    _assert_projector_equal(actual.Y, expected_Y)
    _assert_projector_equal(actual.Z, expected_Z)
    if compare_range_root_repr:
        if compare_dominant_penalty_spectrum:
            _assert_dominant_penalty_spectrum_close(
                np.asarray(actual.E, dtype=np.float64).T
                @ np.asarray(actual.E, dtype=np.float64),
                expected_E.T @ expected_E,
                atol=penalty_atol,
            )
            _assert_dominant_penalty_spectrum_close(
                np.asarray(actual.Eb, dtype=np.float64).T
                @ np.asarray(actual.Eb, dtype=np.float64),
                expected_Eb.T @ expected_Eb,
                atol=penalty_atol,
            )
        else:
            _assert_root_gram_equal(actual.E, expected_E)
            _assert_root_gram_equal(actual.Eb, expected_Eb)

    expected_UrS = _as_matrix_list(expected.get("UrS", []))
    assert len(actual.UrS) == len(expected_UrS)
    if compare_range_root_repr:
        for a_UrS, e_UrS in zip(actual.UrS, expected_UrS, strict=True):
            assert a_UrS.shape == e_UrS.shape
            if compare_dominant_penalty_spectrum:
                _assert_dominant_penalty_spectrum_close(
                    np.asarray(a_UrS, dtype=np.float64).T
                    @ np.asarray(a_UrS, dtype=np.float64),
                    np.asarray(e_UrS, dtype=np.float64).T
                    @ np.asarray(e_UrS, dtype=np.float64),
                    atol=penalty_atol,
                )
            else:
                _assert_root_gram_equal(a_UrS, e_UrS, atol=penalty_atol)


@pytest.mark.parametrize(
    "case_id, data_factory, formula, family, method, select, compare_design_space_only",
    PREOPT_CASES,
    ids=[case[0] for case in PREOPT_CASES],
)
def test_preoptimization_blocks_match_mgcv(
    case_id, data_factory, formula, family, method, select, compare_design_space_only
):
    """Verify that preoptimization blocks match mgcv."""
    data = data_factory()
    actual = _fit_nampy_preoptimization(
        data,
        formula,
        family,
        method,
        select=select,
    )
    expected = _run_mgcv_preoptimization(
        data,
        formula,
        family,
        method,
        select=select,
    )

    align_basis_columns = preoptimization_blocks_align_basis_columns(case_id)
    compare_range_root_repr = preoptimization_blocks_compare_range_root_representation(
        case_id
    )
    projector_atol = 1e-9 if case_id == "gaussian_fs" else 1e-10
    if case_id == "gaussian_fs":
        penalty_atol = 1e-8
    elif align_basis_columns:
        penalty_atol = 1e-10
    else:
        penalty_atol = 1e-12

    _assert_preoptimization_parity(
        actual,
        expected,
        compare_dominant_penalty_spectrum=gam_setup_compares_dominant_penalty_spectrum(
            case_id
        ),
        compare_design_space_only=compare_design_space_only,
        align_basis_columns=align_basis_columns,
        compare_range_root_repr=compare_range_root_repr,
        penalty_atol=penalty_atol,
        projector_atol=projector_atol,
    )
