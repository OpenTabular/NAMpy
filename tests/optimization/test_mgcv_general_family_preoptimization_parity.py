import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
)
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gammals_by_data,
    _gammals_data,
    _gaulss_by_data,
    _gaulss_data,
)
from tests.mgcv_parity_utils import _family_specs, _fit_nampy_model_fixed_sp

R_SCRIPT = shutil.which("Rscript")
MGCV_GENERAL_PREOPT_SCRIPT = PARITY_DIR / "mgcv_general_family_preoptimization.R"
MGCV_INITIAL_SPG_SCRIPT = PARITY_DIR / "mgcv_initial_spg.R"


def _run_mgcv_general_preoptimization(
    data, formula, family, method, *, select=False, sp=None
):
    family_nampy, family_token = _family_specs(family)
    del family_nampy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "general_preopt.json"
        data.to_csv(csv_path, index=False)
        command = [
            R_SCRIPT,
            str(MGCV_GENERAL_PREOPT_SCRIPT),
            str(csv_path),
            str(json_path),
            str(formula),
            family_token,
            method,
            "true" if select else "false",
        ]
        if sp is not None:
            command.append(json.dumps(np.asarray(sp, dtype=np.float64).tolist()))
        subprocess.run(
            command,
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _run_mgcv_initial_spg(data, formula, family, method, *, select=False):
    family_nampy, family_token = _family_specs(family)
    del family_nampy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "initial_spg.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_INITIAL_SPG_SCRIPT),
                str(csv_path),
                str(json_path),
                str(formula),
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


def _as_matrix(value):
    return np.asarray(value, dtype=np.float64)


def _as_matrix_list(value):
    return [np.asarray(item, dtype=np.float64) for item in (value or [])]


def _gaulss_fs_data():
    rng = np.random.default_rng(248)
    n = 132
    row = np.arange(n)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    f = np.asarray(["a", "b", "c"])[row % 3]
    f1 = np.asarray(["u", "v", "w"])[row % 3]
    f2 = np.asarray(["left", "right"])[(row // 3) % 2]
    f_effect = np.asarray(
        [{"a": -0.3, "b": 0.05, "c": 0.25}[value] for value in f]
    )
    cell_effect = np.asarray(
        [
            {
                ("u", "left"): -0.16,
                ("u", "right"): 0.08,
                ("v", "left"): 0.18,
                ("v", "right"): -0.1,
                ("w", "left"): 0.06,
                ("w", "right"): -0.04,
            }[(left, right)]
            for left, right in zip(f1, f2, strict=True)
        ]
    )
    signal = 0.3 * np.sin(1.4 * x0) + f_effect + cell_effect
    sigma = np.exp(-0.45 + 0.12 * x0)
    y = rng.normal(signal, sigma, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "f": f, "f1": f1, "f2": f2})


def _assert_root_gram_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual.T @ actual,
        expected.T @ expected,
        rtol=0.0,
        atol=atol,
    )


def _assert_matrix_space_equal(actual, expected, *, atol=1e-10):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(
        actual @ actual.T,
        expected @ expected.T,
        rtol=0.0,
        atol=atol,
    )


def _assert_sl_block_parity(actual, expected, *, s_atol=5e-12):
    assert actual.start == int(expected["start"])
    assert actual.stop == int(expected["stop"])
    assert actual.repara is bool(expected["repara"])
    assert actual.linear is bool(expected["linear"])
    if expected.get("rank", None) is None:
        assert actual.rank is None
    else:
        assert int(actual.rank) == int(expected["rank"])
    np.testing.assert_allclose(
        np.asarray(actual.lambda_, dtype=np.float64),
        np.asarray(expected["lambda"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    if expected.get("ldet", None) is not None:
        assert float(actual.ldet) == pytest.approx(float(expected["ldet"]), abs=1e-12)

    expected_ind = expected.get("ind", None)
    if expected_ind is None:
        assert actual.ind is None
    else:
        np.testing.assert_array_equal(
            np.asarray(actual.ind, dtype=bool),
            np.asarray(expected_ind, dtype=bool),
        )

    expected_S = _as_matrix_list(expected.get("S", []))
    assert len(actual.S) == len(expected_S)
    for a_S, e_S in zip(actual.S, expected_S, strict=True):
        np.testing.assert_allclose(
            np.asarray(a_S, dtype=np.float64),
            e_S,
            rtol=0.0,
            atol=s_atol,
        )

    expected_D = expected.get("D", None)
    if expected_D is None:
        assert actual.D is None
    else:
        actual_D = np.asarray(actual.D, dtype=np.float64)
        expected_D = np.asarray(expected_D, dtype=np.float64)
        if actual_D.ndim == 1:
            np.testing.assert_allclose(actual_D, expected_D, rtol=0.0, atol=1e-12)
        else:
            _assert_matrix_space_equal(actual_D, expected_D)

    expected_Di = expected.get("Di", None)
    if expected_Di is None:
        assert actual.Di is None
    else:
        actual_Di = np.asarray(actual.Di, dtype=np.float64)
        expected_Di = np.asarray(expected_Di, dtype=np.float64)
        if actual_Di.ndim == 1:
            np.testing.assert_allclose(actual_Di, expected_Di, rtol=0.0, atol=1e-12)
        else:
            _assert_root_gram_equal(actual_Di, expected_Di)

    expected_rS = _as_matrix_list(expected.get("rS", []))
    assert len(actual.rS) == len(expected_rS)
    for a_rS, e_rS in zip(actual.rS, expected_rS, strict=True):
        _assert_root_gram_equal(np.asarray(a_rS, dtype=np.float64), e_rS)

    expected_St = expected.get("St", None)
    if expected_St is None:
        assert actual.St is None
    else:
        np.testing.assert_allclose(
            np.asarray(actual.St, dtype=np.float64),
            np.asarray(expected_St, dtype=np.float64),
            rtol=0.0,
            atol=5e-12,
        )


def _assert_sl_setup_parity(actual, expected, *, s_atol=5e-12):
    assert len(actual) == len(expected["blocks"])
    for a_block, e_block in zip(
        list(actual), list(expected["blocks"]), strict=True
    ):
        _assert_sl_block_parity(a_block, e_block, s_atol=s_atol)

    _assert_root_gram_equal(
        np.asarray(actual.E, dtype=np.float64),
        np.asarray(expected["E"], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(actual.S, dtype=np.float64),
        np.asarray(expected["S"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.lambda_, dtype=np.float64),
        np.asarray(expected["lambda"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert bool(actual.cholesky) is bool(expected["cholesky"])


def _assert_general_fit5_setup_parity(
    actual,
    expected,
    *,
    compare_x_space_only=False,
    st_rtol=2e-15,
    s_block_atol=5e-12,
):
    actual_X_full = np.asarray(actual.X_full, dtype=np.float64)
    expected_X_full = np.asarray(expected["X_full"], dtype=np.float64)
    assert actual_X_full.shape == expected_X_full.shape
    if compare_x_space_only:
        _assert_matrix_space_equal(actual_X_full, expected_X_full)
        _assert_matrix_space_equal(
            np.asarray(actual.X_initial, dtype=np.float64),
            np.asarray(expected["X_initial"], dtype=np.float64),
        )
    else:
        np.testing.assert_allclose(
            actual_X_full,
            expected_X_full,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(actual.X_initial, dtype=np.float64),
            np.asarray(expected["X_initial"], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    assert len(actual.jj) == len(expected["jj"])
    for a_jj, e_jj in zip(actual.jj, expected["jj"], strict=True):
        np.testing.assert_array_equal(
            np.asarray(a_jj, dtype=np.int64),
            np.asarray(e_jj, dtype=np.int64),
        )

    expected_offsets = expected.get("offset_list", None)
    if expected_offsets is None:
        assert actual.offset_list is None
    else:
        assert actual.offset_list is not None
        assert len(actual.offset_list) == len(expected_offsets)
        for a_off, e_off in zip(
            actual.offset_list, expected_offsets, strict=True
        ):
            if e_off is None:
                assert a_off is None
            else:
                np.testing.assert_allclose(
                    np.asarray(a_off, dtype=np.float64),
                    np.asarray(e_off, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )

    np.testing.assert_allclose(
        np.asarray(actual.smoothing_params, dtype=np.float64),
        np.asarray(expected["smoothing_params"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.log_sp, dtype=np.float64),
        np.asarray(expected["log_sp"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(actual.St, dtype=np.float64),
        np.asarray(expected["St"], dtype=np.float64),
        rtol=st_rtol,
        atol=2e-8,
    )

    expected_S_blocks = _as_matrix_list(expected.get("S_blocks", []))
    assert len(actual.S_blocks) == len(expected_S_blocks)
    for a_S, e_S in zip(actual.S_blocks, expected_S_blocks, strict=True):
        np.testing.assert_allclose(
            np.asarray(a_S, dtype=np.float64),
            e_S,
            rtol=0.0,
            atol=s_block_atol,
        )

    assert float(actual.ldetS) == pytest.approx(float(expected["ldetS"]), abs=1e-10)
    np.testing.assert_allclose(
        np.asarray(actual.ldetS1, dtype=np.float64),
        np.asarray(expected["ldetS1"], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(actual.ldetS2, dtype=np.float64),
        np.asarray(expected["ldetS2"], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )
    assert int(actual.Mp) == int(expected["Mp"])
    assert str(actual.score_type) == str(expected["score_type"])
    _assert_sl_setup_parity(actual.Sl, expected["Sl"], s_atol=s_block_atol)


GENERAL_PREOPT_CASES = [
    ("gaulss_cr", "gaulss", GAULSS_FORMULA, _gaulss_data, "ML", False, True),
    (
        "gaulss_fs",
        "gaulss",
        ['y ~ s(f, x0, bs="fs", k=5, xt="cr")', "~ 1"],
        _gaulss_fs_data,
        "ML",
        False,
        True,
    ),
    (
        "gaulss_select_true_cr",
        "gaulss",
        GAULSS_FORMULA,
        _gaulss_data,
        "ML",
        True,
        True,
    ),
    (
        "gaulss_numeric_by",
        "gaulss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gaulss_by_data,
        "ML",
        False,
        True,
    ),
    (
        "gammals_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        False,
        True,
    ),
    (
        "gammals_select_true_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        True,
        True,
    ),
    (
        "gammals_numeric_by",
        "gammals",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gammals_by_data,
        "ML",
        False,
        True,
    ),
]


_GENERAL_FAMILY_SET = {"gaulss", "gammals"}


def test_general_family_preoptimization_case_matrix_covers_requested_surface():
    """
    Verify that general family preoptimization case matrix covers requested surface.
    """
    families = {case[1] for case in GENERAL_PREOPT_CASES}
    assert families >= _GENERAL_FAMILY_SET

    for family in _GENERAL_FAMILY_SET:
        family_cases = [case for case in GENERAL_PREOPT_CASES if case[1] == family]
        ids = {case[0] for case in family_cases}
        assert any(case_id.endswith("_cr") for case_id in ids)
        assert any("select_true" in case_id for case_id in ids)
        assert any("numeric_by" in case_id for case_id in ids)


@pytest.mark.parametrize(
    (
        "case_id",
        "family",
        "formula",
        "data_factory",
        "method",
        "select",
        "compare_x_space_only",
    ),
    GENERAL_PREOPT_CASES,
    ids=[case[0] for case in GENERAL_PREOPT_CASES],
)
def test_general_family_preoptimization_setup_matches_mgcv(
    case_id, family, formula, data_factory, method, select, compare_x_space_only
):
    """Verify that general family preoptimization setup matches mgcv."""
    data = data_factory()
    expected = _run_mgcv_general_preoptimization(
        data, formula, family, method, select=select
    )
    sp = np.asarray(expected["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)
    actual = build_general_family_setup_state(gam, sp, score_type=method)
    st_rtol = 2e-15
    s_block_atol = 5e-12

    _assert_general_fit5_setup_parity(
        actual,
        expected,
        compare_x_space_only=compare_x_space_only,
        st_rtol=st_rtol,
        s_block_atol=s_block_atol,
    )


def test_gaulss_fs_fixed_sp_preoptimization_setup_matches_mgcv():
    """Keep unequal fs penalties aligned before gaulss initialization."""
    data = _gaulss_fs_data()
    formula = ['y ~ s(f, x0, bs="fs", k=5, xt="cr")', "~ 1"]
    sp = np.array([0.7, 0.9, 1.1], dtype=np.float64)
    expected = _run_mgcv_general_preoptimization(
        data,
        formula,
        "gaulss",
        "REML",
        sp=sp,
    )
    gam = _fit_nampy_model_fixed_sp(data, formula, "gaulss", sp)
    actual = build_general_family_setup_state(gam, sp, score_type="REML")

    _assert_general_fit5_setup_parity(
        actual,
        expected,
        compare_x_space_only=True,
    )


def test_gammals_select_true_initial_spg_matches_mgcv():
    """Keep the optimized select=True path on mgcv's two-penalty start."""
    data = _gammals_data()
    formula = ['y ~ s(x, bs="cr", k=6)', "~ 1"]
    expected = _run_mgcv_initial_spg(
        data,
        formula,
        "gammals",
        "ML",
        select=True,
    )

    gam = GAM(
        formula=formula,
        family="gammals",
        select=True,
        optimize_smoothing=False,
        smoothing_method="ML",
    )
    gam.fit(data=data)
    actual = _initial_smoothing_params_from_design(gam, gam.y_)

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected["initial_sp"], dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
