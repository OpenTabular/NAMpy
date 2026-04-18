import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.fit.solvers.general_fit5 import build_gam_fit5_setup_state
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gammals_by_data,
    _gammals_data,
    _gammals_tensor_data,
    _gammals_two_smooth_data,
    _gaulss_by_data,
    _gaulss_data,
    _gaulss_tensor_data,
    _gaulss_two_smooth_data,
    _gevlss_by_data,
    _gevlss_data,
    _gevlss_tensor_data,
    _gevlss_two_smooth_data,
    _shashlss_by_data,
    _shashlss_data,
    _shashlss_tensor_data,
    _shashlss_two_smooth_data,
    _ziplss_by_data,
    _ziplss_data,
    _ziplss_tensor_data,
    _ziplss_two_smooth_data,
)
from tests.mgcv_parity_utils import _family_specs, _fit_nampy_model_fixed_sp

R_SCRIPT = shutil.which("Rscript")
MGCV_GENERAL_PREOPT_SCRIPT = PARITY_DIR / "mgcv_general_family_preoptimization.R"


def _run_mgcv_general_preoptimization(data, formula, family, method, *, select=False):
    family_nampy, family_token = _family_specs(family)
    del family_nampy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "general_preopt.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_GENERAL_PREOPT_SCRIPT),
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
    for a_S, e_S in zip(actual.S, expected_S):
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
    for a_rS, e_rS in zip(actual.rS, expected_rS):
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
    for a_block, e_block in zip(list(actual), list(expected["blocks"])):
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
    else:
        np.testing.assert_allclose(
            actual_X_full,
            expected_X_full,
            rtol=0.0,
            atol=1e-12,
        )
    _assert_matrix_space_equal(
        np.asarray(actual.X_initial, dtype=np.float64),
        np.asarray(expected["X_initial"], dtype=np.float64),
    )

    assert len(actual.jj) == len(expected["jj"])
    for a_jj, e_jj in zip(actual.jj, expected["jj"]):
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
        for a_off, e_off in zip(actual.offset_list, expected_offsets):
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
    for a_S, e_S in zip(actual.S_blocks, expected_S_blocks):
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
    ("gaulss_cr", "gaulss", GAULSS_FORMULA, _gaulss_data, "ML", False, False),
    (
        "gaulss_select_true_cr",
        "gaulss",
        GAULSS_FORMULA,
        _gaulss_data,
        "ML",
        True,
        False,
    ),
    (
        "gaulss_numeric_by",
        "gaulss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gaulss_by_data,
        "ML",
        False,
        False,
    ),
    (
        "gaulss_t2_full_false",
        "gaulss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _gaulss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gaulss_t2_full_true",
        "gaulss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _gaulss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gaulss_two_cr",
        "gaulss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gaulss_two_smooth_data,
        "ML",
        False,
        False,
    ),
    (
        "gammals_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        False,
        False,
    ),
    (
        "gammals_select_true_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _gammals_data,
        "ML",
        True,
        False,
    ),
    (
        "gammals_numeric_by",
        "gammals",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _gammals_by_data,
        "ML",
        False,
        False,
    ),
    (
        "gammals_t2_full_false",
        "gammals",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _gammals_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gammals_t2_full_true",
        "gammals",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _gammals_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gammals_two_cr",
        "gammals",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _gammals_two_smooth_data,
        "ML",
        False,
        False,
    ),
    (
        "gevlss_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "ML",
        False,
        False,
    ),
    (
        "gevlss_select_true_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_data,
        "ML",
        True,
        False,
    ),
    (
        "gevlss_numeric_by",
        "gevlss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_by_data,
        "ML",
        False,
        False,
    ),
    (
        "gevlss_t2_full_false",
        "gevlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1", "~ 1"],
        _gevlss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gevlss_t2_full_true",
        "gevlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1", "~ 1"],
        _gevlss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "gevlss_two_cr",
        "gevlss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1"],
        _gevlss_two_smooth_data,
        "ML",
        False,
        False,
    ),
    (
        "shashlss_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "ML",
        False,
        False,
    ),
    (
        "shashlss_select_true_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_data,
        "ML",
        True,
        False,
    ),
    (
        "shashlss_numeric_by",
        "shashlss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_by_data,
        "ML",
        False,
        False,
    ),
    (
        "shashlss_t2_full_false",
        "shashlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1", "~ 1", "~ 1"],
        _shashlss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "shashlss_t2_full_true",
        "shashlss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1", "~ 1", "~ 1"],
        _shashlss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "shashlss_two_cr",
        "shashlss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
        _shashlss_two_smooth_data,
        "ML",
        False,
        False,
    ),
    (
        "ziplss_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "ML",
        False,
        False,
    ),
    (
        "ziplss_select_true_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        _ziplss_data,
        "ML",
        True,
        False,
    ),
    (
        "ziplss_numeric_by",
        "ziplss",
        ['y ~ s(x, by=z, bs="cr", k=6)', "~ 1"],
        _ziplss_by_data,
        "ML",
        False,
        False,
    ),
    (
        "ziplss_t2_full_false",
        "ziplss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6])', "~ 1"],
        _ziplss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "ziplss_t2_full_true",
        "ziplss",
        ['y ~ t2(x0, x1, bs=["tp", "cr"], k=[6, 6], full=True)', "~ 1"],
        _ziplss_tensor_data,
        "ML",
        False,
        True,
    ),
    (
        "ziplss_two_cr",
        "ziplss",
        ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"],
        _ziplss_two_smooth_data,
        "ML",
        False,
        False,
    ),
]


_GENERAL_FAMILY_SET = {"gaulss", "gammals", "gevlss", "shashlss", "ziplss"}


def test_general_family_preoptimization_case_matrix_covers_requested_surface():
    families = {case[1] for case in GENERAL_PREOPT_CASES}
    assert families >= _GENERAL_FAMILY_SET

    for family in _GENERAL_FAMILY_SET:
        family_cases = [case for case in GENERAL_PREOPT_CASES if case[1] == family]
        ids = {case[0] for case in family_cases}
        assert any(case_id.endswith("_cr") for case_id in ids)
        assert any("select_true" in case_id for case_id in ids)
        assert any("numeric_by" in case_id for case_id in ids)
        assert any("t2_full_false" in case_id for case_id in ids)
        assert any("t2_full_true" in case_id for case_id in ids)
        assert any("two_cr" in case_id for case_id in ids)


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
    data = data_factory()
    expected = _run_mgcv_general_preoptimization(
        data, formula, family, method, select=select
    )
    sp = np.asarray(expected["smoothing_params"], dtype=np.float64)

    gam = _fit_nampy_model_fixed_sp(data, formula, family, sp, select=select)
    actual = build_gam_fit5_setup_state(gam, sp, score_type=method)
    st_rtol = 2e-15
    s_block_atol = 5e-12
    if case_id == "ziplss_t2_full_false":
        s_block_atol = 1e-11
    elif case_id == "ziplss_t2_full_true":
        st_rtol = 5e-15
        s_block_atol = 1e-11

    _assert_general_fit5_setup_parity(
        actual,
        expected,
        compare_x_space_only=compare_x_space_only,
        st_rtol=st_rtol,
        s_block_atol=s_block_atol,
    )
