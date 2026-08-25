"""Constructor, fit, prediction, and Sl parity for adaptive smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
)
from nampy.gam.smooths.univariate.ad import AdaptiveSmoothTerm
from tests.mgcv_parity_utils import (
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _fit_nampy_snapshot,
    _run_mgcv_raw_constructor,
    _run_mgcv_smoothcon_matrix,
    _run_mgcv_smoothcon_penalties,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def _adaptive_data(seed: int, *, two_dimensional: bool, n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.uniform(size=n), "z": rng.uniform(size=n)})
    return frame if two_dimensional else frame[["x"]]


@pytest.mark.parametrize(
    ("seed", "n", "formula", "features", "options"),
    [
        pytest.param(
            44,
            90,
            's(x,bs="ad",k=12,m=4)',
            ["x"],
            {"feature": "x", "k": 12, "m": 4},
            id="one-dimensional",
        ),
        pytest.param(
            45,
            90,
            's(x,z,bs="ad",k=[6,7],m=[2,2])',
            ["x", "z"],
            {"feature": ["x", "z"], "k": [6, 7], "m": [2, 2]},
            id="two-dimensional",
        ),
        pytest.param(
            61,
            100,
            's(x,bs="ad",k=13,m=4,xt={"bs":"cr"})',
            ["x"],
            {"feature": "x", "k": 13, "m": 4, "xt": {"bs": "cr"}},
            id="cubic-regression-base",
        ),
        pytest.param(
            62,
            100,
            's(x,bs="ad",k=13,m=4,xt={"bs":"cp"})',
            ["x"],
            {"feature": "x", "k": 13, "m": 4, "xt": {"bs": "cp"}},
            id="cyclic-pspline-base",
        ),
        pytest.param(
            63,
            100,
            's(x,bs="ad",k=13,m=4,xt={"bs":"cc"})',
            ["x"],
            {"feature": "x", "k": 13, "m": 4, "xt": {"bs": "cc"}},
            id="cyclic-cubic-base",
        ),
        pytest.param(
            64,
            110,
            's(x,z,bs="ad",k=[6,6],m=[3,3])',
            ["x", "z"],
            {"feature": ["x", "z"], "k": [6, 6], "m": [3, 3]},
            id="two-dimensional-default-penalty-basis",
        ),
        pytest.param(
            65,
            90,
            's(x,bs="ad",k=12,m=1)',
            ["x"],
            {"feature": "x", "k": 12, "m": 1},
            id="one-dimensional-nonadaptive-penalty",
        ),
    ],
)
def test_ad_raw_constructor_basis_penalties_and_ranks_match_mgcv(
    seed, n, formula, features, options
):
    data = _adaptive_data(seed, two_dimensional=len(features) == 2, n=n)
    term = AdaptiveSmoothTerm(constraint_mode="never", **options).fit(
        data.to_numpy(dtype=np.float64), features
    )
    expected = _run_mgcv_raw_constructor(data, formula)

    np.testing.assert_allclose(
        term._raw_basis_train, expected["X"], atol=2e-15, rtol=2e-15
    )
    assert len(term._raw_penalties) == len(expected["S"])
    for actual, reference in zip(term._raw_penalties, expected["S"], strict=True):
        np.testing.assert_allclose(actual, reference, atol=5e-15, rtol=5e-15)
    expected_rank = np.asarray(expected["rank"], dtype=int).reshape(-1).tolist()
    assert term.rank == expected_rank


@pytest.mark.parametrize(
    ("seed", "formula", "features", "options"),
    [
        pytest.param(
            44,
            's(x,bs="ad",k=12,m=4)',
            ["x"],
            {"feature": "x", "k": 12, "m": 4},
            id="one-dimensional",
        ),
        pytest.param(
            45,
            's(x,z,bs="ad",k=c(6,7),m=c(2,2))',
            ["x", "z"],
            {"feature": ["x", "z"], "k": [6, 7], "m": [2, 2]},
            id="two-dimensional",
        ),
    ],
)
def test_ad_absorbed_scaled_constructor_and_prediction_match_mgcv(
    seed, formula, features, options
):
    data = _adaptive_data(seed, two_dimensional=len(features) == 2)
    term = AdaptiveSmoothTerm(**options).fit(data.to_numpy(dtype=np.float64), features)
    expected_basis = np.asarray(
        _run_mgcv_smoothcon_matrix(data, formula)["X"], dtype=np.float64
    )
    expected_penalties = _run_mgcv_smoothcon_penalties(
        data, formula, absorb_cons=True, scale_penalty=True
    )["S"]

    np.testing.assert_allclose(term.basis_train, expected_basis, atol=3e-15, rtol=3e-15)
    for actual, reference in zip(term.penalties, expected_penalties, strict=True):
        np.testing.assert_allclose(actual, reference, atol=3e-15, rtol=3e-15)
    np.testing.assert_allclose(
        term.transform_new(data.to_numpy(dtype=np.float64)),
        term.basis_train,
        atol=3e-15,
        rtol=3e-15,
    )


def test_ad_reml_fit_and_public_summary_match_mgcv():
    rng = np.random.default_rng(51)
    x = np.sort(rng.uniform(size=150))
    data = pd.DataFrame(
        {"x": x, "y": np.sin(5.0 * x) + rng.normal(scale=0.15, size=x.size)}
    )
    formula = 'y ~ s(x, bs="ad", k=14, m=4)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-5,
        pred_rtol=1e-5,
        edf_atol=1e-5,
        criterion_atol=1e-5,
        sp_atol=1e-6,
        sp_rtol=1e-6,
        log_sp_atol=1e-5,
    )
    summary = _fit_nampy_model(data, formula, "gaussian", "REML").summary()
    assert list(summary.s_table["label"]) == ["s(x)"]


def test_ad_multi_penalty_term_materializes_as_one_general_family_sl_block():
    rng = np.random.default_rng(55)
    x = np.linspace(-1.25, 1.25, 120)
    mu = 0.3 + np.sin(np.pi * x)
    sigma = np.exp(-0.35 + 0.25 * x)
    data = pd.DataFrame({"y": rng.normal(mu, sigma), "x": x})
    formula = ['y ~ s(x, bs="ad", k=12, m=4)', "~ 1"]
    smoothing = np.array([0.2, 0.5, 1.0, 2.0])
    model = _fit_nampy_model_fixed_sp(data, formula, "gaulss", smoothing)
    setup = build_general_family_setup_state(model, smoothing, score_type="REML")

    assert len(setup.Sl) == 1
    block = setup.Sl[0]
    assert block.linear is True
    assert len(block.S) == 4
    assert tuple(block.penalty_indices) == (0, 1, 2, 3)
    assert (block.start, block.stop) == (2, 12)


def test_ad_fixed_and_selection_penalty_ownership_is_term_local():
    data = _adaptive_data(71, two_dimensional=False)
    X = data.to_numpy(dtype=np.float64)

    fixed = AdaptiveSmoothTerm("x", k=12, m=0).fit(X, ["x"])
    assert fixed.fixed is True
    assert fixed.penalties == []
    assert fixed.get_penalty_definitions() == []

    selected = AdaptiveSmoothTerm(
        "x", k=12, m=4, select=True, smoothing_id="adaptive"
    ).fit(X, ["x"])
    definitions = selected.get_penalty_definitions()
    assert len(selected.penalties) == 4
    assert len(definitions) == 5
    assert [
        definition.metadata["adaptive_penalty_index"] for definition in definitions[:4]
    ] == [0, 1, 2, 3]
    assert definitions[-1].is_null_space_penalty is True
