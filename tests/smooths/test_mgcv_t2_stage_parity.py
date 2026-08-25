"""Focused construction, prediction, fit, and summary parity for ``t2``."""

from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.smooths.tensor.t2 import AlternativeTensorProductSplineTerm
from tests.mgcv_parity_utils import (
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _make_gaussian_data,
    _run_mgcv_raw_constructor,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def _assert_equal_up_to_column_sign(actual, expected, *, atol=5e-12):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    signs = np.sign(np.sum(actual * expected, axis=0))
    signs[signs == 0.0] = 1.0
    np.testing.assert_allclose(
        actual * signs[np.newaxis, :],
        expected,
        atol=atol,
        rtol=atol,
    )


@pytest.mark.parametrize(
    ("seed", "formula", "term_options", "expected_labels", "expected_orders"),
    [
        pytest.param(
            330,
            't2(x0, x1, k=[5, 5], bs=["cr", "cr"])',
            {},
            ("rr", "nr", "rn"),
            (2, 1, 1, 0),
            id="default",
        ),
        pytest.param(
            331,
            't2(x0, x1, k=[5, 5], bs=["cr", "cr"], full=True)',
            {"full": True},
            ("rr", "1r", "2r", "r1", "r2"),
            (2, 1, 1, 1, 1, 0),
            id="full-null-columns",
        ),
        pytest.param(
            331,
            't2(x0, x1, k=[5, 5], bs=["cr", "cr"], ord=[1])',
            {"ord": [1]},
            ("nr", "rn"),
            (1, 1),
            id="first-order-only",
        ),
    ],
)
def test_t2_raw_block_order_and_disjoint_penalties_match_mgcv(
    seed, formula, term_options, expected_labels, expected_orders
):
    data = _make_gaussian_data(seed=seed, n=120)[["x0", "x1"]]
    term = AlternativeTensorProductSplineTerm(
        feature=["x0", "x1"],
        k=[5, 5],
        basis=["cr", "cr"],
        **term_options,
    ).fit(data.to_numpy(dtype=np.float64), ["x0", "x1"])
    expected = _run_mgcv_raw_constructor(data, formula)

    assert term._penalty_labels == expected_labels
    assert term._penalty_orders == expected_orders[: len(expected_labels)]
    _assert_equal_up_to_column_sign(term._raw_basis_train, expected["X"])

    assert list(expected["rank"]) == [
        int(np.linalg.matrix_rank(penalty)) for penalty in term._raw_penalties
    ]
    assert int(expected["null_space_dim"]) == term._null_space_dim
    assert set(expected["S"]) == set(expected_labels)
    for label, penalty in zip(expected_labels, term._raw_penalties, strict=True):
        np.testing.assert_array_equal(
            np.asarray(penalty, dtype=np.float64),
            np.asarray(expected["S"][label], dtype=np.float64),
        )
        diagonal = np.diag(penalty)
        assert set(np.unique(diagonal)).issubset({0.0, 1.0})

    support = np.sum(
        [np.diag(penalty) != 0.0 for penalty in term._raw_penalties], axis=0
    )
    assert np.all(support <= 1)


def test_t2_reml_fit_prediction_and_summary_match_mgcv():
    data = _make_gaussian_data(seed=333, n=180)
    formula = 'y ~ t2(x0, x1, k=[5, 5], bs=["cr", "cr"])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    _assert_exact_mgcv_snapshot_parity(
        actual,
        expected,
        pred_atol=1e-5,
        pred_rtol=1e-5,
        edf_atol=1e-5,
        criterion_atol=1e-5,
        sp_atol=2e-6,
        sp_rtol=2e-6,
        log_sp_atol=5e-6,
    )

    model = _fit_nampy_model(data, formula, "gaussian", "REML")
    summary = model.summary()
    assert list(summary.s_table["label"]) == ["t2(x0, x1)"]
    np.testing.assert_allclose(
        summary.s_table["edf"].to_numpy(dtype=np.float64),
        np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
        atol=2e-10,
        rtol=0.0,
    )


def test_t2_training_prediction_reuses_the_fitted_null_space_parameterization():
    data = _make_gaussian_data(seed=334, n=110)[["x0", "x1"]]
    X = data.to_numpy(dtype=np.float64)
    term = AlternativeTensorProductSplineTerm(
        feature=["x0", "x1"],
        k=[5, 6],
        basis=["tp", "cr"],
    ).fit(X, ["x0", "x1"])

    np.testing.assert_allclose(
        term.transform_new(X), term.basis_train, atol=5e-9, rtol=5e-9
    )
    assert term.basis_train.shape[1] == 5 * 6 - 1
    assert len(term.penalties) == 3
