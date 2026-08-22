"""Invariant parity for wide, structured, and joint-parameter inference.

Raw TP/SZ coefficient, covariance, and lpmatrix representations are not unique:
legal eigenspace rotations and sign changes alter all three together.  These
tests therefore compare identified quantities only: fitted behavior, total
EDF/EDF2, summary/ANOVA statistics, unconditional prediction standard errors,
and the newdata function-space covariance ``L @ Vc @ L.T``.

The partial-zero-weight wide case explicitly excludes EDF2 and the ANOVA
Wald/p-value pair: upstream ``getRpqr`` makes those values change under a pure
row permutation, while every stable fit and inference surface remains strict.

Upstream references:

* ``mgcv/R/gam.fit3.r::gam.fit3.post.proc`` and ``Vb.corr``;
* ``mgcv/R/gam.fit3.r::newton`` joint scale/theta handling;
* ``mgcv/R/mgcv.r::testStat``;
* ``mgcv/R/smooth.r::smooth.construct.sz.smooth.spec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.summary import summary_gam
from nampy.gam.results.snapshots import _normalize_reference_term_label
from tests.mgcv_parity_utils import (
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


@dataclass(frozen=True)
class _InferenceCase:
    case_id: str
    data_factory: Callable[[], pd.DataFrame]
    formula: str
    family: object = "gaussian"
    method: str = "REML"
    select: bool = False
    weights_column: str | None = None
    expect_wide: bool = False
    compare_edf2: bool = True
    compare_term_inference: bool = True
    compare_anova_statistic: bool = True
    prediction_types: tuple[str, ...] = ("link",)
    atol: float = 2e-5
    rtol: float = 2e-5


def _make_wide_gaussian_data(seed: int = 20260820, n: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.7, 1.7, size=n)
    x1 = rng.uniform(-1.4, 1.4, size=n)
    weights = np.geomspace(0.3, 2.5, num=n)
    y = (
        np.sin(1.3 * x0)
        + 0.35 * np.cos(1.7 * x1)
        + 0.2 * x0 * x1
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": weights})


def _make_wide_gaussian_zero_weight_data() -> pd.DataFrame:
    data = _make_wide_gaussian_data(seed=20260830, n=40)
    data.loc[data.index[::9], "w"] = 0.0
    return data


def _make_one_factor_sz_data(seed: int = 20260821) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = np.asarray(["a"] * 9 + ["b"] * 19 + ["c"] * 32, dtype=object)
    rng.shuffle(levels)
    x = rng.uniform(-1.5, 1.5, size=levels.size)
    shift = {"a": -0.35, "b": 0.15, "c": 0.4}
    curve = {"a": -0.2, "b": 0.3, "c": -0.1}
    y = (
        np.sin(1.2 * x)
        + np.asarray([shift[str(level)] for level in levels])
        + np.asarray([curve[str(level)] for level in levels]) * x
        + rng.normal(scale=0.13, size=levels.size)
    )
    return pd.DataFrame({"y": y, "x": x, "f": levels})


def _make_poisson_sz_missing_cell_data(seed: int = 20260822) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    combinations = [
        (f1, f2)
        for f1 in ("a", "b", "c")
        for f2 in ("u", "v", "w")
        if (f1, f2) != ("c", "w")
    ]
    pairs = combinations * 3
    rng.shuffle(pairs)
    f1 = np.asarray([pair[0] for pair in pairs], dtype=object)
    f2 = np.asarray([pair[1] for pair in pairs], dtype=object)
    x = rng.uniform(-1.4, 1.4, size=len(pairs))
    f1_effect = {"a": -0.25, "b": 0.1, "c": 0.3}
    f2_effect = {"u": -0.2, "v": 0.05, "w": 0.25}
    eta = (
        0.35 * np.sin(1.5 * x)
        + np.asarray([f1_effect[str(value)] for value in f1])
        + np.asarray([f2_effect[str(value)] for value in f2])
    )
    y = rng.poisson(np.exp(0.35 + eta))
    return pd.DataFrame({"y": y, "x": x, "f1": f1, "f2": f2})


def _make_poisson_three_factor_sz_data(seed: int = 20260823) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    combinations = [
        (f1, f2, f3)
        for f1 in ("a", "b")
        for f2 in ("u", "v")
        for f3 in ("low", "high")
    ]
    triples = combinations * 6
    rng.shuffle(triples)
    f1 = np.asarray([triple[0] for triple in triples], dtype=object)
    f2 = np.asarray([triple[1] for triple in triples], dtype=object)
    f3 = np.asarray([triple[2] for triple in triples], dtype=object)
    x = rng.uniform(-1.2, 1.2, size=len(triples))
    eta = (
        0.2
        + 0.3 * np.cos(1.4 * x)
        + 0.15 * (f1 == "b")
        - 0.12 * (f2 == "v")
        + 0.18 * (f3 == "high")
        + 0.1 * x * (f1 == "b")
    )
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x": x, "f1": f1, "f2": f2, "f3": f3})


def _make_poisson_structured_data(
    seed: int = 20260826, n: int = 180
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.6, 1.6, size=n)
    x1 = rng.uniform(-1.4, 1.4, size=n)
    eta = 0.25 + 0.45 * np.sin(1.3 * x0) - 0.3 * np.cos(1.1 * x1)
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_positive_gaussian_structured_data(
    seed: int = 20260827, n: int = 170
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-1.3, 1.3, size=n)
    mu = np.exp(0.4 + 0.3 * np.sin(1.4 * x0) + 0.18 * x1**2)
    y = np.maximum(mu * (1.0 + 0.1 * rng.standard_normal(n)), 1e-5)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_negbin_joint_data(
    seed: int = 20260828, n: int = 190, theta: float = 1.6
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.7, 1.7, size=n)
    x1 = rng.uniform(-1.4, 1.4, size=n)
    mu = np.exp(0.2 + 0.5 * np.sin(1.2 * x0) - 0.2 * x1)
    probability = theta / (theta + mu)
    y = rng.negative_binomial(theta, probability, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


_WIDE_CASES = [
    _InferenceCase(
        case_id="wide_weighted_te_reml",
        data_factory=_make_wide_gaussian_data,
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[8, 8])',
        weights_column="w",
        expect_wide=True,
        atol=3e-5,
        rtol=3e-5,
    ),
    _InferenceCase(
        case_id="wide_aliased_te_ml",
        data_factory=lambda: _make_wide_gaussian_data(seed=20260824, n=44),
        formula='y ~ x0 + I(2*x0) + te(x0, x1, bs=["cr", "cr"], k=[8, 8])',
        method="ML",
        expect_wide=True,
        # With a perfect parametric alias the fitted function and its
        # covariance are identified, but the allocation to individual terms
        # (and hence term-wise ANOVA) is not.
        compare_term_inference=False,
        atol=4e-5,
        rtol=4e-5,
    ),
    _InferenceCase(
        case_id="wide_fixed_te",
        data_factory=lambda: _make_wide_gaussian_data(seed=20260825, n=40),
        formula=(
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[8, 8], sp=[0.7, 1.1])'
        ),
        method="fixed",
        expect_wide=True,
        atol=2e-7,
        rtol=2e-7,
    ),
    _InferenceCase(
        case_id="wide_fixed_te_partial_zero_weights",
        data_factory=_make_wide_gaussian_zero_weight_data,
        formula=(
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[8, 8], sp=[0.7, 1.1])'
        ),
        method="fixed",
        weights_column="w",
        expect_wide=True,
        # mgcv/src/mat.c::getRpqr() reads a square factor from a packed
        # rank-deficient 40x64 QR here. Its EDF2 total changes under a pure row
        # permutation, so it is not an identified parity quantity. All stable
        # fit, covariance, prediction, summary, and ANOVA outputs remain strict.
        compare_edf2=False,
        compare_anova_statistic=False,
        atol=2e-7,
        rtol=2e-7,
    ),
]


_SZ_CASES = [
    _InferenceCase(
        case_id="sz_one_factor_cr_select_unbalanced",
        data_factory=_make_one_factor_sz_data,
        formula=(
            'y ~ s(x, bs="cr", k=6) + '
            's(x, f, bs="sz", k=6, xt="cr")'
        ),
        select=True,
        atol=3e-5,
        rtol=3e-5,
    ),
    _InferenceCase(
        case_id="sz_poisson_two_factor_tp_missing_cell_wide",
        data_factory=_make_poisson_sz_missing_cell_data,
        formula='y ~ s(x, bs="tp", k=5) + s(x, f1, f2, bs="sz", k=5)',
        family="poisson",
        expect_wide=True,
        prediction_types=("link", "response", "terms"),
        atol=8e-5,
        rtol=8e-5,
    ),
    _InferenceCase(
        case_id="sz_poisson_three_factor_ps",
        data_factory=_make_poisson_three_factor_sz_data,
        formula=(
            'y ~ s(x, bs="ps", k=5) + '
            's(x, f1, f2, f3, bs="sz", k=5, xt="ps", m=2)'
        ),
        family="poisson",
        atol=8e-5,
        rtol=8e-5,
    ),
]


_PIRLS_STRUCTURED_CASES = [
    _InferenceCase(
        case_id="poisson_mixed_tensor_reml",
        data_factory=_make_poisson_structured_data,
        formula='y ~ te(x0, x1, bs=["cr", "ps"], k=[5, 6])',
        family="poisson",
        prediction_types=("link", "response", "terms"),
        atol=8e-5,
        rtol=8e-5,
    ),
    _InferenceCase(
        case_id="poisson_linked_cr_reml",
        data_factory=lambda: _make_poisson_structured_data(seed=20260829),
        formula=(
            'y ~ s(x0, bs="cr", k=7, id="shared") + '
            's(x1, bs="cr", k=7, id="shared")'
        ),
        family="poisson",
        prediction_types=("link", "response", "terms"),
        atol=8e-5,
        rtol=8e-5,
    ),
]


_JOINT_PARAMETER_CASES = [
    _InferenceCase(
        case_id="gaussian_log_joint_scale_tensor_reml",
        data_factory=_make_positive_gaussian_structured_data,
        formula='y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6])',
        family={"name": "gaussian", "link": "log"},
        prediction_types=("link", "response", "terms"),
        atol=1e-4,
        rtol=1e-4,
    ),
    _InferenceCase(
        case_id="negbin_joint_theta_cr_reml",
        data_factory=_make_negbin_joint_data,
        formula='y ~ s(x0, bs="cr", k=7) + x1',
        family={"name": "negbin", "theta": 1.6, "estimate_theta": True},
        prediction_types=("link", "response", "terms"),
        atol=2e-4,
        rtol=2e-4,
    ),
]


def _fit_case(case: _InferenceCase, data: pd.DataFrame) -> GAM:
    weights = (
        None
        if case.weights_column is None
        else data[case.weights_column].to_numpy(dtype=np.float64)
    )
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=case.method.lower() != "fixed",
        smoothing_method=case.method,
        smoothing_optimizer="outer_newton",
    )
    return gam.fit(data=data, sample_weight=weights)


def _newdata(case: _InferenceCase, data: pd.DataFrame) -> pd.DataFrame:
    drop = ["y"]
    if case.weights_column is not None:
        drop.append(case.weights_column)
    # Keep all observed combinations for structured terms.  This compares the
    # identified function on the whole observed SZ support, independently of
    # coefficient/basis orientation.
    if case.case_id == "sz_one_factor_cr_select_unbalanced":
        indices = list(range(len(data)))
    elif case.case_id == "sz_poisson_two_factor_tp_missing_cell_wide":
        indices = list(range(len(data)))
    elif case.case_id == "sz_poisson_three_factor_ps":
        indices = list(range(len(data)))
    else:
        indices = list(range(0, len(data), 6))
    return data.iloc[indices].drop(columns=drop).copy()


def _optional_matrix(value) -> np.ndarray | None:
    if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 0:
        return None
    return matrix


def _numeric_matrix(value) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim == 1:
        raw = raw[None, :]

    def coerce(item):
        return np.nan if item is None or item == "NA" else float(item)

    return np.vectorize(coerce, otypes=[np.float64])(raw)


def _function_space_covariance(lpmatrix, covariance) -> np.ndarray:
    L = np.asarray(lpmatrix, dtype=np.float64)
    V = np.asarray(covariance, dtype=np.float64)
    out = L @ V @ L.T
    return np.asarray(0.5 * (out + out.T), dtype=np.float64)


def _assert_invariant_parity(case: _InferenceCase) -> None:
    data = case.data_factory()
    gam = _fit_case(case, data)
    snapshot = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        select=case.select,
        weights_column=case.weights_column,
        allow_live_run=True,
        optimizer="outer_newton" if case.method.lower() != "fixed" else None,
    )

    fit_state = gam.gam_result_.fit_core_solution.fit_state
    assert fit_state.X is not None
    if case.expect_wide:
        assert fit_state.X.shape[1] > fit_state.X.shape[0]

    feature_columns = [
        column
        for column in data.columns
        if column not in {"y", case.weights_column}
    ]
    actual_response = np.asarray(
        gam.predict(data[feature_columns], type="response"), dtype=np.float64
    ).ravel()
    np.testing.assert_allclose(
        actual_response,
        np.asarray(snapshot["predictions"]["response"], dtype=np.float64).ravel(),
        atol=case.atol,
        rtol=case.rtol,
    )

    fit_result = gam.gam_result_.fit_core_solution.fit_result
    np.testing.assert_allclose(
        float(np.sum(np.asarray(fit_result.edf, dtype=np.float64))),
        float(snapshot["fit"]["edf_total"]),
        atol=case.atol,
        rtol=case.rtol,
    )

    actual_edf2 = _optional_matrix(fit_result.edf2)
    expected_edf2 = _optional_matrix(snapshot["fit"].get("edf2"))
    assert (actual_edf2 is None) == (expected_edf2 is None)
    if (
        case.compare_edf2
        and actual_edf2 is not None
        and expected_edf2 is not None
    ):
        # Individual coefficient EDF2 values change with basis gauge; the
        # trace/total is the identified inferential quantity.
        np.testing.assert_allclose(
            float(np.sum(actual_edf2)),
            float(np.sum(expected_edf2)),
            atol=max(case.atol, 8e-5),
            rtol=max(case.rtol, 8e-5),
        )

    newdata = _newdata(case, data)
    expected_lpmatrix = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method=case.method,
        type="lpmatrix",
        select=case.select,
        weights_column=case.weights_column,
        optimizer="outer_newton" if case.method.lower() != "fixed" else None,
        allow_live_run=True,
    )["pred"]
    actual_lpmatrix = gam.predict(newdata, type="lpmatrix")

    expected_covariance = _optional_matrix(
        snapshot["fit"].get("cov_unconditional")
    )
    if expected_covariance is None:
        expected_covariance = _optional_matrix(
            snapshot["fit"].get("vcov_unconditional")
        )
    assert expected_covariance is not None
    actual_covariance = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)

    # This is invariant to any simultaneous coefficient-basis rotation or sign
    # flip in the lpmatrix and covariance.
    np.testing.assert_allclose(
        _function_space_covariance(actual_lpmatrix, actual_covariance),
        _function_space_covariance(expected_lpmatrix, expected_covariance),
        atol=max(case.atol, 5e-7),
        rtol=max(case.rtol, 5e-5),
    )

    for prediction_type in case.prediction_types:
        expected_prediction = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            case.formula,
            family=case.family,
            method=case.method,
            type=prediction_type,
            return_se=True,
            unconditional=True,
            select=case.select,
            weights_column=case.weights_column,
            optimizer="outer_newton" if case.method.lower() != "fixed" else None,
            allow_live_run=True,
        )
        actual_prediction, actual_se = gam.predict(
            newdata,
            type=prediction_type,
            return_se=True,
            cov=actual_covariance,
        )
        actual_prediction = np.asarray(actual_prediction, dtype=np.float64)
        expected_prediction_values = np.asarray(
            expected_prediction["pred"], dtype=np.float64
        )
        actual_se = np.asarray(actual_se, dtype=np.float64)
        expected_se = np.asarray(expected_prediction["se"], dtype=np.float64)
        if prediction_type != "terms":
            actual_prediction = actual_prediction.ravel()
            expected_prediction_values = expected_prediction_values.ravel()
            actual_se = actual_se.ravel()
            expected_se = expected_se.ravel()
        np.testing.assert_allclose(
            actual_prediction,
            expected_prediction_values,
            atol=case.atol,
            rtol=case.rtol,
        )
        np.testing.assert_allclose(
            actual_se,
            expected_se,
            atol=max(case.atol, 5e-7),
            rtol=max(case.rtol, 5e-5),
        )

    if case.compare_term_inference:
        expected_anova = snapshot["parity"]["diagnostics"].get("anova_smooth")
        assert expected_anova is not None
        actual_anova = gam.anova(freq=False).smooth_table
        actual_labels = [
            _normalize_reference_term_label(value)
            for value in actual_anova["label"].tolist()
        ]
        expected_labels = [
            _normalize_reference_term_label(value)
            for value in np.atleast_1d(expected_anova["labels"]).tolist()
        ]
        assert actual_labels == expected_labels
        actual_anova_values = actual_anova[
            ["edf", "ref_df", "wald_stat", "p_value"]
        ].to_numpy(dtype=np.float64)
        expected_anova_values = _numeric_matrix(expected_anova["values"])
        if case.compare_anova_statistic:
            np.testing.assert_allclose(
                actual_anova_values,
                expected_anova_values,
                atol=max(case.atol, 1e-6),
                rtol=max(case.rtol, 2e-4),
                equal_nan=True,
            )
        else:
            # EDF and reference DF are invariant here. The Wald statistic and
            # its p-value consume the same non-identified wide pqr.R factor as
            # EDF2 and therefore are deliberately outside this parity contract.
            np.testing.assert_allclose(
                actual_anova_values[:, :2],
                expected_anova_values[:, :2],
                atol=max(case.atol, 1e-6),
                rtol=max(case.rtol, 2e-4),
                equal_nan=True,
            )

    # Wide/structured summary coverage intentionally stays on scalar behavior;
    # aliased coefficient tables are representation-level quantities.
    actual_summary = summary_gam(gam)
    expected_summary = snapshot["parity"]["diagnostics"].get("summary")
    assert expected_summary is not None
    np.testing.assert_allclose(
        actual_summary.residual_df,
        float(expected_summary["residual_df"]),
        atol=max(case.atol, 1e-6),
        rtol=max(case.rtol, 1e-5),
    )
    if actual_summary.dev_expl is not None and expected_summary.get("dev_expl") is not None:
        np.testing.assert_allclose(
            actual_summary.dev_expl,
            float(np.ravel(expected_summary["dev_expl"])[0]),
            atol=max(case.atol, 1e-6),
            rtol=max(case.rtol, 1e-5),
        )


@pytest.mark.smooth_te
@pytest.mark.parametrize("case", _WIDE_CASES, ids=lambda case: case.case_id)
def test_wide_unconditional_inference_matches_mgcv_on_identified_quantities(
    case: _InferenceCase,
):
    _assert_invariant_parity(case)


@pytest.mark.smooth_sz
@pytest.mark.parametrize("case", _SZ_CASES, ids=lambda case: case.case_id)
def test_sz_unconditional_inference_matches_mgcv_on_identified_quantities(
    case: _InferenceCase,
):
    _assert_invariant_parity(case)


@pytest.mark.parametrize(
    "case", _PIRLS_STRUCTURED_CASES, ids=lambda case: case.case_id
)
def test_pirls_tensor_and_linked_unconditional_inference_match_mgcv(
    case: _InferenceCase,
):
    _assert_invariant_parity(case)


@pytest.mark.parametrize(
    "case", _JOINT_PARAMETER_CASES, ids=lambda case: case.case_id
)
def test_joint_scale_and_theta_unconditional_inference_match_mgcv(
    case: _InferenceCase,
):
    _assert_invariant_parity(case)
