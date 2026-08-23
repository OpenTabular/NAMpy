"""Behavioral parity coverage for supported combinations with thin ownership."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.families.test_general_family_mgcv_parity import (
    _gammals_tensor_data,
    _gaulss_tensor_data,
)
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _make_binomial_data,
    _make_gaussian_data,
    _make_gaussian_data_3col,
    _make_gaussian_link_data,
    _make_poisson_data,
    _normalize_python_formula_text,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def _tensor_newdata(data: pd.DataFrame) -> pd.DataFrame:
    newdata = data.iloc[2::19].drop(columns=["y"]).copy()
    newdata.loc[newdata.index[0], "x0"] = float(data["x0"].min()) - 0.25
    newdata.loc[newdata.index[-1], "x1"] = float(data["x1"].max()) + 0.25
    return newdata


def _assert_fixed_newdata_response_se(
    data: pd.DataFrame, formula: str, *, atol: float
) -> GAM:
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = _tensor_newdata(data)
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual, np.asarray(expected["pred"]).ravel(), atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        actual_se, np.asarray(expected["se"]).ravel(), atol=atol, rtol=atol
    )
    return gam


_TENSOR_BASES = ("cr", "cs", "cc", "ps", "tp", "ts")
_PORTABLE_TENSOR_PAIRS = tuple(
    pair for pair in itertools.product(_TENSOR_BASES, repeat=2) if "cs" not in pair
)


@pytest.mark.parametrize("special", ["te", "ti"])
@pytest.mark.parametrize(
    ("left", "right"),
    _PORTABLE_TENSOR_PAIRS,
    ids=[f"{left}_{right}" for left, right in _PORTABLE_TENSOR_PAIRS],
)
def test_tensor_portable_cartesian_basis_matrix_matches_mgcv_response_and_se(
    special, left, right
):
    """Portable ordered tensor-basis pairs match mgcv response and SE."""
    data = _make_gaussian_data(seed=243, n=84)
    formula = (
        f'y ~ {special}(x0, x1, bs=["{left}", "{right}"], '
        "k=[5, 5], sp=[0.7, 1.1])"
    )
    _assert_fixed_newdata_response_se(data, formula, atol=3e-6)


def test_tensor_cs_pairs_are_owned_by_representation_invariant_stage_checks():
    """Route platform-indeterminate cs tensor margins to invariant parity tests.

    The shrinkage direction added by ``smooth.construct.cs.smooth.spec`` is
    built from the two numerically null eigenvectors of the cubic-regression
    penalty. Their orientation is LAPACK-dependent, so a static response
    vector from another platform is not a valid oracle. The te/ti stage suites
    instead compare the identified subspace and penalty spectrum for cs
    margins, as required by the repository's representation policy.
    """
    cs_pairs = {
        pair for pair in itertools.product(_TENSOR_BASES, repeat=2) if "cs" in pair
    }
    assert len(_PORTABLE_TENSOR_PAIRS) == 25
    assert len(cs_pairs) == 11


def test_three_margin_ti_optimized_reml_fit_matches_mgcv():
    """Three-margin ti has live optimized fit coverage, not construction only."""
    data = _make_gaussian_data_3col(seed=241, n=110)
    formula = (
        'y ~ ti(x0, x1, x2, bs=["cr", "ps", "tp"], k=[4, 4, 4])'
    )
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(formula),
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=5e-5,
        pred_rtol=5e-5,
        sp_log_atol=8e-3,
        criterion_atol=5e-3,
    )


@pytest.mark.parametrize("special", ["te", "ti"])
def test_multivariate_tensor_margin_fixed_fit_matches_mgcv(special):
    """A d=[2,1] multivariate margin is covered by a fitted parity contract."""
    data = _make_gaussian_data_3col(seed=242, n=130)
    formula = (
        f'y ~ {special}(x0, x1, x2, d=[2, 1], bs=["cr", "cr"], '
        "k=[6, 5], sp=[0.7, 1.1])"
    )
    _assert_fixed_newdata_response_se(data, formula, atol=2e-6)


def _four_column_gaussian_data(*, seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-1.4, 1.4, size=n)
    x2 = rng.uniform(-1.3, 1.3, size=n)
    x3 = rng.uniform(-1.2, 1.2, size=n)
    y = (
        0.4 * np.sin(1.2 * x0)
        + 0.2 * x1 * x2
        - 0.15 * x2**2
        + 0.1 * np.cos(1.4 * x3)
        + rng.normal(scale=0.14, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "x2": x2, "x3": x3})


_THREE_MARGIN_BASIS_COVER = [
    ("cr", "ps", "tp"),
    ("cs", "tp", "cc"),
    ("cc", "ts", "ps"),
    ("ps", "cr", "ts"),
    ("tp", "cs", "cr"),
    ("ts", "cc", "cs"),
]


@pytest.mark.parametrize("special", ["te", "ti"])
@pytest.mark.parametrize(
    "bases",
    _THREE_MARGIN_BASIS_COVER,
    ids=["_".join(bases) for bases in _THREE_MARGIN_BASIS_COVER],
)
def test_three_margin_tensor_basis_cover_fixed_fit_matches_live_mgcv(
    special, bases
):
    """Each supported basis occurs in every three-margin position against mgcv."""
    data = _four_column_gaussian_data(seed=244, n=118)
    basis_text = ", ".join(f'"{basis}"' for basis in bases)
    formula = (
        f"y ~ {special}(x0, x1, x2, bs=[{basis_text}], "
        "k=[4, 4, 4], sp=[0.7, 0.9, 1.1])"
    )
    _assert_fixed_newdata_response_se(data, formula, atol=4e-6)


@pytest.mark.parametrize("special", ["te", "ti"])
def test_four_margin_tensor_fixed_fit_matches_live_mgcv(special):
    """Four-margin tensor block assembly is checked through fitted predictions."""
    data = _four_column_gaussian_data(seed=245, n=190)
    formula = (
        f'y ~ {special}(x0, x1, x2, x3, bs=["cr", "ps", "tp", "cc"], '
        "k=[3, 4, 3, 4], sp=[0.6, 0.8, 1.0, 1.2])"
    )
    _assert_fixed_newdata_response_se(data, formula, atol=5e-6)


def _higher_tensor_family_data(family_id: str, *, seed: int, n: int = 160):
    data = _four_column_gaussian_data(seed=seed, n=n)
    rng = np.random.default_rng(seed + 1)
    eta = (
        0.15
        + 0.25 * np.sin(data["x0"].to_numpy())
        - 0.12 * data["x1"].to_numpy()
        + 0.1 * data["x2"].to_numpy() * data["x3"].to_numpy()
    )
    if family_id == "poisson":
        data["y"] = rng.poisson(np.exp(eta))
    elif family_id == "gamma":
        mu = np.exp(eta)
        data["y"] = rng.gamma(shape=3.0, scale=mu / 3.0)
    elif family_id == "binomial":
        data["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    elif family_id == "negbin":
        theta = 1.7
        mu = np.exp(eta)
        data["y"] = rng.negative_binomial(theta, theta / (theta + mu), size=n)
    else:
        raise AssertionError(f"Unhandled family {family_id!r}")
    return data


@pytest.mark.parametrize(
    ("family_id", "family", "special", "bases", "atol"),
    [
        pytest.param(
            "poisson", "poisson", "te", ("cr", "ps", "tp"), 4e-4, id="poisson_te"
        ),
        pytest.param(
            "gamma",
            {"name": "gamma", "link": "log"},
            "ti",
            ("cs", "cc", "ts"),
            8e-4,
            id="gamma_ti",
        ),
    ],
)
def test_three_margin_non_gaussian_optimized_fit_matches_live_mgcv(
    family_id, family, special, bases, atol
):
    """Three-margin optimized tensors are covered outside Gaussian fitting."""
    data = _higher_tensor_family_data(family_id, seed=246, n=120)
    basis_text = ", ".join(f'"{basis}"' for basis in bases)
    formula = (
        f"y ~ {special}(x0, x1, x2, bs=[{basis_text}], k=[3, 4, 3])"
    )
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        family,
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=atol,
        pred_rtol=atol,
        sp_log_atol=8e-2,
        criterion_atol=8e-2,
    )


@pytest.mark.parametrize("special", ["te", "ti"])
@pytest.mark.parametrize(
    ("family_id", "family", "atol"),
    [
        pytest.param("poisson", "poisson", 5e-4, id="poisson"),
        pytest.param(
            "gamma", {"name": "gamma", "link": "log"}, 8e-4, id="gamma"
        ),
    ],
)
def test_multivariate_margin_non_gaussian_optimized_fit_matches_live_mgcv(
    special, family_id, family, atol
):
    """Optimized d=[2,1] tensor margins retain non-Gaussian fit parity."""
    data = _higher_tensor_family_data(family_id, seed=247)
    formula = (
        f'y ~ {special}(x0, x1, x2, d=[2, 1], bs=["cr", "ps"], '
        "k=[6, 5])"
    )
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        family,
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=atol,
        pred_rtol=atol,
        sp_log_atol=8e-2,
        criterion_atol=8e-2,
    )


@dataclass(frozen=True)
class _StructuredTermCase:
    term_id: str
    formula: str


@dataclass(frozen=True)
class _InferenceFamilyCase:
    family_id: str
    family: object
    atol: float


def _with_factor_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    row = np.arange(len(data))
    data["f"] = pd.Categorical(np.asarray(["a", "b", "c", "d"])[row % 4])
    data["f1"] = pd.Categorical(np.asarray(["a", "b", "c"])[row % 3])
    data["f2"] = pd.Categorical(np.asarray(["u", "v"])[(row // 3) % 2])
    return data


_STRUCTURED_TERMS = [
    _StructuredTermCase("re", 'y ~ offset(off) + s(f, bs="re")'),
    _StructuredTermCase(
        "fs", 'y ~ offset(off) + s(f, x0, bs="fs", k=5, xt="cr")'
    ),
    _StructuredTermCase(
        "sz", 'y ~ offset(off) + s(f1, f2, x0, bs="sz", k=5)'
    ),
    _StructuredTermCase(
        "te", 'y ~ offset(off) + te(x0, x1, bs=["cr", "ps"], k=[5, 5])'
    ),
    _StructuredTermCase(
        "ti", 'y ~ offset(off) + ti(x0, x1, bs=["cr", "ps"], k=[5, 5])'
    ),
]
_INFERENCE_FAMILIES = [
    _InferenceFamilyCase("gaussian", "gaussian", 4e-4),
    _InferenceFamilyCase("binomial", "binomial", 1e-3),
    _InferenceFamilyCase("poisson", "poisson", 8e-4),
    _InferenceFamilyCase("gamma", {"name": "gamma", "link": "log"}, 1e-3),
    _InferenceFamilyCase(
        "negbin", {"name": "negbin", "theta": 1.7}, 1.5e-3
    ),
]


@dataclass(frozen=True)
class _GeneralStructuredCase:
    case_id: str
    family: str
    formula: list[str]
    atol: float


_GENERAL_STRUCTURED_CASES = [
    _GeneralStructuredCase(
        "gaulss_re_fixed",
        "gaulss",
        ['y ~ s(f, bs="re", sp=0.8)', "~ 1"],
        2e-5,
    ),
    _GeneralStructuredCase(
        "gaulss_fs_fixed",
        "gaulss",
        [
            'y ~ s(f, x0, bs="fs", k=5, xt="cr",'
            ' sp=[0.8, 0.8, 0.8])',
            "~ 1",
        ],
        2e-5,
    ),
    _GeneralStructuredCase(
        "gammals_sz_fixed",
        "gammals",
        [
            'y ~ s(f1, f2, x0, bs="sz", k=5,'
            ' id="general_sz", sp=0.8)',
            "~ 1",
        ],
        4e-4,
    ),
    _GeneralStructuredCase(
        "gaulss_fs_second_predictor_fixed",
        "gaulss",
        [
            "y ~ 1",
            '~ s(f, x0, bs="fs", k=5, xt="cr",'
            ' sp=[0.8, 0.8, 0.8])',
        ],
        5e-5,
    ),
]


def _structured_inference_data(family_id: str, *, seed: int, n: int = 144):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.6, 1.6, size=n)
    x1 = rng.uniform(-1.4, 1.4, size=n)
    row = np.arange(n)
    f = np.asarray(["a", "b", "c", "d"])[row % 4]
    f1 = np.asarray(["a", "b", "c"])[row % 3]
    f2 = np.asarray(["u", "v"])[(row // 3) % 2]
    f_effect = np.asarray(
        [{"a": -0.3, "b": 0.0, "c": 0.2, "d": 0.4}[value] for value in f]
    )
    off = 0.08 * np.sin(0.7 * x1)
    eta = 0.15 + 0.45 * np.sin(1.3 * x0) - 0.18 * x1 + f_effect + off
    if family_id == "gaussian":
        y = eta + rng.normal(scale=0.18, size=n)
    elif family_id == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    elif family_id == "poisson":
        y = rng.poisson(np.exp(eta))
    elif family_id == "gamma":
        mu = np.exp(eta)
        y = rng.gamma(shape=3.0, scale=mu / 3.0)
    elif family_id == "negbin":
        theta = 1.7
        mu = np.exp(eta)
        y = rng.negative_binomial(theta, theta / (theta + mu), size=n)
    else:
        raise AssertionError(f"Unhandled family {family_id!r}")
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "x1": x1,
            "f": pd.Categorical(f),
            "f1": pd.Categorical(f1),
            "f2": pd.Categorical(f2),
            "off": off,
            "w": np.exp(np.linspace(-0.8, 0.8, n)),
        }
    )


def _general_structured_data(
    family: str, *, seed: int, n: int = 132
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
    if family == "gaulss":
        sigma = np.exp(-0.45 + 0.12 * x0)
        y = rng.normal(signal, sigma, size=n)
    elif family == "gammals":
        mu = np.exp(0.3 + signal)
        shape = np.exp(0.55 + 0.1 * np.cos(x0))
        y = rng.gamma(shape=shape, scale=mu / shape, size=n)
    else:
        raise AssertionError(f"Unhandled general family {family!r}")
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "f": pd.Categorical(f),
            "f1": pd.Categorical(f1),
            "f2": pd.Categorical(f2),
        }
    )


@pytest.mark.parametrize(
    "case", _GENERAL_STRUCTURED_CASES, ids=lambda case: case.case_id
)
def test_single_structured_term_general_family_fit_and_newdata_match_mgcv(case):
    """Supported single re/fs/sz blocks reach general-family fit and prediction."""
    data = _general_structured_data(case.family, seed=248)
    gam = GAM(
        family=case.family,
        formula=case.formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        "fixed",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.atol,
        pred_rtol=case.atol,
        sp_log_atol=1e-10,
        check_sp=False,
        check_sp_count=False,
        check_criterion=False,
    )

    newdata = data.iloc[5::19].drop(columns=["y"]).copy()
    actual_prediction, actual_se = gam.predict(
        newdata, type="response", return_se=True
    )
    expected_prediction = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        case.formula,
        family=case.family,
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual_prediction, dtype=np.float64),
        np.asarray(expected_prediction["pred"], dtype=np.float64),
        atol=case.atol,
        rtol=case.atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se, dtype=np.float64),
        np.asarray(expected_prediction["se"], dtype=np.float64),
        atol=2.0 * case.atol,
        rtol=2.0 * case.atol,
    )


def test_gaulss_fs_optimized_reml_fit_matches_mgcv_in_identified_space():
    """Automatic fs smoothing may permute its null SPs but not fitted behavior."""
    data = _general_structured_data("gaulss", seed=248)
    formula = ['y ~ s(f, x0, bs="fs", k=5, xt="cr")', "~ 1"]
    gam = GAM(
        family="gaulss",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="reml",
    ).fit(data=data)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "gaulss",
        "REML",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=2e-4,
        pred_rtol=2e-4,
        sp_log_atol=2e-2,
        check_sp=False,
        check_sp_count=False,
        check_criterion=False,
    )


def _numeric_matrix(value) -> np.ndarray:
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]

    def _coerce(item):
        return np.nan if item is None or item == "NA" else float(item)

    return np.vectorize(_coerce, otypes=[np.float64])(arr)


def _assert_structured_diagnostics(gam, expected_snapshot, *, atol):
    expected_diag = expected_snapshot["parity"]["diagnostics"]
    actual_summary = gam.summary()
    expected_summary = expected_diag["summary"]
    for actual, expected in (
        (actual_summary.residual_df, expected_summary["residual_df"]),
        (actual_summary.scale, expected_summary["scale"]),
        (actual_summary.r_sq, expected_summary["r_sq"]),
        (actual_summary.dev_expl, expected_summary["dev_expl"]),
    ):
        if actual is not None and expected is not None:
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=atol)

    expected_anova = expected_diag["anova_smooth"]
    actual_anova = gam.anova(freq=False).smooth_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=np.float64)
    expected_anova_values = _numeric_matrix(expected_anova["values"])
    assert actual_anova.shape == expected_anova_values.shape
    np.testing.assert_allclose(
        actual_anova[:, :2],
        expected_anova_values[:, :2],
        atol=max(2e-3, 10.0 * atol),
        rtol=2e-2,
    )
    # mgcv/R/mgcv.r::testStat documents two valid fractional-rank test
    # statistics.  It returns one arbitrarily but averages their p-values.
    # Eigenbasis orientation can therefore move the displayed statistic while
    # leaving the inferential quantity invariant.
    np.testing.assert_allclose(
        actual_anova[:, 2],
        expected_anova_values[:, 2],
        atol=max(0.06, 10.0 * atol),
        rtol=0.15,
    )
    np.testing.assert_allclose(
        actual_anova[:, 3],
        expected_anova_values[:, 3],
        atol=max(2e-3, 10.0 * atol),
        rtol=5e-2,
        equal_nan=True,
    )

    expected_k = expected_diag["k_check"]
    actual_k = gam.k_check(subsample=120, n_rep=8, seed=0)[
        ["k_prime", "edf", "k_index", "p_value"]
    ].to_numpy(dtype=np.float64)
    expected_k_values = _numeric_matrix(expected_k["values"])
    assert actual_k.shape == expected_k_values.shape
    np.testing.assert_array_equal(
        actual_k[:, 0].astype(int), np.rint(expected_k_values[:, 0]).astype(int)
    )
    np.testing.assert_allclose(
        actual_k[:, 1], expected_k_values[:, 1], atol=max(atol, 2e-4), rtol=0.0
    )
    finite = np.isfinite(expected_k_values[:, 2])
    np.testing.assert_allclose(
        actual_k[finite, 2],
        expected_k_values[finite, 2],
        atol=1.0 / np.sqrt(8.0),
        rtol=0.5,
    )
    assert np.all(np.isnan(actual_k[~finite, 2:]))
    assert np.all((actual_k[finite, 3] >= 0.0) & (actual_k[finite, 3] <= 1.0))


def test_gaussian_inverse_reml_postfit_surfaces_match_mgcv():
    """Noncanonical Gaussian post-fit outputs remain aligned with mgcv.

    The optimizer lifecycle already covers this family/link cell.  This test
    owns the downstream ``logLik.gam``/AIC/BIC, ``vcov.gam``, residual,
    ``summary.gam``/``anova.gam``, and ``k.check`` surfaces that consume the
    fitted joint scale state.
    """
    data = _make_gaussian_link_data(seed=252, n=150)
    family = {"name": "gaussian", "link": "inverse"}
    formula = 'y ~ s(x0, bs="cr", k=8)'
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        family,
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    expected_fit = expected["fit"]
    expected_diagnostics = expected["parity"]["diagnostics"]

    assert gam.loglik() == pytest.approx(
        float(expected_fit["loglik"]), rel=2e-6, abs=2e-7
    )
    assert gam.aic() == pytest.approx(
        float(expected_fit["aic"]), rel=2e-6, abs=2e-7
    )
    assert gam.bic() == pytest.approx(
        float(expected_diagnostics["bic"]), rel=2e-6, abs=2e-7
    )

    covariance_cases = (
        (gam.vcov(), expected_fit["cov_bayes"]),
        (gam.vcov(freq=True), expected_fit["cov_freq"]),
        (gam.vcov(sandwich=True), expected_fit["cov_sandwich_bayes"]),
        (
            gam.vcov(sandwich=True, freq=True),
            expected_fit["cov_sandwich_freq"],
        ),
        (
            gam.vcov(unconditional=True),
            expected_fit["vcov_unconditional"],
        ),
    )
    for actual_covariance, expected_covariance in covariance_cases:
        assert expected_covariance is not None
        np.testing.assert_allclose(
            np.asarray(actual_covariance, dtype=np.float64),
            np.asarray(expected_covariance, dtype=np.float64),
            atol=2e-6,
            rtol=2e-4,
        )

    for snapshot_key, residual_type in (
        ("response", "response"),
        ("working", "working"),
        ("pearson", "pearson"),
        ("scaled_pearson", "scaled.pearson"),
        ("deviance", "deviance"),
    ):
        np.testing.assert_allclose(
            np.asarray(gam.residuals(type=residual_type), dtype=np.float64),
            np.asarray(
                expected_diagnostics["residuals"][snapshot_key],
                dtype=np.float64,
            ),
            atol=2e-6,
            rtol=2e-6,
        )

    _assert_structured_diagnostics(gam, expected, atol=2e-5)


@pytest.mark.parametrize("term", _STRUCTURED_TERMS, ids=lambda case: case.term_id)
@pytest.mark.parametrize(
    "family_case", _INFERENCE_FAMILIES, ids=lambda case: case.family_id
)
def test_structured_term_family_inference_cartesian_matrix_matches_mgcv(
    term, family_case
):
    """All re/fs/sz/te/ti × ordinary-family inference surfaces match mgcv."""
    seed = 250 + 10 * _INFERENCE_FAMILIES.index(family_case) + _STRUCTURED_TERMS.index(term)
    data = _structured_inference_data(family_case.family_id, seed=seed)
    gam = GAM(
        family=family_case.family,
        formula=term.formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    actual_snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    expected_snapshot = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(term.formula),
        family_case.family,
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual_snapshot,
        expected_snapshot,
        pred_atol=family_case.atol,
        pred_rtol=family_case.atol,
        sp_log_atol=5e-2,
        check_sp=term.term_id not in {"fs", "sz"},
        criterion_atol=5e-2,
    )

    actual_edf2 = np.asarray(actual_snapshot["fit"]["edf2"], dtype=np.float64)
    expected_edf2 = np.asarray(expected_snapshot["fit"]["edf2"], dtype=np.float64)
    np.testing.assert_allclose(
        np.sum(actual_edf2),
        np.sum(expected_edf2),
        atol=max(2e-3, 10.0 * family_case.atol),
        rtol=2e-3,
    )

    rows = np.arange(3, len(data), 19)
    actual_lpmatrix = np.asarray(
        gam.lpmatrix(data.iloc[rows].drop(columns=["y"])), dtype=np.float64
    )
    expected_lpmatrix = np.asarray(
        expected_snapshot["predictions"]["lpmatrix"], dtype=np.float64
    )[rows]
    actual_vc = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
    expected_vc = np.asarray(
        expected_snapshot["fit"]["cov_unconditional"], dtype=np.float64
    )
    np.testing.assert_allclose(
        actual_lpmatrix @ actual_vc @ actual_lpmatrix.T,
        expected_lpmatrix @ expected_vc @ expected_lpmatrix.T,
        atol=max(3e-3, 20.0 * family_case.atol),
        rtol=3e-2,
    )

    newdata = data.iloc[5::23].drop(columns=["y"]).copy()
    if term.term_id != "re":
        newdata.loc[newdata.index[0], "x0"] = float(data["x0"].min()) - 0.2
        newdata.loc[newdata.index[-1], "x1"] = float(data["x1"].max()) + 0.2
    actual_terms, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    expected_terms = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        term.formula,
        family=family_case.family,
        method="REML",
        type="terms",
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    actual_terms = _numeric_matrix(actual_terms)
    actual_se = _numeric_matrix(actual_se)
    expected_term_values = _numeric_matrix(expected_terms["pred"])
    expected_se_values = _numeric_matrix(expected_terms["se"])
    np.testing.assert_allclose(
        actual_terms,
        expected_term_values,
        atol=max(2e-3, 10.0 * family_case.atol),
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        actual_se,
        expected_se_values,
        atol=max(3e-3, 15.0 * family_case.atol),
        rtol=3e-2,
    )
    _assert_structured_diagnostics(
        gam, expected_snapshot, atol=max(family_case.atol, 2e-4)
    )


@pytest.mark.parametrize(
    ("term", "family_case"),
    list(zip(_STRUCTURED_TERMS, _INFERENCE_FAMILIES, strict=True)),
    ids=[
        f"{term.term_id}_{family.family_id}"
        for term, family in zip(
            _STRUCTURED_TERMS, _INFERENCE_FAMILIES, strict=True
        )
    ],
)
def test_structured_newdata_term_filters_and_unconditional_se_match_mgcv(
    term, family_case
):
    """terms=/exclude= compose with structured, non-Gaussian unconditional SE."""
    data = _structured_inference_data(
        family_case.family_id,
        seed=310 + _STRUCTURED_TERMS.index(term),
    )
    gam = GAM(
        family=family_case.family,
        formula=term.formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = data.iloc[4::21].drop(columns=["y"]).copy()
    if term.term_id not in {"re", "fs", "sz"}:
        newdata.loc[newdata.index[0], "x0"] = float(data["x0"].min()) - 0.3
        newdata.loc[newdata.index[-1], "x1"] = float(data["x1"].max()) + 0.3

    all_expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        term.formula,
        family=family_case.family,
        method="REML",
        type="terms",
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    names = all_expected["term_names"]
    term_name = str(names[0] if isinstance(names, list) else names)
    actual_term, actual_term_se = gam.predict(
        newdata,
        type="terms",
        terms=[term_name],
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_term),
        _numeric_matrix(all_expected["pred"]),
        atol=max(3e-3, 12.0 * family_case.atol),
        rtol=3e-2,
    )
    np.testing.assert_allclose(
        _numeric_matrix(actual_term_se),
        _numeric_matrix(all_expected["se"]),
        atol=max(4e-3, 16.0 * family_case.atol),
        rtol=4e-2,
    )

    actual_excluded, actual_excluded_se = gam.predict(
        newdata,
        type="response",
        exclude=[term_name],
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    expected_excluded = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        term.formula,
        family=family_case.family,
        method="REML",
        type="response",
        exclude=[term_name],
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual_excluded,
        np.asarray(expected_excluded["pred"], dtype=np.float64).ravel(),
        atol=max(2e-3, 10.0 * family_case.atol),
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        actual_excluded_se,
        np.asarray(expected_excluded["se"], dtype=np.float64).ravel(),
        atol=max(3e-3, 15.0 * family_case.atol),
        rtol=3e-2,
    )


@pytest.mark.parametrize(
    "term",
    _STRUCTURED_TERMS[:3],
    ids=lambda term: term.term_id,
)
def test_structured_newdata_unseen_factor_levels_match_mgcv_behavior(term):
    """RE zero rows and FS/SZ NA rows match mgcv for unseen factor levels."""
    data = _structured_inference_data("gaussian", seed=318)
    formula = {
        "re": 'y ~ offset(off) + s(f, bs="re", sp=0.8)',
        "fs": (
            'y ~ offset(off) + s(f, x0, bs="fs", k=5, xt="cr", '
            "sp=[0.7, 0.9, 1.1])"
        ),
        "sz": (
            'y ~ offset(off) + s(f1, f2, x0, bs="sz", k=5, '
            "sp=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2])"
        ),
    }[term.term_id]
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[:3].drop(columns=["y"]).copy()
    source = "f1" if term.term_id == "sz" else "f"
    newdata[source] = pd.Categorical(["unseen", "unseen", "unseen"])
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual,
        _numeric_matrix(expected["pred"]).ravel(),
        atol=3e-5,
        rtol=3e-5,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual_se,
        _numeric_matrix(expected["se"]).ravel(),
        atol=3e-5,
        rtol=3e-5,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("formula", "data_factory", "atol"),
    [
        pytest.param(
            'y ~ s(f, x0, bs="fs", k=5, xt="cr", sp=[0.7, 0.9, 1.1])',
            lambda: _with_factor_columns(_make_gaussian_data(seed=255, n=120)),
            3e-7,
            id="fs",
        ),
        pytest.param(
            'y ~ s(f1, f2, x0, bs="sz", k=5,'
            " sp=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2])",
            lambda: _with_factor_columns(_make_gaussian_data(seed=256, n=120)),
            3e-7,
            id="sz",
        ),
    ],
)
def test_fixed_structured_fs_sz_newdata_response_and_se_match_mgcv(
    formula, data_factory, atol
):
    """Fixed-SP fs/sz transforms retain seen-level newdata covariance behavior."""
    data = data_factory()
    _assert_fixed_newdata_response_se(data, formula, atol=atol)


def _positive_gaussian_tensor_data() -> pd.DataFrame:
    rng = np.random.default_rng(260)
    x0 = rng.uniform(-1.4, 1.4, size=150)
    x1 = rng.uniform(-1.2, 1.2, size=150)
    mu = np.exp(0.4 + 0.35 * np.sin(1.3 * x0) + 0.2 * x1**2)
    y = np.maximum(mu * (1.0 + 0.08 * rng.standard_normal(len(mu))), 1e-4)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _boundary_random_effect_data() -> pd.DataFrame:
    rng = np.random.default_rng(263)
    n = 192
    row = np.arange(n)
    x0 = rng.uniform(-1.0, 1.0, size=n)
    off = 0.05 * x0
    y = 0.4 + off + rng.normal(scale=0.16, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "f": pd.Categorical(np.asarray([f"g{i}" for i in range(12)])[row % 12]),
            "off": off,
            "w": np.power(10.0, np.linspace(-0.5, 0.5, n)),
        }
    )


@dataclass(frozen=True)
class _OptimizerCrossCase:
    case_id: str
    data_factory: object
    family: object
    formula: str
    method: str
    optimizer: str
    select: bool
    weighted: bool
    check_sp: bool
    atol: float
    boundary: bool = False


_OPTIMIZER_CRITERION_LINK_CASES = [
    _OptimizerCrossCase(
        "gaussian_log_ml_bfgs_select_mixed_ti",
        _positive_gaussian_tensor_data,
        {"name": "gaussian", "link": "log"},
        'y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6])',
        "ML",
        "bfgs",
        True,
        False,
        True,
        8e-5,
    ),
    _OptimizerCrossCase(
        "poisson_sqrt_reml_efs_mixed_te",
        lambda: _make_poisson_data(seed=261, n=170),
        {"name": "poisson", "link": "sqrt"},
        'y ~ te(x0, x1, bs=["cc", "ps"], k=[5, 6])',
        "REML",
        "efs",
        False,
        False,
        True,
        2e-4,
    ),
    _OptimizerCrossCase(
        "binomial_cauchit_ml_optim_fs",
        lambda: _with_factor_columns(_make_binomial_data(seed=262, n=180)),
        {"name": "binomial", "link": "cauchit"},
        'y ~ s(f, x0, bs="fs", k=5, xt="cr")',
        "ML",
        "optim",
        False,
        False,
        False,
        4e-4,
    ),
    _OptimizerCrossCase(
        "gaussian_identity_gcv_newton_weighted_offset_re",
        lambda: _structured_inference_data("gaussian", seed=264),
        "gaussian",
        'y ~ offset(off) + s(f, bs="re")',
        "GCV",
        "outer_newton",
        False,
        True,
        True,
        4e-3,
    ),
    _OptimizerCrossCase(
        "binomial_probit_reml_bfgs_select_weighted_offset_sz",
        lambda: _structured_inference_data("binomial", seed=265),
        {"name": "binomial", "link": "probit"},
        'y ~ offset(off) + s(f1, f2, x0, bs="sz", k=5)',
        "REML",
        "bfgs",
        True,
        True,
        False,
        1e-3,
    ),
    _OptimizerCrossCase(
        "gamma_inverse_ml_optim_weighted_offset_te",
        lambda: _structured_inference_data("gamma", seed=266),
        {"name": "gamma", "link": "inverse"},
        'y ~ offset(off) + te(x0, x1, bs=["tp", "ps"], k=[5, 5])',
        "ML",
        "optim",
        False,
        True,
        False,
        1.2e-3,
    ),
    _OptimizerCrossCase(
        "negbin_sqrt_reml_efs_weighted_offset_re",
        lambda: _structured_inference_data("negbin", seed=267),
        {"name": "negbin", "link": "sqrt", "theta": 1.7},
        'y ~ offset(off) + s(f, bs="re")',
        "REML",
        "efs",
        False,
        True,
        True,
        1.5e-3,
    ),
    _OptimizerCrossCase(
        "poisson_log_reml_optim_select_weighted_offset_fs",
        lambda: _structured_inference_data("poisson", seed=268),
        "poisson",
        'y ~ offset(off) + s(f, x0, bs="fs", k=5, xt="cr")',
        "REML",
        "optim",
        True,
        True,
        False,
        1e-3,
    ),
    _OptimizerCrossCase(
        "gaussian_reml_newton_boundary_re",
        _boundary_random_effect_data,
        "gaussian",
        'y ~ offset(off) + s(f, bs="re")',
        "REML",
        "outer_newton",
        False,
        True,
        False,
        8e-4,
        boundary=True,
    ),
]


@pytest.mark.parametrize(
    "case",
    _OPTIMIZER_CRITERION_LINK_CASES,
    ids=lambda case: case.case_id,
)
def test_optimizer_criterion_link_structured_crosses_match_mgcv(case):
    """Every optimizer/criterion/link axis is crossed with structured terms."""
    data = case.data_factory()
    weights = data["w"].to_numpy(dtype=np.float64) if case.weighted else None
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer=case.optimizer,
    ).fit(data=data, sample_weight=weights)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(case.formula),
        case.family,
        "GCV.Cp" if case.method.upper() == "GCV" else case.method,
        select=case.select,
        weights_column="w" if case.weighted else None,
        optimizer=case.optimizer,
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.atol,
        pred_rtol=case.atol,
        # The weighted random-effect GCV objective is very flat at its
        # endpoint; fit, criterion and uncertainty remain checked below.
        sp_log_atol=(
            2.5e-1
            if case.case_id
            == "gaussian_identity_gcv_newton_weighted_offset_re"
            else 3e-2
        ),
        check_sp=case.check_sp,
        criterion_atol=6e-2,
    )
    if case.case_id == "gaussian_identity_gcv_newton_weighted_offset_re":
        shared = GAM(
            family=case.family,
            formula=case.formula,
            select=case.select,
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=np.asarray(
                expected["fit"]["smoothing_params"], dtype=np.float64
            ),
        ).fit(data=data, sample_weight=weights)
        shared_snapshot = shared.parity_snapshot(X=data, include_covariances=False)
        for pred_type in ("response", "link"):
            np.testing.assert_allclose(
                np.asarray(
                    shared_snapshot["predictions"][pred_type], dtype=np.float64
                ),
                np.asarray(expected["predictions"][pred_type], dtype=np.float64),
                rtol=0.0,
                atol=6e-4,
            )
    if case.boundary:
        actual_sp = np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        expected_sp = np.asarray(
            expected["fit"]["smoothing_params"], dtype=np.float64
        )
        # At a variance-component boundary the exact log-SP endpoint is not
        # identified.  Check the shared boundary state, not optimizer position.
        assert np.max(actual_sp) > 1e4
        assert np.max(expected_sp) > 1e4


def _clustered_weighted_data() -> pd.DataFrame:
    rng = np.random.default_rng(270)
    centers = np.repeat(np.linspace(-1.5, 1.5, 28), 8)
    x0 = centers + 2e-4 * rng.standard_normal(len(centers))
    x1 = x0 + 7e-4 * rng.standard_normal(len(centers))
    y = np.sin(2.1 * x0) + 0.15 * x1**2 + 0.04 * rng.standard_normal(len(x0))
    weights = np.power(10.0, np.linspace(-2.0, 2.0, len(x0)))
    weights = weights[rng.permutation(len(weights))]
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "w": weights})


def test_clustered_large_k_weighted_near_rank_reml_matches_mgcv_invariantly():
    """Numerical stress is checked through fit and row-order invariant behavior."""
    data = _clustered_weighted_data()
    formula = 'y ~ s(x0, bs="cr", k=16) + s(x1, bs="ps", k=16)'
    weights = data["w"].to_numpy(dtype=np.float64)
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data, sample_weight=weights)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "gaussian",
        "REML",
        weights_column="w",
        optimizer="newton",
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=2e-3,
        pred_rtol=2e-3,
        sp_log_atol=0.1,
        # Near-collinear smooths do not uniquely allocate complexity term by
        # term; the total penalized response and uncertainty are identified.
        check_sp=False,
        criterion_atol=5e-2,
    )

    evaluation = data.iloc[::31].drop(columns=["y", "w"]).copy()
    reference, reference_se = gam.predict(
        evaluation, type="response", return_se=True
    )
    permutation = np.random.default_rng(271).permutation(len(data))
    permuted = data.iloc[permutation].reset_index(drop=True)
    permuted_gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray(
            actual["fit"]["smoothing_params"], dtype=np.float64
        ),
    ).fit(data=permuted, sample_weight=permuted["w"].to_numpy(dtype=np.float64))
    permuted_prediction, permuted_se = permuted_gam.predict(
        evaluation, type="response", return_se=True
    )
    np.testing.assert_allclose(
        permuted_prediction, reference, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(permuted_se, reference_se, atol=2e-6, rtol=2e-6)


def _large_n_k_stress_data() -> pd.DataFrame:
    data = _make_gaussian_data(seed=272, n=600)
    rng = np.random.default_rng(272)
    data["w"] = np.exp(rng.uniform(-2.0, 2.0, size=len(data)))
    return data


def _severe_knot_cluster_data() -> pd.DataFrame:
    rng = np.random.default_rng(273)
    centers = np.repeat(np.linspace(-1.0, 1.0, 24), 12)
    x0 = centers + 1e-10 * rng.standard_normal(len(centers))
    x1 = x0 + 2e-7 * rng.standard_normal(len(centers))
    y = np.sin(2.4 * x0) + 0.03 * rng.standard_normal(len(x0))
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "x1": x1,
            "w": np.exp(rng.uniform(-4.0, 4.0, size=len(x0))),
        }
    )


def _extreme_poisson_weight_data() -> pd.DataFrame:
    rng = np.random.default_rng(274)
    n = 360
    x0 = np.linspace(-2.5, 2.5, n)
    x1 = rng.uniform(-2.0, 2.0, size=n)
    eta = np.clip(0.25 + 1.1 * x0 + 0.3 * np.sin(2.0 * x1), -3.0, 4.0)
    return pd.DataFrame(
        {
            "y": rng.poisson(np.exp(eta)),
            "x0": x0,
            "x1": x1,
            "w": np.power(10.0, np.linspace(-4.0, 4.0, n)),
        }
    )


def _sparse_factor_cell_data() -> pd.DataFrame:
    rng = np.random.default_rng(275)
    counts = np.asarray([38, 30, 24, 20, 16, 12, 9, 7, 5, 4, 3, 2, 1, 1])
    f = np.concatenate(
        [np.repeat(f"g{i}", int(count)) for i, count in enumerate(counts)]
    )
    n = len(f)
    x0 = rng.uniform(-1.5, 1.5, size=n)
    x1 = rng.uniform(-1.2, 1.2, size=n)
    f1 = np.asarray([f"a{i % 5}" for i in range(n)])
    f2 = np.asarray([f"b{(i // 17) % 4}" for i in range(n)])
    effect = np.asarray([0.04 * int(value[1:]) for value in f])
    y = 0.3 * np.sin(1.4 * x0) + effect + rng.normal(scale=0.15, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "x1": x1,
            "f": pd.Categorical(f),
            "f1": pd.Categorical(f1),
            "f2": pd.Categorical(f2),
        }
    )


def _multi_near_rank_data(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 230
    latent = rng.uniform(-1.5, 1.5, size=n)
    x0 = latent + 1e-5 * rng.standard_normal(n)
    x1 = latent + 2e-5 * rng.standard_normal(n)
    y = np.sin(1.8 * latent) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


@dataclass(frozen=True)
class _NumericalStressCase:
    case_id: str
    data_factory: object
    family: object
    formula: str
    weighted: bool
    atol: float
    optimized: bool = False


_NUMERICAL_STRESS_CASES = [
    _NumericalStressCase(
        "large_n_large_k",
        _large_n_k_stress_data,
        "gaussian",
        'y ~ s(x0, bs="cr", k=26) + s(x1, bs="ps", k=24)',
        True,
        2e-3,
        optimized=True,
    ),
    _NumericalStressCase(
        "severe_knot_clustering",
        _severe_knot_cluster_data,
        "gaussian",
        (
            'y ~ s(x0, bs="cr", k=22, sp=0.7)'
            ' + s(x1, bs="ps", k=22, sp=1.1)'
        ),
        True,
        4e-3,
    ),
    _NumericalStressCase(
        "extreme_poisson_response_weights",
        _extreme_poisson_weight_data,
        "poisson",
        'y ~ ti(x0, x1, bs=["cr", "ps"], k=[8, 8], sp=[0.8, 1.0])',
        True,
        8e-3,
    ),
    _NumericalStressCase(
        "sparse_fs_cells",
        _sparse_factor_cell_data,
        "gaussian",
        'y ~ s(f, x0, bs="fs", k=5, xt="cr", sp=[0.7, 0.9, 1.1])',
        False,
        4e-3,
    ),
    _NumericalStressCase(
        "sparse_sz_cells",
        _sparse_factor_cell_data,
        "gaussian",
        'y ~ s(f1, f2, x0, bs="sz", k=5, id="shared", sp=0.8)',
        False,
        5e-3,
    ),
    *[
        _NumericalStressCase(
            f"near_rank_seed_{seed}",
            lambda seed=seed: _multi_near_rank_data(seed),
            "gaussian",
            (
                'y ~ s(x0, bs="cr", k=18, sp=0.7)'
                ' + s(x1, bs="ps", k=18, sp=1.1)'
            ),
            False,
            3e-3,
        )
        for seed in (276, 277, 278)
    ],
]


@pytest.mark.parametrize(
    "case", _NUMERICAL_STRESS_CASES, ids=lambda case: case.case_id
)
def test_numerical_stress_matrix_matches_live_mgcv_in_identified_space(case):
    """Large, extreme, sparse, clustered and multi-seed fits retain parity."""
    data = case.data_factory()
    weights = data["w"].to_numpy(dtype=np.float64) if case.weighted else None
    gam = GAM(
        family=case.family,
        formula=case.formula,
        optimize_smoothing=case.optimized,
        smoothing_method="REML" if case.optimized else "fixed",
        smoothing_optimizer="outer_newton",
    ).fit(data=data, sample_weight=weights)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        "REML" if case.optimized else "fixed",
        weights_column="w" if case.weighted else None,
        optimizer="newton" if case.optimized else None,
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.atol,
        pred_rtol=case.atol,
        sp_log_atol=0.2,
        check_sp=case.optimized,
        check_sp_count=case.optimized,
        # NAMpy reports expanded parametric factor blocks in this vector;
        # mgcv reports smooth terms only. Total EDF is the identified surface.
        check_edf_by_term=False,
        criterion_atol=0.12,
    )
    actual_edf2 = np.asarray(actual["fit"]["edf2"], dtype=np.float64)
    expected_edf2_raw = expected["fit"]["edf2"]
    if not isinstance(expected_edf2_raw, dict):
        expected_edf2 = np.asarray(expected_edf2_raw, dtype=np.float64)
        np.testing.assert_allclose(
            np.sum(actual_edf2),
            np.sum(expected_edf2),
            atol=max(0.02, 5.0 * case.atol),
            rtol=5e-3,
        )


def _formula_intersection_data() -> pd.DataFrame:
    data = _make_gaussian_data(seed=280, n=135)
    row = np.arange(len(data))
    data["f"] = pd.Categorical(np.asarray(["a", "b", "c"])[row % 3])
    data["z"] = 0.8 + 0.15 * np.cos(data["x0"].to_numpy(dtype=np.float64))
    data["off"] = 0.08 * np.sin(data["x1"].to_numpy(dtype=np.float64))
    level_effect = np.asarray(
        [{"a": -0.15, "b": 0.05, "c": 0.2}[str(value)] for value in data["f"]]
    )
    data["y"] = data["y"].to_numpy(dtype=np.float64) + data["off"] + level_effect
    return data


def test_formula_feature_intersection_matches_mgcv_on_newdata_behavior():
    """Transforms, linked factor-by, offset, and ti constraints coexist in one fit."""
    data = _formula_intersection_data()
    formula = (
        'y ~ f + I(x1**2) + offset(off)'
        ' + s(I(x0 + 0.1*x0**2), by=f, bs="cr", k=5, id="curve", sp=0.8)'
        ' + s(I(x1 + 0.1*x1**2), by=f, bs="cr", k=5, id="curve")'
        ' + ti(x0, x1, by=z, bs=["cs", "ps"], k=[5, 6],'
        " mc=[True, False], sp=[0.7, 1.1])"
    )
    _assert_fixed_newdata_response_se(data, formula, atol=2e-6)


def _ordinary_optimized_formula_intersection_data(family_id: str) -> pd.DataFrame:
    data = _formula_intersection_data().iloc[:105].copy()
    rng = np.random.default_rng(281)
    x0 = data["x0"].to_numpy(dtype=np.float64)
    x1 = data["x1"].to_numpy(dtype=np.float64)
    eta = 0.25 + 0.35 * np.sin(1.2 * x0) - 0.15 * x1 + data["off"].to_numpy()
    if family_id == "binomial":
        probability = 1.0 - np.exp(-np.exp(eta))
        data["y"] = rng.binomial(1, probability)
    elif family_id == "gamma":
        mu = np.exp(eta)
        data["y"] = rng.gamma(shape=3.0, scale=mu / 3.0)
    else:
        raise AssertionError(f"Unhandled family {family_id!r}")
    data["w"] = np.exp(np.linspace(-0.7, 0.7, len(data)))
    return data


def _general_formula_intersection_data(family_id: str) -> pd.DataFrame:
    if family_id == "gaulss":
        data = _gaulss_tensor_data(seed=282, n=115)
    elif family_id == "gammals":
        data = _gammals_tensor_data(seed=283, n=115)
    else:
        raise AssertionError(f"Unhandled family {family_id!r}")
    x0 = data["x0"].to_numpy(dtype=np.float64)
    x1 = data["x1"].to_numpy(dtype=np.float64)
    data["z"] = 0.9 + 0.15 * np.cos(x0)
    data["off1"] = 0.04 * np.sin(x1)
    data["off2"] = 0.03 * np.cos(x0)
    data["w"] = np.exp(np.linspace(-0.5, 0.5, len(data)))
    return data


@dataclass(frozen=True)
class _FormulaIntersectionCase:
    case_id: str
    data_factory: object
    family: object
    formula: object
    method: str
    optimizer: str
    select: bool
    check_sp: bool
    atol: float


_OPTIMIZED_FORMULA_INTERSECTIONS = [
    _FormulaIntersectionCase(
        "binomial_cloglog_linked_factor_by_tensor_select_weights",
        lambda: _ordinary_optimized_formula_intersection_data("binomial"),
        {"name": "binomial", "link": "cloglog"},
        (
            'y ~ f + I(x1**2) + offset(off)'
            ' + s(I(x0 + 0.1*x0**2), by=f, bs="cr", k=4, id="curve")'
            ' + s(I(x1 + 0.1*x1**2), by=f, bs="cr", k=4, id="curve")'
            ' + ti(x0, x1, by=z, bs=["cs", "ps"], k=[4, 4],'
            " mc=[True, False])"
        ),
        "REML",
        "outer_newton",
        True,
        True,
        2e-3,
    ),
    _FormulaIntersectionCase(
        "gamma_log_linked_factor_by_tensor_ml_weights",
        lambda: _ordinary_optimized_formula_intersection_data("gamma"),
        {"name": "gamma", "link": "log"},
        (
            'y ~ f + I(x1**2) + offset(off)'
            ' + s(I(x0 + 0.1*x0**2), by=f, bs="cr", k=4, id="curve")'
            ' + s(I(x1 + 0.1*x1**2), by=f, bs="cr", k=4, id="curve")'
            ' + ti(x0, x1, by=z, bs=["cr", "ps"], k=[4, 4],'
            " mc=[True, False])"
        ),
        "ML",
        "outer_newton",
        False,
        True,
        2e-3,
    ),
    _FormulaIntersectionCase(
        "gaulss_formula_list_tensor_offsets_select_weights",
        lambda: _general_formula_intersection_data("gaulss"),
        "gaulss",
        [
            (
                'y ~ I(x0 + 0.1*x0**2) + offset(off1)'
                ' + ti(x0, x1, by=z, bs=["cr", "ps"], k=[4, 4])'
            ),
            "~ I(x1**2) + offset(off2)",
        ],
        "ML",
        "outer_newton",
        True,
        False,
        2e-3,
    ),
    _FormulaIntersectionCase(
        "gammals_formula_list_transform_by_offsets_weights",
        lambda: _general_formula_intersection_data("gammals"),
        "gammals",
        [
            (
                'y ~ I(x1**2) + offset(off1)'
                ' + s(I(x0 + 0.1*x0**2), by=z, bs="cr", k=5)'
            ),
            "~ I(x0*x1) + offset(off2)",
        ],
        "ML",
        "outer_newton",
        False,
        True,
        3e-3,
    ),
]


@pytest.mark.parametrize(
    "case",
    _OPTIMIZED_FORMULA_INTERSECTIONS,
    ids=lambda case: case.case_id,
)
def test_optimized_formula_feature_intersections_match_live_mgcv(case):
    """Demanding formula features compose across links, families and optimizers."""
    data = case.data_factory()
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer=case.optimizer,
    ).fit(data=data, sample_weight=data["w"].to_numpy(dtype=np.float64))
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        case.formula,
        case.family,
        case.method,
        select=case.select,
        weights_column="w",
        optimizer=case.optimizer,
        allow_live_run=True,
    )
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.atol,
        pred_rtol=case.atol,
        sp_log_atol=0.2,
        check_sp=case.check_sp,
        # NAMpy reports expanded parametric factor blocks in this vector;
        # mgcv reports smooth terms only. Total EDF is the identified surface.
        check_edf_by_term=False,
        criterion_atol=0.12,
    )
    if not case.check_sp:
        actual_log_sp = np.log(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        expected_log_sp = np.log(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        actual_boundary = actual_log_sp > 10.0
        expected_boundary = expected_log_sp > 10.0
        # The raw endpoint is not identified on mgcv's high-penalty flat tail;
        # the boundary classification and every interior component are.
        np.testing.assert_array_equal(actual_boundary, expected_boundary)
        assert np.any(actual_boundary)
        np.testing.assert_allclose(
            actual_log_sp[~actual_boundary],
            expected_log_sp[~expected_boundary],
            atol=0.2,
            rtol=0.0,
        )
