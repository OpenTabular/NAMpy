"""Strict xfail registry for remaining concrete mgcv parity gaps.

This file intentionally tracks only genuine failing or explicitly unsupported
surfaces. Do not add xfails for behavior that is already green.

Other remaining gap buckets already live elsewhere:
- post-fit / final-fit gaps: ``tests/optimization/test_mgcv_postprocessing_final_fit_parity.py``
- raw constructor gaps: ``tests/smooths/test_mgcv_raw_constructor_parity.py``
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.fit.backends import solve_pirls_given_smoothing
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.smoothing_selection.criteria.pirls.derivatives import _gdi2_joint_kernel
from nampy.gam.specs.build import build_formula_model
from tests._paths import REPO_ROOT
from tests.families.test_general_family_mgcv_parity import GAULSS_FORMULA, _gaulss_data
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _make_random_effect_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_snapshot as _coverage_fit_nampy_snapshot,
)
from tests.mgcv_parity_utils import (
    _make_binomial_data as _coverage_make_binomial_data,
)
from tests.mgcv_parity_utils import (
    _make_gamma_data as _coverage_make_gamma_data,
)
from tests.mgcv_parity_utils import (
    _make_gaussian_data as _coverage_make_gaussian_data,
)
from tests.mgcv_parity_utils import (
    _make_gaussian_data_3col as _coverage_make_gaussian_data_3col,
)
from tests.mgcv_parity_utils import (
    _make_negbin_data as _coverage_make_negbin_data,
)
from tests.mgcv_parity_utils import (
    _make_poisson_data as _coverage_make_poisson_data,
)
from tests.mgcv_parity_utils import (
    _run_mgcv_snapshot as _coverage_run_mgcv_snapshot,
)

R_SCRIPT = shutil.which("Rscript")


def _build_formula_only(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def _run_mgcv_random_effect_id_error(data: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        script_path = tmpdir_path / "re_id_error.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(
            """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
payload <- tryCatch(
  {
    gam(y ~ s(f, bs="re", id="shared"), data = d, method = "REML")
    list(ok = TRUE, message = NULL)
  },
  error = function(e) list(ok = FALSE, message = conditionMessage(e))
)
write_json(payload, args[[2]], auto_unbox = TRUE, digits = 17)
""".strip(),
            encoding="utf-8",
        )
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def test_formula_dot_shorthand_builds_with_data_context():
    """
    Known-gap coverage verifying that formula dot shorthand builds with data context.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
            "z": [2.0, 1.0, 0.0],
        }
    )

    built = _build_formula_only("y ~ .", data)

    assert built.feature_names == ["x", "z"]
    assert [term.label for term in built.predictor_specs[0].terms] == ["x", "z"]


def test_formula_mixed_positional_and_keyword_list_args_build():
    """
    Known-gap coverage verifying that formula mixed positional and keyword list args
    build.
    """
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

    built = _build_formula_only('y ~ s(x, bs="tp", xt=list(1, bs="ps"))', data)

    smooth = built.predictor_specs[0].terms[0]
    assert smooth.kind == "smooth"
    assert smooth.smooth_spec is not None
    assert smooth.smooth_spec.xt == {0: 1, "bs": "ps"}


def test_formula_transformed_smooth_covariate_builds_end_to_end():
    """
    Known-gap coverage verifying that formula transformed smooth covariate builds end to
    end.
    """
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

    built = _build_formula_only('y ~ s(I(x**2), k=5, bs="cr")', data)

    term = built.predictor_specs[0].terms[0]
    assert term.features[0] in built.working_data.columns


def test_formula_multi_predictor_distinct_offsets_build():
    """
    Known-gap coverage verifying that formula multi predictor distinct offsets build.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [0.5, 1.5, 2.5, 3.5],
            "o1": [0.1, 0.2, 0.3, 0.4],
            "o2": [0.4, 0.3, 0.2, 0.1],
        }
    )

    built = _build_formula_only(
        [
            'y ~ s(x, bs="cr", k=5) + offset(o1)',
            '~ s(z, bs="cr", k=5) + offset(o2)',
        ],
        data,
    )

    assert [pred.offset_name for pred in built.predictor_specs] == ["o1", "o2"]
    assert built.preprocess_state["offset_names"] == ("o1", "o2")
    assert isinstance(built.offsets, list)
    assert_allclose(built.offsets[0], data["o1"].to_numpy(dtype=np.float64))
    assert_allclose(built.offsets[1], data["o2"].to_numpy(dtype=np.float64))


def test_general_family_formula_multi_predictor_offsets_predict_with_defaults():
    """
    Known-gap coverage verifying that general family formula multi predictor offsets
    predict with defaults.
    """
    rng = np.random.default_rng(12)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    o1 = np.linspace(-0.25, 0.25, n)
    o2 = np.linspace(0.2, -0.2, n)
    mu = 0.4 + 0.3 * x + o1
    sigma = np.exp(-0.25 + o2)
    y = rng.normal(mu, sigma, size=n)
    data = pd.DataFrame({"y": y, "x": x, "o1": o1, "o2": o2})

    gam = GAM(
        family="gaulss",
        formula=["y ~ x + offset(o1)", "~ 1 + offset(o2)"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)

    eta_default = np.asarray(gam.predict(type="link"), dtype=np.float64)
    eta_from_data = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    eta_zero = np.asarray(
        gam.predict(data.assign(o1=np.zeros(n), o2=np.zeros(n)), type="link"),
        dtype=np.float64,
    )

    assert eta_default.shape == (n, 2)
    assert_allclose(eta_default, eta_from_data, atol=1e-10, rtol=1e-10)
    assert_allclose(eta_default[:, 0] - eta_zero[:, 0], o1, atol=1e-10, rtol=1e-10)
    assert_allclose(eta_default[:, 1] - eta_zero[:, 1], o2, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity")
def test_random_effect_id_linkage_is_explicitly_unsupported_like_mgcv():
    """
    Known-gap coverage verifying that random effect id linkage is explicitly unsupported
    like mgcv.
    """
    data = _make_random_effect_data()
    expected = _run_mgcv_random_effect_id_error(data)

    assert expected["ok"] is False
    assert "random effects don't work with ids" in str(expected["message"]).lower()

    gam = GAM(
        family="gaussian",
        formula='y ~ s(f, bs="re", id="shared")',
        optimize_smoothing=True,
        smoothing_method="REML",
    )

    with pytest.raises(NotImplementedError, match="random effects don't work with ids"):
        gam.fit(data=data)


def test_general_family_generic_gdi2_kernel_available_for_gaulss():
    """
    Known-gap coverage verifying that general family generic GDI2 kernel available for
    gaulss.
    """
    data = _gaulss_data()
    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", [1.0])
    y = data["y"].to_numpy(dtype=float)
    sp = gam.smoothing_params
    sol = solve_pirls_given_smoothing(gam, y, sp)

    _gdi2_joint_kernel(gam, y, sol, sp, method="REML", need_hessian=True)


def _coverage_factor_by_data(seed=901, n=180):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, size=n)
    z = rng.uniform(-1.2, 1.2, size=n)
    f = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    scale = {"a": 0.5, "b": -0.25, "c": 0.8}
    y = (np.sin(x) + 0.2 * z**2) * np.array([scale[str(v)] for v in f]) + rng.normal(
        scale=0.08,
        size=n,
    )
    return pd.DataFrame({"y": y, "x": x, "z": z, "f": f})


def _coverage_poisson_factor_data(seed=902, n=180):
    rng = np.random.default_rng(seed)
    data = _coverage_make_poisson_data(seed=seed, n=n).copy()
    data["f"] = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    return data


def _coverage_binomial_sz_data(seed=903, n=180):
    rng = np.random.default_rng(seed)
    data = _coverage_make_binomial_data(seed=seed, n=n).copy()
    data["f1"] = rng.choice(np.array(["a", "b", "c"], dtype=object), size=n)
    data["f2"] = rng.choice(np.array(["u", "v"], dtype=object), size=n)
    return data


def _coverage_random_slope_data(seed=904, n=160):
    rng = np.random.default_rng(seed)
    f = rng.choice(np.array(["a", "b", "c", "d"], dtype=object), size=n)
    x = rng.uniform(-1.5, 1.5, size=n)
    intercept = {"a": -0.4, "b": 0.2, "c": 0.6, "d": -0.1}
    slope = {"a": 0.1, "b": -0.2, "c": 0.35, "d": -0.45}
    y = np.array([intercept[str(v)] + slope[str(v)] * x[i] for i, v in enumerate(f)])
    y = y + rng.normal(scale=0.06, size=n)
    return pd.DataFrame({"y": y, "x": x, "f": f})


_REMAINING_SNAPSHOT_GAP_CASES = [
    pytest.param(
        "tensor_te_ps_ps_optimized",
        _coverage_make_gaussian_data,
        'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 6])',
        "gaussian",
        "REML",
        id="tensor_te_ps_ps_optimized",
    ),
    pytest.param(
        "tensor_ti_ps_ps_optimized",
        _coverage_make_gaussian_data,
        'y ~ ti(x0, x1, bs=["ps", "ps"], k=[6, 6])',
        "gaussian",
        "REML",
        id="tensor_ti_ps_ps_optimized",
    ),
    pytest.param(
        "tensor_te_three_margin",
        _coverage_make_gaussian_data_3col,
        'y ~ te(x0, x1, x2, bs=["cr", "cr", "cr"], k=[5, 5, 5])',
        "gaussian",
        "REML",
        id="tensor_te_three_margin",
    ),
    pytest.param(
        "tensor_linked_id",
        _coverage_make_gaussian_data,
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], id="shared") + te(x0, x1, bs=["cr", "cr"], k=[5, 5], id="shared")',
        "gaussian",
        "REML",
        id="tensor_linked_id",
    ),
    pytest.param(
        "tensor_factor_by",
        _coverage_factor_by_data,
        'y ~ f + te(x, z, by=f, bs=["cr", "cr"], k=[5, 5])',
        "gaussian",
        "REML",
        id="tensor_factor_by",
    ),
    pytest.param(
        "fs_poisson",
        _coverage_poisson_factor_data,
        'y ~ s(x0, f, bs="fs", k=5)',
        "poisson",
        "REML",
        id="fs_poisson",
    ),
    pytest.param(
        "sz_binomial",
        _coverage_binomial_sz_data,
        'y ~ s(x0, f1, f2, bs="sz", k=5)',
        "binomial",
        "REML",
        id="sz_binomial",
    ),
    pytest.param(
        "random_slope_gaussian",
        _coverage_random_slope_data,
        'y ~ s(f, by=x, bs="re")',
        "gaussian",
        "REML",
        id="random_slope_gaussian",
    ),
    pytest.param(
        "random_effect_interaction_gaussian",
        _coverage_random_slope_data,
        'y ~ s(f, x, bs="re")',
        "gaussian",
        "REML",
        id="random_effect_interaction_gaussian",
    ),
    pytest.param(
        "random_effect_poisson",
        _coverage_poisson_factor_data,
        'y ~ s(f, bs="re")',
        "poisson",
        "REML",
        id="random_effect_poisson",
    ),
    pytest.param(
        "negbin_estimated_theta_identity",
        lambda: _coverage_make_negbin_data(seed=910, n=220, theta=1.4),
        'y ~ s(x0, bs="cr", k=8)',
        {"name": "negbin", "theta": 1.4, "estimate_theta": True, "link": "identity"},
        "REML",
        id="negbin_estimated_theta_identity",
    ),
    pytest.param(
        "gamma_joint_scale_bfgs",
        _coverage_make_gamma_data,
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
        "gamma",
        "REML",
        id="gamma_joint_scale_bfgs",
    ),
]


@pytest.mark.parametrize(
    "case_id,data_factory,formula,family,method",
    _REMAINING_SNAPSHOT_GAP_CASES,
)
def test_remaining_snapshot_gap_surface_matches_mgcv(case_id, data_factory, formula, family, method):
    """Strict xfail registry for remaining full-fit/snapshot parity surfaces."""
    data = data_factory()
    actual = _coverage_fit_nampy_snapshot(data, formula, family, method)
    expected = _coverage_run_mgcv_snapshot(data, formula, family, method)
    for key in ("response", "link"):
        assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=2e-5,
            rtol=2e-5,
        )


def test_transformed_by_smooth_is_rejected_explicitly_until_supported():
    """Verify transformed by-variable smooths are explicit unsupported behavior."""
    rng = np.random.default_rng(962)
    n = 60
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = np.sin(x) * (z > 0.0) + rng.normal(scale=0.05, size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z})
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, by=I(z > 0), bs="cr", k=6)',
    )

    with pytest.raises((NotImplementedError, ValueError)):
        gam.fit(data=data)


@pytest.mark.xfail(
    strict=True,
    reason="Exact gam.check diagnostic parity remains a tracked gap.",
)
def test_exact_gam_check_mgcv_parity_gap_is_tracked():
    """Strict xfail for exact gam_check parity beyond local report shape."""
    data = _coverage_make_gaussian_data(seed=970, n=120)
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    report = gam.gam_check(type="deviance", k_sample=80, k_rep=8, seed=7)
    expected = _coverage_run_mgcv_snapshot(data, gam.formula, "gaussian", "REML")
    assert_allclose(
        np.asarray(report["mgcv_comparable"]["k_check"], dtype=np.float64),
        np.asarray(expected["parity"]["diagnostics"]["gam_check"], dtype=np.float64),
    )


@pytest.mark.xfail(
    strict=True,
    reason="Exact summary.gam text/statistic parity remains a tracked gap.",
)
def test_exact_summary_mgcv_parity_gap_is_tracked():
    """Strict xfail for full summary.gam text/statistic parity."""
    data = _coverage_make_poisson_data(seed=971, n=140)
    gam = GAM(
        family="poisson",
        formula='y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)',
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)
    expected = _coverage_run_mgcv_snapshot(data, gam.formula, "poisson", "REML")
    text = gam.summary()
    assert str(expected["parity"]["diagnostics"]["summary_text"]) == text


def test_exact_bic_mgcv_parity_gap_is_tracked():
    """Verify direct BIC parity against mgcv."""
    data = _coverage_make_gaussian_data(seed=972, n=120)
    formula = 'y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)'
    gam = GAM(family="gaussian", formula=formula, optimize_smoothing=True, smoothing_method="REML")
    gam.fit(data=data)
    expected = _coverage_run_mgcv_snapshot(data, formula, "gaussian", "REML")
    assert gam.bic() == pytest.approx(float(expected["parity"]["diagnostics"]["bic"]))
