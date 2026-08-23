"""Strict regressions promoted from the former mgcv known-gap registry.

There are currently no active expected failures on the targeted GAM slices;
future upstream-localized gaps belong beside their exact owning surface.
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
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.fit.selection.criteria.pirls.joint_kernels import gdi2_joint_kernel
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model
from tests._paths import REPO_ROOT
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gammals_data,
    _gaulss_data,
)
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _fit_nampy_model_fixed_sp,
    _make_random_effect_data,
    _run_mgcv_predict_on_newdata,
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
from tests.reference_fixtures import (
    load_reference,
    portable_dataframe_identity,
    reference_key,
    save_reference,
)

R_SCRIPT = shutil.which("Rscript")


def _build_formula_only(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def _run_mgcv_random_effect_id_error(data: pd.DataFrame):
    key = reference_key(
        "random_effect_id_error", {"data": portable_dataframe_identity(data)}
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
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
        result = json.loads(json_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


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
    The generic gdi2 kernel must decompose the gaulss REML score consistently.

    Beyond mere availability, the kernel's ``Dp/(2*gamma) + K`` split
    (mgcv/R/gam.fit4.r::gam.fit5) must recombine to the parity-tested
    criterion derivatives, and the exact gradient must agree with central
    finite differences of the criterion itself.
    """
    data = _gaulss_data()
    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", [1.0])
    y = data["y"].to_numpy(dtype=float)
    sp = gam.smoothing_params
    log_sp = np.log(np.asarray(sp, dtype=np.float64))
    sol = solve_pirls_given_smoothing(gam, y, sp)

    kernel = gdi2_joint_kernel(gam, y, sol, sp, method="REML", need_hessian=True)

    gamma = float(gam.score_gamma)
    assert np.isfinite(float(kernel.Dp))
    grad_kernel = np.asarray(kernel.Dp1, dtype=np.float64) / (
        2.0 * gamma
    ) + np.asarray(kernel.K1_full, dtype=np.float64)
    hess_kernel = np.asarray(kernel.Dp2, dtype=np.float64) / (
        2.0 * gamma
    ) + np.asarray(kernel.K2_full, dtype=np.float64)
    assert grad_kernel.shape == log_sp.shape
    assert hess_kernel.shape == (log_sp.size, log_sp.size)
    assert np.all(np.isfinite(grad_kernel))
    assert np.all(np.isfinite(hess_kernel))
    assert_allclose(hess_kernel, hess_kernel.T, atol=1e-10, rtol=0.0)

    grad_dispatch = np.asarray(
        criterion_gradient(gam, y, log_sp, method="reml"), dtype=np.float64
    )
    hess_dispatch = np.asarray(
        criterion_hessian(gam, y, log_sp, method="reml"), dtype=np.float64
    )
    assert_allclose(grad_kernel, grad_dispatch, atol=1e-8, rtol=1e-8)
    assert_allclose(hess_kernel, hess_dispatch, atol=1e-8, rtol=1e-8)

    # Independent derivative check: central finite differences of the
    # criterion value itself.
    step = 1e-4
    fd_grad = np.empty_like(log_sp)
    for i in range(log_sp.size):
        lo = log_sp.copy()
        hi = log_sp.copy()
        lo[i] -= step
        hi[i] += step
        fd_grad[i] = (
            float(criterion_value(gam, y, hi, method="reml"))
            - float(criterion_value(gam, y, lo, method="reml"))
        ) / (2.0 * step)
    assert_allclose(grad_kernel, fd_grad, atol=5e-5, rtol=5e-5)


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


_REGRESSION_SNAPSHOT_CASES = [
    pytest.param(
        "tensor_te_vector_fx",
        _coverage_make_gaussian_data,
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6], fx=[True, False])',
        "gaussian",
        "REML",
        id="tensor_te_vector_fx",
    ),
    pytest.param(
        "tensor_ti_vector_fx",
        _coverage_make_gaussian_data,
        'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6], fx=[True, False], mc=[True, False])',
        "gaussian",
        "REML",
        id="tensor_ti_vector_fx",
    ),
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
        'y ~ s(x0, f, bs="fs", k=5, xt="cc")',
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
    _REGRESSION_SNAPSHOT_CASES,
)
def test_promoted_snapshot_surface_matches_mgcv(
    case_id, data_factory, formula, family, method
):
    """Verify formerly tracked full-fit surfaces remain green against mgcv."""
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


def test_transformed_numeric_by_smooth_matches_mgcv():
    """Verify transformed numeric by-variables match mgcv at fit and prediction."""
    rng = np.random.default_rng(962)
    n = 60
    x = rng.uniform(-1.0, 1.0, size=n)
    z = rng.uniform(-1.0, 1.0, size=n)
    by = np.log(z + 2.0)
    y = np.sin(x) * by + rng.normal(scale=0.05, size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z})
    formula = 'y ~ s(x, by=log(z + 2), bs="cr", k=6)'
    actual = _coverage_fit_nampy_snapshot(
        data,
        formula,
        "gaussian",
        "REML",
    )
    expected = _coverage_run_mgcv_snapshot(
        data,
        formula,
        "gaussian",
        "REML",
    )

    for key in ("response", "link"):
        assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=2e-7,
            rtol=2e-7,
        )
    assert_allclose(
        np.asarray(actual["fit"]["log_smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=2e-6,
        rtol=0.0,
    )

    newdata = pd.DataFrame(
        {
            "x": np.linspace(-0.8, 0.8, 17),
            "z": np.linspace(-0.7, 0.7, 17),
        }
    )
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")
    for pred_type in ("link", "response"):
        expected_new = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family="gaussian",
            method="REML",
            type=pred_type,
        )
        assert_allclose(
            np.asarray(gam.predict(newdata, type=pred_type), dtype=np.float64).ravel(),
            np.asarray(expected_new["pred"], dtype=np.float64).ravel(),
            atol=2e-7,
            rtol=2e-7,
        )


def test_logical_transformed_by_smooth_is_rejected_explicitly():
    """Mirror mgcv::get.var(), which rejects non-numeric logical by results."""
    data = pd.DataFrame(
        {
            "y": [0.1, 0.3, 0.2, 0.5, 0.4, 0.7],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "z": [-1.0, 0.5, -0.3, 0.8, -0.7, 1.0],
        }
    )
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, by=I(z > 0), bs="cr", k=5)',
    )

    with pytest.raises(NotImplementedError, match="Unsupported formula expression"):
        gam.fit(data=data)


def test_predict_iterms_matches_mgcv_on_newdata():
    """Verify constrained-term SEs include mgcv's uncertainty about the mean."""
    data = _coverage_make_gaussian_data(seed=963, n=140)
    formula = 'y ~ x1 + s(x0, bs="cr", k=7)'
    snapshot = _coverage_run_mgcv_snapshot(data, formula, "gaussian", "REML")
    sp = np.asarray(snapshot["fit"]["smoothing_params"], dtype=np.float64)
    gam = _fit_nampy_model_fixed_sp(data, formula, "gaussian", sp)
    newdata = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, 19),
            "x1": np.linspace(0.8, -0.8, 19),
        }
    )

    actual_fit, actual_se = gam.predict(newdata, type="iterms", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="iterms",
        return_se=True,
    )

    assert_allclose(
        np.asarray(actual_fit, dtype=np.float64),
        np.asarray(expected["pred"], dtype=np.float64),
        atol=2e-8,
        rtol=2e-8,
    )
    assert_allclose(
        np.asarray(actual_se, dtype=np.float64),
        np.asarray(expected["se"], dtype=np.float64),
        atol=2e-8,
        rtol=2e-8,
    )


def test_bic_matches_mgcv():
    """Verify direct BIC parity against mgcv."""
    data = _coverage_make_gaussian_data(seed=972, n=120)
    formula = 'y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)'
    gam = GAM(family="gaussian", formula=formula, optimize_smoothing=True, smoothing_method="REML")
    gam.fit(data=data)
    expected = _coverage_run_mgcv_snapshot(data, formula, "gaussian", "REML")
    assert gam.bic() == pytest.approx(float(expected["parity"]["diagnostics"]["bic"]))


@pytest.mark.parametrize(
    ("case_id", "family", "formula", "data_factory", "method"),
    [
        (
            "poisson_reml",
            "poisson",
            'y ~ s(x0, bs="cr", k=8)',
            _coverage_make_poisson_data,
            "REML",
        ),
        (
            "negbin_fixed_theta_reml",
            {"name": "negbin", "theta": 1.8},
            'y ~ s(x0, bs="cr", k=8)',
            _coverage_make_negbin_data,
            "REML",
        ),
        (
            "negbin_est_theta_reml",
            {"name": "negbin", "theta": 1.8, "estimate_theta": True},
            'y ~ s(x0, bs="cr", k=8)',
            _coverage_make_negbin_data,
            "REML",
        ),
        (
            "gaulss_ml",
            "gaulss",
            GAULSS_FORMULA,
            _gaulss_data,
            "ML",
        ),
        # gamma, non-identity-link gaussian, and gammals were the remaining
        # implemented loglik/bic surfaces without parity coverage.
        (
            "gamma_log_reml",
            {"name": "gamma", "link": "log"},
            'y ~ s(x0, bs="cr", k=8)',
            _coverage_make_gamma_data,
            "REML",
        ),
        (
            "gaussian_log_reml",
            {"name": "gaussian", "link": "log"},
            'y ~ s(x0, bs="cr", k=8)',
            _coverage_make_gamma_data,
            "REML",
        ),
        (
            "gammals_ml",
            "gammals",
            ['y ~ s(x, bs="cr", k=6)', "~ 1"],
            _gammals_data,
            "ML",
        ),
    ],
    ids=[
        "poisson_reml",
        "negbin_fixed_theta_reml",
        "negbin_est_theta_reml",
        "gaulss_ml",
        "gamma_log_reml",
        "gaussian_log_reml",
        "gammals_ml",
    ],
)
def test_loglik_aic_bic_match_mgcv(case_id, family, formula, data_factory, method):
    """
    logLik/AIC/BIC must follow mgcv's `logLik.gam` semantics for the
    non-`object$aic` families too (negbin, general families take the
    fallback path in `GAM.loglik()`/`aic()`/`bic()`).
    """
    data = data_factory()
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    expected = _coverage_run_mgcv_snapshot(data, formula, family, method)

    assert gam.loglik() == pytest.approx(float(expected["fit"]["loglik"]), rel=1e-6)
    assert gam.aic() == pytest.approx(float(expected["fit"]["aic"]), rel=1e-6)
    assert gam.bic() == pytest.approx(
        float(expected["parity"]["diagnostics"]["bic"]), rel=1e-6
    )
