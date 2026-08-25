"""Owner tests for the remaining upstream optimizer and criterion routes."""

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM, gam_control, normalize_nei
from nampy.gam.fit.selection.criteria import criterion_gradient, criterion_value
from nampy.gam.observations import ar1_log_determinant_correction


def _gaussian_data(seed=42, n=36):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    return pd.DataFrame({"x": x, "y": np.sin(1.7 * x) + rng.normal(0.0, 0.12, n)})


def test_gam_control_upstream_aliases_and_nested_defaults():
    control = gam_control(
        {
            "ncv.threads": 2,
            "scale.est": "deviance",
            "edge.correct": 0.01,
            "b.notexp": 1.5,
            "threshold.notexp": 18,
            "bfgs": {"maxNstep": 4, "gradtol.bfgs": 2e-6},
        },
        **{"efs.lspmax": 12},
    )
    assert control.ncv_threads == 2
    assert control.scale_est == "deviance"
    assert control.edge_correct == 0.01
    assert control.efs_lspmax == 12
    assert control.scam_b_notexp == 1.5
    assert control.scam_threshold_notexp == 18
    assert control.scam_bfgs["max_n_step"] == 4
    assert control.scam_bfgs["gradtol_bfgs"] == 2e-6
    assert control.nlm["iterlim"] == 200
    assert control.optim["factr"] == 1e7


def test_nei_accepts_mgcv_and_python_index_bases():
    one = normalize_nei({"a": [1, 2], "ma": [1, 2], "d": [1, 2], "md": [1, 2]}, 2)
    zero = normalize_nei(
        {
            "a": [0, 1],
            "ma": [1, 2],
            "d": [0, 1],
            "md": [1, 2],
            "index_base": 0,
        },
        2,
    )
    np.testing.assert_array_equal(one["a"], zero["a"])
    np.testing.assert_array_equal(one["d"], zero["d"])


def test_ncv_and_qncv_analytic_gradients_match_local_value_check():
    data = _gaussian_data()
    model = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="gaussian",
        smoothing_params=[1.0],
        optimize_smoothing=False,
    ).fit(data=data)
    point = np.array([0.2])
    step = 1e-5
    for method in ("ncv", "qncv"):
        actual = criterion_gradient(model, data.y.to_numpy(), point, method=method)
        check = (
            criterion_value(model, data.y.to_numpy(), point + step, method=method)
            - criterion_value(model, data.y.to_numpy(), point - step, method=method)
        ) / (2.0 * step)
        np.testing.assert_allclose(actual, [check], rtol=2e-6, atol=2e-8)


def test_known_scale_and_distinct_magic_nlm_identities():
    data = _gaussian_data(n=42)
    known = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="gaussian",
        scale=0.25,
        smoothing_method="GCV.Cp",
        optimize_smoothing=True,
    ).fit(data=data)
    assert known.family.known_scale == 0.25
    assert known._optim_method == "aic"
    assert known._optim_result.outer_info["optimizer"] == "magic"
    for optimizer in ("magic", "nlm"):
        model = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family="gaussian",
            smoothing_method="gcv",
            smoothing_optimizer=optimizer,
            optimize_smoothing=True,
        ).fit(data=data)
        assert model._optim_result.outer_info["optimizer"] == optimizer


def test_gacv_and_pearson_likelihood_derivatives_and_public_routes():
    data = _gaussian_data(seed=91, n=40)
    fixed = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="gaussian",
        smoothing_params=[1.0],
        optimize_smoothing=False,
    ).fit(data=data)
    point = np.array([0.15])
    step = 2e-5
    for method in ("gacv", "p-ml", "p-reml"):
        actual = criterion_gradient(fixed, data.y.to_numpy(), point, method=method)
        check = (
            criterion_value(fixed, data.y.to_numpy(), point + step, method=method)
            - criterion_value(fixed, data.y.to_numpy(), point - step, method=method)
        ) / (2.0 * step)
        np.testing.assert_allclose(actual, [check], rtol=3e-5, atol=3e-7)

    for method in ("P-ML", "P-REML"):
        fitted = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family="gaussian",
            smoothing_method=method,
            optimize_smoothing=True,
        ).fit(data=data)
        assert fitted._optim_method == method.lower()
        assert np.isfinite(fitted.smoothing_score_)
        assert fitted.sp_vcov() is not None
        assert fitted.fit_result().cov_unconditional is not None
        assert fitted.fit_result().edf2 is not None


def test_new_criterion_endpoints_match_vendored_mgcv_1_9_4():
    x = np.linspace(0.02, 0.98, 40)
    y = 1.0 + 0.35 * np.sin(2.0 * np.pi * x) + 0.04 * np.cos(11.0 * x)
    data = pd.DataFrame({"x": x, "y": y})
    # Generated directly from the vendored mgcv 1.9-4 sources with
    # gam(y ~ s(x, bs="cr", k=6), method=<key>).
    expected = {
        "GACV.Cp": (3.052098717645989e-05, 0.004478168658632537, 5.997325699650938),
        "P-ML": (-30.67857511660512, 22.45862323247254, 3.776363326853716),
        "P-REML": (-24.5464962732863, 25.35773440627658, 3.703734335889844),
        "NCV": (0.001553310031570951, 1.233626458657467e-06, 5.99999926224917),
        "QNCV": (0.001553310031570962, 1.233626458610274e-06, 5.99999926224917),
    }
    pearson_postfit = {
        "P-ML": (0.119455672746176, 4.37917161734447),
        "P-REML": (0.138060157301853, 4.29724688456656),
    }
    for method, (score, sp, edf) in expected.items():
        fitted = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family="gaussian",
            smoothing_method=method,
            optimize_smoothing=True,
        ).fit(data=data)
        np.testing.assert_allclose(fitted.smoothing_score_, score, rtol=2e-11)
        np.testing.assert_allclose(fitted.smoothing_params, [sp], rtol=2e-9)
        np.testing.assert_allclose(fitted.fit_result().edf_total, edf, rtol=2e-11)
        if method in pearson_postfit:
            sp_var, edf2 = pearson_postfit[method]
            np.testing.assert_allclose(fitted.sp_vcov(), [[sp_var]], rtol=2e-11)
            np.testing.assert_allclose(
                np.sum(fitted.fit_result().edf2), edf2, rtol=2e-11
            )


def test_explicit_nei_ncv_endpoints_match_vendored_mgcv_1_9_4():
    n = 40
    x = np.linspace(0.02, 0.98, n)
    y = 1.0 + 0.35 * np.sin(2.0 * np.pi * x) + 0.04 * np.cos(11.0 * x)
    dropped = []
    for index in range(n):
        dropped.extend([index + 1, (index + 1) % n + 1])
    nei = {
        "a": dropped,
        "ma": np.arange(2, 2 * n + 1, 2),
        "d": np.arange(1, n + 1),
        "md": np.arange(1, n + 1),
    }
    expected = {
        "NCV": (0.002453467873637, 5.6540568140832164e-07),
        "QNCV": (0.0024534678736369917, 5.654056813866827e-07),
    }
    for method, (score, sp) in expected.items():
        fitted = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family="gaussian",
            smoothing_method=method,
            nei=nei,
            optimize_smoothing=True,
        ).fit(data=pd.DataFrame({"x": x, "y": y}))
        np.testing.assert_allclose(fitted.smoothing_score_, score, rtol=2e-12)
        np.testing.assert_allclose(fitted.smoothing_params, [sp], rtol=2e-10)
        np.testing.assert_allclose(
            fitted.fit_result().edf_total, 5.999999661867985, rtol=2e-12
        )


def test_magic_known_scale_gaussian_endpoint_matches_vendored_mgcv_1_9_4():
    x = np.linspace(0.02, 0.98, 40)
    y = 1.0 + 0.35 * np.sin(2.0 * np.pi * x) + 0.04 * np.cos(11.0 * x)
    fitted = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="gaussian",
        scale=0.2,
        smoothing_method="GCV.Cp",
        smoothing_optimizer="magic",
        optimize_smoothing=True,
    ).fit(data=pd.DataFrame({"x": x, "y": y}))
    assert fitted._optim_method == "aic"
    assert fitted._optim_result.outer_info["optimizer"] == "magic"
    np.testing.assert_allclose(fitted.smoothing_score_, -0.1617024934179115, rtol=2e-12)
    np.testing.assert_allclose(fitted.smoothing_params, [1133965.41757369], rtol=2e-10)
    np.testing.assert_allclose(
        fitted.fit_result().edf_total, 2.00021432467268, rtol=2e-12
    )


def test_known_scale_gamma_ubre_endpoint_matches_vendored_mgcv_1_9_4():
    x = np.linspace(0.02, 0.98, 40)
    y = np.exp(0.2 + 0.6 * np.sin(2.0 * np.pi * x)) * (1.0 + 0.03 * np.cos(13.0 * x))
    fitted = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="gamma",
        scale=0.02,
        smoothing_method="GCV.Cp",
        optimize_smoothing=True,
    ).fit(data=pd.DataFrame({"x": x, "y": y}))
    assert fitted._optim_method == "aic"
    assert fitted.family.known_scale == 0.02
    np.testing.assert_allclose(fitted.smoothing_score_, -0.0134659137919778, rtol=2e-12)
    np.testing.assert_allclose(fitted.smoothing_params, [0.813191611193724], rtol=2e-10)
    np.testing.assert_allclose(
        fitted.fit_result().edf_total, 5.69998710611251, rtol=2e-12
    )


def test_remaining_scam_outer_and_coefficient_optimizer_routes():
    rng = np.random.default_rng(117)
    x = np.linspace(-1.0, 1.0, 36)
    data = pd.DataFrame(
        {"x": x, "y": 1.0 + np.exp(0.35 * x) + rng.normal(0.0, 0.06, x.size)}
    )
    routes = (
        ("efs", "newton"),
        ("optim", "newton"),
        ("nlm", "newton"),
        ("nlm.fd", "newton"),
        ("efs", "bfgs"),
    )
    for optimizer, coefficient_optimizer in routes:
        fitted = GAM(
            formula='y ~ s(x, bs="mpi", k=6)',
            family="gaussian",
            smoothing_method="gcv",
            smoothing_optimizer=optimizer,
            coefficient_optimizer=coefficient_optimizer,
            optimize_smoothing=True,
        ).fit(data=data)
        assert fitted._optim_result.outer_info["optimizer"] == optimizer
        assert np.all(np.isfinite(fitted.fit_result().coef_full))
        if coefficient_optimizer == "bfgs":
            assert fitted.coefficient_optimizer == "bfgs"


def test_intentional_scam_and_multipredictor_boundaries_remain_guarded():
    data = _gaussian_data(n=30)
    with pytest.raises(NotImplementedError, match="GCV/UBRE"):
        GAM(
            formula='y ~ s(x, bs="mpi", k=6)',
            family="gaussian",
            smoothing_method="REML",
            optimize_smoothing=True,
        ).fit(data=data)

    with pytest.raises(NotImplementedError, match="transformed Laplace derivatives"):
        GAM(
            formula=[
                'y ~ s(x, bs="mpi", k=6)',
                '~ s(x, bs="cr", k=5)',
            ],
            family="gaulss",
            smoothing_method="REML",
            optimize_smoothing=True,
        ).fit(data=data)

    with pytest.raises(NotImplementedError, match="exposed upstream by SCAM"):
        GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family="gaussian",
            smoothing_method="GCV.Cp",
            smoothing_optimizer="nlm.fd",
            optimize_smoothing=True,
        ).fit(data=data)


def test_tweedie_efs_supports_min_sp_and_fixed_or_estimated_power():
    x = np.linspace(0.02, 0.98, 40)
    y = np.exp(0.15 + 0.5 * np.sin(2.0 * np.pi * x))
    y[::7] = 0.0
    data = pd.DataFrame({"x": x, "y": y})
    for theta in (1.5, -1.3):
        fitted = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family={"name": "tw", "theta": theta},
            smoothing_method="REML",
            smoothing_optimizer="efs",
            min_sp=[0.02],
            optimize_smoothing=True,
        ).fit(data=data)
        assert fitted._optim_result.outer_info["optimizer"] == "efs"
        assert fitted.smoothing_params[0] >= 0.02
        assert fitted.family.a < fitted.family.p < fitted.family.b


def test_laml_scores_additional_upstream_extended_family_classes():
    x = np.linspace(0.03, 0.97, 36)
    wave = np.sin(2.0 * np.pi * x)
    cases = (
        ("poisson", np.maximum(0, np.rint(np.exp(0.5 + 0.4 * wave)))),
        ({"name": "betar", "theta": 8.0}, np.clip(0.5 + 0.25 * wave, 0.02, 0.98)),
        (
            {"name": "ocat", "theta": [0.8, 0.8, 0.8]},
            np.asarray(1 + np.searchsorted([-0.5, 0.3, 1.1], wave)),
        ),
        (
            {"name": "tw", "theta": 1.5},
            np.where(np.arange(x.size) % 8 == 0, 0.0, np.exp(0.2 + 0.3 * wave)),
        ),
    )
    for family, y in cases:
        fitted = GAM(
            formula='y ~ s(x, bs="cr", k=6)',
            family=family,
            smoothing_method="LAML",
            optimize_smoothing=True,
        ).fit(data=pd.DataFrame({"x": x, "y": y}))
        assert fitted._optim_method == "laml"
        assert np.isfinite(fitted.smoothing_score_)
        assert np.all(np.isfinite(fitted.smoothing_params))


def test_bam_ar1_determinant_formula_counts_series_starts():
    model = type("M", (), {})()
    model.ar1_rho = 0.4
    model.n_samples_ = 20
    model.ar_start_ = np.array([True] + [False] * 9 + [True] + [False] * 9)
    expected = -(20 - 2) * np.log(1.0 / np.sqrt(1.0 - 0.4**2))
    assert ar1_log_determinant_correction(model) == expected
