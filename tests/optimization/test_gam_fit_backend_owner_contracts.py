from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.fit import backends as backends_module
from nampy.gam.fit.backends import (
    solve_gaussian_given_smoothing,
    solve_pirls_given_smoothing,
)

pytestmark = [
    pytest.mark.surface_backend,
    pytest.mark.surface_regression,
]


def _backend_model(
    *,
    use_stacked_qr: bool = False,
    family_name: str = "stub",
    supports_closed_form_solve: bool = False,
    supports_pirls: bool = False,
    family_class: str = "",
):
    return SimpleNamespace(
        _use_stacked_qr=use_stacked_qr,
        family=SimpleNamespace(
            name=family_name,
            supports_closed_form_solve=supports_closed_form_solve,
            supports_pirls=supports_pirls,
            family_class=family_class,
        ),
    )


def test_available_fit_backends_reports_all_supported_routes_in_priority_order():
    """
    Owner-contract coverage verifying that available fit backends reports all supported
    routes in priority order.
    """
    model = _backend_model(
        use_stacked_qr=True,
        supports_closed_form_solve=True,
        supports_pirls=True,
        family_class="general",
    )

    assert backends_module.available_fit_backends(model) == (
        "stacked_qr",
        "gaussian_exact",
        backends_module.GENERAL_FAMILY_BACKEND,
        "pirls",
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            _backend_model(use_stacked_qr=True, supports_closed_form_solve=True),
            "stacked_qr",
        ),
        (_backend_model(supports_closed_form_solve=True), "gaussian_exact"),
        (
            _backend_model(family_class="general"),
            backends_module.GENERAL_FAMILY_BACKEND,
        ),
        (_backend_model(supports_pirls=True), "pirls"),
    ],
    ids=["stacked_qr", "gaussian_exact", "general_family", "pirls"],
)
def test_resolve_fit_backend_uses_expected_precedence(model, expected):
    """
    Owner-contract coverage verifying that resolve fit backend uses expected precedence.
    """
    assert backends_module.resolve_fit_backend(model) == expected


def test_resolve_fit_backend_raises_for_unsupported_family():
    """
    Owner-contract coverage verifying that resolve fit backend raises for unsupported
    family.
    """
    model = _backend_model(family_name="mystery")

    with pytest.raises(NotImplementedError, match="No supported fitting backend"):
        backends_module.resolve_fit_backend(model)


def test_solve_fit_dispatches_to_selected_solver(monkeypatch):
    """
    Owner-contract coverage verifying that solve fit dispatches to selected solver.
    """
    calls: list[tuple[str, np.ndarray | None]] = []

    def _gaussian(model, y, smoothing_params, weights=None):
        calls.append(("gaussian", None if weights is None else np.asarray(weights)))
        return "gaussian"

    def _general(model, y, smoothing_params, weights=None):
        calls.append(("general", None if weights is None else np.asarray(weights)))
        return "general"

    def _pirls(model, y, smoothing_params, weights=None):
        calls.append(("pirls", None if weights is None else np.asarray(weights)))
        return "pirls"

    monkeypatch.setattr(backends_module, "solve_gaussian_fit", _gaussian)
    monkeypatch.setattr(backends_module, "solve_general_family_fit", _general)
    monkeypatch.setattr(backends_module, "solve_pirls_fit", _pirls)

    model = _backend_model(supports_closed_form_solve=True)
    y = np.array([1.0], dtype=np.float64)
    sp = np.array([0.5], dtype=np.float64)
    w = np.array([2.0], dtype=np.float64)

    assert (
        backends_module.solve_fit(model, y, sp, backend="stacked_qr", weights=w)
        == "gaussian"
    )
    assert (
        backends_module.solve_fit(model, y, sp, backend="gaussian_exact", weights=w)
        == "gaussian"
    )
    assert (
        backends_module.solve_fit(
            _backend_model(family_class="general"),
            y,
            sp,
            backend=backends_module.GENERAL_FAMILY_BACKEND,
            weights=w,
        )
        == "general"
    )
    assert (
        backends_module.solve_fit(
            _backend_model(supports_pirls=True),
            y,
            sp,
            backend="pirls",
            weights=w,
        )
        == "pirls"
    )

    assert [name for name, _ in calls] == ["gaussian", "gaussian", "general", "pirls"]
    for _name, weights in calls:
        np.testing.assert_allclose(weights, w)


def test_solve_fit_rejects_unknown_backend():
    """Owner-contract coverage verifying that solve fit rejects unknown backend."""
    model = _backend_model(supports_closed_form_solve=True)

    with pytest.raises(ValueError, match="Unknown fit backend"):
        backends_module.solve_fit(
            model,
            np.array([1.0], dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            backend="mystery",
        )


def test_fixed_smoothing_wrappers_forward_prior_weights_and_scale(monkeypatch):
    """
    Owner-contract coverage verifying that fixed smoothing wrappers forward prior
    weights.
    """
    calls: list[tuple[str, np.ndarray, float | None]] = []

    def _gaussian(model, y, smoothing_params, weights=None):
        calls.append(("gaussian", np.asarray(weights, dtype=np.float64), None))
        return "gaussian-sol"

    def _pirls(
        model, y, smoothing_params, weights=None, *, scale_reference=None
    ):
        calls.append(
            (
                "pirls",
                np.asarray(weights, dtype=np.float64),
                scale_reference,
            )
        )
        return "pirls-sol"

    monkeypatch.setattr(backends_module, "solve_gaussian_fit", _gaussian)
    monkeypatch.setattr(backends_module, "solve_pirls_fit", _pirls)

    model = SimpleNamespace(prior_weights_=np.array([1.5, 2.5], dtype=np.float64))
    y = np.array([1.0, 2.0], dtype=np.float64)
    sp = np.array([0.4], dtype=np.float64)

    assert solve_gaussian_given_smoothing(model, y, sp) == "gaussian-sol"
    assert (
        solve_pirls_given_smoothing(model, y, sp, scale_reference=0.75)
        == "pirls-sol"
    )
    assert [name for name, _, _ in calls] == ["gaussian", "pirls"]
    for _name, weights, _scale_reference in calls:
        np.testing.assert_allclose(weights, model.prior_weights_)
    assert calls[1][2] == pytest.approx(0.75, abs=0.0)


def test_unconditional_covariance_efs_gate_mirrors_mgcv_postproc_split():
    """
    mgcv splits Vc by post-processor: gam.fit3.post.proc leaves
    `V.sp <- edf2 <- Vc <- NULL` without db.drho (mgcv/R/gam.fit3.r:1053, the
    efs/optim deriv=0 case), while gam.fit5.post.proc always returns
    Vc == Vb + 0 with edf2 (mgcv/R/gam.fit4.r:1685-1690, 1714-1715).
    """
    import pandas as pd

    from nampy.gam import GAM

    rng = np.random.default_rng(5)
    n = 80
    x = rng.uniform(size=n)
    data = pd.DataFrame({"x": x})
    data["y"] = rng.poisson(np.exp(0.3 + np.sin(2.0 * x)))

    glm_fit = GAM(
        formula='y ~ s(x, bs="cr", k=6)',
        family="poisson",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    ).fit(data=data)
    glm_result = glm_fit.fit_core_solution_.fit_result
    assert glm_result.cov_unconditional is None
    assert glm_result.edf2 is None

    data_g = pd.DataFrame({"x": x})
    data_g["y"] = np.sin(2.0 * x) + 0.2 * rng.standard_normal(n)
    general_fit = GAM(
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1"],
        family="gaulss",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="efs",
    ).fit(data=data_g)
    general_result = general_fit.fit_core_solution_.fit_result
    assert general_result.cov_unconditional is not None
    assert general_result.edf2 is not None
    # Vc == Vb + 0 when the deriv=0 correction is suppressed; compare through
    # the public vcov surface so both sides are in the same parameterization.
    # The two matrices reach public space through separate mapping paths, so
    # allow last-bit float noise only.
    np.testing.assert_allclose(
        np.asarray(general_fit.vcov(unconditional=True), dtype=np.float64),
        np.asarray(general_fit.vcov(), dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
