from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import nampy.gam.engine as engine_module
from nampy.gam.fit import backends as backends_module
from nampy.gam.fit.solve_ops import (
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
        (_backend_model(use_stacked_qr=True, supports_closed_form_solve=True), "stacked_qr"),
        (_backend_model(supports_closed_form_solve=True), "gaussian_exact"),
        (_backend_model(family_class="general"), backends_module.GENERAL_FAMILY_BACKEND),
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

    assert backends_module.solve_fit(model, y, sp, backend="stacked_qr", weights=w) == "gaussian"
    assert backends_module.solve_fit(model, y, sp, backend="gaussian_exact", weights=w) == "gaussian"
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


def test_fixed_smoothing_wrappers_forward_prior_weights(monkeypatch):
    """
    Owner-contract coverage verifying that fixed smoothing wrappers forward prior
    weights.
    """
    calls: list[tuple[str, np.ndarray]] = []

    def _gaussian(model, y, smoothing_params, weights=None):
        calls.append(("gaussian", np.asarray(weights, dtype=np.float64)))
        return "gaussian-sol"

    def _pirls(model, y, smoothing_params, weights=None):
        calls.append(("pirls", np.asarray(weights, dtype=np.float64)))
        return "pirls-sol"

    monkeypatch.setattr(engine_module, "solve_gaussian_fit", _gaussian)
    monkeypatch.setattr(engine_module, "solve_pirls_fit", _pirls)

    model = SimpleNamespace(prior_weights_=np.array([1.5, 2.5], dtype=np.float64))
    y = np.array([1.0, 2.0], dtype=np.float64)
    sp = np.array([0.4], dtype=np.float64)

    assert solve_gaussian_given_smoothing(model, y, sp) == "gaussian-sol"
    assert solve_pirls_given_smoothing(model, y, sp) == "pirls-sol"
    assert [name for name, _ in calls] == ["gaussian", "pirls"]
    for _name, weights in calls:
        np.testing.assert_allclose(weights, model.prior_weights_)
