from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.diagnostics.concurvity import concurvity
from nampy.gam.diagnostics.residuals import residuals_gam
from nampy.gam.diagnostics.summary import summary_text
from nampy.gam.smoothing_selection.postfit import one_se_rule, sp_vcov

concurvity_module = importlib.import_module("nampy.gam.diagnostics.concurvity")
residuals_module = importlib.import_module("nampy.gam.diagnostics.residuals")
summary_module = importlib.import_module("nampy.gam.diagnostics.summary")
postfit_module = importlib.import_module("nampy.gam.smoothing_selection.postfit")

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_residuals_working_restores_gaussian_offset(monkeypatch):
    """
    Owner-contract coverage verifying that residuals working restores gaussian offset.
    """
    model = SimpleNamespace(
        y_=np.array([1.0, 3.0], dtype=np.float64),
        n_samples_=2,
        prior_weights_=None,
        family=SimpleNamespace(name="gaussian"),
    )

    monkeypatch.setattr(residuals_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        residuals_module,
        "_fitted_mu",
        lambda model: np.array([0.5, 2.5], dtype=np.float64),
    )
    monkeypatch.setattr(
        residuals_module,
        "_fitted_eta",
        lambda model: np.array([0.6, 2.6], dtype=np.float64),
    )
    monkeypatch.setattr(
        residuals_module,
        "_fit_state",
        lambda model: SimpleNamespace(
            working_response=np.array([0.9, 2.8], dtype=np.float64),
            offset=np.array([0.1, 0.2], dtype=np.float64),
        ),
    )

    out = residuals_gam(model, type="working")

    np.testing.assert_allclose(out, np.array([0.4, 0.4], dtype=np.float64))


def test_residuals_pearson_warns_and_falls_back_to_deviance(monkeypatch):
    """
    Owner-contract coverage verifying that residuals pearson warns and falls back to
    deviance.
    """
    model = SimpleNamespace(
        y_=np.array([1.0, 2.0], dtype=np.float64),
        n_samples_=2,
        prior_weights_=None,
        family=SimpleNamespace(name="gaussian", variance=None),
    )

    monkeypatch.setattr(residuals_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        residuals_module,
        "_fitted_mu",
        lambda model: np.array([0.5, 1.5], dtype=np.float64),
    )
    monkeypatch.setattr(
        residuals_module,
        "_fitted_eta",
        lambda model: np.array([0.5, 1.5], dtype=np.float64),
    )
    monkeypatch.setattr(
        residuals_module,
        "_deviance_residuals",
        lambda model: np.array([7.0, 8.0], dtype=np.float64),
    )

    with pytest.warns(RuntimeWarning, match="Pearson residuals not available"):
        out = residuals_gam(model, type="pearson")

    np.testing.assert_allclose(out, np.array([7.0, 8.0], dtype=np.float64))


def test_residuals_general_family_requires_family_specific_residuals(monkeypatch):
    """
    Owner-contract coverage verifying that general families do not use a generic
    primary-predictor fallback for mgcv-specific residuals.
    """
    model = SimpleNamespace(
        y_=np.array([4.0, 8.0], dtype=np.float64),
        n_samples_=2,
        prior_weights_=None,
        family=SimpleNamespace(
            name="mock_general",
            family_class="general",
            linfo=[
                SimpleNamespace(
                    linkinv=lambda eta: np.asarray(eta, dtype=np.float64) + 0.5
                )
            ],
        ),
    )

    monkeypatch.setattr(residuals_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        residuals_module,
        "_fitted_mu",
        lambda model: np.array([[99.0, 1.0], [98.0, 2.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        residuals_module,
        "_fitted_eta",
        lambda model: np.array([[3.0, 0.0], [7.0, 0.0]], dtype=np.float64),
    )

    with pytest.raises(NotImplementedError, match="Residual type 'deviance'"):
        residuals_gam(model, type="deviance")


def test_concurvity_returns_pairwise_measure_matrices_with_parametric_block(
    monkeypatch,
):
    """
    Owner-contract coverage verifying that concurvity returns pairwise measure matrices
    with parametric block.
    """
    monkeypatch.setattr(concurvity_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        concurvity_module,
        "build_lpmatrix",
        lambda model: np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )
    monkeypatch.setattr(concurvity_module, "_compiled_model", lambda model: None)
    monkeypatch.setattr(concurvity_module, "_coef_column_offset", lambda model: 0)
    monkeypatch.setattr(concurvity_module, "_fit_intercept", lambda model: False)
    monkeypatch.setattr(
        concurvity_module,
        "_coef_full",
        lambda model: np.array([0.5, 1.25], dtype=np.float64),
    )
    monkeypatch.setattr(
        concurvity_module,
        "_term_blocks_seq",
        lambda model: (
            SimpleNamespace(
                term_type="smooth",
                label='s(x1)',
                coef_slice=slice(1, 2),
            ),
        ),
    )

    out = concurvity(SimpleNamespace(), full=False)

    assert out["measure_names"] == ("worst", "observed", "estimate")
    assert out["labels"] == ["para", "s(x1)"]
    for name in out["measure_names"]:
        mat = np.asarray(out["values"][name], dtype=np.float64)
        assert mat.shape == (2, 2)
        np.testing.assert_allclose(np.diag(mat), np.ones(2, dtype=np.float64))


def test_concurvity_parametric_block_uses_mgcv_first_column_indexing(monkeypatch):
    """
    Verify the upstream mgcv/R/mgcv.r::concurvity parametric block indexing.
    """
    monkeypatch.setattr(concurvity_module, "_compiled_model", lambda model: None)
    monkeypatch.setattr(concurvity_module, "_coef_column_offset", lambda model: 0)
    monkeypatch.setattr(
        concurvity_module,
        "_term_blocks_seq",
        lambda model: (
            SimpleNamespace(
                term_type="parametric",
                label="f[b]",
                coef_slice=slice(0, 1),
            ),
            SimpleNamespace(
                term_type="parametric",
                label="f[c]",
                coef_slice=slice(1, 2),
            ),
            SimpleNamespace(
                term_type="smooth",
                label="s(x)",
                coef_slice=slice(2, 5),
            ),
        ),
    )

    blocks = concurvity_module._term_indices_for_concurvity(SimpleNamespace(), 5)

    assert blocks[0][0] == "para"
    np.testing.assert_array_equal(blocks[0][1], np.array([0], dtype=int))


def test_concurvity_raises_when_no_components_available(monkeypatch):
    """
    Owner-contract coverage verifying that concurvity raises when no components
    available.
    """
    monkeypatch.setattr(concurvity_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        concurvity_module,
        "build_lpmatrix",
        lambda model: np.ones((3, 0), dtype=np.float64),
    )
    monkeypatch.setattr(concurvity_module, "_compiled_model", lambda model: None)
    monkeypatch.setattr(concurvity_module, "_coef_column_offset", lambda model: 0)
    monkeypatch.setattr(concurvity_module, "_fit_intercept", lambda model: False)
    monkeypatch.setattr(
        concurvity_module, "_coef_full", lambda model: np.zeros(0, dtype=np.float64)
    )
    monkeypatch.setattr(concurvity_module, "_term_blocks_seq", lambda model: ())

    with pytest.raises(ValueError, match="No smooth or parametric components available"):
        concurvity(SimpleNamespace(), full=True)


def test_summary_text_includes_offset_and_gaussian_rss(monkeypatch):
    """
    Owner-contract coverage verifying that summary text includes offset and gaussian
    RSS.
    """
    fit_summary = SimpleNamespace(
        edf_by_term=np.array([2.5, 1.0], dtype=np.float64),
        intercept=0.75,
        edf_total=3.5,
        scale=0.8,
        rss=1.25,
        deviance=99.0,
    )
    term_blocks = (
        SimpleNamespace(
            basis_name="cr",
            term_type="smooth",
            label='s(x0)',
            coef_slice=slice(0, 4),
            smoothing_indices=[0],
        ),
        SimpleNamespace(
            basis_name="re",
            term_type="random_effect",
            label='s(region)',
            coef_slice=slice(4, 6),
            smoothing_indices=[1, 2],
        ),
    )
    model = SimpleNamespace(
        family=SimpleNamespace(name="gaussian", link_name="identity"),
        _optim_method="REML",
        n_samples_=123,
        smoothing_score_=4.2,
        offset_train_=np.array([1.0], dtype=np.float64),
        smoothing_params=np.array([0.4, 2.0, 3.0], dtype=np.float64),
    )

    monkeypatch.setattr(summary_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(summary_module, "_fit_summary", lambda model: fit_summary)
    monkeypatch.setattr(summary_module, "_term_blocks_seq", lambda model: term_blocks)
    monkeypatch.setattr(summary_module, "_fit_intercept", lambda model: True)

    text = summary_text(model)

    assert "Family : gaussian" in text
    assert "Link : identity" in text
    assert "Smoothing method : REML" in text
    assert "Offset : yes" in text
    assert "RSS : 1.25" in text
    assert "s(x0)" in text
    assert "s(s(x0))" not in text
    assert "s(region)" in text
    assert "[2, 3]" in text


def test_sp_vcov_returns_none_outside_ml_reml_family(monkeypatch):
    """
    Owner-contract coverage verifying that sp vcov returns none outside ML REML family.
    """
    model = SimpleNamespace(_optim_method="fixed")
    monkeypatch.setattr(postfit_module, "_require_fitted", lambda model: None)

    assert sp_vcov(model) is None


def test_sp_vcov_uses_joint_gaussian_outer_hessian(monkeypatch):
    """
    Owner-contract coverage verifying that sp vcov uses joint gaussian outer hessian.
    """
    H = np.array(
        [
            [4.0, 1.0, 0.5],
            [1.0, 5.0, 0.25],
            [0.5, 0.25, 3.0],
        ],
        dtype=np.float64,
    )
    model = SimpleNamespace(
        _optim_method="reml",
        _optim_result=SimpleNamespace(
            joint_gaussian_reml_outer=True,
            joint_x=np.array([0.1, 0.2, -1.0], dtype=np.float64),
            outer_info={"hess": H},
        ),
    )

    monkeypatch.setattr(postfit_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        postfit_module,
        "_postfit_hessian",
        lambda model, method, edge_correct: (_ for _ in ()).throw(
            AssertionError("profiled Hessian should not be used")
        ),
    )

    out = sp_vcov(model, edge_correct=False, reg=1e-3)

    np.testing.assert_allclose(
        out,
        np.linalg.solve(H + 1e-3, np.eye(3)),
        atol=1e-12,
        rtol=0.0,
    )


def test_one_se_rule_updates_only_requested_free_smoothing_parameters(monkeypatch):
    """
    Owner-contract coverage verifying that one-standard-error rule updates only
    requested free smoothing parameters.
    """
    model = SimpleNamespace(
        _optim_method="reml",
        smoothing_params=np.array([2.0, 3.0, 5.0], dtype=np.float64),
        smoothing_fixed_mask_=np.array([False, True, False]),
    )

    monkeypatch.setattr(postfit_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(
        postfit_module,
        "_n_smoothing_params",
        lambda model: 3,
    )
    monkeypatch.setattr(
        postfit_module,
        "sp_vcov",
        lambda model, edge_correct=False: np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float64),
    )

    out = one_se_rule(model, candidate_indices=[2])

    expected = np.array([2.0, 3.0, 5.0], dtype=np.float64)
    expected[2] = np.exp(np.log(5.0) + (np.sqrt(2.0) * 3.0))
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)


def test_one_se_rule_recycles_free_log_sp_for_joint_gaussian_covariance(monkeypatch):
    """
    Owner-contract coverage verifying that one-standard-error rule recycles free log sp
    for joint gaussian covariance.
    """
    V = np.array(
        [
            [0.39731808, 0.00948097, 0.01212533],
            [0.00948097, 0.45646423, 0.01121854],
            [0.01212533, 0.01121854, 0.01193115],
        ],
        dtype=np.float64,
    )
    model = SimpleNamespace(
        _optim_method="reml",
        smoothing_params=np.array([2.88601658, 10.72390627], dtype=np.float64),
        smoothing_fixed_mask_=np.array([False, False]),
        _optim_result=SimpleNamespace(joint_gaussian_reml_outer=True),
    )

    monkeypatch.setattr(postfit_module, "_require_fitted", lambda model: None)
    monkeypatch.setattr(postfit_module, "sp_vcov", lambda model, edge_correct=False: V)

    out = one_se_rule(model)

    d = np.sqrt(np.diag(V))
    alpha = np.sqrt(2.0 * len(d)) / (d @ np.linalg.solve(V, d))
    expected = np.exp(np.resize(np.log(model.smoothing_params), 3) + alpha * d)
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)
