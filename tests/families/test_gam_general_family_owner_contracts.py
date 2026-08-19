from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    _general_family_term_start_stop,
    _GeneralPredictorLayout,
    build_general_penalty_setup,
)
from nampy.gam.fit.solvers.general_family.newton import (
    _sl_ldetS,
    postprocess_general_newton_fit,
)
from nampy.gam.predict import general as general_predict_module

pytestmark = [
    pytest.mark.surface_derivatives,
    pytest.mark.surface_regression,
]


def _simple_general_fit_dict(*, reml2: float | None):
    fit = {
        "lbb": np.array([[-2.0]], dtype=np.float64),
        "L": np.array([[np.sqrt(2.0)]], dtype=np.float64),
        "D": np.array([1.0], dtype=np.float64),
        "bdrop": np.array([False]),
        "St_full": np.array([[0.0]], dtype=np.float64),
        "db_drho": np.array([[1.0]], dtype=np.float64),
        "piv": np.array([0], dtype=int),
        "ipiv": np.array([0], dtype=int),
    }
    if reml2 is not None:
        fit["REML2"] = np.array([[reml2]], dtype=np.float64)
    return fit


def _simple_general_model(*, terms=(), penalties=()):
    return SimpleNamespace(
        gam_result_=SimpleNamespace(
            compiled_model=SimpleNamespace(
                compiled_terms=tuple(terms),
                compiled_penalties=tuple(penalties),
            ),
            fit_core_solution=None,
            fit_summary=None,
        )
    )


def _simple_general_layout(
    full_idx: np.ndarray, *, n_full: int
) -> _GeneralPredictorLayout:
    return _GeneralPredictorLayout(
        X_full=np.zeros((1, n_full), dtype=np.float64),
        jj=[],
        reduced_to_full_idx=np.asarray(full_idx, dtype=int),
        predictor_full_slices=[slice(0, n_full)],
    )


@pytest.mark.parametrize(
    ("outer_hess", "outer_info", "fit_reml2", "expected_source"),
    [
        (4.0, {"hess": np.array([[9.0]], dtype=np.float64)}, 16.0, 4.0),
        (None, {"hess": np.array([[9.0]], dtype=np.float64)}, 16.0, 9.0),
        (None, {}, 16.0, 16.0),
    ],
    ids=["outer_hess", "outer_info_hess", "fit_reml2"],
)
def test_postprocess_general_newton_fit_hessian_source_precedence(
    outer_hess, outer_info, fit_reml2, expected_source
):
    """
    Owner-contract coverage verifying that postprocess general newton fit hessian source
    precedence.
    """
    fit = _simple_general_fit_dict(reml2=fit_reml2)

    out = postprocess_general_newton_fit(
        fit,
        outer_hess=(
            None if outer_hess is None else np.array([[outer_hess]], dtype=np.float64)
        ),
        outer_info=dict(outer_info),
        smoothing_params=np.array([1.0], dtype=np.float64),
    )

    expected_v_sp = 1.0 / (expected_source + (1.0 / 50.0))
    assert float(out["V_sp"][0, 0]) == pytest.approx(expected_v_sp, rel=0.0, abs=1e-12)
    assert float(out["db_drho"][0, 0]) == pytest.approx(1.0, abs=0.0)
    assert out["Vp"].shape == (1, 1)
    assert out["Vc"].shape == (1, 1)
    assert out["edf"].shape == (1,)
    assert out["edf1"].shape == (1,)
    assert out["edf2"].shape == (1,)


def test_general_family_term_start_stop_rejects_noncontiguous_coefficients():
    """
    Owner-contract coverage verifying that general family term start stop rejects
    noncontiguous coefficients.
    """
    term = SimpleNamespace(coef_slice=slice(0, 2))

    with pytest.raises(
        NotImplementedError,
        match="General-family nonlinear Sl setup requires contiguous term coefficient blocks",
    ):
        _general_family_term_start_stop(term, full_idx=np.array([0, 2], dtype=int))


def test_build_general_penalty_setup_rejects_noncontiguous_term_penalty_blocks():
    """
    Owner-contract coverage verifying that build general penalty setup rejects
    noncontiguous term penalty blocks.
    """
    model = _simple_general_model(
        terms=(SimpleNamespace(coef_slice=slice(0, 2), metadata={}),),
        penalties=(
            SimpleNamespace(
                term_index=0,
                coef_slice=slice(0, 2),
                matrix=np.eye(2, dtype=np.float64),
                rank=2,
                smoothing_index=0,
            ),
        ),
    )
    layout = _simple_general_layout(np.array([0, 2], dtype=int), n_full=3)

    with pytest.raises(
        NotImplementedError,
        match="General-family Sl setup requires contiguous term penalty blocks",
    ):
        build_general_penalty_setup(model, layout)


def test_build_general_penalty_setup_rejects_noncontiguous_fallback_penalty_blocks():
    """
    Owner-contract coverage verifying that build general penalty setup rejects
    noncontiguous fallback penalty blocks.
    """
    model = _simple_general_model(
        terms=(),
        penalties=(
            SimpleNamespace(
                term_index=0,
                coef_slice=slice(0, 2),
                matrix=np.eye(2, dtype=np.float64),
                rank=2,
                smoothing_index=0,
            ),
        ),
    )
    layout = _simple_general_layout(np.array([0, 2], dtype=int), n_full=3)

    with pytest.raises(
        NotImplementedError,
        match="General-family Sl setup requires contiguous fallback penalty blocks",
    ):
        build_general_penalty_setup(model, layout)


def test_sl_ldetS_rejects_nonreparameterized_single_penalty_blocks():
    """
    Owner-contract coverage verifying that sl ldetS rejects nonreparameterized single
    penalty blocks.
    """
    block = SimpleNamespace(
        start=1,
        stop=2,
        rank=2,
        S=[np.eye(2, dtype=np.float64)],
        lambda_=np.zeros(1, dtype=np.float64),
        repara=False,
        linear=True,
        ldet=0.0,
        ind=np.array([True, True], dtype=bool),
        rS=[],
    )

    with pytest.raises(
        NotImplementedError,
        match="Non-reparameterized single-penalty general-family Sl blocks are unsupported",
    ):
        _sl_ldetS(
            [block],
            rho=np.array([0.0], dtype=np.float64),
            fixed=np.array([False]),
            np_=2,
        )


def test_sl_ldetS_rejects_nonreparameterized_multi_penalty_blocks():
    """
    Owner-contract coverage verifying that sl ldetS rejects nonreparameterized multi
    penalty blocks.
    """
    root = np.eye(2, dtype=np.float64)
    block = SimpleNamespace(
        start=1,
        stop=2,
        rank=2,
        S=[root, root],
        lambda_=np.zeros(2, dtype=np.float64),
        repara=False,
        linear=True,
        ldet=0.0,
        ind=np.array([True, True], dtype=bool),
        rS=[root, root],
    )

    with pytest.raises(
        NotImplementedError,
        match="Non-reparameterized multi-penalty general-family Sl blocks are unsupported",
    ):
        _sl_ldetS(
            [block],
            rho=np.array([0.0, 0.1], dtype=np.float64),
            fixed=np.array([False, False]),
            np_=2,
        )


def test_predict_general_terms_rejects_raw_prediction_basis(monkeypatch):
    """
    Owner-contract coverage verifying that predict general terms rejects raw prediction
    basis.
    """
    monkeypatch.setattr(
        general_predict_module,
        "_term_blocks_seq",
        lambda model: (
            SimpleNamespace(metadata={"expose_raw_prediction_basis": True}),
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="type='terms' is not supported for general-family models whose prediction parameterization is wider than the fitted coefficient space",
    ):
        general_predict_module.predict_general_values(SimpleNamespace(), type="terms")


def test_postprocess_general_newton_fit_uses_exact_penalty_derivatives_for_vb_corr():
    """
    Owner-contract coverage verifying that postprocess general newton fit uses exact
    penalty derivatives for VB corr.
    """
    fit = _simple_general_fit_dict(reml2=None)
    fit["db_drho"] = np.zeros((1, 1), dtype=np.float64)

    out = postprocess_general_newton_fit(
        fit,
        outer_hess=np.array([[4.0]], dtype=np.float64),
        smoothing_params=np.array([1.0], dtype=np.float64),
        penalty_matrix=np.array([[3.0]], dtype=np.float64),
        penalty_derivatives=[np.array([[2.0]], dtype=np.float64)],
    )

    expected_v_sp = 1.0 / (4.0 + (1.0 / 50.0))
    expected_correction = expected_v_sp / 125.0
    assert float(out["V_sp"][0, 0]) == pytest.approx(expected_v_sp, rel=0.0, abs=1e-12)
    assert float(out["Vp"][0, 0]) == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert float(out["Vc"][0, 0]) == pytest.approx(
        0.5 + expected_correction, rel=0.0, abs=1e-12
    )
