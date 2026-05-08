from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    _GeneralPredictorLayout,
    build_general_penalty_setup,
)
from nampy.gam.fit.solvers.general_family.newton import _sl_ldetS
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model
from tests.mgcv_parity_utils import _make_random_effect_data

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _simple_general_model(*, terms=(), penalties=()):
    return SimpleNamespace(
        compiled_model_=SimpleNamespace(
            compiled_terms=tuple(terms),
            compiled_penalties=tuple(penalties),
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


def _build_from_formula(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def test_general_family_contiguous_penalty_guard_rejects_noncontiguous_blocks():
    """
    Guard coverage verifying that general family contiguous penalty guard rejects
    noncontiguous blocks.
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


def test_nonreparameterized_single_penalty_sl_guard_raises_explicitly():
    """
    Guard coverage verifying that nonreparameterized single penalty sl guard raises
    explicitly.
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


def test_nonreparameterized_multi_penalty_sl_guard_raises_explicitly():
    """
    Guard coverage verifying that nonreparameterized multi penalty sl guard raises
    explicitly.
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


def test_formula_unsupported_expression_function_guard_raises_explicitly():
    """
    Guard coverage verifying that formula unsupported expression function guard raises
    explicitly.
    """
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

    with pytest.raises(
        NotImplementedError,
        match="Unsupported formula expression function",
    ):
        _build_from_formula("y ~ foo(x)", data)


def test_formula_list_data_aware_dot_shorthand_guard_raises_explicitly():
    """
    Guard coverage verifying that formula list data aware dot shorthand guard raises
    explicitly.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Data-aware '\\.' shorthand is unsupported for formula-list / general-family models",
    ):
        _build_from_formula(["y ~ .", "~ 1"], data)


def test_general_family_multi_smooth_fit_guard_raises_explicitly():
    """
    Guard coverage verifying that multi-smooth general-family fits are unsupported.
    """
    data = pd.DataFrame(
        {
            "y": [0.7, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0, 2.2],
            "x": [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0],
            "z": [1.0, 0.6, 0.3, 0.1, -0.2, -0.5, -0.7, -1.0],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Multi-smooth general-family models are not supported",
    ):
        GAM(
            family="gaulss",
            formula=['y ~ s(x, bs="cr", k=5) + s(z, bs="cr", k=5)', "~ 1"],
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, x, bs="fs", xt="cs")',
        'y ~ s(f, x, bs="fs", xt="ts")',
    ],
)
def test_fs_full_rank_shrinkage_surface_guard_raises_explicitly(formula):
    """
    Guard coverage verifying that fs full rank shrinkage surface guard raises
    explicitly.
    """
    data = pd.DataFrame(
        {
            "y": [0.4, 0.9, 1.2, 1.8, 2.1, 2.4],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "f": ["a", "b", "c", "a", "b", "c"],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match=r'`bs="fs"` with base bs=.* is unsupported\. Upstream mgcv::smooth\.construct\.fs\.smooth\.spec also rejects this full-rank shrinkage base',
    ):
        GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)


def test_random_effect_linked_id_guard_raises_explicitly():
    """Guard coverage verifying that random effect linked id guard raises explicitly."""
    data = _make_random_effect_data()
    gam = GAM(
        family="gaussian",
        formula='y ~ s(f, bs="re", id="shared")',
        optimize_smoothing=True,
        smoothing_method="REML",
    )

    with pytest.raises(NotImplementedError, match="random effects don't work with ids"):
        gam.fit(data=data)
