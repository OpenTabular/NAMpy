from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.compiler.compile_predictors import compile_predictors
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    _GeneralPredictorLayout,
    build_general_penalty_setup,
)
from nampy.gam.fit.solvers.general_family.newton import _sl_ldetS
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.predict.predictions import predict_values
from nampy.gam.specs.build import build_formula_model
from tests.mgcv_parity_utils import _make_negbin_data, _make_random_effect_data

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


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


def test_negbin_estimated_theta_ml_optim_guard_raises_explicitly():
    """Guard the R/SciPy L-BFGS-B mismatch at the flat joint-ML boundary."""
    data = _make_negbin_data()
    gam = GAM(
        family={"name": "negbin", "theta": 1.8, "estimate_theta": True},
        formula='y ~ s(x0, bs="cr", k=8)',
        optimize_smoothing=True,
        smoothing_method="ML",
        smoothing_optimizer="optim",
    )

    with pytest.raises(
        NotImplementedError,
        match=r"exact R stats::optim L-BFGS-B boundary behavior is ported",
    ):
        gam.fit(data=data)


def test_t2_smooth_guard_raises_explicitly():
    """Guard coverage verifying that t2(...) smooths raise a clear error."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match=r"t2\(\.\.\.\) tensor product smooths are not supported",
    ):
        _build_from_formula("y ~ t2(x, z)", data)


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        ('y ~ s(f, bs="re", pc=0)', r"pc= is not supported for s\(\.\.\., bs='re'\)"),
        ('y ~ s(x, f, bs="fs", pc=0)', r"pc= is not supported for s\(\.\.\., bs='fs'\)"),
        ('y ~ s(x, f, bs="sz", pc=0)', r"pc= is not supported for s\(\.\.\., bs='sz'\)"),
    ],
)
def test_pc_guard_raises_explicitly_for_unsupported_s_bases(formula, match):
    """pc= must fail loudly wherever the point constraint is not implemented."""
    data = pd.DataFrame(
        {
            "y": [0.4, 0.9, 1.2, 1.8, 2.1, 2.4],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "z": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "f": ["a", "b", "c", "a", "b", "c"],
        }
    )

    with pytest.raises(NotImplementedError, match=match):
        _build_from_formula(formula, data)


def test_pc_guard_raises_for_multiple_id_linked_tensor_terms():
    """mgcv cannot construct differently named linked tensor terms with pc=."""
    data = pd.DataFrame(
        {
            "y": [0.4, 0.9, 1.2, 1.8, 2.1, 2.4],
            "x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            "z": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
            "w": [-0.5, 0.1, 0.8, -0.2, 0.4, 1.1],
            "v": [0.2, -0.3, 0.6, 1.0, -0.8, 0.5],
        }
    )
    formula = (
        'y ~ te(x, z, k=c(3, 3), id="g", pc=c(0, 0)) + '
        'te(w, v, k=c(3, 3), id="g", pc=c(0, 0))'
    )

    with pytest.raises(
        NotImplementedError,
        match=(
            r"pc= is not supported across id-linked te\(\)/ti\(\) terms "
            "with different feature sets"
        ),
    ):
        built = _build_from_formula(formula, data)
        compile_predictors(built.X, built.feature_names, built.predictor_specs)


def test_gam_unknown_constructor_arguments_raise_explicitly():
    """Unknown GAM kwargs (e.g. unported mgcv arguments) must fail loudly."""
    with pytest.raises(TypeError, match=r"Unknown GAM argument\(s\): \['paraPen'\]"):
        GAM(family="gaussian", formula="y ~ s(x)", paraPen={"X": [np.eye(2)]})

    with pytest.raises(
        TypeError, match=r"Unknown GAM argument\(s\): \['absorb_cons', 'gamma'\]"
    ):
        GAM(family="gaussian", absorb_cons=False, gamma=1.4)

    data = pd.DataFrame({"y": [0.1, 0.5, 0.9, 1.4], "x": [0.0, 0.4, 0.8, 1.2]})
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x, bs="cr", k=4)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        apply_side_conditions=True,
    )
    gam.fit(data=data)
    assert gam.fit_result() is not None


def test_gacv_cp_method_guard_raises_explicitly():
    """mgcv maps GACV.Cp to the GACV criterion, which NAMpy does not implement."""
    from nampy.gam.fit.selection.criteria.dispatch import criterion_value

    with pytest.raises(ValueError, match="method must be one of"):
        criterion_value(
            SimpleNamespace(family=None), np.zeros(1), np.zeros(1), method="gacv.cp"
        )


def test_general_family_prediction_term_filter_guard_raises_explicitly():
    """Guard filters until multi-predictor coefficient blocks mirror predict.gam."""
    model = SimpleNamespace(
        _fitted=True,
        family=SimpleNamespace(family_class="general"),
    )

    with pytest.raises(
        NotImplementedError,
        match="multi-predictor general-family models",
    ):
        predict_values(model, type="link", terms=["s(x)"])


def test_shared_lpi_component_compiles_one_overlapping_block():
    """Shared formula components fit once and target both predictors."""
    rng = np.random.default_rng(31)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, size=n)
    y = rng.normal(0.3 * x, np.exp(-0.2), size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z})

    gam = GAM(
        family="gaulss",
        formula=[
            'y ~ s(x, bs="cr", k=6)',
            "~ 1",
            '1 + 2 ~ s(z, bs="cr", k=5)',
        ],
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0],
    ).fit(data=data)
    compiled = gam.gam_result_.require_compiled_model()
    assert len(compiled.compiled_terms) == 2
    assert len(compiled.compiled_penalties) == 2
    assert compiled.compiled_terms[1].predictor_indices == (0, 1)
    assert len(compiled.predictor_full_indices) == 2


def test_shared_lpi_olid_drops_one_redundant_parametric_column():
    rng = np.random.default_rng(32)
    n = 80
    x = rng.normal(size=n)
    data = pd.DataFrame({"y": rng.normal(0.2 * x, 0.7, size=n), "x": x})

    with pytest.warns(
        UserWarning, match="dropping unidentifiable parametric terms from model"
    ):
        gam = GAM(
            family="gaulss",
            formula=["y ~ x", "~ x", "1 + 2 ~ x - 1"],
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)

    compiled = gam.gam_result_.require_compiled_model()
    assert [term.basis_train.shape[1] for term in compiled.compiled_terms] == [
        0,
        1,
        1,
    ]
    assert compiled.compiled_terms[2].predictor_indices == (0, 1)
    assert compiled.predictor_full_indices[0].size == 2
    assert compiled.predictor_full_indices[1].size == 3
    assert gam.predict(data, type="lpmatrix").shape == (n, 4)
