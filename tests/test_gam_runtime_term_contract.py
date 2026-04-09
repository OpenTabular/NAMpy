import warnings

import numpy as np
import pytest

from nampy.gam.runtime.factory import instantiate_term
from nampy.gam.smooths.base import RUNTIME_TERM_INTERFACE_CHECKLIST
from nampy.gam.smooths.categorical.factor_smooth import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
)
from nampy.gam.smooths.categorical.mrf import MarkovRandomFieldTerm
from nampy.gam.smooths.categorical.random_effect import RandomEffectTerm
from nampy.gam.smooths.tensor.t2 import TensorANOVASplineTerm
from nampy.gam.smooths.tensor.te import TensorProductSplineTerm
from nampy.gam.smooths.tensor.ti import InteractionTensorProductSplineTerm
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D
from nampy.gam.smooths.univariate.gp import GPSmoothTerm
from nampy.gam.smooths.univariate.pspline import PSplineTerm1D
from nampy.gam.smooths.univariate.thin_plate import ThinPlateSplineTerm
from nampy.gam.specs import TermSpec
from nampy.gam.terms.linear import LinearTerm


def _build_mixed_data(n=40):
    rng = np.random.default_rng(123)
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(0.0, 2.0, size=n)
    by = rng.uniform(0.5, 1.5, size=n)
    fac = np.where(rng.random(n) > 0.5, "A", "B")
    area = np.array(
        ["r0", "r1", "r2", "r3"] * (n // 4) + ["r0"] * (n % 4), dtype=object
    )

    X = np.empty((n, 5), dtype=object)
    X[:, 0] = x0
    X[:, 1] = x1
    X[:, 2] = fac
    X[:, 3] = area
    X[:, 4] = by
    feature_names = ["x0", "x1", "fac", "area", "by"]
    return X, feature_names


def _assert_term_contract(term):
    for attr in RUNTIME_TERM_INTERFACE_CHECKLIST:
        assert hasattr(term, attr), f"missing runtime contract attribute: {attr}"
    assert callable(term.transform_new)
    assert callable(term.get_penalty_definitions)
    assert term.resolved_feature_names is not None


def test_runtime_term_contract_and_prediction_coercion():
    X, feature_names = _build_mixed_data()
    X_new = X.copy()
    # Unrelated categorical columns must not break numeric-term prediction.
    X_new[:, 2] = "UNUSED_FACTOR"
    X_new[:, 3] = "UNUSED_REGION"

    terms = [
        LinearTerm(feature="x0", label="lin_x0"),
        SplineTerm1D(feature="x0", k=8, basis="cr", label="s_x0"),
        PSplineTerm1D(feature="x0", k=8, basis="ps", label="ps_x0", by="by"),
        ThinPlateSplineTerm(feature=["x0", "x1"], basis="tp", k=15, label="tp_x0_x1"),
        GPSmoothTerm(feature=["x0", "x1"], basis="gp", k=15, label="gp_x0_x1"),
        TensorProductSplineTerm(
            feature=["x0", "x1"], basis=["cr", "cr"], k=[6, 6], label="te_x0_x1"
        ),
        InteractionTensorProductSplineTerm(
            feature=["x0", "x1"], basis=["cr", "cr"], k=[6, 6], label="ti_x0_x1"
        ),
        TensorANOVASplineTerm(
            feature=["x0", "x1"], basis=["cr", "cr"], k=[6, 6], label="t2_x0_x1"
        ),
        RandomEffectTerm(feature=["fac"], label="re_fac"),
        MarkovRandomFieldTerm(
            feature=["area"], label="mrf_area", xt={"penalty": np.eye(4)}
        ),
        FSmoothInteractionTerm(feature=["x0", "fac"], k=7, label="fs_x0_fac", xt="cr"),
        SZSmoothInteractionTerm(feature=["x0", "fac"], k=7, label="sz_x0_fac", xt="cr"),
    ]

    for term in terms:
        term.fit(X, feature_names)
        _assert_term_contract(term)
        B = term.transform_new(X_new)
        assert isinstance(B, np.ndarray)
        assert B.shape[0] == X_new.shape[0]


def test_select_penalty_metadata_uses_canonical_runtime_state():
    X, feature_names = _build_mixed_data()

    terms = [
        PSplineTerm1D(feature="x0", k=8, basis="ps", label="ps_sel", select=True),
        ThinPlateSplineTerm(
            feature=["x0", "x1"], basis="ts", k=15, label="tp_sel", select=True
        ),
        GPSmoothTerm(
            feature=["x0", "x1"], basis="gp", k=15, label="gp_sel", select=True
        ),
    ]

    found_selection_penalty = False
    for term in terms:
        term.fit(X, feature_names)
        defs = term.get_penalty_definitions()
        selection_defs = [d for d in defs if bool(d.is_null_space_penalty)]
        for pdef in selection_defs:
            found_selection_penalty = True
            meta = pdef.metadata
            assert meta["by_name"] == getattr(term._by_state, "feature_name", None)
            assert meta["by_is_constant"] is bool(
                getattr(term._by_state, "is_constant", True)
            )
            assert meta["constraint_kind"] == getattr(term, "constraint_kind", None)
    assert found_selection_penalty


def test_factory_supports_select_for_categorical_runtime_terms():
    X, feature_names = _build_mixed_data()

    specs = [
        TermSpec(
            kind="smooth",
            features=("area",),
            basis_options={
                "special": "s",
                "bs": "mrf",
                "k": 4,
                "select": True,
                "xt": {"penalty": np.eye(4)},
            },
            label="mrf_sel",
        ),
        TermSpec(
            kind="smooth",
            features=("fac",),
            basis_options={"special": "s", "bs": "re", "select": True},
            label="re_sel",
        ),
        TermSpec(
            kind="smooth",
            features=("x0", "fac"),
            basis_options={
                "special": "s",
                "bs": "fs",
                "k": 7,
                "select": True,
                "xt": "cr",
            },
            label="fs_sel",
        ),
        TermSpec(
            kind="smooth",
            features=("x0", "fac"),
            basis_options={
                "special": "s",
                "bs": "sz",
                "k": 7,
                "select": True,
                "xt": "cr",
            },
            label="sz_sel",
        ),
    ]

    terms = [instantiate_term(spec) for spec in specs]
    for term in terms:
        assert getattr(term, "select", False) is True
        term.fit(X, feature_names)
        defs = term.get_penalty_definitions()
        assert defs

    mrf_defs = terms[0].get_penalty_definitions()
    assert len(mrf_defs) == 1

    re_defs = terms[1].get_penalty_definitions()
    assert not any(d.is_null_space_penalty for d in re_defs)

    fs_defs = terms[2].get_penalty_definitions()
    assert any(d.is_null_space_penalty for d in fs_defs)

    sz_defs = terms[3].get_penalty_definitions()
    assert any(d.is_null_space_penalty for d in sz_defs)


def test_random_effect_term_rejects_linked_ids_with_mgcv_message():
    with pytest.raises(
        NotImplementedError, match=r"random effects don't work with ids\."
    ):
        RandomEffectTerm(feature=["fac"], label="re_linked", smoothing_id="group_re")

    spec = TermSpec(
        kind="smooth",
        features=("fac",),
        basis_options={"special": "s", "bs": "re"},
        smoothing_id="group_re",
        label="re_linked_spec",
    )
    with pytest.raises(
        NotImplementedError, match=r"random effects don't work with ids\."
    ):
        instantiate_term(spec)


def test_t2_invalid_term_sp_length_warns_and_is_ignored():
    X, feature_names = _build_mixed_data()
    spec = TermSpec(
        kind="smooth",
        features=("x0", "x1"),
        basis_options={
            "special": "t2",
            "bs": ["ps", "ps"],
            "k": [5, 5],
            "sp": [0.7, 1.3],
        },
        label="t2_ps_ps_bad_sp",
    )
    term = instantiate_term(spec)
    term.fit(X, feature_names)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        defs = term.get_penalty_definitions()

    assert any(
        "length of sp incorrect in t2: ignored" in str(w.message) for w in caught
    )
    assert len(defs) == 3
    assert all(d.sp_mode is None for d in defs)


def test_t2_full_true_matches_mgcv_penalty_count():
    X, feature_names = _build_mixed_data()
    spec = TermSpec(
        kind="smooth",
        features=("x0", "x1"),
        basis_options={"special": "t2", "bs": ["tp", "cr"], "k": [6, 6], "full": True},
        label="t2_tp_cr_full",
    )
    term = instantiate_term(spec)
    term.fit(X, feature_names)
    defs = term.get_penalty_definitions()
    assert len(defs) == 5


def test_t2_select_adds_one_null_space_penalty():
    X, feature_names = _build_mixed_data()
    spec = TermSpec(
        kind="smooth",
        features=("x0", "x1"),
        basis_options={
            "special": "t2",
            "bs": ["tp", "cr"],
            "k": [6, 6],
            "select": True,
        },
        label="t2_select_spec",
    )
    term = instantiate_term(spec)
    term.fit(X, feature_names)
    defs = term.get_penalty_definitions()
    assert len(defs) == 4
    assert sum(bool(d.is_null_space_penalty) for d in defs) == 1
