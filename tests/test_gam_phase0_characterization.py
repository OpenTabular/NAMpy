"""Phase 0.1 — characterization of the currently supported GAM surface (structure-first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam.parity import build_parity_snapshot, compare_parity_snapshots
from nampy.models.gam import GAMRegressor

from gam_phase0_utils import (
    design_structure,
    fit_gaussian_formula,
    parity_snapshot_structure,
    predict_type_shapes,
    roundtrip_parity_snapshot,
    term_contribution_shapes,
)


def _base_df(n=48, seed=2):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    fac = np.array(["a", "b", "c", "a", "b", "c"] * (n // 6 + 1), dtype=object)[:n]
    y = np.sin(1.1 * x0) + 0.15 * x1 + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "fac": fac})


def _mrf_df():
    regions = np.array(["A", "B", "C", "A", "B", "C", "A", "B"], dtype=object)
    vals = {"A": 1.0, "B": -0.5, "C": 0.3}
    y = np.array([vals[r] for r in regions], dtype=np.float64)
    return pd.DataFrame({"y": y, "region": regions})


def _fs_df():
    rng = np.random.default_rng(27)
    levels = ["a", "b", "c"]
    n = 18
    f = np.array([levels[i % len(levels)] for i in range(n)], dtype=object)
    x = np.linspace(0.1, 1.7, n)
    offsets = {"a": 1.0, "b": -0.5, "c": 0.9}
    y = 0.4 * np.sin(2 * x) + np.array([offsets[v] for v in f])
    return pd.DataFrame({"y": y, "f": f, "x": x})


def _sz_df():
    return pd.DataFrame(
        {
            "y": [0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.15, 0.05],
            "f1": ["a", "b", "b", "c", "c", "a", "b", "c"],
            "f2": ["x", "x", "y", "x", "y", "y", "x", "y"],
            "x": np.linspace(0, 1.5, 8),
        }
    )


def _re_df_noisy():
    rng = np.random.default_rng(21)
    g = rng.choice(np.array(["a", "b", "c"], dtype=object), size=40)
    u = {"a": 1.1, "b": -0.4, "c": 0.55}
    signal = np.array([u[str(x)] for x in g], dtype=np.float64)
    return pd.DataFrame({"y": signal + rng.normal(0, 0.35, size=40), "g": g})


def _assert_fs_sz_structure(m, family: str):
    ds = design_structure(m)
    hits = [b for b in ds["term_blocks"] if b["basis_name"] == family]
    assert hits, f"expected a {family!r} term block"
    expected_tt = f"factor_smooth_{family}"
    ok = False
    for b in hits:
        if b["n_penalties"] > 0:
            ok = True
        assert b["term_type"] == expected_tt
        assert family in b["label"] or family in str(b["basis_name"]).lower()
        assert b["coef_width"] > 0
        assert b["train_shape"][1] > 0
    assert ok, f"{family}: expected at least one block with n_penalties > 0"


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(x0, bs="cr", k=8)',
        'y ~ s(x0, bs="cs", k=8)',
        'y ~ s(x0, bs="cc", k=8)',
        'y ~ s(x0, bs="ps", k=8)',
        'y ~ s(x0, x1, bs="tp", k=12)',
        'y ~ s(x0, x1, bs="ts", k=12)',
        'y ~ s(x0, bs="gp", k=8)',
    ],
)
def test_univariate_and_mv_smooth_bases_fit(formula):
    data = _base_df()
    m = fit_gaussian_formula(data, formula)
    ds = design_structure(m)
    assert ds["n_samples"] == len(data)
    assert ds["n_coef"] > 0
    assert len(ds["term_blocks"]) >= 1
    tb0 = ds["term_blocks"][0]
    assert tb0["train_shape"][0] == len(data)
    assert tb0["n_penalties"] >= 1
    assert tb0["coef_width"] == tb0["train_shape"][1] > 0
    for sh in tb0["penalty_shapes"]:
        assert len(sh) == 2 and sh[0] == sh[1]


def test_tensor_te_ti_t2_cr_marginals():
    data = _base_df()
    for formula in (
        "y ~ te(x0, x1, k=[6, 6])",
        "y ~ ti(x0, x1, k=[6, 6])",
        "y ~ t2(x0, x1, k=[6, 6])",
    ):
        m = fit_gaussian_formula(data, formula)
        ds = design_structure(m)
        assert len(ds["term_blocks"]) == 1
        bn = str(ds["term_blocks"][0]["basis_name"])
        assert bn.startswith("te") or bn.startswith("ti") or bn.startswith("t2")


def test_re_fit_structure():
    m = fit_gaussian_formula(_re_df_noisy(), 'y ~ s(g, bs="re")')
    ds = design_structure(m)
    assert len(ds["term_blocks"]) == 1
    b0 = ds["term_blocks"][0]
    assert b0["basis_name"] == "re"
    assert b0["term_type"] == "random_effect"
    assert "re" in b0["label"].lower()
    assert b0["coef_width"] > 0
    assert b0["n_penalties"] >= 1


def test_mrf_fit_structure():
    f_mrf = (
        'y ~ s(region, bs="mrf", k=3, '
        'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
    )
    m = fit_gaussian_formula(_mrf_df(), f_mrf)
    ds = design_structure(m)
    b0 = ds["term_blocks"][0]
    assert b0["basis_name"] == "mrf"
    assert b0["term_type"] == "smooth"
    assert "mrf" in b0["label"].lower()
    assert b0["coef_width"] > 0
    assert b0["n_penalties"] >= 1


def test_fs_fit_structure():
    m = fit_gaussian_formula(_fs_df(), 'y ~ s(f, x, bs="fs", k=6)')
    _assert_fs_sz_structure(m, "fs")


def test_sz_fit_structure():
    m = fit_gaussian_formula(_sz_df(), 'y ~ s(f1, f2, x, bs="sz", k=6)')
    _assert_fs_sz_structure(m, "sz")


def test_factor_by_expansion():
    data = _base_df()
    m = fit_gaussian_formula(data, 'y ~ s(x0, by=fac)')
    ds = design_structure(m)
    smooth_blocks = [b for b in ds["term_blocks"] if b["term_type"] == "smooth"]
    fac_levels = sorted(set(str(v) for v in data["fac"].values))
    assert len(smooth_blocks) == len(fac_levels)
    labels = [b["label"] for b in smooth_blocks]
    assert len(labels) == len(set(labels))
    for b in smooth_blocks:
        assert b["coef_width"] > 0
        assert b["train_shape"][1] > 0
        assert b["runtime_constraint_kind"] == "factor_by"
        assert b["runtime_by_name"] is not None
        assert str(b["runtime_by_name"]).startswith("__gam_by__")


def test_parametric_factor_expansion():
    data = _base_df()
    m = fit_gaussian_formula(data, 'y ~ fac + s(x0, bs="cr", k=8)')
    ds = design_structure(m)
    param_blocks = [b for b in ds["term_blocks"] if b["term_type"] == "parametric"]
    smooth_blocks = [b for b in ds["term_blocks"] if b["term_type"] == "smooth"]
    assert len(param_blocks) >= 1
    assert len(smooth_blocks) >= 1
    assert any(b["basis_name"] == "cr" for b in smooth_blocks)
    for b in param_blocks:
        assert b["has_parametric_expansion_meta"] is True
        assert b["coef_width"] == 1


def test_linked_id_shared_basis_cr():
    data = _base_df()
    m = fit_gaussian_formula(
        data,
        'y ~ s(x0, bs="cr", k=7, id="pool") + s(x1, bs="cr", k=7, id="pool")',
    )
    setups = []
    widths = []
    for tb in m.term_blocks_:
        rt = tb.smooth.runtime
        s = getattr(rt, "shared_basis_setup", None)
        setups.append(s)
        widths.append(int(tb.coef_slice.stop - tb.coef_slice.start))
    assert all(s is not None and s.get("mode") == "pooled_cr_1d" for s in setups)
    assert setups[0].get("id") == "pool"
    assert setups[0].get("k") == 7 == setups[1].get("k")
    assert widths[0] == widths[1] > 0
    px = setups[0].get("pooled_x", [])
    assert isinstance(px, list) and len(px) == 2 * len(data)


def test_linked_id_smoothing_map_shares_sp_for_pool():
    data = _base_df()
    m = fit_gaussian_formula(
        data,
        'y ~ s(x0, bs="cr", k=7, id="pool") + s(x1, bs="cr", k=7, id="pool")',
    )
    sp_idx = [pb.smoothing_index for pb in m.penalty_blocks_]
    assert min(sp_idx) == max(sp_idx) == 0
    assert m.n_smoothing_params_ == 1


def test_predict_types_and_shapes_training_rows():
    data = _base_df()
    m = fit_gaussian_formula(data, 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="ps", k=8)')
    sh = predict_type_shapes(m, data)
    n = len(data)
    p = int(m.n_coef_) + (1 if m.fit_intercept else 0)
    assert sh["response"] == (n,)
    assert sh["link"] == (n,)
    assert sh["terms"][0] == n
    assert sh["lpmatrix"] == (n, p)


def test_predict_types_new_data_predictor_only_columns():
    data = _base_df()
    m = fit_gaussian_formula(data, 'y ~ s(x0, bs="cr", k=8)')
    new_df = pd.DataFrame({"x0": [0.0, 0.5]})
    sh = predict_type_shapes(m, new_df)
    assert sh["response"] == (2,)
    assert sh["link"] == (2,)
    assert sh["terms"][0] == 2
    assert sh["lpmatrix"][0] == 2


def test_term_contribution_shapes_and_intercept():
    data = _base_df().drop(columns=["fac"])
    m = fit_gaussian_formula(data, 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)')
    n = len(data)
    ts = term_contribution_shapes(m, data)
    assert ts["output"] == (n,)
    assert m.fit_intercept is True
    assert "intercept" in ts
    # Scalar intercept is stored as a 0-d array -> shape () in ``term_contribution_shapes``.
    assert ts["intercept"] == ()
    assert np.size(m.intercept_) == 1
    for tb in m.term_blocks_:
        assert ts[tb.term_id] == (n,)


def test_term_contributions_formula_offset_when_X_none():
    rng = np.random.default_rng(3)
    n = 16
    d = pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x0": rng.uniform(0, 1, n),
            "off": rng.uniform(0.05, 0.15, n),
        }
    )
    m = fit_gaussian_formula(d, 'y ~ offset(off) + s(x0, bs="cr", k=6)')
    ts = term_contribution_shapes(m, X=None)
    assert ts["offset"] == (n,)


def test_intercept_only_model():
    d = pd.DataFrame({"y": [1.0, 2.0, 0.5]})
    m = fit_gaussian_formula(d, "y ~ 1")
    assert len(m.term_blocks_) == 0
    assert m.n_coef_ == 0
    assert m.lpmatrix(d).shape == (len(d), 1)
    assert np.asarray(m.predict(X=d, type="link")).shape == (len(d),)
    assert np.asarray(m.predict(X=d, type="response")).shape == (len(d),)


def test_no_intercept_smooth_only():
    rng = np.random.default_rng(4)
    d = pd.DataFrame({"y": rng.standard_normal(25), "x0": rng.uniform(-1, 1, 25)})
    m = fit_gaussian_formula(d, 'y ~ 0 + s(x0, bs="cr", k=8)')
    assert m.fit_intercept is False
    assert m.lpmatrix(d).shape == (len(d), m.n_coef_)
    sh = predict_type_shapes(m, d)
    assert sh["lpmatrix"] == (len(d), m.n_coef_)


def test_random_effect_unseen_factor_level_zero_basis_rows():
    train = pd.DataFrame({"y": [0.0, 1.0, 0.5], "g": ["a", "b", "a"]})
    m = fit_gaussian_formula(train, 'y ~ s(g, bs="re")')
    new = pd.DataFrame({"g": ["a", "unseen_level"]})
    B = m.design_.build_new_matrix(new.to_numpy(dtype=object))
    assert B.shape == (2, m.n_coef_)
    assert np.all(B[1, :] == 0.0)


def test_factor_by_predict_new_data_shapes_and_finite():
    rng = np.random.default_rng(5)
    n = 36
    d = pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x0": rng.uniform(-1, 1, n),
            "fac": np.array(["p", "q"] * (n // 2), dtype=object),
        }
    )
    m = fit_gaussian_formula(d, 'y ~ s(x0, by=fac, bs="cr", k=8)')
    new = pd.DataFrame({"x0": [0.0, 0.25], "fac": ["p", "q"]})
    eta = np.asarray(m.predict(X=new, type="link"), dtype=np.float64).ravel()
    assert eta.shape == (2,)
    assert np.isfinite(eta).all()
    Zn = np.asarray(m.lpmatrix(new), dtype=np.float64)
    assert Zn.shape[0] == 2 and np.isfinite(Zn).all()


def test_linked_id_incompatible_fx_raises():
    data = _base_df()
    with pytest.raises(NotImplementedError, match="Linked id|fx"):
        fit_gaussian_formula(
            data,
            'y ~ s(x0, bs="cr", k=7, id="g", fx=True) + s(x1, bs="cr", k=7, id="g", fx=False)',
        )


def test_parity_snapshot_criterion_view_keys():
    data = _base_df()
    m = fit_gaussian_formula(data, 'y ~ s(x0, bs="cr", k=8)')
    snap = build_parity_snapshot(m, X=data[["x0"]], include_covariances=False)
    assert "parity" in snap
    cv = snap["parity"]["criterion_view"]
    expected = {
        "criterion_name",
        "stored_criterion_value",
        "recomputed_criterion_value",
        "recomputed_criterion_source",
        "criterion_backend",
    }
    assert set(cv.keys()) == expected


def test_parity_snapshot_build_compare_roundtrip(tmp_path):
    data = _base_df()
    m = fit_gaussian_formula(data, "y ~ te(x0, x1, k=[5, 5])")
    Xp = data[["x0", "x1"]]
    snap_a = build_parity_snapshot(m, X=Xp, include_covariances=False)
    snap_b = build_parity_snapshot(m, X=Xp, include_covariances=False)
    struct_a = parity_snapshot_structure(snap_a)
    struct_b = parity_snapshot_structure(snap_b)
    assert struct_a == struct_b
    rep = compare_parity_snapshots(snap_a, snap_b, atol=0.0, rtol=0.0)
    assert rep["passed"]
    roundtrip_parity_snapshot(m, tmp_path, X=Xp)


def test_gam_regressor_parity_api():
    data = _base_df()
    reg = GAMRegressor()
    reg.fit(data=data, formula='y ~ s(x0, bs="cr", k=8)')
    Xp = data[["x0"]]
    s1 = reg.parity_snapshot(X=Xp)
    s2 = build_parity_snapshot(reg.model, X=Xp)
    assert parity_snapshot_structure(s1) == parity_snapshot_structure(s2)


def test_fixed_sp_exact_characterization():
    data = _base_df()
    m = fit_gaussian_formula(
        data,
        'y ~ s(x0, bs="cr", k=8, sp=1.0)',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    assert np.isfinite(m.intercept_)
    assert np.isfinite(m.scale_)
    assert m.deviance_ >= 0.0
    assert m._optim_method == "fixed" or m.smoothing_method == "fixed"
    assert m.n_smoothing_params_ == 1
    sp = np.asarray(m.smoothing_params, dtype=np.float64).ravel()
    assert sp.shape == (1,)
    np.testing.assert_allclose(sp, [1.0], rtol=0.0, atol=0.0)
    coef = np.asarray(m.coef_, dtype=np.float64)
    assert coef.shape == (m.n_coef_,)
    assert np.isfinite(coef).all()
    assert m.term_blocks_[0].n_penalties == 1
    pbs = [pb for pb in m.penalty_blocks_ if pb.coef_slice == m.term_blocks_[0].coef_slice]
    assert len(pbs) == 1
