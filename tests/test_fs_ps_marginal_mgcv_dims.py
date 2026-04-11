"""Fast structural checks: fs + ps marginal vs mgcv smoothCon layout (no fit, no R)."""

from __future__ import annotations

from nampy.gam.design.compiler import compile_predictor_designs
from nampy.gam.formula import compile_predictor_specs_from_formula, parse_gam_formula
from mgcv_parity_utils import _make_fs_data


def test_fs_ps_marginal_design_matches_mgcv_smoothcon_structure():
    # Reference: mgcv smoothCon(s(f,x,bs="fs",xt=list(bs="ps",m=2,k=7)), ...,
    #   absorb.cons=TRUE) on _make_fs_data() gives ncol(X)=30, length(S)=3,
    #   ranks (24, 3, 3).  Outer s() default k is 10 (same as compile default_k).
    data = _make_fs_data()
    formula = 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'
    parsed = parse_gam_formula(formula)
    specs = compile_predictor_specs_from_formula(parsed, default_k=10, default_select=False)
    X = data[["f", "x"]].to_numpy(dtype=object)
    des = compile_predictor_designs(X, ["f", "x"], specs)[0]

    assert des.n_smoothing_params == 3
    assert des.design_matrix.shape == (len(data), 30)

    pdefs = des.compiled_terms[0].smooth.penalty_specs
    assert len(pdefs) == 3
    assert [int(p.rank) for p in pdefs] == [24, 3, 3]
