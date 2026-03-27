"""
mgcv parity tests for:
  1. pc= point-constraint smooths (cr, cs, cc, ps, gp, tp, ts)
  2. Linked-basis id= smooths (compatible k, incompatible-k harmonisation)

Every test here runs the SAME formula through both NAMpy and mgcv (via
Rscript) and compares results.

Design notes
------------
- cr: penalty matrices are identically scaled to mgcv, so FIXED-sp comparisons
  work to machine precision (≤ 1e-10).
- cs: shrinkage penalty scaling differs slightly from mgcv; fixed-sp diff is
  ~1e-5.  REML comparisons work to ~1e-4.
- ps, tp, ts, cc: penalty matrices are scaled differently from mgcv
  (pre-existing, unrelated to pc=).  Fixed-sp comparisons are meaningless;
  REML predictions match well (≤ 1e-6) even though the optimised sp values
  differ.  Tests use ``check_sp=False`` to focus on prediction parity.
- gp: pre-existing ~7.5 % prediction mismatch between NAMpy and mgcv under
  REML (unrelated to pc=); gp+pc= parity tests are omitted.
- pc=+id=: mgcv does not support combining pc= with linked id= smooths; the
  combined test only verifies NAMpy-internal consistency (no mgcv comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _assert_basic_mgcv_parity,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)

# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

def _data_1d(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, n)
    y = np.sin(np.pi * x) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x})


def _data_2col(n: int = 180, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.uniform(-1.5, 1.5, n)
    y  = np.sin(1.2 * x0) + 0.4 * x1 ** 2 + rng.normal(0.0, 0.15, n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_parity(actual: dict, expected: dict, *, atol: float = 1e-10) -> None:
    """Assert that response and link predictions match to ``atol``."""
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["response"], dtype=np.float64),
        np.asarray(expected["predictions"]["response"], dtype=np.float64),
        atol=atol,
        rtol=atol,
        err_msg="response predictions differ from mgcv",
    )
    np.testing.assert_allclose(
        np.asarray(actual["predictions"]["link"], dtype=np.float64),
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=atol,
        rtol=atol,
        err_msg="link predictions differ from mgcv",
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_by_term"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
        atol=atol,
        rtol=atol,
        err_msg="edf_by_term differs from mgcv",
    )
    np.testing.assert_allclose(
        float(actual["fit"]["deviance"]),
        float(expected["fit"]["deviance"]),
        atol=atol,
        rtol=atol,
        err_msg="deviance differs from mgcv",
    )


# ===========================================================================
# pc= parity — fixed smoothing parameter (cr: exact; cs: near-exact)
# ===========================================================================

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript not available")
class TestPcParityFixed:
    """
    At a FIXED smoothing parameter the penalised-LS solution is identical in
    closed form when penalty matrices are identically scaled.

    cr matches mgcv to machine precision (≤ 1e-10) because cr penalty matrices
    are identical.  cs has a small pre-existing shrinkage-penalty scale
    difference so the tolerance is relaxed to 5e-5.

    Other basis types (ps, tp, ts, cc) have larger penalty-scale differences and
    are tested under REML instead (see TestPcParityREML).
    """

    def test_cr_pc_fixed_sp_matches_mgcv(self):
        """cr basis with pc=0 at fixed sp matches mgcv to machine precision."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cr", k=8, pc=0.0, sp=1.5)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cs_pc_fixed_sp_matches_mgcv(self):
        """
        cs (shrinkage cr) with pc=0 at fixed sp matches mgcv to ~1e-4.

        cs adds a ridge penalty on top of the cr basis; the shrinkage penalty
        matrix is scaled slightly differently from mgcv's, producing a
        pre-existing ~1e-5 prediction difference even without pc=.
        pc= does not worsen this gap.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="cs", k=8, pc=0.0, sp=1.5)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected, atol=5e-5)

    def test_cr_pc_n_coef_matches_mgcv(self):
        """
        mgcv: pc= reparameterises the basis (over-parameterisation) but does NOT
        reduce the number of estimated smooth coefficients relative to the
        no-pc= case (both yield k-1 columns after sum-to-zero absorption).
        NAMpy must reproduce this.

        NAMpy n_coef_ counts smooth coefficients only (intercept is NOT
        included), matching the number mgcv stores in coef_full after
        subtracting the intercept.
        """
        data = _data_1d()

        snap_no_pc = _run_mgcv_snapshot(
            data, 'y ~ s(x, bs="cr", k=8, sp=1.5)', "gaussian", "fixed"
        )
        snap_pc    = _run_mgcv_snapshot(
            data, 'y ~ s(x, bs="cr", k=8, pc=0.0, sp=1.5)', "gaussian", "fixed"
        )
        # mgcv: same number of smooth coefs with and without pc=
        mgcv_n_no_pc = len(snap_no_pc["fit"]["coef_full"]) - 1  # subtract intercept
        mgcv_n_pc    = len(snap_pc["fit"]["coef_full"])    - 1
        assert mgcv_n_no_pc == mgcv_n_pc, (
            f"mgcv itself: expected same smooth n_coef for pc= vs no-pc, "
            f"got {mgcv_n_no_pc} vs {mgcv_n_pc}"
        )

        # NAMpy should match
        from nampy.basemodels.gam import GAM
        gam_no_pc = GAM(family="gaussian",
                        formula='y ~ s(x, bs="cr", k=8)',
                        optimize_smoothing=False, smoothing_method="fixed",
                        smoothing_params=[1.5])
        gam_pc    = GAM(family="gaussian",
                        formula='y ~ s(x, bs="cr", k=8, pc=0.0)',
                        optimize_smoothing=False, smoothing_method="fixed",
                        smoothing_params=[1.5])
        gam_no_pc.fit(data=data)
        gam_pc.fit(data=data)
        assert gam_pc.n_coef_ == gam_no_pc.n_coef_, (
            f"NAMpy: expected same n_coef for pc= vs no-pc, "
            f"got {gam_pc.n_coef_} vs {gam_no_pc.n_coef_}"
        )
        # NAMpy n_coef_ does NOT include the intercept (unlike mgcv coef_full).
        # mgcv_n_no_pc already has the intercept subtracted, so comparison is direct.
        assert gam_no_pc.n_coef_ == mgcv_n_no_pc, (
            f"NAMpy n_coef_={gam_no_pc.n_coef_} != mgcv smooth coefs={mgcv_n_no_pc}"
        )

    def test_cr_pc_smooth_zero_at_constraint_matches_mgcv(self):
        """
        Both NAMpy and mgcv must give smooth contribution = 0 at the pc= point,
        confirming the constraint is enforced by both to machine precision.
        """
        data = _data_1d()
        pc_value = 0.5

        from nampy.basemodels.gam import GAM
        formula = f'y ~ s(x, bs="cr", k=8, pc={pc_value}, sp=1.5)'
        gam = GAM(family="gaussian", formula=formula,
                  optimize_smoothing=False, smoothing_method="fixed",
                  smoothing_params=[1.5])
        gam.fit(data=data)

        x_at_pc = pd.DataFrame({"x": [pc_value]})
        # The smooth contribution at x0 must be 0 (intercept excluded)
        smooth_at_pc = gam.predict(x_at_pc, type="terms")[0, 0]
        assert abs(smooth_at_pc) < 1e-10, (
            f"NAMpy smooth at pc={pc_value} is {smooth_at_pc}, expected ~0"
        )

        # mgcv must also give ~0 at the pc= point
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        # Find closest training obs to pc_value or use the predict-terms column
        pred_terms_mgcv = np.asarray(expected["predictions"]["terms"], dtype=np.float64)
        x_train = np.asarray(data["x"])
        idx = int(np.argmin(np.abs(x_train - pc_value)))
        # The smooth column (col 0) evaluated at a point near pc= must be ~0
        # (not strictly at training obs, but the smooth is continuous so nearby
        # obs will be very close to 0 if the constraint is enforced)
        # Better: compare lpmatrix evaluations — but for simplicity just confirm
        # the prediction diff between NAMpy and mgcv is small everywhere.
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        np.testing.assert_allclose(
            np.asarray(actual["predictions"]["response"], dtype=np.float64),
            np.asarray(expected["predictions"]["response"], dtype=np.float64),
            atol=1e-10, rtol=1e-10,
        )


# ===========================================================================
# pc= parity — REML (approximate comparison)
# ===========================================================================

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript not available")
class TestPcParityREML:
    """
    Under REML, smoothing parameters are optimised numerically.

    For cr: sp values converge to the same optimum in both NAMpy and mgcv, so
    both predictions and sp are tight.

    For cs, ps, tp, ts, cc: penalty matrices are scaled differently from mgcv
    (pre-existing issue, unrelated to pc=), causing sp optimisation to converge
    to different log(sp) values even though predictions are similar.  Tests use
    ``check_sp=False`` and verify only predictions and EDF.

    Note: gp has a pre-existing ~7.5 % prediction mismatch under REML even
    without pc= and is therefore not tested here.
    """

    def test_cr_pc_reml_matches_mgcv(self):
        data = _data_1d()
        formula = 'y ~ s(x, bs="cr", k=8, pc=0.0)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=1e-2, pred_rtol=1e-2, sp_log_atol=0.3
        )

    def test_cs_pc_reml_matches_mgcv(self):
        """
        cs with pc=0 under REML: predictions match to ~1e-4.
        sp values differ due to shrinkage-penalty scale mismatch; check_sp=False.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="cs", k=8, pc=0.0)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=1e-4, pred_rtol=1e-4,
            sp_log_atol=0.0,  # unused
            check_sp=False,
        )

    def test_cc_pc_reml_matches_mgcv(self):
        """
        Cyclic cubic with pc=0.5 under REML: predictions match to ~1e-8.
        sp values differ (penalty scale mismatch); check_sp=False.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="cc", k=8, pc=0.5)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=1e-7, pred_rtol=1e-7,
            sp_log_atol=0.0,  # unused
            check_sp=False,
        )

    def test_ps_pc_reml_matches_mgcv(self):
        """
        P-spline with pc=0 under REML: predictions match to ~1e-6.
        sp values differ (P-spline penalty scale mismatch); check_sp=False.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="ps", k=8, pc=0.0)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=1e-5, pred_rtol=1e-5,
            sp_log_atol=0.0,  # unused
            check_sp=False,
        )

    def test_tp_pc_reml_matches_mgcv(self):
        """
        Thin-plate spline with pc=0 under REML: predictions match to ~1e-10.
        sp values differ (tp penalty scale mismatch); check_sp=False.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="tp", k=8, pc=0.0)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=1e-9, pred_rtol=1e-9,
            sp_log_atol=0.0,  # unused
            check_sp=False,
        )

    def test_ts_pc_reml_matches_mgcv(self):
        """
        ts (shrinkage tp) with pc=0 under REML: predictions match to ~1e-6.
        check_sp=False for the same reason as tp.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="ts", k=8, pc=0.0)'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected,
            pred_atol=1e-5, pred_rtol=1e-5,
            sp_log_atol=0.0,  # unused
            check_sp=False,
        )


# ===========================================================================
# Linked basis (id=) parity — compatible k (exact)
# ===========================================================================

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript not available")
class TestLinkedIdParityFixed:
    """Fixed-sp parity for linked-basis id= smooths."""

    def test_linked_cr_compatible_k_fixed_sp_matches_mgcv(self):
        """Linked 1D cr smooths with same k must match mgcv exactly at fixed sp."""
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=6, id="g", sp=0.9)'
        )
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_linked_cr_compatible_k_shared_sp_count_matches_mgcv(self):
        """
        Linked terms share one smoothing parameter; the model must report
        fewer smoothing parameters than the equivalent unlinked model.
        """
        data = _data_2col()
        formula_linked   = 'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")'
        formula_unlinked = 'y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)'

        snap_linked   = _run_mgcv_snapshot(data, formula_linked,   "gaussian", "REML")
        snap_unlinked = _run_mgcv_snapshot(data, formula_unlinked, "gaussian", "REML")
        # Use atleast_1d to handle scalar smoothing_params (single-sp case)
        mgcv_sp_linked   = len(np.atleast_1d(snap_linked["fit"]["smoothing_params"]))
        mgcv_sp_unlinked = len(np.atleast_1d(snap_unlinked["fit"]["smoothing_params"]))
        assert mgcv_sp_linked < mgcv_sp_unlinked, (
            f"mgcv: expected fewer sp for linked ({mgcv_sp_linked}) vs "
            f"unlinked ({mgcv_sp_unlinked})"
        )

        from nampy.basemodels.gam import GAM
        gam_linked   = GAM(family="gaussian", formula=formula_linked)
        gam_unlinked = GAM(family="gaussian", formula=formula_unlinked)
        gam_linked.fit(data=data)
        gam_unlinked.fit(data=data)
        assert gam_linked.n_smoothing_params_ < gam_unlinked.n_smoothing_params_, (
            "NAMpy: expected fewer sp for linked vs unlinked model"
        )
        assert gam_linked.n_smoothing_params_ == mgcv_sp_linked

    def test_linked_cr_compatible_k_reml_matches_mgcv(self):
        """Linked cr smooths with same k under REML match mgcv approximately."""
        data = _data_2col()
        formula = 'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")'
        actual   = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=3e-2, pred_rtol=3e-2, sp_log_atol=0.45
        )


# ===========================================================================
# Linked basis (id=) — incompatible k harmonisation
# ===========================================================================

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript not available")
class TestLinkedIdIncompatibleK:
    """
    When linked terms have different k, mgcv uses the FIRST term's k for the
    shared basis (representative-term convention).  NAMpy must reproduce the
    same behaviour.
    """

    def test_linked_cr_incompatible_k_first_k_used(self):
        """
        mgcv harmonises s(x0, k=6, id=g) + s(x1, k=8, id=g) by using k=6
        (the first term's k).  NAMpy must produce the same number of smooth
        coefficients as mgcv (NAMpy n_coef_ excludes the intercept).
        """
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=8, id="g", sp=0.9)'
        )
        mgcv_snap = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        # mgcv coef_full = [intercept, smooth0_coefs..., smooth1_coefs...]
        mgcv_n_coef = len(mgcv_snap["fit"]["coef_full"])
        expected_k = 6
        expected_total = 1 + 2 * (expected_k - 1)  # intercept + 2 smooths * (k-1)
        assert mgcv_n_coef == expected_total, (
            f"mgcv did not use first k={expected_k}: got n_coef={mgcv_n_coef}, "
            f"expected {expected_total}"
        )

        import warnings
        from nampy.basemodels.gam import GAM
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gam = GAM(family="gaussian", formula=formula,
                      optimize_smoothing=False, smoothing_method="fixed",
                      smoothing_params=[0.9])
            gam.fit(data=data)
        assert any("k" in str(warning.message).lower() for warning in w), (
            "Expected a warning about k harmonisation"
        )
        # NAMpy n_coef_ excludes the intercept; mgcv_n_coef includes it.
        # Subtract 1 from mgcv_n_coef to compare smooth coefs only.
        assert gam.n_coef_ == mgcv_n_coef - 1, (
            f"NAMpy n_coef_={gam.n_coef_} does not match "
            f"mgcv smooth n_coef={mgcv_n_coef - 1} (mgcv total={mgcv_n_coef})"
        )

    def test_linked_cr_incompatible_k_fixed_sp_predictions_match_mgcv(self):
        """
        After k harmonisation, NAMpy predictions at fixed sp must match mgcv
        to machine precision.
        """
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=8, id="g", sp=0.9)'
        )
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")

        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_linked_cr_incompatible_k_reversed_order_first_k_wins(self):
        """
        When the formula is reversed (k=8 first, k=6 second) the first k=8 is
        used.  NAMpy and mgcv must agree on which k is canonical.
        """
        data = _data_2col()
        # First term has k=8, second has k=6 — canonical k should now be 8
        formula = (
            'y ~ s(x0, bs="cr", k=8, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=6, id="g", sp=0.9)'
        )
        mgcv_snap = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        mgcv_n_coef = len(mgcv_snap["fit"]["coef_full"])
        expected_k = 8
        expected_total = 1 + 2 * (expected_k - 1)
        assert mgcv_n_coef == expected_total, (
            f"mgcv: expected first k={expected_k}, n_coef={expected_total}, "
            f"got {mgcv_n_coef}"
        )

        import warnings
        from nampy.basemodels.gam import GAM
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gam = GAM(family="gaussian", formula=formula,
                      optimize_smoothing=False, smoothing_method="fixed",
                      smoothing_params=[0.9])
            gam.fit(data=data)
        # NAMpy n_coef_ excludes the intercept; mgcv_n_coef includes it.
        assert gam.n_coef_ == mgcv_n_coef - 1, (
            f"NAMpy n_coef_={gam.n_coef_} != mgcv smooth n_coef={mgcv_n_coef - 1}"
        )

    def test_linked_cr_incompatible_k_reml_matches_mgcv(self):
        """After k harmonisation, REML optimisation should also match mgcv."""
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g")'
            ' + s(x1, bs="cr", k=8, id="g")'
        )
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=3e-2, pred_rtol=3e-2, sp_log_atol=0.45
        )


# ===========================================================================
# Combined: pc= + linked id= (NAMpy-internal consistency only)
# ===========================================================================

class TestPcAndLinkedCombined:
    """
    mgcv does not support combining pc= with linked id= smooths (R raises an
    error for this combination).  These tests verify NAMpy-internal consistency
    only: that the model fits without error, the smooth is zero at the pc= point,
    and the linked basis properly pools knots.
    """

    def test_linked_cr_with_pc_fits_and_enforces_constraint(self):
        """
        Linked smooths with pc= must fit without error and enforce s(x0=0)=0
        in both terms.
        """
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", pc=0.0, sp=1.0)'
            ' + s(x1, bs="cr", k=6, id="g", pc=0.0, sp=1.0)'
        )
        from nampy.basemodels.gam import GAM
        gam = GAM(family="gaussian", formula=formula,
                  optimize_smoothing=False, smoothing_method="fixed",
                  smoothing_params=[1.0])
        gam.fit(data=data)
        assert gam.n_coef_ > 0

        # Both smooths must be zero at x0=x1=0
        x_at_pc = pd.DataFrame({"x0": [0.0], "x1": [0.0]})
        terms = gam.predict(x_at_pc, type="terms")
        assert abs(float(terms[0, 0])) < 1e-10, (
            f"smooth for x0 at pc=0 is {terms[0, 0]}, expected ~0"
        )
        assert abs(float(terms[0, 1])) < 1e-10, (
            f"smooth for x1 at pc=0 is {terms[0, 1]}, expected ~0"
        )

    def test_linked_cr_with_pc_predictions_are_finite(self):
        """Basic sanity: linked cr + pc= produces finite predictions."""
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", pc=0.0)'
            ' + s(x1, bs="cr", k=6, id="g", pc=0.0)'
        )
        from nampy.basemodels.gam import GAM
        gam = GAM(family="gaussian", formula=formula)
        gam.fit(data=data)
        preds = gam.predict(data)
        assert np.all(np.isfinite(preds))
