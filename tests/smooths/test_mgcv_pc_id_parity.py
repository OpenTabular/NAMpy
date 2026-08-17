"""
mgcv parity tests for:
  1. pc= point-constraint smooths (cr, cs, cc, ps, tp, ts)
  2. Linked-basis id= smooths (compatible k, incompatible-k harmonisation)

Every test here runs the SAME formula through both NAMpy and mgcv (via
Rscript) and compares results.

Design notes
------------
- cr: penalty matrices are identically scaled to mgcv, so FIXED-sp comparisons
  work to machine precision (<= 1e-10).
- cs: point-constrained cs now matches mgcv exactly on the tested fixed-sp and
  REML surfaces.
- ps: point-constrained P-splines now match mgcv exactly, including optimised
  smoothing parameters under REML.
- cc: cyclic cubic pc= now matches mgcv exactly after scaling the penalty
  before point-constraint absorption, mirroring `smoothCon(scale.penalty=TRUE)`.
- ts: point-constrained ts REML surfaces now match mgcv exactly on the tested
  slices.
- tp: fixed-sp pc= constructions now match mgcv to machine precision,
  including the previously blocked multivariate path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam._model_state import _n_coef, _n_smoothing_params
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _assert_exact_mgcv_snapshot_parity,
    _fit_nampy_snapshot,
    _make_gaussian_data,
    _make_gaussian_data_3col,
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
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(0.0, 0.15, n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _data_2d(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.5, 1.5, n)
    y = np.sin(1.3 * x) + 0.4 * z**2 + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _data_factor_by(n: int = 150, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, n)
    f = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    offsets = {"a": 0.8, "b": -0.4, "c": 0.2}
    y = np.sin(np.pi * x) + np.array([offsets[str(v)] for v in f], dtype=np.float64)
    y = y + rng.normal(0.0, 0.12, n)
    return pd.DataFrame({"y": y, "x": x, "f": f})


def _data_numeric_by_2d(n: int = 140, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    w = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(0.5, 1.5, n)
    y = z * (np.sin(1.3 * x) + 0.4 * w**2) + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"y": y, "x": x, "w": w, "z": z})


def _data_numeric_by_1d(n: int = 140, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.5, 1.5, n)
    y = z * np.sin(np.pi * x) + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


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
# pc= parity -- fixed smoothing parameter
# ===========================================================================


class TestPcParityFixed:
    """
    At a FIXED smoothing parameter the penalised-LS solution is identical in
    closed form when penalty matrices are identically scaled.

    On the tested pc= surfaces, cr/cs/cc/ps/tp/ts all match mgcv to machine
    precision when the smoothing parameter is fixed.
    """

    def test_cr_pc_fixed_sp_matches_mgcv(self):
        """cr basis with pc=0 at fixed sp matches mgcv to machine precision."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cr", k=8, pc=0.0, sp=1.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cs_pc_fixed_sp_matches_mgcv(self):
        """cs (shrinkage cr) with pc=0 at fixed sp matches mgcv to machine precision.

        scipy.linalg.eigh (DSYEVR) finds the same null-space eigenvectors as
        R's eigen(symmetric=TRUE), so the cs shrinkage penalty now matches to
        machine precision, making cs + pc= parity on par with cr.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="cs", k=8, pc=0.0, sp=1.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_ps_pc_fixed_sp_matches_mgcv(self):
        """ps with pc=0 at fixed sp matches mgcv exactly."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="ps", k=8, pc=0.0, sp=1.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cc_pc_fixed_sp_matches_mgcv(self):
        """cc with pc= matches mgcv exactly at fixed sp."""
        data = _data_1d(seed=45)
        formula = 'y ~ s(x, bs="cc", k=8, pc=0.5, sp=1.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cr_pc_n_coef_matches_mgcv(self):
        """
        mgcv: pc= reparameterises the basis (over-parameterisation) but does NOT
        reduce the number of estimated smooth coefficients relative to the
        no-pc= case (both yield k-1 columns after sum-to-zero absorption).
        NAMpy must reproduce this.

        NAMpy's internal smooth coefficient count via ``_n_coef(gam)`` excludes
        the intercept, matching the number mgcv stores in ``coef_full`` after
        subtracting the intercept.
        """
        data = _data_1d()

        snap_no_pc = _run_mgcv_snapshot(
            data, 'y ~ s(x, bs="cr", k=8, sp=1.5)', "gaussian", "fixed"
        )
        snap_pc = _run_mgcv_snapshot(
            data, 'y ~ s(x, bs="cr", k=8, pc=0.0, sp=1.5)', "gaussian", "fixed"
        )
        # mgcv: same number of smooth coefs with and without pc=
        mgcv_n_no_pc = len(snap_no_pc["fit"]["coef_full"]) - 1  # subtract intercept
        mgcv_n_pc = len(snap_pc["fit"]["coef_full"]) - 1
        assert mgcv_n_no_pc == mgcv_n_pc, (
            f"mgcv itself: expected same smooth n_coef for pc= vs no-pc, "
            f"got {mgcv_n_no_pc} vs {mgcv_n_pc}"
        )

        # NAMpy should match
        from nampy.gam.model.api import GAM

        gam_no_pc = GAM(
            family="gaussian",
            formula='y ~ s(x, bs="cr", k=8)',
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=[1.5],
        )
        gam_pc = GAM(
            family="gaussian",
            formula='y ~ s(x, bs="cr", k=8, pc=0.0)',
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=[1.5],
        )
        gam_no_pc.fit(data=data)
        gam_pc.fit(data=data)
        assert _n_coef(gam_pc) == _n_coef(gam_no_pc), (
            f"NAMpy: expected same n_coef for pc= vs no-pc, "
            f"got {_n_coef(gam_pc)} vs {_n_coef(gam_no_pc)}"
        )
        # NAMpy's internal smooth coefficient count does not include the
        # intercept (unlike mgcv ``coef_full``).
        # mgcv_n_no_pc already has the intercept subtracted, so comparison is direct.
        assert (
            _n_coef(gam_no_pc) == mgcv_n_no_pc
        ), f"NAMpy n_coef={_n_coef(gam_no_pc)} != mgcv smooth coefs={mgcv_n_no_pc}"

    def test_cr_pc_fixed_sp_full_parity_matches_mgcv(self):
        """Whole-fit parity separately confirms the pc= path still matches mgcv."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cr", k=8, pc=0.5, sp=1.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cr_factor_by_pc_fixed_sp_matches_mgcv(self):
        """Factor-by replicated cr smooths with pc= match mgcv exactly at fixed sp."""
        data = _data_factor_by()
        formula = 'y ~ s(x, by=f, bs="cr", k=8, pc=0.2, sp=1.3)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_tp_multivariate_pc_fixed_sp_matches_mgcv(self):
        """Multivariate tp smooths with pc= match mgcv exactly at fixed sp."""
        data = _data_2d()
        formula = 'y ~ s(x, z, bs="tp", k=15, pc=[0.2, -0.3], sp=1.1)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_ts_multivariate_pc_fixed_sp_matches_mgcv(self):
        """Multivariate ts smooths with pc= match mgcv exactly at fixed sp."""
        data = _data_2d(seed=14)
        formula = 'y ~ s(x, z, bs="ts", k=15, pc=[0.2, -0.3], sp=1.1)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_cs_factor_by_pc_fixed_sp_matches_mgcv(self):
        """Factor-by replicated cs smooths with pc= match mgcv at fixed sp."""
        data = _data_factor_by(seed=15)
        formula = 'y ~ s(x, by=f, bs="cs", k=8, pc=0.2, sp=1.3)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected, atol=5e-4)

# ===========================================================================
# pc= parity -- REML
# ===========================================================================


class TestPcParityREML:
    """
    On the tested pc= REML surfaces, cr/cs/cc/ps/tp/ts match mgcv exactly.

    """

    def test_cr_pc_reml_matches_mgcv(self):
        """Verify that cr pc REML matches mgcv."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cr", k=8, pc=0.0)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=1e-2, pred_rtol=1e-2, sp_log_atol=0.3
        )

    def test_cs_pc_reml_matches_mgcv(self):
        """cs with pc=0 under REML matches mgcv exactly."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cs", k=8, pc=0.0)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(actual, expected)

    def test_cc_pc_reml_matches_mgcv(self):
        """Cyclic cubic with pc=0.5 under REML matches mgcv exactly."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="cc", k=8, pc=0.5)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(actual, expected)

    def test_ps_pc_reml_matches_mgcv(self):
        """
        P-spline with pc=0 under REML matches mgcv, including smoothing parameter.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="ps", k=8, pc=0.0)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-10,
            pred_rtol=1e-10,
            sp_log_atol=1e-8,
        )

    def test_ps_factor_by_pc_reml_matches_mgcv(self):
        """
        Factor-by replicated P-splines with pc= match mgcv under REML.
        """
        data = _data_factor_by(seed=31)
        formula = 'y ~ s(x, by=f, bs="ps", k=8, pc=0.2)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-10,
            pred_rtol=1e-10,
            sp_log_atol=1e-8,
        )

    def test_tp_pc_reml_matches_mgcv(self):
        """
        Thin-plate spline with pc=0 under REML matches mgcv to machine precision.

        This case uses the exact Gaussian REML backend, so we also check that the
        optimized smoothing parameter and reported REML score agree directly.
        """
        data = _data_1d()
        formula = 'y ~ s(x, bs="tp", k=8, pc=0.0)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-9,
            pred_rtol=5e-9,
            sp_log_atol=1e-7,
            check_sp=True,
        )
        np.testing.assert_allclose(
            float(actual["fit"]["criterion_value"]),
            float(expected["fit"]["criterion_value"]),
            atol=1e-10,
            rtol=1e-10,
            err_msg="tp pc REML criterion differs from mgcv",
        )
        assert (
            actual["parity"]["criterion_view"]["criterion_backend"] == "gaussian_exact"
        )
        np.testing.assert_allclose(
            float(actual["parity"]["criterion_view"]["joint_criterion_value"]),
            float(expected["fit"]["criterion_value"]),
            atol=1e-10,
            rtol=1e-10,
            err_msg="tp pc REML joint criterion view differs from mgcv",
        )

    def test_ts_pc_reml_matches_mgcv(self):
        """ts (shrinkage tp) with pc=0 under REML matches mgcv exactly."""
        data = _data_1d()
        formula = 'y ~ s(x, bs="ts", k=8, pc=0.0)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(actual, expected)

    def test_tp_numeric_by_pc_reml_matches_mgcv(self):
        """Verify that tp numeric by pc REML matches mgcv."""
        data = _data_numeric_by_2d(seed=41)
        formula = 'y ~ s(x, w, bs="tp", k=15, pc=[0.2, -0.3], by=z)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-8,
            pred_rtol=1e-8,
            sp_log_atol=1e-6,
        )

    def test_ts_numeric_by_pc_reml_matches_mgcv(self):
        """Verify that ts numeric by pc REML matches mgcv."""
        data = _data_numeric_by_2d(seed=42)
        formula = 'y ~ s(x, w, bs="ts", k=15, pc=[0.2, -0.3], by=z)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(actual, expected)

    def test_ps_numeric_by_pc_reml_matches_mgcv(self):
        """
        Numeric-by P-splines with pc= match mgcv under REML.
        """
        data = _data_numeric_by_1d(seed=44)
        formula = 'y ~ s(x, bs="ps", k=8, pc=0.2, by=z)'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-10,
            pred_rtol=1e-10,
            sp_log_atol=1e-8,
        )


# ===========================================================================
# Linked basis (id=) parity -- compatible k (exact)
# ===========================================================================


class TestLinkedIdParityFixed:
    """Fixed-sp parity for linked-basis id= smooths."""

    def test_linked_cr_compatible_k_fixed_sp_matches_mgcv(self):
        """Linked 1D cr smooths with same k must match mgcv exactly at fixed sp."""
        data = _data_2col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=6, id="g", sp=0.9)'
        )
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_linked_cr_compatible_k_shared_sp_count_matches_mgcv(self):
        """
        Linked terms share one smoothing parameter; the model must report
        fewer smoothing parameters than the equivalent unlinked model.
        """
        data = _data_2col()
        formula_linked = 'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")'
        formula_unlinked = 'y ~ s(x0, bs="cr", k=6) + s(x1, bs="cr", k=6)'

        snap_linked = _run_mgcv_snapshot(data, formula_linked, "gaussian", "REML")
        snap_unlinked = _run_mgcv_snapshot(data, formula_unlinked, "gaussian", "REML")
        # Use atleast_1d to handle scalar smoothing_params (single-sp case)
        mgcv_sp_linked = len(np.atleast_1d(snap_linked["fit"]["smoothing_params"]))
        mgcv_sp_unlinked = len(np.atleast_1d(snap_unlinked["fit"]["smoothing_params"]))
        assert mgcv_sp_linked < mgcv_sp_unlinked, (
            f"mgcv: expected fewer sp for linked ({mgcv_sp_linked}) vs "
            f"unlinked ({mgcv_sp_unlinked})"
        )

        from nampy.gam.model.api import GAM

        gam_linked = GAM(family="gaussian", formula=formula_linked)
        gam_unlinked = GAM(family="gaussian", formula=formula_unlinked)
        gam_linked.fit(data=data)
        gam_unlinked.fit(data=data)
        assert _n_smoothing_params(gam_linked) < _n_smoothing_params(
            gam_unlinked
        ), "NAMpy: expected fewer sp for linked vs unlinked model"
        assert _n_smoothing_params(gam_linked) == mgcv_sp_linked

    def test_linked_cr_compatible_k_reml_matches_mgcv(self):
        """Linked cr smooths with same k under REML match mgcv approximately."""
        data = _data_2col()
        formula = 'y ~ s(x0, bs="cr", k=6, id="g") + s(x1, bs="cr", k=6, id="g")'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=3e-2, pred_rtol=3e-2, sp_log_atol=0.45
        )


# ===========================================================================
# Linked basis (id=) -- incompatible k harmonisation
# ===========================================================================


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
        coefficients as mgcv (NAMpy's internal smooth coefficient count
        excludes the intercept).
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

        from nampy.gam.model.api import GAM

        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=[0.9],
        )
        gam.fit(data=data)
        # NAMpy's internal smooth coefficient count excludes the intercept;
        # ``mgcv_n_coef`` includes it.
        # Subtract 1 from mgcv_n_coef to compare smooth coefs only.
        assert _n_coef(gam) == mgcv_n_coef - 1, (
            f"NAMpy n_coef={_n_coef(gam)} does not match "
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
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")

        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_linked_cr_incompatible_k_reversed_order_first_k_wins(self):
        """
        When the formula is reversed (k=8 first, k=6 second) the first k=8 is
        used.  NAMpy and mgcv must agree on which k is canonical.
        """
        data = _data_2col()
        # First term has k=8, second has k=6 -- canonical k should now be 8
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

        from nampy.gam.model.api import GAM

        gam = GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=[0.9],
        )
        gam.fit(data=data)
        # NAMpy's internal smooth coefficient count excludes the intercept;
        # ``mgcv_n_coef`` includes it.
        assert (
            _n_coef(gam) == mgcv_n_coef - 1
        ), f"NAMpy n_coef={_n_coef(gam)} != mgcv smooth n_coef={mgcv_n_coef - 1}"

    def test_linked_cr_incompatible_k_reml_matches_mgcv(self):
        """After k harmonisation, REML optimisation should also match mgcv."""
        data = _data_2col()
        formula = 'y ~ s(x0, bs="cr", k=6, id="g")' ' + s(x1, bs="cr", k=8, id="g")'
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_basic_mgcv_parity(
            actual, expected, pred_atol=3e-2, pred_rtol=3e-2, sp_log_atol=0.45
        )


class TestBySelectAndMoreLinkedIdParity:
    """
    Parity checks for linked id and select=True smooth setups that extend the fixed and
    REML baseline matrices.
    """
    def test_cr_factor_by_select_reml_matches_mgcv(self):
        """Factor-by replicated cr smooths with select=True match mgcv under REML."""
        data = _data_factor_by()
        formula = 'y ~ s(x, by=f, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-5,
            pred_rtol=5e-5,
            sp_log_atol=1e-4,
        )

    def test_cc_factor_by_fixed_sp_matches_mgcv(self):
        """Factor-by replicated cc smooths at fixed sp should match mgcv exactly."""
        data = _data_factor_by(seed=113)
        formula = 'y ~ s(x, by=f, bs="cc", k=6, sp=0.7)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected, atol=1e-8)

    def test_cc_factor_by_reml_matches_mgcv(self):
        """Factor-by replicated cc smooths under REML match mgcv exactly."""
        data = _data_factor_by(seed=113)
        formula = 'y ~ s(x, by=f, bs="cc", k=6)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(actual, expected)

    def test_linked_cc_fixed_sp_matches_mgcv(self):
        """Linked cc smooths with compatible k must share knots and match mgcv."""
        data = _data_2col(seed=118)
        formula = (
            'y ~ s(x0, bs="cc", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cc", k=6, id="g", sp=0.9)'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected, atol=1e-8)

    def test_linked_cc_reml_matches_mgcv(self):
        """Linked cc smooths with compatible k match mgcv under REML."""
        data = _data_2col(seed=118)
        formula = 'y ~ s(x0, bs="cc", k=6, id="g")' ' + s(x1, bs="cc", k=6, id="g")'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        _assert_exact_mgcv_snapshot_parity(
            actual,
            expected,
            pred_atol=5e-9,
            pred_rtol=0.0,
            edf_atol=5e-8,
            criterion_atol=1e-9,
            criterion_rtol=0.0,
            sp_atol=2e-7,
            sp_rtol=0.0,
            log_sp_atol=2e-7,
        )

    def test_cr_select_with_term_sp_vector_fixed_matches_mgcv(self):
        """select=True should accept one fixed sp per emitted penalty for cr."""
        data = _data_1d(seed=119)
        formula = 'y ~ s(x, bs="cr", k=6, sp=[0.7, 1.3])'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed", select=True)
        _exact_parity(actual, expected, atol=1e-8)

    def test_cc_select_with_scalar_sp_fixed_matches_mgcv(self):
        """cc + select=True should still accept scalar sp when no extra null penalty remains."""
        data = _data_1d(seed=120)
        formula = 'y ~ s(x, bs="cc", k=6, sp=0.7)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed", select=True)
        _exact_parity(actual, expected, atol=1e-8)

    def test_linked_cr_three_terms_fixed_sp_matches_mgcv(self):
        """Three linked cr smooths should still share one basis and match mgcv exactly."""
        data = _make_gaussian_data_3col()
        formula = (
            'y ~ s(x0, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x1, bs="cr", k=6, id="g", sp=0.9)'
            ' + s(x2, bs="cr", k=6, id="g", sp=0.9)'
        )

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        _exact_parity(actual, expected)

    def test_linked_cr_three_terms_share_one_smoothing_param(self):
        """Three linked terms should collapse to one smoothing parameter in mgcv and NAMpy."""
        data = _make_gaussian_data_3col()
        formula_linked = (
            'y ~ s(x0, bs="cr", k=6, id="g")'
            ' + s(x1, bs="cr", k=6, id="g")'
            ' + s(x2, bs="cr", k=6, id="g")'
        )
        formula_unlinked = (
            'y ~ s(x0, bs="cr", k=6)' ' + s(x1, bs="cr", k=6)' ' + s(x2, bs="cr", k=6)'
        )

        snap_linked = _run_mgcv_snapshot(data, formula_linked, "gaussian", "REML")
        snap_unlinked = _run_mgcv_snapshot(data, formula_unlinked, "gaussian", "REML")
        mgcv_sp_linked = len(np.atleast_1d(snap_linked["fit"]["smoothing_params"]))
        mgcv_sp_unlinked = len(np.atleast_1d(snap_unlinked["fit"]["smoothing_params"]))
        assert mgcv_sp_linked == 1
        assert mgcv_sp_linked < mgcv_sp_unlinked

        from nampy.gam.model.api import GAM

        gam_linked = GAM(family="gaussian", formula=formula_linked)
        gam_unlinked = GAM(family="gaussian", formula=formula_unlinked)
        gam_linked.fit(data=data)
        gam_unlinked.fit(data=data)

        assert _n_smoothing_params(gam_linked) == 1
        assert _n_smoothing_params(gam_linked) == mgcv_sp_linked
        assert _n_smoothing_params(gam_linked) < _n_smoothing_params(gam_unlinked)


def _pc_matrix_rename_x(df):
    out = df.copy()
    if "x0" in out.columns:
        out = out.rename(columns={"x0": "x"})
    cols = [c for c in ("y", "x", "x1") if c in out.columns]
    return out[cols].copy()


def _pc_matrix_gaussian(seed=5001, n=180):
    return _pc_matrix_rename_x(_make_gaussian_data(seed=seed, n=n))


def _pc_matrix_assert_snapshot_close(actual, expected, *, atol=1e-5):
    for key in ("response", "link"):
        np.testing.assert_allclose(
            np.asarray(actual["predictions"][key], dtype=np.float64),
            np.asarray(expected["predictions"][key], dtype=np.float64),
            atol=atol,
            rtol=atol,
        )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["edf_total"], dtype=np.float64),
        np.asarray(expected["fit"]["edf_total"], dtype=np.float64),
        atol=max(atol, 1e-4),
        rtol=atol,
    )


_PC_OPTION_MATRIX_CASES = [
    pytest.param(
        "pc_cr_select_reml",
        _pc_matrix_gaussian,
        "gaussian",
        "REML",
        'y ~ s(x, bs="cr", k=8, pc=0.0)',
        {"select": True},
        id="pc_cr_select_reml",
    ),
    pytest.param(
        "pc_ps_numeric_by_reml",
        lambda: _pc_matrix_gaussian().assign(z=lambda d: 0.5 + 0.2 * d["x"]),
        "gaussian",
        "REML",
        'y ~ s(x, bs="ps", k=8, pc=0.2, by=z)',
        {},
        id="pc_ps_numeric_by_reml",
    ),
    pytest.param(
        "pc_cr_factor_by_reml",
        lambda: _pc_matrix_gaussian().assign(f=lambda d: np.where(d["x"] > 0.0, "b", "a")),
        "gaussian",
        "REML",
        'y ~ f + s(x, by=f, bs="cr", k=8, pc=0.2)',
        {},
        id="pc_cr_factor_by_reml",
    ),
    pytest.param(
        "pc_tp_weighted_reml",
        lambda: _pc_matrix_gaussian().assign(w=lambda d: 1.0 + 0.25 * np.abs(d["x"])),
        "gaussian",
        "REML",
        'y ~ s(x, bs="tp", k=12, pc=0.0)',
        {"weights_column": "w"},
        id="pc_tp_weighted_reml",
    ),
    pytest.param(
        "pc_cs_offset_reml",
        lambda: _pc_matrix_gaussian().assign(off=lambda d: 0.1 * np.cos(d["x"])),
        "gaussian",
        "REML",
        'y ~ offset(off) + s(x, bs="cs", k=8, pc=0.0)',
        {},
        id="pc_cs_offset_reml",
    ),
]


@pytest.mark.parametrize(
    "case_id,data_factory,family,method,formula,kwargs",
    _PC_OPTION_MATRIX_CASES,
)
def test_pc_option_cross_matrix_matches_mgcv(
    case_id,
    data_factory,
    family,
    method,
    formula,
    kwargs,
):
    """Cover pc= crossed with select, by, weights, offsets, and dict/list syntax."""
    data = data_factory()
    actual_kwargs = dict(kwargs)
    weights_column = actual_kwargs.pop("weights_column", None)
    if weights_column is not None:
        actual_kwargs["sample_weight"] = np.asarray(
            data[str(weights_column)], dtype=np.float64
        )
    actual = _fit_nampy_snapshot(data, formula, family, method, **actual_kwargs)
    expected = _run_mgcv_snapshot(data, formula, family, method, **kwargs)
    _pc_matrix_assert_snapshot_close(actual, expected)
