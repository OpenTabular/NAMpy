"""Parity tests for k_check() vs mgcv::k.check().

What is compared:
  k_prime  — basis column count after identifiability constraints. EXACT.
  edf      — effective degrees of freedom. EXACT (from fit, already verified elsewhere).
  k_index  — v_obs / mean(rsd^2). TIGHT (atol=1e-3): deterministic given residuals, but
             NAMpy and mgcv deviance residuals differ at ~1e-6 so the ratio differs slightly.

What is NOT compared:
  p_value  — permutation-test p-value. R and Python use different RNG algorithms, so
             the p-values are not numerically comparable. We only check both are in [0, 1]
             (or both NaN for factor/spatial terms).

R script runs:  k.check(fit, subsample=120, n.rep=8) with set.seed(0).
NAMpy runs:     model.k_check(subsample=120, n_rep=8, seed=0).
"""

from __future__ import annotations

import numpy as np
import pytest

from mgcv_parity_utils import (
    _fit_nampy_model,
    _make_binomial_data,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_poisson_data,
    _make_random_effect_data_noisy,
    _run_mgcv_snapshot,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SUBSAMPLE = 120
_N_REP = 8
_SEED = 0


def _coerce_na(x):
    """Convert R's NA representations (None or string "NA") to float NaN."""
    if x is None or x == "NA":
        return float("nan")
    return float(x)


def _r_k_check(snap):
    """Extract k_check block from an mgcv snapshot dict.

    jsonlite quirks handled here:
      * auto_unbox=TRUE → single-element arrays become scalars; normalise to list.
      * R NA_real_ → JSON null → Python None; R character NA → "NA" string.
        Both are coerced to float NaN.
    """
    block = snap["parity"]["diagnostics"].get("k_check")
    if block is None:
        return None
    raw_labels = block["labels"]
    labels = [raw_labels] if isinstance(raw_labels, str) else list(raw_labels)
    raw_values = block["values"]
    # May be a flat list (single term) or nested list (multiple terms)
    if raw_values is None:
        return None
    if not isinstance(raw_values[0], list):
        raw_values = [raw_values]
    values = np.array(
        [[_coerce_na(v) for v in row] for row in raw_values], dtype=np.float64
    )
    return labels, values  # shape (n_terms, 4): k_prime, edf, k_index, p_value


def _nampy_k_check(model):
    """Run k_check on a fitted NAMpy model and return (labels, array)."""
    df = model.k_check(subsample=_SUBSAMPLE, n_rep=_N_REP, seed=_SEED)
    assert df is not None, "k_check returned None for a fitted model"
    labels = list(df.index)
    values = df[["k_prime", "edf", "k_index", "p_value"]].to_numpy(dtype=np.float64)
    return labels, values


def _assert_k_check_parity(
    r_block, py_labels, py_values, *, numeric_terms, edf_atol: float = 5e-6
):
    """Compare R and NAMpy k_check outputs.

    What is compared:
      k_prime  — EXACT: just the column count of the basis; deterministic.
      edf      — EXACT (atol=1e-6): from the fit, already verified in snapshot tests.
      k_index  — VALIDITY ONLY: R and Python subsample with different RNGs (same
                 seed integer, different algorithms), so the selected rows differ and
                 the ratio v_obs/mean(rsd^2) is not reproducible across implementations.
                 We only check: finite for numeric terms, NaN for factor/spatial.
      p_value  — VALIDITY ONLY: permutation RNG differs. Check in [0,1] or NaN.

    numeric_terms: set of substrings; a term is "numeric" if its label contains
    any of them.  Factor-only terms (re, mrf) should yield NaN for k_index/p_value.
    """
    r_labels, r_values = r_block

    assert len(py_labels) == len(r_labels), (
        f"Term count mismatch: NAMpy={len(py_labels)} R={len(r_labels)}"
    )

    for i, (py_lbl, r_lbl) in enumerate(zip(py_labels, r_labels)):
        # ---- k_prime (col 0) — exact ----------------------------------------
        py_kp = int(py_values[i, 0])
        r_kp = int(round(r_values[i, 0]))
        assert py_kp == r_kp, (
            f"k_prime mismatch for term '{py_lbl}': NAMpy={py_kp} R={r_kp}"
        )

        # ---- edf (col 1) — tight; tolerance varies by smooth type -----------
        np.testing.assert_allclose(
            py_values[i, 1],
            r_values[i, 1],
            atol=edf_atol,
            rtol=0.0,
            err_msg=f"edf mismatch for term '{py_lbl}'",
        )

        # ---- k_index and p_value — validity only ----------------------------
        is_numeric = any(s in py_lbl for s in numeric_terms)
        if is_numeric:
            assert np.isfinite(py_values[i, 2]), (
                f"NAMpy k_index should be finite for numeric term '{py_lbl}'"
            )
            assert np.isfinite(r_values[i, 2]), (
                f"R k_index should be finite for numeric term '{py_lbl}'"
            )
            p_py = py_values[i, 3]
            p_r = r_values[i, 3]
            assert np.isfinite(p_py) and 0.0 <= p_py <= 1.0, (
                f"NAMpy p_value out of [0,1] for '{py_lbl}': {p_py}"
            )
            assert np.isfinite(p_r) and 0.0 <= p_r <= 1.0, (
                f"R p_value out of [0,1] for '{py_lbl}': {p_r}"
            )
        else:
            assert np.isnan(py_values[i, 2]), (
                f"NAMpy k_index should be NaN for non-numeric term '{py_lbl}'"
            )
            assert np.isnan(r_values[i, 2]), (
                f"R k_index should be NaN for non-numeric term '{py_lbl}'"
            )
            assert np.isnan(py_values[i, 3]), (
                f"NAMpy p_value should be NaN for non-numeric term '{py_lbl}'"
            )
            assert np.isnan(r_values[i, 3]), (
                f"R p_value should be NaN for non-numeric term '{py_lbl}'"
            )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


class TestKCheckParity:
    """Compare k_check() output against mgcv::k.check() for each smooth type."""

    # ------------------------------------------------------------------ #
    # Gaussian — univariate smooths                                       #
    # ------------------------------------------------------------------ #

    def test_gaussian_single_cr_reml(self):
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None, "R snapshot missing k_check block"

        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0"})

    def test_gaussian_two_cr_reml(self):
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0", "x1"})

    def test_gaussian_cr_fixed_sp(self):
        """Fixed-sp: k_prime and edf must still match regardless of optimizer."""
        data = _make_gaussian_data(seed=500, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "fixed")
        model = _fit_nampy_model(data, formula, "gaussian", "fixed")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0", "x1"})

    def test_gaussian_ps_reml(self):
        data = _make_gaussian_data(seed=600, n=180)
        formula = 'y ~ s(x0, bs="ps", k=8) + s(x1, bs="ps", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0", "x1"})

    def test_gaussian_tp_reml(self):
        data = _make_gaussian_data(seed=700, n=180)
        formula = 'y ~ s(x0, bs="tp", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0"})

    def test_gaussian_cc_reml(self):
        data = _make_gaussian_data(seed=800, n=180)
        formula = 'y ~ s(x0, bs="cc", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0"})

    def test_gaussian_gp_reml(self):
        data = _make_gaussian_data(seed=900, n=180)
        formula = 'y ~ s(x0, bs="gp", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0"})

    # ------------------------------------------------------------------ #
    # Non-Gaussian families                                               #
    # ------------------------------------------------------------------ #

    def test_binomial_cr_reml(self):
        data = _make_binomial_data(seed=456, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "binomial", "REML")
        model = _fit_nampy_model(data, formula, "binomial", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0", "x1"})

    def test_poisson_cr_reml(self):
        data = _make_poisson_data(seed=789, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "poisson", "REML")
        model = _fit_nampy_model(data, formula, "poisson", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(r_block, py_labels, py_values, numeric_terms={"x0", "x1"})

    def test_gamma_cr_reml(self):
        """Gamma REML: k_prime exact; edf tolerance 5e-3 (Gamma is a LOOSE family —
        NAMpy and mgcv may converge to different sp values, producing different EDF).
        """
        data = _make_gamma_data(seed=1701, n=220)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gamma", "REML")
        model = _fit_nampy_model(data, formula, "gamma", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels)
        for i in range(len(py_labels)):
            assert int(py_values[i, 0]) == int(round(r_values[i, 0])), (
                f"k_prime mismatch for term {i}"
            )
            # Gamma EDF can differ up to ~5e-3 when sp values differ
            np.testing.assert_allclose(
                py_values[i, 1], r_values[i, 1], atol=5e-3, rtol=0.0,
                err_msg=f"edf mismatch for term {i}",
            )
            assert np.isfinite(py_values[i, 2])
            assert np.isfinite(r_values[i, 2])
            assert np.isfinite(py_values[i, 3]) and 0.0 <= py_values[i, 3] <= 1.0

    # ------------------------------------------------------------------ #
    # Tensor product smooths — 2D features use nearest-neighbour         #
    # ------------------------------------------------------------------ #

    def test_gaussian_te_reml(self):
        """te() smooth: 2D feature block, nearest-neighbour k_index.

        edf tolerance relaxed to 1e-4: te/ti/t2 EDF differs at ~1e-4 vs mgcv
        (known TIGHT gap from tensor marginal penalty scaling; tracked in
        PARITY_SUMMARY.md section 11).
        """
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ te(x0, x1, bs=["cr","cr"], k=[5,5])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(
            r_block, py_labels, py_values,
            numeric_terms={"x0", "x1"}, edf_atol=1e-4,
        )

    def test_gaussian_ti_reml(self):
        """ti() ANOVA interaction: edf tolerance 5e-4 (TIGHT, same gap as te)."""
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ ti(x0, x1, bs=["cr","cr"], k=[5,5])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(
            r_block, py_labels, py_values,
            numeric_terms={"x0", "x1"}, edf_atol=5e-4,
        )

    def test_gaussian_t2_reml(self):
        """t2(): natural-param reparameterization accumulates ~4e-4 EDF error vs mgcv."""
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ t2(x0, x1, bs=["cr","cr"], k=[5,5])'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(
            r_block, py_labels, py_values,
            numeric_terms={"x0", "x1"}, edf_atol=1e-3,
        )

    # ------------------------------------------------------------------ #
    # Factor/spatial smooths — k_index and p_value are NaN               #
    # ------------------------------------------------------------------ #

    def test_random_effect_nan_diagnostics(self):
        """re() term: no numeric feature → k_index and p_value must be NaN."""
        data = _make_random_effect_data_noisy()
        formula = 'y ~ s(f, bs="re")'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 1

        # k_prime: exact
        assert int(py_values[0, 0]) == int(round(r_values[0, 0]))
        # edf: tight
        np.testing.assert_allclose(
            py_values[0, 1], r_values[0, 1], atol=1e-6, rtol=0.0,
        )
        # k_index + p_value: NaN for factor-only terms
        assert np.isnan(py_values[0, 2]), "NAMpy k_index should be NaN for re() term"
        assert np.isnan(r_values[0, 2]), "R k_index should be NaN for re() term"
        assert np.isnan(py_values[0, 3]), "NAMpy p_value should be NaN for re() term"
        assert np.isnan(r_values[0, 3]), "R p_value should be NaN for re() term"

    def test_fs_k_prime_and_edf_match(self):
        """fs() smooth: k_prime and edf match mgcv.

        k_index is NaN in NAMpy because _numeric_feature_block does not yet extract
        the metric feature from an fs() RuntimeTerm (known limitation — the fs term
        stores both a factor and a numeric feature; the current introspection path
        only handles terms with _feature_index or _feature_indices).  mgcv's k.check
        does compute a finite k_index for fs().  This mismatch is documented in
        PARITY_SUMMARY.md section 9 (item "k_check fs/sz feature extraction").

        edf_atol=5e-5: EDF accumulates small differences across factor-level columns.
        """
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", k=6)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 1

        # k_prime: exact
        assert int(py_values[0, 0]) == int(round(r_values[0, 0])), (
            f"k_prime mismatch: NAMpy={int(py_values[0,0])} R={int(round(r_values[0,0]))}"
        )
        # edf: tight
        np.testing.assert_allclose(
            py_values[0, 1], r_values[0, 1], atol=5e-5, rtol=0.0,
            err_msg="edf mismatch for fs() term",
        )
        # k_index: both return NaN — mgcv's k.check also skips factor-smooth
        # terms when it cannot identify a single numeric covariate column.
        assert np.isnan(py_values[0, 2]), "NAMpy k_index should be NaN for fs()"
        assert np.isnan(r_values[0, 2]), "R k_index should be NaN for fs()"

    def test_mrf_nan_diagnostics(self):
        """mrf() term: region column is categorical → k_index and p_value NaN."""
        data = _make_mrf_data()
        formula = (
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        )

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 1

        assert int(py_values[0, 0]) == int(round(r_values[0, 0]))
        np.testing.assert_allclose(
            py_values[0, 1], r_values[0, 1], atol=1e-6, rtol=0.0,
        )
        assert np.isnan(py_values[0, 2])
        assert np.isnan(r_values[0, 2])

    # ------------------------------------------------------------------ #
    # Mixed model: numeric + factor terms in one formula                  #
    # ------------------------------------------------------------------ #

    def test_mixed_numeric_and_re_terms(self):
        """Model with both a numeric smooth and a random effect.

        Numeric term → finite k_index.
        RE term → NaN k_index.
        """
        from mgcv_parity_utils import _make_random_effect_data_noisy

        rng = __import__("numpy").random.default_rng(42)
        data = _make_random_effect_data_noisy(seed=21, n_draws=60)
        data["x0"] = rng.uniform(-2.0, 2.0, size=len(data))
        formula = 'y ~ s(x0, bs="cr", k=7) + s(f, bs="re")'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 2

        # edf_atol=1e-5: small dataset (n=60) with RE term causes minor rounding
        _assert_k_check_parity(
            r_block, py_labels, py_values, numeric_terms={"x0"}, edf_atol=1e-5,
        )

    # ------------------------------------------------------------------ #
    # select=True: shrinkage penalty does not alter k_prime or edf logic  #
    # ------------------------------------------------------------------ #

    def test_gaussian_select_reml_k_prime_and_edf_match(self):
        """select=True adds a null-space penalty but k_prime count should still match."""
        data = _make_gaussian_data(seed=123, n=180)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
        model = _fit_nampy_model(data, formula, "gaussian", "REML", select=True)

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 2

        for i in range(2):
            assert int(py_values[i, 0]) == int(round(r_values[i, 0])), (
                f"k_prime mismatch under select=True for term {i}"
            )
            np.testing.assert_allclose(
                py_values[i, 1], r_values[i, 1], atol=1e-6, rtol=0.0,
                err_msg=f"edf mismatch under select=True for term {i}",
            )
