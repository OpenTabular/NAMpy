"""Parity tests for k_check() vs mgcv::k.check().

What is compared:
  k_prime  — basis column count after identifiability constraints. EXACT.
  edf      — effective degrees of freedom. EXACT (from fit, already verified elsewhere).
  k_index  — v_obs / mean(rsd^2). Approximate with independent RNG path.
  p_value  — permutation-test p-value. Validity checks are probabilistic-range checks.

R script runs:  k.check(fit, subsample=120, n.rep=8) with set.seed(0).
NAMpy runs:     model.k_check(subsample=120, n_rep=8, seed=0).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _make_fs_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_random_effect_data_noisy,
    _run_mgcv_snapshot,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SUBSAMPLE = 120
_N_REP = 8
_SEED = 0
_K_INDEX_TOL_ATOL = 1.0 / np.sqrt(_N_REP)
_K_INDEX_TOL_RTOL = 0.5
_KCHECK_PGRID = 1.0 / _N_REP


def _compact_kcheck_label(label: str) -> str:
    """Normalize k_check labels to term-identity strings for comparison.

    NAMpy exposes full constructor args in term labels (e.g.
    ``s(x0, bs="cr", k=8)``) while mgcv snapshots emit condensed labels like
    ``s(x0)``.  Normalize both sides to the same identity form so alignment checks
    validate term order rather than formatting details.
    """

    text = str(label).strip()
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx < 0 or close_idx <= open_idx:
        return text
    fn = text[:open_idx].strip()
    inner = text[open_idx + 1 : close_idx]

    args: list[str] = []
    current = []
    depth = 0
    for ch in inner:
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                args.append(part)
            current = []
            continue
        current.append(ch)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    part = "".join(current).strip()
    if part:
        args.append(part)

    kept = []
    for part in args:
        if "=" in part:
            break
        kept.append(part)
    if not kept:
        kept = args[:1]
    if not kept:
        return f"{fn}()"
    return f"{fn}({','.join(kept)})"


def _assert_kcheck_p_value(
    value: float, *, n_rep: int, label: str, source: str
) -> None:
    assert np.isfinite(
        value
    ), f"{source} k_check p_value is non-finite for {label}: {value}"
    assert (
        0.0 <= value <= 1.0
    ), f"{source} k_check p_value out of range for {label}: {value}"
    scaled = value * n_rep
    nearest = np.rint(scaled)
    assert np.isclose(scaled, nearest, atol=1e-12), (
        f"{source} k_check p_value for {label} is not on mgcv grid "
        f"({_KCHECK_PGRID:g} increments): value={value}"
    )
    assert (
        0.0 <= nearest <= n_rep
    ), f"{source} k_check p_value for {label} maps to invalid grid index: value={value}, n_rep={n_rep}"


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
    r_block,
    py_labels,
    py_values,
    *,
    numeric_terms,
    edf_atol: float = 5e-6,
    k_index_atol: float = _K_INDEX_TOL_ATOL,
    k_index_rtol: float = _K_INDEX_TOL_RTOL,
):
    """Compare R and NAMpy k_check outputs.

    What is compared:
      k_prime  — EXACT: just the column count of the basis; deterministic.
      edf      — EXACT (atol=1e-6): from the fit, already verified in snapshot tests.
      k_index  — approximate for supported terms.
      p_value  — finite-range check for supported terms.

    numeric_terms: set of substrings; a term is "numeric" if its label contains
    any of them.  Factor-only terms (re, mrf) should yield NaN for k_index/p_value.
    """
    r_labels, r_values = r_block

    assert len(py_labels) == len(
        r_labels
    ), f"Term count mismatch: NAMpy={len(py_labels)} R={len(r_labels)}"
    assert [_compact_kcheck_label(x) for x in py_labels] == [
        _compact_kcheck_label(x) for x in r_labels
    ], (
        "Term labels diverged between NAMpy and mgcv k_check outputs.\n"
        f"NAMpy labels: {list(py_labels)}\n"
        f"R labels: {list(r_labels)}"
    )

    for i, (py_lbl, _r_lbl) in enumerate(zip(py_labels, r_labels)):
        # ---- k_prime (col 0) — exact ----------------------------------------
        py_kp = int(py_values[i, 0])
        r_kp = int(round(r_values[i, 0]))
        assert (
            py_kp == r_kp
        ), f"k_prime mismatch for term '{py_lbl}': NAMpy={py_kp} R={r_kp}"

        # ---- edf (col 1) — tight; tolerance varies by smooth type -----------
        np.testing.assert_allclose(
            py_values[i, 1],
            r_values[i, 1],
            atol=edf_atol,
            rtol=0.0,
            err_msg=f"edf mismatch for term '{py_lbl}'",
        )

        # ---- k_index and p_value -------------------------------------------
        is_numeric = any(s in py_lbl for s in numeric_terms)
        if is_numeric:
            np.testing.assert_allclose(
                py_values[i, 2],
                r_values[i, 2],
                atol=k_index_atol,
                rtol=k_index_rtol,
                err_msg=f"k_index mismatch for term '{py_lbl}'",
            )
            _assert_kcheck_p_value(
                float(py_values[i, 3]), n_rep=_N_REP, label=py_lbl, source="actual"
            )
            _assert_kcheck_p_value(
                float(r_values[i, 3]), n_rep=_N_REP, label=py_lbl, source="R"
            )
        else:
            assert np.isnan(
                py_values[i, 2]
            ), f"NAMpy k_index should be NaN for non-numeric term '{py_lbl}'"
            assert np.isnan(
                r_values[i, 2]
            ), f"R k_index should be NaN for non-numeric term '{py_lbl}'"
            assert np.isnan(
                py_values[i, 3]
            ), f"NAMpy p_value should be NaN for non-numeric term '{py_lbl}'"
            assert np.isnan(
                r_values[i, 3]
            ), f"R p_value should be NaN for non-numeric term '{py_lbl}'"


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


class TestKCheckParity:
    """Compare k_check() output against mgcv::k.check() for each smooth type."""

    @pytest.mark.parametrize(
        ("data_factory", "formula", "family", "method", "numeric_terms", "edf_atol"),
        [
            (
                lambda: _make_gaussian_data(seed=123, n=180),
                'y ~ s(x0, bs="cr", k=8)',
                "gaussian",
                "REML",
                {"x0"},
                5e-6,
            ),
            (
                lambda: _make_gaussian_data(seed=500, n=180),
                'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)',
                "gaussian",
                "fixed",
                {"x0", "x1"},
                1e-4,
            ),
            (
                lambda: _make_gaussian_data(seed=600, n=180),
                'y ~ s(x0, bs="ps", k=8) + s(x1, bs="ps", k=8)',
                "gaussian",
                "REML",
                {"x0", "x1"},
                1e-4,
            ),
            (
                lambda: _make_gaussian_data(seed=123, n=180),
                'y ~ te(x0, x1, bs=["cr","cr"], k=[5,5])',
                "gaussian",
                "REML",
                {"x0", "x1"},
                1e-4,
            ),
        ],
        ids=[
            "gaussian_cr",
            "gaussian_cr_fixed",
            "gaussian_ps",
            "gaussian_te",
        ],
    )
    def test_numeric_k_check_parity_representative_cases(
        self, data_factory, formula, family, method, numeric_terms, edf_atol
    ):
        """
        Verify that numeric-term k-check diagnostics match mgcv on the representative
        case matrix in this file.
        """
        data = data_factory()
        snap = _run_mgcv_snapshot(data, formula, family, method)
        model = _fit_nampy_model(data, formula, family, method)

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        _assert_k_check_parity(
            r_block,
            py_labels,
            py_values,
            numeric_terms=numeric_terms,
            edf_atol=edf_atol,
        )

    @pytest.mark.parametrize(
        ("data_factory", "formula", "edf_atol"),
        [
            (_make_random_effect_data_noisy, 'y ~ s(f, bs="re")', 1e-6),
            pytest.param(
                _make_fs_data,
                'y ~ s(f, x, bs="fs", k=6)',
                5e-5,
            ),
            (
                _make_mrf_data,
                'y ~ s(region, bs="mrf", k=3, xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))',
                1e-6,
            ),
        ],
        ids=["re", "fs", "mrf"],
    )
    def test_factor_like_terms_have_nan_k_diagnostics(
        self, data_factory, formula, edf_atol
    ):
        """
        Verify that factor-like smooths surface NaN k-index diagnostics, matching the
        mgcv convention for nonnumeric terms.
        """
        data = data_factory()
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")

        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 1
        assert int(py_values[0, 0]) == int(round(r_values[0, 0]))
        np.testing.assert_allclose(
            py_values[0, 1], r_values[0, 1], atol=edf_atol, rtol=0.0
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
        from tests.mgcv_parity_utils import _make_random_effect_data_noisy

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
            r_block,
            py_labels,
            py_values,
            numeric_terms={"x0"},
            edf_atol=1e-5,
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
            assert int(py_values[i, 0]) == int(
                round(r_values[i, 0])
            ), f"k_prime mismatch under select=True for term {i}"
            np.testing.assert_allclose(
                py_values[i, 1],
                r_values[i, 1],
                atol=1e-6,
                rtol=0.0,
                err_msg=f"edf mismatch under select=True for term {i}",
            )
