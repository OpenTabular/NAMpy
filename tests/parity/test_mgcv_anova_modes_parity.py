"""Multi-model anova test-mode parity (F/LRT/None), dispersion and guards.

Upstream references: mgcv/R/mgcv.r::anova.gam (multi-model branch delegates to
stats::anova.glmlist / stats::stat.anova), mirrored by
nampy/gam/inference/anova.py::_comparison_table.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import (
    _build_r_command,
    _df_fixture_repr,
    _family_specs,
    _fit_nampy_model,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _mgcv_fixture_key,
    _mgcv_fixture_load,
    _mgcv_fixture_save,
    _run_mgcv_anova,
)

pytestmark = [pytest.mark.surface_output]

_ANOVA_OPTIONS_FIXTURE_VERSION = 1

_GAUSSIAN_FORMULAS = [
    'y ~ s(x0, bs="cr", k=8)',
    'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
]
# Note: a parametric-only first model (e.g. "y ~ x1") is not usable here —
# nampy's REML driver raises NotImplementedError for smooth-free formulas
# (initial.spg has no penalty to initialize), so all compared models carry a
# smooth.
_GAUSSIAN_FORMULAS_3 = [
    'y ~ s(x0, bs="cr", k=5)',
    'y ~ x1 + s(x0, bs="cr", k=5)',
    'y ~ s(x0, bs="cr", k=5) + s(x1, bs="cr", k=8)',
]
_POISSON_FORMULAS = [
    'y ~ s(x0, bs="cr", k=8)',
    'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
]


def _run_mgcv_anova_options(
    data,
    formulas,
    family,
    method: str,
    *,
    test: str | None,
    dispersion: float | None = None,
    freq: bool = False,
):
    """anova.gam reference with dispersion= / freq= (not in mgcv_anova.R)."""
    _family_nampy, family_token = _family_specs(family)
    formula_texts = [str(formula) for formula in list(formulas)]
    cache_key = _mgcv_fixture_key(
        "anova_options",
        {
            "version": _ANOVA_OPTIONS_FIXTURE_VERSION,
            "data": _df_fixture_repr(data),
            "formulas": formula_texts,
            "family_token": family_token,
            "method": method,
            "test": test,
            "dispersion": dispersion,
            "freq": freq,
        },
    )
    cached = _mgcv_fixture_load(cache_key)
    if cached is not None:
        return cached

    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
out <- args[[2]]
formula_texts <- fromJSON(args[[3]])
family_name <- tolower(args[[4]])
method_name <- args[[5]]
test_name <- args[[6]]
dispersion_arg <- args[[7]]
freq_flag <- tolower(args[[8]]) %in% c("true", "1", "yes")
family_obj <- switch(
  family_name,
  gaussian = gaussian(),
  binomial = binomial(),
  poisson = poisson(),
  gamma = Gamma(link = "log"),
  stop(sprintf("Unsupported family: %s", family_name))
)
fits <- lapply(formula_texts, function(txt) {
  gam(as.formula(txt), data = d, family = family_obj, method = method_name)
})
disp <- if (dispersion_arg == "NULL") NULL else as.numeric(dispersion_arg)
if (length(fits) == 1) {
  a <- anova(fits[[1]], dispersion = disp, freq = freq_flag)
  payload <- list(
    smooth = if (is.null(a$s.table)) NULL else list(
      labels = unname(as.character(rownames(a$s.table))),
      values = unname(as.matrix(a$s.table))
    ),
    parametric = if (is.null(a$pTerms.table)) NULL else list(
      labels = unname(as.character(rownames(a$pTerms.table))),
      values = unname(as.matrix(a$pTerms.table))
    )
  )
} else {
  test_arg <- if (test_name == "NULL") NULL else test_name
  a <- do.call(
    anova,
    c(list(object = fits[[1]]), fits[-1], list(test = test_arg, dispersion = disp))
  )
  payload <- list(
    table = list(columns = colnames(a), values = unname(as.matrix(a)))
  )
}
write_json(payload, out, auto_unbox = TRUE, digits = 17, null = "null")
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "anova_options.json"
        script_path = tmpdir_path / "anova_options.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(r_code, encoding="utf-8")
        subprocess.run(
            _build_r_command(
                script_path,
                str(csv_path),
                str(json_path),
                json.dumps(formula_texts),
                family_token,
                method,
                "NULL" if test is None else str(test),
                "NULL" if dispersion is None else repr(float(dispersion)),
                "true" if freq else "false",
            ),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))

    _mgcv_fixture_save(cache_key, result)
    return result


def _normalized_smooth_label(label: str) -> str:
    text = str(label)
    if "(" not in text:
        return text
    head, args = text.split("(", 1)
    first_arg = args.rstrip(")").split(",")[0].strip()
    return f"{head}({first_arg})"


def _expected_matrix(values) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 1:
        arr = arr[:, None]

    def _coerce(item):
        if item is None or item == "NA" or item == "NaN":
            return np.nan
        return float(item)

    return np.vectorize(_coerce, otypes=[np.float64])(arr)


def _assert_comparison_table_matches(actual, expected_payload, *, p_rtol=1e-8):
    expected_columns = [str(c) for c in expected_payload["table"]["columns"]]
    expected = _expected_matrix(expected_payload["table"]["values"])
    assert list(actual.table.columns) == expected_columns
    actual_values = actual.table.to_numpy(dtype=np.float64)
    assert actual_values.shape == expected.shape
    for j, column in enumerate(expected_columns):
        if column in {"Pr(>Chi)", "Pr(>F)"}:
            np.testing.assert_allclose(
                actual_values[:, j],
                expected[:, j],
                rtol=p_rtol,
                atol=1e-12,
                equal_nan=True,
            )
        elif column == "F":
            np.testing.assert_allclose(
                actual_values[:, j],
                expected[:, j],
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            )
        elif column in {"Resid. Df", "Df"}:
            np.testing.assert_allclose(
                actual_values[:, j],
                expected[:, j],
                atol=5e-6,
                rtol=1e-8,
                equal_nan=True,
            )
        else:
            np.testing.assert_allclose(
                actual_values[:, j],
                expected[:, j],
                atol=1e-9,
                rtol=1e-9,
                equal_nan=True,
            )


def _fit_models(data, formulas, family, method="REML"):
    return [_fit_nampy_model(data, formula, family, method) for formula in formulas]


def test_anova_gaussian_pair_f_test_matches_mgcv():
    """gaussian nested pair with test="F" reproduces mgcv's F table."""
    data = _make_gaussian_data()
    models = _fit_models(data, _GAUSSIAN_FORMULAS, "gaussian")
    actual = models[0].anova(models[1], test="F")
    expected = _run_mgcv_anova(data, _GAUSSIAN_FORMULAS, "gaussian", "REML", test="F")
    assert actual.test == "F"
    _assert_comparison_table_matches(actual, expected)


def test_anova_gaussian_three_model_f_test_matches_mgcv():
    """Three nested gaussian models with test="F" match mgcv row by row."""
    data = _make_gaussian_data()
    models = _fit_models(data, _GAUSSIAN_FORMULAS_3, "gaussian")
    actual = models[0].anova(models[1], models[2], test="F")
    expected = _run_mgcv_anova(data, _GAUSSIAN_FORMULAS_3, "gaussian", "REML", test="F")
    _assert_comparison_table_matches(actual, expected)


def test_anova_gaussian_pair_default_test_matches_mgcv():
    """test=None defaults to F for estimated-dispersion families (R >= 4.4)."""
    data = _make_gaussian_data()
    models = _fit_models(data, _GAUSSIAN_FORMULAS, "gaussian")
    actual = models[0].anova(models[1], test=None)
    expected = _run_mgcv_anova(data, _GAUSSIAN_FORMULAS, "gaussian", "REML", test=None)
    assert actual.test == "F"
    _assert_comparison_table_matches(actual, expected)


def test_anova_poisson_pair_default_test_matches_mgcv():
    """test=None defaults to Chisq for known-dispersion families (R >= 4.4)."""
    data = _make_poisson_data()
    models = _fit_models(data, _POISSON_FORMULAS, "poisson")
    actual = models[0].anova(models[1], test=None)
    expected = _run_mgcv_anova(data, _POISSON_FORMULAS, "poisson", "REML", test=None)
    assert actual.test == "CHISQ"
    _assert_comparison_table_matches(actual, expected)


def test_anova_negbin_pair_default_no_test_matches_mgcv():
    """Extended families get no default test columns, matching mgcv/R."""
    data = _make_negbin_data()
    family = {"name": "negbin", "theta": 1.6}
    models = _fit_models(data, _POISSON_FORMULAS, family)
    actual = models[0].anova(models[1], test=None)
    expected = _run_mgcv_anova(data, _POISSON_FORMULAS, family, "REML", test=None)
    assert actual.test is None
    assert list(actual.table.columns) == [
        "Resid. Df",
        "Resid. Dev",
        "Df",
        "Deviance",
    ]
    _assert_comparison_table_matches(actual, expected)


def test_anova_poisson_pair_lrt_matches_mgcv_and_chisq_alias():
    """test="LRT" matches mgcv and is the documented alias of test="Chisq"."""
    data = _make_poisson_data()
    models = _fit_models(data, _POISSON_FORMULAS, "poisson")
    actual_lrt = models[0].anova(models[1], test="LRT")
    expected_lrt = _run_mgcv_anova(
        data, _POISSON_FORMULAS, "poisson", "REML", test="LRT"
    )
    _assert_comparison_table_matches(actual_lrt, expected_lrt)

    actual_chisq = models[0].anova(models[1], test="Chisq")
    np.testing.assert_allclose(
        actual_lrt.table.to_numpy(dtype=np.float64),
        actual_chisq.table.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_anova_poisson_pair_f_test_matches_mgcv():
    """Known-scale families still honor an explicit test="F" like mgcv."""
    data = _make_poisson_data()
    models = _fit_models(data, _POISSON_FORMULAS, "poisson")
    actual = models[0].anova(models[1], test="F")
    expected = _run_mgcv_anova(data, _POISSON_FORMULAS, "poisson", "REML", test="F")
    _assert_comparison_table_matches(actual, expected)


def test_anova_dispersion_override_matches_mgcv():
    """An explicit dispersion= rescales the Chisq comparison as in mgcv."""
    data = _make_gaussian_data()
    models = _fit_models(data, _GAUSSIAN_FORMULAS, "gaussian")
    actual = models[0].anova(models[1], test="Chisq", dispersion=0.05)
    expected = _run_mgcv_anova_options(
        data,
        _GAUSSIAN_FORMULAS,
        "gaussian",
        "REML",
        test="Chisq",
        dispersion=0.05,
    )
    _assert_comparison_table_matches(actual, expected)


def test_anova_single_model_freq_matches_mgcv():
    """anova(freq=True) uses the frequentist covariance like mgcv."""
    data = _make_gaussian_data()
    model = _fit_nampy_model(data, _GAUSSIAN_FORMULAS[1], "gaussian", "REML")
    actual = model.anova(freq=True)
    expected = _run_mgcv_anova_options(
        data,
        [_GAUSSIAN_FORMULAS[1]],
        "gaussian",
        "REML",
        test=None,
        freq=True,
    )
    expected_smooth = _expected_matrix(expected["smooth"]["values"])
    actual_smooth = actual.smooth_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=np.float64)
    # mgcv prints smooth labels without basis/k arguments; nampy keeps the
    # full term text, so compare on the variable-only normalization.
    assert [
        _normalized_smooth_label(label) for label in expected["smooth"]["labels"]
    ] == [_normalized_smooth_label(label) for label in actual.smooth_table["label"]]
    np.testing.assert_allclose(
        actual_smooth, expected_smooth, rtol=1e-6, atol=1e-8, equal_nan=True
    )


def test_anova_comparison_guards_reject_incompatible_models():
    """Family, sample-size, method and test-name guards raise explicitly."""
    data = _make_gaussian_data()
    poisson_data = _make_poisson_data()
    gaussian_model = _fit_nampy_model(data, _GAUSSIAN_FORMULAS[0], "gaussian", "REML")
    gaussian_model_full = _fit_nampy_model(
        data, _GAUSSIAN_FORMULAS[1], "gaussian", "REML"
    )
    poisson_model = _fit_nampy_model(
        poisson_data, _POISSON_FORMULAS[0], "poisson", "REML"
    )
    subset_model = _fit_nampy_model(
        data.iloc[:120].reset_index(drop=True),
        _GAUSSIAN_FORMULAS[0],
        "gaussian",
        "REML",
    )
    gcv_model = _fit_nampy_model(data, _GAUSSIAN_FORMULAS[0], "gaussian", "GCV")

    with pytest.raises(ValueError, match="same family"):
        gaussian_model.anova(poisson_model, test="Chisq")
    with pytest.raises(ValueError, match="same sample size"):
        gaussian_model.anova(subset_model, test="Chisq")
    with pytest.raises(ValueError, match="same smoothing selection method"):
        gaussian_model.anova(gcv_model, test="Chisq")
    with pytest.raises(ValueError, match="test must be one of"):
        gaussian_model.anova(gaussian_model_full, test="wald")


def test_anova_single_model_dispersion_override_matches_mgcv():
    """anova(dispersion=) on a single model rescales like summary.gam.

    The multi-model dispersion override was already covered; the single-model
    path (anova.gam == summary.gam, mgcv/R/mgcv.r:4153) was not.
    """
    data = _make_gaussian_data()
    model = _fit_nampy_model(data, _GAUSSIAN_FORMULAS[1], "gaussian", "REML")
    actual = model.anova(dispersion=2.5)
    expected = _run_mgcv_anova_options(
        data,
        [_GAUSSIAN_FORMULAS[1]],
        "gaussian",
        "REML",
        test=None,
        dispersion=2.5,
    )
    expected_smooth = _expected_matrix(expected["smooth"]["values"])
    actual_smooth = actual.smooth_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=np.float64)
    assert [
        _normalized_smooth_label(label) for label in expected["smooth"]["labels"]
    ] == [_normalized_smooth_label(label) for label in actual.smooth_table["label"]]
    np.testing.assert_allclose(
        actual_smooth, expected_smooth, rtol=1e-6, atol=1e-8, equal_nan=True
    )
    # The override must genuinely differ from the estimated-scale table.
    baseline = model.anova().smooth_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=np.float64)
    assert not np.allclose(actual_smooth, baseline, rtol=1e-10, atol=1e-12)
