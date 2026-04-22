"""Strict xfail registry for remaining concrete mgcv parity gaps.

This file intentionally tracks only genuine failing or explicitly unsupported
surfaces. Do not add xfails for behavior that is already green.

Other remaining gap buckets already live elsewhere:
- post-fit / final-fit gaps: ``tests/optimization/test_mgcv_postprocessing_final_fit_parity.py``
- raw constructor gaps: ``tests/smooths/test_mgcv_raw_constructor_parity.py``
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nampy.gam import GAM
from nampy.gam.fit.solve_ops import solve_pirls_given_smoothing
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.smoothing_selection.criteria.pirls_deriv import _gdi2_joint_kernel
from nampy.gam.specs.build import build_formula_model
from tests._paths import REPO_ROOT
from tests.families.test_general_family_mgcv_parity import GAULSS_FORMULA, _gaulss_data
from tests.mgcv_parity_utils import (
    _fit_nampy_model_fixed_sp,
    _make_gaussian_data,
    _make_random_effect_data,
)

R_SCRIPT = shutil.which("Rscript")


def _build_formula_only(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def _run_mgcv_random_effect_id_error(data: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "out.json"
        script_path = tmpdir_path / "re_id_error.R"
        data.to_csv(csv_path, index=False)
        script_path.write_text(
            """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
payload <- tryCatch(
  {
    gam(y ~ s(f, bs="re", id="shared"), data = d, method = "REML")
    list(ok = TRUE, message = NULL)
  },
  error = function(e) list(ok = FALSE, message = conditionMessage(e))
)
write_json(payload, args[[2]], auto_unbox = TRUE, digits = 17)
""".strip(),
            encoding="utf-8",
        )
        subprocess.run(
            [R_SCRIPT, str(script_path), str(csv_path), str(json_path)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def test_formula_dot_shorthand_builds_with_data_context():
    """
    Known-gap coverage verifying that formula dot shorthand builds with data context.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
            "z": [2.0, 1.0, 0.0],
        }
    )

    built = _build_formula_only("y ~ .", data)

    assert built.feature_names == ["x", "z"]
    assert [term.label for term in built.predictor_specs[0].terms] == ["x", "z"]


def test_formula_mixed_positional_and_keyword_list_args_build():
    """
    Known-gap coverage verifying that formula mixed positional and keyword list args
    build.
    """
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

    built = _build_formula_only('y ~ s(x, bs="tp", xt=list(1, bs="ps"))', data)

    smooth = built.predictor_specs[0].terms[0]
    assert smooth.kind == "smooth"
    assert smooth.smooth_spec is not None
    assert smooth.smooth_spec.xt == {0: 1, "bs": "ps"}


def test_formula_transformed_smooth_covariate_builds_end_to_end():
    """
    Known-gap coverage verifying that formula transformed smooth covariate builds end to
    end.
    """
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})

    built = _build_formula_only('y ~ s(I(x**2), k=5, bs="cr")', data)

    term = built.predictor_specs[0].terms[0]
    assert term.features[0] in built.working_data.columns


def test_formula_multi_predictor_distinct_offsets_build():
    """
    Known-gap coverage verifying that formula multi predictor distinct offsets build.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [0.5, 1.5, 2.5, 3.5],
            "o1": [0.1, 0.2, 0.3, 0.4],
            "o2": [0.4, 0.3, 0.2, 0.1],
        }
    )

    built = _build_formula_only(
        [
            'y ~ s(x, bs="cr", k=5) + offset(o1)',
            '~ s(z, bs="cr", k=5) + offset(o2)',
        ],
        data,
    )

    assert [pred.offset_name for pred in built.predictor_specs] == ["o1", "o2"]
    assert built.preprocess_state["offset_names"] == ("o1", "o2")
    assert isinstance(built.offsets, list)
    assert_allclose(built.offsets[0], data["o1"].to_numpy(dtype=np.float64))
    assert_allclose(built.offsets[1], data["o2"].to_numpy(dtype=np.float64))


def test_general_family_formula_multi_predictor_offsets_predict_with_defaults():
    """
    Known-gap coverage verifying that general family formula multi predictor offsets
    predict with defaults.
    """
    rng = np.random.default_rng(12)
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    o1 = np.linspace(-0.25, 0.25, n)
    o2 = np.linspace(0.2, -0.2, n)
    mu = 0.4 + 0.3 * x + o1
    sigma = np.exp(-0.25 + o2)
    y = rng.normal(mu, sigma, size=n)
    data = pd.DataFrame({"y": y, "x": x, "o1": o1, "o2": o2})

    gam = GAM(
        family="gaulss",
        formula=["y ~ x + offset(o1)", "~ 1 + offset(o2)"],
        optimize_smoothing=False,
    )
    gam.fit(data=data)

    eta_default = np.asarray(gam.predict(type="link"), dtype=np.float64)
    eta_from_data = np.asarray(gam.predict(data, type="link"), dtype=np.float64)
    eta_zero = np.asarray(
        gam.predict(data.assign(o1=np.zeros(n), o2=np.zeros(n)), type="link"),
        dtype=np.float64,
    )

    assert eta_default.shape == (n, 2)
    assert_allclose(eta_default, eta_from_data, atol=1e-10, rtol=1e-10)
    assert_allclose(eta_default[:, 0] - eta_zero[:, 0], o1, atol=1e-10, rtol=1e-10)
    assert_allclose(eta_default[:, 1] - eta_zero[:, 1], o2, atol=1e-10, rtol=1e-10)


def test_formula_vector_valued_fx_build():
    """Known-gap coverage verifying that formula vector valued fx build."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x0": [0.0, 1.0, 2.0, 3.0],
            "x1": [0.5, 1.5, 2.5, 3.5],
        }
    )

    for formula in (
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], fx=[True, False])',
        'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], fx=[True, False], mc=[True, False])',
    ):
        built = _build_formula_only(formula, data)
        smooth = built.predictor_specs[0].terms[0].smooth_spec
        assert smooth is not None
        assert smooth.fx == [True, False]


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript required for mgcv parity")
def test_random_effect_id_linkage_is_explicitly_unsupported_like_mgcv():
    """
    Known-gap coverage verifying that random effect id linkage is explicitly unsupported
    like mgcv.
    """
    data = _make_random_effect_data()
    expected = _run_mgcv_random_effect_id_error(data)

    assert expected["ok"] is False
    assert "random effects don't work with ids" in str(expected["message"]).lower()

    gam = GAM(
        family="gaussian",
        formula='y ~ s(f, bs="re", id="shared")',
        optimize_smoothing=True,
        smoothing_method="REML",
    )

    with pytest.raises(NotImplementedError, match="random effects don't work with ids"):
        gam.fit(data=data)


def test_exact_reparam_state_drops_legacy_sl_block_bookkeeping():
    """
    Known-gap coverage verifying that exact reparam state drops legacy sl block
    bookkeeping.
    """
    data = _make_gaussian_data(seed=123, n=120)
    gam = GAM(
        family="gaussian",
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.3, 0.9])',
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)

    state = gam.reparam_state_
    assert state is not None
    assert getattr(state, "sl_blocks", None) in (None, ())


def test_general_family_generic_gdi2_kernel_available_for_gaulss():
    """
    Known-gap coverage verifying that general family generic GDI2 kernel available for
    gaulss.
    """
    data = _gaulss_data()
    gam = _fit_nampy_model_fixed_sp(data, GAULSS_FORMULA, "gaulss", [1.0])
    y = data["y"].to_numpy(dtype=float)
    sp = gam.smoothing_params
    sol = solve_pirls_given_smoothing(gam, y, sp)

    _gdi2_joint_kernel(gam, y, sol, sp, method="REML", need_hessian=True)
