"""Regression checks for the reviewed reference contracts and preview semantics."""

import ast
import inspect
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))


def test_example_execution_evidence_tracks_library_changes(tmp_path):
    module = runpy.run_path(str(ROOT / "docs/scripts/check_reference_examples.py"))
    package = tmp_path / "nampy"
    package.mkdir()
    source = package / "contracts.py"
    source.write_text("value = 1\n")
    fingerprint = module["implementation_fingerprint"]
    before = fingerprint(tmp_path)
    output = tmp_path / "docs/_generated/output.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    assert fingerprint(tmp_path) == before
    source.write_text("value = 2\n")
    assert fingerprint(tmp_path) != before


def test_inventory_dimensions_match_smooth_purpose():
    from smooth_contracts import smooth_inputs

    import nampy.gam.smooths  # noqa: F401
    import nampy.gam.specs  # noqa: F401
    from nampy.gam.basis_registry import basis_descriptors

    for descriptor in basis_descriptors():
        text = smooth_inputs(descriptor)
        if descriptor.name in {"te", "ti", "t2"}:
            assert text == "Two or more covariates, grouped into margins"
        elif (
            descriptor.runtime_class
            and descriptor.runtime_class.__name__ == "BivariateShapePSplineTerm"
        ):
            assert text == "Two covariates"
        elif descriptor.name == "ad":
            assert text == "One or two covariates"


def test_formula_tables_cover_compiler_keywords_and_aliases():
    from smooth_contracts import formula_option_rows

    from nampy.gam.basis_registry import basis_descriptors, require_basis_descriptor
    from nampy.gam.specs.build import _build_predictor_spec

    source = ast.parse(inspect.getsource(_build_predictor_spec))
    accepted = {
        node.args[0].value
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "kw"
        and node.func.attr == "pop"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    documented = {
        row[0].strip("`")
        for descriptor in basis_descriptors()
        for row in formula_option_rows(descriptor)
    }
    assert accepted <= documented | {"basis", "smoothing_id"}
    for name in ["te", "ti", "t2"]:
        options = {
            row[0].strip("`")
            for row in formula_option_rows(require_basis_descriptor(name))
        }
        assert {"bs", "k", "m", "d", "xt", "pc", "by", "id", "sp"} <= options
        assert ("fx" in options) == (name != "t2")
        assert ("mc" in options) == (name == "ti")
        assert ("ord" in options) == (name == "t2")
        assert "np" not in options


def test_previews_preserve_scalar_and_matrix_shapes():
    helpers = runpy.run_path(str(ROOT / "docs/scripts/check_reference_examples.py"))
    preview = helpers["result_preview"]
    assert preview(np.array(1.25)) == 1.25
    array = preview(np.arange(16).reshape(8, 2))
    assert array["shape"] == [8, 2]
    assert array["values"] == [[0, 1], [2, 3]]
    assert array["omitted_by_axis"] == [6, 0]
    covariance = preview(np.eye(4), field_name="coefficient_covariance")
    assert covariance["values"] == "omitted; see shape contract"
    assert covariance["shape"] == [4, 4]


def test_local_importance_preview_marks_both_truncation_levels():
    preview = runpy.run_path(str(ROOT / "docs/scripts/check_reference_examples.py"))[
        "result_preview"
    ]
    result = [
        [{"term": ("x",), "order": 1, "contribution": float(i)} for i in range(3)]
        for _ in range(8)
    ]
    output = preview(result, field_name="local_term_importance")
    assert output["length"] == 8 and output["omitted"] == 7
    assert output["items"][0]["length"] == 3
    assert output["items"][0]["omitted"] == 1


def test_controls_have_one_combined_example_and_no_scalar_output_dumps():
    from reference_organization import CONTROL_GROUPS

    for slug, (_, names) in CONTROL_GROUPS.items():
        text = (ROOT / f"docs/statistical_reference/controls/{slug}.rst").read_text()
        assert text.count("Combined example\n") == 1
        assert text.count(".. literalinclude::") == 1
        assert "Observed result preview" not in text
        assert "Output types and shapes" not in text
        for name in names:
            assert f"``{name}``" in text
    assert (
        "scale_est"
        not in (ROOT / "docs/statistical_reference/controls/execution.rst").read_text()
    )


@pytest.mark.parametrize(
    "name,reference", [("student_lss", "TF"), ("beta_lss", "BE"), ("nb_lss", "NBI"), ("lognormal_lss", "LOGNO"), ("weibull_lss", "WEI"), ("betabinomial_lss", "BB"), ("zaga_lss", "ZAGA"), ("beinf_lss", "BEINF")]
)
def test_extension_family_leads_with_supported_parameterization(name, reference):
    from family_contracts import mathematical_contract

    text = mathematical_contract(name, "statistical")
    assert f"gamlss.dist::{reference}" in text
    assert "References: `mgcv" not in text
    if name == "student_lss":
        assert r"2<\nu<\infty" in text
        assert r"\nu>0" not in text
        assert ":doc:`/statistical_reference/families/theory`" in text
        assert text.count(r"\sigma^2\nu/(\nu-2)") == 1
    elif name == "beta_lss":
        assert r"\frac{\mu(1-\mu)}{1+\phi}" in text
    elif name == "nb_lss":
        assert r"\mu+\alpha\mu^2" in text
    elif name == "betabinomial_lss":
        assert "sigma=1/kappa" in text
        assert "automatic joint REML" in text
        assert "neural fitting uses its normalized likelihood" in text
    elif name == "zaga_lss":
        assert "mass at zero and density elsewhere" in text
        assert "automatic general-family REML" in text
    elif name == "beinf_lss":
        assert "coupled category log ratios" in text
        assert "all five natural errors" in text
        assert "variance hook" in text
    else:
        assert "coercion to REML" in text
        assert "optim route is unavailable" in text


def test_multinomial_contract_and_adapter_inventory_are_symbolic():
    text = (ROOT / "docs/statistical_reference/families/multinom.rst").read_text()
    assert "     - K\n" in text
    assert "eta1, ..., etaK" in text
    assert "K=1: two response categories" in text
    from nampy import GAMLSS, GAMRegressor
    from nampy.gam.families.registry import make_gam_family

    family = make_gam_family({"name": "multinom", "K": 3})
    GAMLSS()._validate_family_role(family)
    with pytest.raises(ValueError):
        GAMRegressor()._validate_family_role(family)
    index = (ROOT / "docs/statistical_reference/families/index.rst").read_text()
    row = index.split(":doc:`multinom <multinom>`", 1)[1].split("   * - ", 1)[0]
    assert "GAMLSS / core GAM" in row


@pytest.mark.parametrize("name", ["ExpectileGAM", "QuantileGAM"])
def test_loss_estimator_method_contracts_are_present_and_route_specific(name):
    text = (ROOT / f"docs/statistical_reference/estimators/{name}.rst").read_text()
    for method in ["fit", "predict", "score", "predict_components", "plot_terms", "lpmatrix"]:
        assert f"model.{method}(" in text
    assert "-sum(w*loss)/sum(w)" in text
    assert "fit_result_" in text and "NotFittedError" in text
    assert ".. autoclass::" not in text
    example = (
        ROOT / f"docs/_examples/reference/statistical_reference/estimators/{name}.py"
    ).read_text()
    if name == "QuantileGAM":
        assert "bandwidth=.1" not in example
        assert (
            "bandwidth=.1"
            in (
                ROOT
                / "docs/_examples/reference/statistical_reference/estimators/QuantileGAM_smooth.py"
            ).read_text()
        )


def test_statistical_guide_is_a_runnable_workflow():
    import matplotlib.pyplot as plt

    text = (ROOT / "docs/user_guide/statistical_gams.rst").read_text()
    assert len(text) < 12000
    for step in range(1, 7):
        assert f"{step}. " in text
    namespace = {}
    lines = text.splitlines()
    try:
        for i, line in enumerate(lines):
            if line == ".. code-block:: python":
                block = []
                for following in lines[i + 2 :]:
                    if following and not following.startswith("   "):
                        break
                    block.append(
                        following[3:] if following.startswith("   ") else following
                    )
                exec(
                    compile("\n".join(block), "statistical workflow", "exec"), namespace
                )
        assert namespace["prediction"].shape == (40,)
        assert np.isfinite(namespace["standard_error"]).all()
    finally:
        plt.close("all")
