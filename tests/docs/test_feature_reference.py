"""Central references must cover registries and be reachable from existing docs."""

import importlib
import posixpath
import sys
from pathlib import Path
from types import SimpleNamespace

from docutils import nodes
from docutils.utils import new_document

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))
reference = importlib.import_module("feature_reference")


def builder():
    return SimpleNamespace(
        get_relative_uri=lambda source, target: posixpath.relpath(
            target + ".html", posixpath.dirname(source) or "."
        )
    )


def test_inline_names_and_existing_roles_share_central_destination():
    tree = new_document("links")
    paragraph = nodes.paragraph()
    paragraph += nodes.literal("", "GAMRegressor")
    paragraph += nodes.literal("", "NAMRegressor")
    tree += paragraph
    existing = nodes.reference(
        "",
        "GAMRegressor",
        refuri="../api/generated/nampy.models.GAMRegressor.html#nampy.models.GAMRegressor",
    )
    tree += existing
    code = nodes.literal_block("", "GAMRegressor()")
    tree += code
    app = SimpleNamespace(builder=builder())
    reference.link_references(app, tree, "tutorials/example")
    links = list(tree.findall(nodes.reference))
    assert links[0]["refuri"] == "../statistical_reference/estimators/GAMRegressor.html"
    assert links[1]["refuri"] == "../neural_reference/estimators/NAMRegressor.html"
    assert existing["refuri"] == links[0]["refuri"]
    assert code.astext() == "GAMRegressor()"
    assert not list(code.findall(nodes.reference))


def test_highlighted_code_links_preserve_source_and_ignore_strings():
    content = '<div class="highlight"><pre><span class="n">GAMRegressor</span><span class="p">()</span><span class="s">"GAMRegressor"</span></pre></div>'
    linked = reference.link_code_html(
        content, "tutorials/example", builder(), reference.reference_targets()
    )
    assert 'href="../statistical_reference/estimators/GAMRegressor.html"' in linked
    assert '<span class="s">"GAMRegressor"</span>' in linked
    assert linked.count("api-code-link") == 1
    assert (
        reference.link_code_html(
            linked, "tutorials/example", builder(), reference.reference_targets()
        )
        == linked
    )


def test_registered_features_have_dedicated_pages_and_parameter_contracts(tmp_path):
    from public_reference import generate

    from nampy.gam.basis_registry import basis_descriptors
    from nampy.neural.distributions.registry import FAMILY_REGISTRY
    from nampy.neural.registry import architectures

    docs = tmp_path / "docs"
    (docs / "api").mkdir(parents=True)
    (docs / "api/gam_families.rst").write_text(
        (ROOT / "docs/api/gam_families.rst").read_text()
    )
    generate(tmp_path)
    reference.generate(docs)
    for descriptor in basis_descriptors():
        page = docs / f"statistical_reference/smooths/{descriptor.name}.rst"
        text = page.read_text()
        assert "Parameters\n" in text and "Outputs\n" in text
        assert "transform_new(X_new)" in text
        assert "design_matrix(" not in text
    for spec in architectures().values():
        for task, suffix in [
            ("regression", "Regressor"),
            ("classification", "Classifier"),
            ("distributional", "LSS"),
        ]:
            if spec.supports(task):
                name = spec.estimator_prefix + suffix
                page = docs / f"neural_reference/estimators/{name}.rst"
                assert page.exists()
                contract = (docs / f"_generated/reference/{name}.inc").read_text()
                assert "Configuration option of type" not in contract
                assert "Predictions and explanations" in contract
    for name in FAMILY_REGISTRY:
        text = (docs / f"neural_reference/families/{name}.rst").read_text()
        assert "Parameters\n" in text and "Outputs\n" in text
        assert "0x" not in text  # callable defaults must be reproducible
    assert (
        "Statistical models" in (docs / "statistical_reference/index.rst").read_text()
    )


def test_every_public_estimator_resolves_to_a_central_page():
    from nampy import models

    targets = reference.reference_targets()
    for name in models.__all__:
        assert name in targets, name
        assert targets[name].startswith(("statistical_reference/", "neural_reference/"))


def test_unindexed_reference_signatures_keep_viewcode_backlinks():
    from sphinx import addnodes

    tree = new_document("signature")
    signature = addnodes.desc_signature()
    signature["module"] = "nampy.models"
    signature["fullname"] = "GAMRegressor.fit"
    tree += signature
    reference.link_references(
        SimpleNamespace(builder=builder()),
        tree,
        "statistical_reference/estimators/GAMRegressor",
    )
    assert signature["ids"] == ["nampy.models.GAMRegressor.fit"]


def test_existing_guides_are_links_instead_of_duplicate_sidebar_branches():
    text = reference.tree("Reference", ["smooths/index", "Formula </api/gam_formula>"])
    assert "   smooths/index" in text
    assert "   Formula" not in text
    assert ":doc:`Formula </api/gam_formula>`" in text


def test_parameter_tables_have_explicit_types_and_effective_factory_defaults():
    from nampy.gam.smooths import ThinPlateSplineTerm
    from nampy.neural.configs import DefaultNAMConfig

    table = reference.parameters(ThinPlateSplineTerm)
    assert "Type / input shape" in table and "Default" in table
    assert "str | int | Sequence[str | int]" in table
    config = reference.parameters(DefaultNAMConfig)
    assert "<factory>" not in config
    assert "[128, 128, 32]" in config


def test_every_core_constructor_keyword_has_type_default_and_meaning():
    from reference_schema import core_parameter_rows

    from nampy.gam.model.api import _GAM_HPARAM_KEYS

    rows = core_parameter_rows()
    assert {row[0].strip("`") for row in rows} == set(_GAM_HPARAM_KEYS)
    assert all(len(row) == 4 and all(row) for row in rows)


def test_private_covariance_correction_is_separate_from_prediction_workflow():
    methods = (ROOT / "docs/statistical_reference/methods/index.rst").read_text()
    assert "Prediction uncertainty" in methods
    assert "Vb.corr" not in methods
    assert reference.reference_targets()["Vb.corr"].endswith(
        "internals/covariance_correction"
    )


def test_all_reference_examples_are_executed_and_have_output_shapes():
    import runpy

    audit = runpy.run_path(str(ROOT / "docs/scripts/check_reference_contracts.py"))
    audit["main"]()


def test_quantiles_do_not_claim_a_scalar_spread_hook():
    import numpy as np
    from reference_schema import neural_family_returns

    from nampy.neural.distributions.registry import FAMILY_REGISTRY

    family, _ = FAMILY_REGISTRY["quantile"].instantiate(np.ones(4))
    rows = neural_family_returns("quantile", family)
    assert any(
        row[0] == "predict_spread" and "NotImplementedError" in row[3] for row in rows
    )


def test_persistence_return_contract_matches_the_verified_path_output():
    from reference_schema import METHOD_RETURNS

    assert METHOD_RETURNS["save_model"][0][1] == "pathlib.Path"


def test_context_specific_parameter_types_do_not_advertise_invalid_inputs():
    import inspect

    from reference_schema import parameter_type

    from nampy.gam.families.gamlss.gaulss import GaulssFamily
    from nampy.gam.smooths import TensorProductSplineTerm, ThinPlateSplineTerm

    assert parameter_type("k", -1, obj=ThinPlateSplineTerm) == "int"
    assert "Sequence" in parameter_type("k", 10, obj=TensorProductSplineTerm)
    assert "Sequence[str]" in parameter_type(
        "link", ("identity", "logb"), obj=GaulssFamily
    )
    assert "None" not in parameter_type("X", inspect.Parameter.empty)


def test_external_diagnostic_draws_use_replicates_by_observations():
    from reference_schema import PARAMETER_TYPES

    assert "(s_rep, n)" in PARAMETER_TYPES["quantile_permutations"]
    assert "(rep, n)" in PARAMETER_TYPES["response_draws"]


def test_category_and_natural_parameter_shapes_remain_distinct():
    from reference_schema import statistical_family_returns

    from nampy.gam.families.registry import make_gam_family

    ordinal = statistical_family_returns(
        "ocat", make_gam_family({"name": "ocat", "R": 3})
    )
    assert any(row[2] == "(n, R)" and "probabilities" in row[3] for row in ordinal)
    multinomial = statistical_family_returns("multinom", make_gam_family("multinom"))
    assert any(
        row[0] == "Natural parameter interface" and row[2] == "(n, K+1)"
        for row in multinomial
    )
    assert any(
        row[0] == "GAM.predict(type='link')" and row[2] == "(n, K)"
        for row in multinomial
    )
