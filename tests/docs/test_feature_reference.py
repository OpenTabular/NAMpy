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
    from reference_symbols import SYMBOLS

    for module, names in SYMBOLS.items():
        for name in names:
            assert (docs / f"api/generated/{module}.{name}.rst").exists()
    for descriptor in basis_descriptors():
        page = docs / f"statistical_reference/smooths/{descriptor.name}.rst"
        text = page.read_text()
        assert "Parameters\n" in text and "Outputs\n" in text
        runtime_cls = (
            descriptor.runtime_class.__name__
            if descriptor.runtime_class
            else {"fs": "FSmoothInteractionTerm", "sz": "SZSmoothInteractionTerm"}[
                descriptor.name
            ]
        )
        runtime = (docs / f"extension_reference/smooths/{runtime_cls}.rst").read_text()
        assert "transform_new(X_new)" in runtime
        assert "predict(type='terms')" in text
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


def test_reexported_config_viewcode_backlinks_keep_cached_import_path():
    from sphinx import addnodes

    from nampy.neural.registry import architectures

    for name, spec in architectures().items():
        config = spec.config
        docname = f"neural_reference/{name}"
        tree = new_document("config signature")
        signature = addnodes.desc_signature()
        signature["module"] = config.__module__
        signature["fullname"] = config.__name__
        tree += signature
        # A clean build sees the public API guide before the dedicated page.
        # Sphinx retains that first refname even after updating the target page.
        entry = ("source", {}, {config.__name__: docname}, "nampy.neural.configs")
        app = SimpleNamespace(
            builder=builder(),
            env=SimpleNamespace(_viewcode_modules={config.__module__: entry}),
        )
        reference.link_references(app, tree, docname)
        reference.link_references(app, tree, docname)
        assert signature["ids"] == [
            f"{config.__module__}.{config.__name__}",
            f"nampy.neural.configs.{config.__name__}",
        ]


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


def test_formula_and_runtime_contracts_have_separate_homes():
    user = (ROOT / "docs/statistical_reference/smooths/cr.rst").read_text()
    runtime = (
        ROOT / "docs/extension_reference/smooths/CubicSplineTerm.rst"
    ).read_text()
    assert "Cubic regression spline" in user
    assert "``shared_basis_setup``" not in user
    assert "``shared_basis_setup``" in runtime
    assert "/extension_reference/smooths/CubicSplineTerm" in user


def test_family_contracts_do_not_leak_other_parameterizations():
    normal = (ROOT / "docs/neural_reference/families/normal.rst").read_text()
    assert "For Beta" not in normal
    assert "class probabilities" not in normal
    assert "sigma^2" in normal
    from family_contracts import mathematical_contract

    nb = mathematical_contract("negativebinom", "neural")
    assert "r=1/alpha" in nb
    quantile = mathematical_contract("quantile", "neural")
    assert "otherwise the arithmetic average" in quantile


def test_control_inventory_has_unique_canonical_anchors():
    from reference_organization import CONTROL_GROUPS, control_anchor

    from nampy.gam import gam_control

    names = [name for _, group in CONTROL_GROUPS.values() for name in group]
    assert len(names) == len(set(names))
    assert set(names) == set(gam_control().__dataclass_fields__)
    for slug, (_, group) in CONTROL_GROUPS.items():
        text = (ROOT / f"docs/statistical_reference/controls/{slug}.rst").read_text()
        for name in group:
            assert f".. _{control_anchor(name)}:" in text
    index = (ROOT / "docs/statistical_reference/controls/index.rst").read_text()
    assert "   epsilon\n" not in index


def test_small_result_previews_preserve_local_term_meanings():
    import runpy

    runner = runpy.run_path(str(ROOT / "docs/scripts/check_reference_examples.py"))
    value = [[{"term": ("x", "z"), "order": 2, "contribution": 0.25}]]
    preview = runner["result_preview"](value)
    term = preview["items"][0]["items"][0]
    assert term["order"] == 2 and term["contribution"] == 0.25
    assert term["term"]["items"] == ["x", "z"]
    assert preview["omitted"] == 0


def test_gammals_output_columns_match_the_live_response_transform():
    import numpy as np
    from reference_schema import statistical_family_returns

    from nampy.gam.families.registry import make_gam_family

    family = make_gam_family("gammals")
    eta = np.array([[np.log(2.0), -1.0]])
    response = family._predict_response_from_eta(eta)
    natural = family.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(response[:, 0], natural[:, 0])
    np.testing.assert_allclose(response[:, 1], np.log(natural[:, 1] ** 2))
    meaning = next(
        row[3]
        for row in statistical_family_returns("gammals", family)
        if row[0] == "GAM.predict(type='response')"
    )
    assert "(mu, log(phi))" in meaning


def test_public_inventory_is_stable_and_covers_registry_entries():
    import json

    text = (ROOT / "docs/_data/public_reference_inventory.json").read_text()
    assert "0x" not in text
    records = json.loads(text)
    assert all(record.get("tests") and record.get("example") for record in records)
    assert len({(r["kind"], r["symbol"]) for r in records}) == len(records)


def test_control_names_link_directly_to_the_grouped_feature_anchor():
    tree = new_document("control link")
    tree += nodes.paragraph("", "", nodes.literal("", "epsilon"))
    reference.reference_targets.cache_clear()
    reference.link_references(
        SimpleNamespace(builder=builder()), tree, "tutorials/example"
    )
    link = next(tree.findall(nodes.reference))
    assert (
        link["refuri"]
        == "../statistical_reference/controls/coefficient_fitting.html#control-epsilon"
    )


def test_underscore_control_names_use_normalized_html_anchors():
    target = reference.reference_targets()["ncv_threads"]
    assert target == "statistical_reference/controls/execution#control-ncv-threads"
    assert (
        reference.relative_reference_uri(builder(), "compatibility", target)
        == "statistical_reference/controls/execution.html#control-ncv-threads"
    )


def test_formula_defaults_follow_the_compiler_not_the_spec_record_default():
    from smooth_contracts import formula_option_rows

    from nampy.gam.basis_registry import require_basis_descriptor

    cr = {row[0]: row for row in formula_option_rows(require_basis_descriptor("cr"))}
    tp = {row[0]: row for row in formula_option_rows(require_basis_descriptor("tp"))}
    te = {row[0]: row for row in formula_option_rows(require_basis_descriptor("te"))}
    assert "model k (10 by default)" in cr["``k``"][2]
    assert "-1 (automatic)" in tp["``k``"][2]
    assert "5**d[j]" in te["``k``"][2]
