"""Check authored helper coverage against signatures and independent identities."""

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))


def test_link_reference_covers_registry_and_round_trips():
    from statistical_helpers_reference import LINKS, link_code

    from nampy.gam.families._function_maps import LINK_REGISTRY

    assert set(LINK_REGISTRY) == {
        ("1/mu^2" if n == "inverse_squared" else n) for n in LINKS if n != "power"
    }
    for slug in LINKS:
        namespace = {}
        source = link_code(slug)
        assert not any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(source)))
        exec(source, namespace)
        np.testing.assert_allclose(
            namespace["recovered_mean"], namespace["mean"], rtol=1e-10, atol=1e-10
        )


def test_quasi_power_and_variance_are_independent_and_documented():
    from feature_reference import PARAMETERS

    from nampy.gam.families.registry import make_gam_family

    mu = np.array([0.5, 1.0, 2.0])
    family = make_gam_family({"name": "quasi", "variance": "mu^3", "power": 0.5})
    np.testing.assert_allclose(family.link(mu), np.sqrt(mu))
    np.testing.assert_allclose(family.variance(mu), mu**3)
    assert "link exponent" in PARAMETERS["power"]
    for power in [0.0, -1.0]:
        np.testing.assert_allclose(
            make_gam_family({"name": "quasi", "power": power}).link(mu), np.log(mu)
        )
    with pytest.raises(ValueError):
        make_gam_family({"name": "quasi", "link": "logit", "power": 0.5})


def test_helper_parameter_contracts_cover_live_signatures():
    from statistical_helpers_reference import helper_contracts, helper_specs

    for name, obj in helper_specs().items():
        parameters, outputs = helper_contracts(name)
        documented = {
            part.strip() for field, *_ in parameters for part in field.split(";")
        }
        assert set(inspect.signature(obj).parameters) <= documented, (name, documented)
        assert outputs
        page = ROOT / f"docs/extension_reference/statistical/{name}.rst"
        assert page.exists()
        assert str(inspect.signature(obj)) in page.read_text()


def test_endpoint_output_fields_have_authored_descriptions():
    import textwrap

    from statistical_helpers_reference import helper_contracts

    from nampy.gam.fit.selection import optimizer_endpoint_diagnostics

    tree = ast.parse(textwrap.dedent(inspect.getsource(optimizer_endpoint_diagnostics)))
    keys = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant)
    }
    documented = " ".join(
        row[0] for row in helper_contracts("optimizer_endpoint_diagnostics")[1]
    )
    assert keys
    assert all(key in documented for key in keys)


def test_mask_is_unit_invariant_and_true_means_exclude():
    from nampy.gam.diagnostics.plots import exclude_too_far

    g = np.array([0.0, 0.5, 1.0])
    mask = exclude_too_far(g, g, [0.5], [0.5], 0.2)
    np.testing.assert_array_equal(mask, [True, False, True])
    np.testing.assert_array_equal(
        mask, exclude_too_far(100 * g, 0.01 * g, [50], [0.005], 0.2)
    )


def test_quadratic_form_helpers_match_an_ordinary_chisquare_case():
    from nampy.gam.inference.chi_square_mixtures import (
        DaviesAlgorithm,
        liu2,
        psum_chisq,
    )

    q = np.array([1.0, 4.0, 8.0])
    np.testing.assert_allclose(psum_chisq(q, [1.0], df=[2]), chi2.sf(q, 2), atol=2e-5)
    np.testing.assert_allclose(liu2(q, [1.0], df=[2]), chi2.sf(q, 2), atol=1e-12)
    probability, trace, status = DaviesAlgorithm().davies(
        [1.0], [0.0], [2], 1, 0.0, 4.0, 100000, 2e-5
    )
    assert status == 0
    assert len(trace) == 7
    assert probability == pytest.approx(chi2.cdf(4.0, 2), abs=2e-5)
