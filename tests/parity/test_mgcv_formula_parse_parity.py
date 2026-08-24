"""Parser-layer parity checks against mgcv::interpret.gam."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam.formula import (
    ParsedParametricTerm,
    ParsedSmoothTerm,
    extract_formula_terms,
    parse_gam_formula,
)
from nampy.gam.specs.build import build_formula_model
from nampy.gam.specs.preprocess import apply_formula_preprocess_to_new_data
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.mgcv_parity_utils import _normalize_python_formula_text
from tests.reference_fixtures import load_reference, reference_key, save_reference

R_SCRIPT = shutil.which("Rscript")
MGCV_INTERPRET_GAM_SCRIPT = PARITY_DIR / "mgcv_interpret_gam.R"


def _normalize_formula_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("**", "^").strip())


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _run_mgcv_interpret_gam(formula):
    key = reference_key("interpret_gam", {"formula": formula})
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.json"
        output_path = tmpdir_path / "output.json"
        input_path.write_text(json.dumps({"formula": formula}), encoding="utf-8")
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_INTERPRET_GAM_SCRIPT),
                str(input_path),
                str(output_path),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _python_smooth_term(term: ParsedSmoothTerm) -> dict:
    k_value = term.kwargs.get("k", None)
    if k_value is None:
        k_list = []
    elif isinstance(k_value, tuple):
        k_list = [int(v) for v in k_value]
    elif isinstance(k_value, list):
        k_list = [int(v) for v in k_value]
    else:
        k_list = [int(k_value)]

    return {
        "kind": str(term.kind),
        "label": _normalize_formula_text(term.label),
        "term": [_normalize_formula_text(v) for v in term.features],
        "by": (
            "NA"
            if term.by_variable is None
            else _normalize_formula_text(str(term.by_variable))
        ),
        "id": None if term.kwargs.get("id") is None else str(term.kwargs.get("id")),
        "k": k_list,
    }


def _python_component(component) -> dict:
    smooth_terms = []

    for term in component.terms:
        if isinstance(term, ParsedParametricTerm):
            continue
        if isinstance(term, ParsedSmoothTerm):
            smooth_terms.append(_python_smooth_term(term))
            continue
        raise TypeError(f"Unknown parsed term type: {type(term)}")

    return {
        "pf": _normalize_formula_text(component.pf),
        "pfok": int(component.pfok),
        "fake_formula": _normalize_formula_text(component.fake_formula),
        "response": component.response_name,
        "fake_names": [_normalize_formula_text(v) for v in component.fake_names],
        "pred_names": [_normalize_formula_text(v) for v in component.pred_names],
        "pred_formula": _normalize_formula_text(component.pred_formula),
        "lpi": [int(v) for v in component.lpi],
        "intercept": bool(component.intercept),
        "smooth_terms": smooth_terms,
    }


def _normalize_mgcv_component(component: dict) -> dict:
    smooth_terms = []
    for term in component["smooth_terms"]:
        smooth_terms.append(
            {
                "kind": str(term["kind"]),
                "label": _normalize_formula_text(term["label"]),
                "term": [
                    _normalize_formula_text(v) for v in _ensure_list(term["term"])
                ],
                "by": (
                    "NA"
                    if term.get("by", None) is None
                    else _normalize_formula_text(str(term["by"]))
                ),
                "id": None if term.get("id", None) is None else str(term["id"]),
                "k": [int(v) for v in _ensure_list(term.get("k", []))],
            }
        )

    return {
        "pf": _normalize_formula_text(component["pf"]),
        "pfok": int(component["pfok"]),
        "fake_formula": _normalize_formula_text(component["fake_formula"]),
        "response": component.get("response", None),
        "fake_names": [
            _normalize_formula_text(v)
            for v in _ensure_list(component.get("fake_names", []))
        ],
        "pred_names": [
            _normalize_formula_text(v)
            for v in _ensure_list(component.get("pred_names", []))
        ],
        "pred_formula": _normalize_formula_text(component["pred_formula"]),
        "lpi": [int(v) for v in _ensure_list(component.get("lpi", []))],
        "intercept": bool(component["intercept"]),
        "smooth_terms": smooth_terms,
    }


def _assert_parser_parity(formula) -> None:
    expected = _run_mgcv_interpret_gam(formula)
    actual = parse_gam_formula(formula)

    assert actual.response_name == expected["response"]
    assert int(actual.nlp) == int(expected["nlp"])
    assert _normalize_formula_text(actual.fake_formula) == _normalize_formula_text(
        expected["fake_formula"]
    )
    assert _normalize_formula_text(actual.pred_formula) == _normalize_formula_text(
        expected["pred_formula"]
    )

    actual_components = [
        _python_component(component) for component in actual.components
    ]
    expected_components = [
        _normalize_mgcv_component(component) for component in expected["components"]
    ]
    assert actual_components == expected_components


class TestMGCVFormulaParseParity:
    """
    Parser and formula-build parity checks against mgcv::interpret.gam, including
    preprocess reconstruction paths.
    """
    def test_dot_shorthand_parse_is_deferred_until_build(self):
        """Verify that dot shorthand parse is deferred until build."""
        parsed = parse_gam_formula("y ~ .")

        assert parsed.response_name == "y"
        assert parsed.components[0].pf == "y ~ ."
        assert parsed.components[0].fake_formula == "y ~ ."
        assert parsed.components[0].pred_names == ()

    def test_single_formula_parser_matches_mgcv_interpret_gam(self):
        """Verify that single formula parser matches mgcv interpret gam."""
        formula = 'y ~ x * z - z + s(w, k=6, bs="cr", by=f, id="g") + offset(o) - 1'
        _assert_parser_parity(formula)

    def test_formula_list_parser_matches_mgcv_interpret_gam(self):
        """Verify that formula list parser matches mgcv interpret gam."""
        formula = [
            'y ~ x + s(z, k=7, bs="cr")',
            "~ 1",
            '1 + 2 ~ te(a, b, bs=["cr", "tp"], k=[5, 6], id="shared")',
        ]
        _assert_parser_parity(formula)

    def test_multi_response_formula_list_matches_mgcv_interpret_gam(self):
        """Verify that multi response formula list matches mgcv interpret gam."""
        formula = [
            "y1 ~ -1",
            "y2 ~ -1",
            '1 + 2 ~ s(x, k=5, bs="cr")',
        ]
        _assert_parser_parity(formula)

    def test_formula_operator_parity_matches_mgcv_interpret_gam(self):
        """Verify that formula operator parity matches mgcv interpret gam."""
        formula = "y ~ (a + b)/c + z:x + x + offset(o1) - offset(o2)"
        _assert_parser_parity(formula)

    def test_transformed_factor_and_power_parser_matches_mgcv(self):
        """Verify that transformed factor and power parser matches mgcv."""
        formula = 'y ~ (0 + x)**2 + I(z**2) + s(I(w**2), k=5, bs="cr")'
        _assert_parser_parity(formula)

    def test_mixed_list_formula_args_preserve_positional_and_named_entries(self):
        """Verify that mixed list formula args preserve positional and named entries."""
        parsed = parse_gam_formula('y ~ s(x, bs="tp", xt=list(1, bs="ps"))')
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        assert term.kwargs["xt"] == {0: 1, "bs": "ps"}

    def test_nested_mixed_list_formula_args_preserve_structure(self):
        """Verify that nested mixed list formula args preserve structure."""
        parsed = parse_gam_formula(
            'y ~ s(x, bs="tp", xt=list(list(1, seed=2), bs="ps"))'
        )
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        assert term.kwargs["xt"] == {0: {0: 1, "seed": 2}, "bs": "ps"}

    def test_r_boolean_formula_values_parse_as_booleans(self):
        """Verify that R TRUE/FALSE option values are parsed as booleans."""
        parsed = parse_gam_formula(
            'y ~ ti(x0, x1, bs=["cr", "ps"], k=[6, 6], mc=[TRUE, FALSE], fx=FALSE)'
        )
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        assert term.kwargs["mc"] == [True, False]
        assert term.kwargs["fx"] is False

    def test_r_matrix_formula_values_preserve_r_fill_order(self):
        """Verify that R matrix(...) values preserve column-major/byrow semantics."""
        parsed = parse_gam_formula(
            'y ~ s(x0, bs="cr", '
            'xt=list(penalty=matrix(c(1,2,3,4),2,2), '
            'polys=list(A=matrix(c(1,2,3,4),2,2,byrow=TRUE))))'
        )
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        np.testing.assert_array_equal(
            term.kwargs["xt"]["penalty"],
            np.array([[1, 3], [2, 4]]),
        )
        np.testing.assert_array_equal(
            term.kwargs["xt"]["polys"]["A"],
            np.array([[1, 2], [3, 4]]),
        )

    def test_r_diag_formula_values_parse_as_identity_matrix(self):
        """Verify that R diag(n) formula values parse as an identity matrix."""
        parsed = parse_gam_formula('y ~ s(f, bs="re", xt=list(S=list(diag(4))))')
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        np.testing.assert_array_equal(term.kwargs["xt"]["S"][0], np.eye(4))

    def test_list_kwargs_expansion_supports_non_identifier_names(self):
        """Verify that list kwargs expansion supports non identifier names."""
        formula = 'y ~ s(x, bs="tp", k=5, xt=list(**{"max.knots": 10}, seed=2))'
        parsed = parse_gam_formula(formula)
        term = parsed.components[0].terms[0]

        assert isinstance(term, ParsedSmoothTerm)
        assert term.kwargs["xt"] == {"max.knots": 10, "seed": 2}

        expected = _run_mgcv_interpret_gam(_normalize_python_formula_text(formula))
        actual_component = _python_component(parsed.components[0])
        expected_component = _normalize_mgcv_component(expected["components"][0])
        assert actual_component == expected_component

    def test_extract_formula_terms_preserves_shared_component_ownership(self):
        """
        Shared ``1 + 2 ~ ...`` remains one component with overlapping LP ids.
        """
        parsed = parse_gam_formula(
            [
                "y1 ~ -1",
                "y2 ~ -1",
                '1 + 2 ~ s(x, k=5, bs="cr")',
            ]
        )

        extracted = extract_formula_terms(parsed)
        assert len(extracted) == 3
        assert [component.lpi for component in extracted] == [(1,), (2,), (1, 2)]
        assert [component.is_base_formula for component in extracted] == [
            True,
            True,
            False,
        ]
        assert len(extracted[2].terms) == 1

    def test_build_formula_model_accepts_transformed_parametric_terms(self):
        """Verify that build formula model accepts transformed parametric terms."""
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
            }
        )

        parsed = parse_gam_formula("y ~ I(x**2)")
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=data)

        term = built.predictor_specs[0].terms[0]
        hidden_name = term.features[0]
        expr_state = built.preprocess_state["formula_expression_columns"]

        assert hidden_name in built.working_data.columns
        np.testing.assert_allclose(
            built.working_data[hidden_name].to_numpy(dtype=float),
            data["x"].to_numpy(dtype=float) ** 2,
        )
        assert len(expr_state) == 1
        assert expr_state == [
            {
                "hidden_name": expr_state[0]["hidden_name"],
                "expr": "I(x**2)",
                "source_variables": ["x"],
            }
        ]
        np.testing.assert_allclose(
            built.X.ravel(),
            data["x"].to_numpy(dtype=float) ** 2,
        )

    def test_build_formula_model_accepts_transformed_offsets(self):
        """Verify that build formula model accepts transformed offsets."""
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
                "o": [0.5, 1.5, 2.5],
            }
        )

        parsed = parse_gam_formula("y ~ x + offset(log(o + 1))")
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=data)

        hidden_name = built.preprocess_state["offset_name"]

        assert hidden_name in built.working_data.columns
        np.testing.assert_allclose(
            built.offsets,
            np.log(data["o"].to_numpy(dtype=float) + 1.0),
        )
        assert built.preprocess_state["formula_expression_columns"] == [
            {
                "hidden_name": hidden_name,
                "expr": "log(o + 1)",
                "source_variables": ["o"],
            }
        ]

    def test_build_formula_model_accepts_transformed_responses(self):
        """Verify that build formula model accepts transformed responses."""
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
            }
        )

        parsed = parse_gam_formula("I(y**2) ~ x")
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=data)

        np.testing.assert_allclose(
            built.response,
            data["y"].to_numpy(dtype=float) ** 2,
        )

    def test_build_formula_model_accepts_transformed_smooth_covariates(self):
        """Verify that build formula model accepts transformed smooth covariates."""
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
            }
        )

        parsed = parse_gam_formula('y ~ s(I(x**2), k=5, bs="cr")')
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=data)

        term = built.predictor_specs[0].terms[0]
        hidden_name = term.features[0]

        assert hidden_name in built.working_data.columns
        np.testing.assert_allclose(
            built.working_data[hidden_name].to_numpy(dtype=float),
            data["x"].to_numpy(dtype=float) ** 2,
        )
        assert built.preprocess_state["formula_expression_columns"] == [
            {
                "hidden_name": hidden_name,
                "expr": "I(x**2)",
                "source_variables": ["x"],
            }
        ]

    def test_apply_formula_preprocess_rebuilds_transformed_smooth_covariates(self):
        """
        Verify that apply formula preprocess rebuilds transformed smooth covariates.
        """
        fit_data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
            }
        )
        new_data = pd.DataFrame({"x": [3.0, 4.0]})

        parsed = parse_gam_formula('y ~ s(I(x**2), k=5, bs="cr")')
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=fit_data)
        hidden_name = built.predictor_specs[0].terms[0].features[0]

        rebuilt = apply_formula_preprocess_to_new_data(new_data, built.preprocess_state)

        np.testing.assert_allclose(
            rebuilt[hidden_name].to_numpy(dtype=float),
            new_data["x"].to_numpy(dtype=float) ** 2,
        )

    def test_apply_formula_preprocess_rebuilds_transformed_parametric_terms(self):
        """
        Verify that apply formula preprocess rebuilds transformed parametric terms.
        """
        fit_data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
            }
        )
        new_data = pd.DataFrame({"x": [3.0, 4.0]})

        parsed = parse_gam_formula("y ~ I(x**2)")
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=fit_data)
        hidden_name = built.predictor_specs[0].terms[0].features[0]

        rebuilt = apply_formula_preprocess_to_new_data(
            new_data, built.preprocess_state
        )

        np.testing.assert_allclose(
            rebuilt[hidden_name].to_numpy(dtype=float),
            new_data["x"].to_numpy(dtype=float) ** 2,
        )

    def test_apply_formula_preprocess_rebuilds_transformed_offsets(self):
        """Verify that apply formula preprocess rebuilds transformed offsets."""
        fit_data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
                "o": [0.5, 1.5, 2.5],
            }
        )
        new_data = pd.DataFrame({"x": [3.0, 4.0], "o": [3.5, 4.5]})

        parsed = parse_gam_formula("y ~ x + offset(log(o + 1))")
        extracted = extract_formula_terms(parsed)
        built = build_formula_model(extracted, data=fit_data)
        hidden_name = built.preprocess_state["offset_name"]

        rebuilt = apply_formula_preprocess_to_new_data(
            new_data, built.preprocess_state
        )

        np.testing.assert_allclose(
            rebuilt[hidden_name].to_numpy(dtype=float),
            np.log(new_data["o"].to_numpy(dtype=float) + 1.0),
        )

    @pytest.mark.parametrize(
        ("formula", "data", "expected_features"),
        [
            (
                "y ~ .",
                pd.DataFrame(
                    {
                        "y": [1.0, 2.0, 3.0],
                        "x": [0.0, 1.0, 2.0],
                        "z": [2.0, 1.0, 0.0],
                    }
                ),
                ["x", "z"],
            ),
            (
                "y ~ . - z",
                pd.DataFrame(
                    {
                        "y": [1.0, 2.0, 3.0],
                        "x": [0.0, 1.0, 2.0],
                        "z": [2.0, 1.0, 0.0],
                    }
                ),
                ["x"],
            ),
            (
                "y ~ .",
                pd.DataFrame(
                    {
                        "y": [1.0, 2.0, 3.0],
                        "z": [2.0, 1.0, 0.0],
                        "x": [0.0, 1.0, 2.0],
                    }
                ),
                ["z", "x"],
            ),
        ],
    )
    def test_build_formula_model_expands_dot_with_data_context(
        self, formula, data, expected_features
    ):
        """Verify that build formula model expands dot with data context."""
        parsed = parse_gam_formula(formula)
        extracted = extract_formula_terms(parsed)

        built = build_formula_model(extracted, data=data)

        assert built.feature_names == expected_features
        assert [term.label for term in built.predictor_specs[0].terms] == (
            expected_features
        )

    @pytest.mark.parametrize(
        "formula",
        [
            [
                "y ~ .",
                "~ z",
            ],
            [
                "y ~ x",
                "~ .",
            ],
            [
                "y1 ~ -1",
                "y2 ~ -1",
                "1 + 2 ~ .",
            ],
        ],
    )
    def test_build_formula_model_rejects_dot_in_formula_lists_for_mgcv_parity(
        self, formula
    ):
        """
        Verify that build formula model rejects dot in formula lists for mgcv parity.
        """
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "y1": [1.0, 2.0, 3.0],
                "y2": [2.0, 3.0, 4.0],
                "x": [0.0, 1.0, 2.0],
                "z": [2.0, 1.0, 0.0],
                "w": [3.0, 4.0, 5.0],
            }
        )
        parsed = parse_gam_formula(formula)

        with pytest.raises(
            NotImplementedError,
            match=(
                "Data-aware '\\.' shorthand is unsupported for formula-list / "
                "multi-predictor models"
            ),
        ):
            extracted = extract_formula_terms(parsed)
            build_formula_model(extracted, data=data)

    @pytest.mark.parametrize(
        ("formula", "error_type", "match"),
        [
            ("y ~ s(.)", NotImplementedError, r"s\(\.\) not supported"),
            ("y ~ . + s(.)", NotImplementedError, r"s\(\.\) not supported"),
            ("y ~ s(x, by=.)", ValueError, r"by=\. not allowed"),
            (
                "y ~ s(x, x)",
                ValueError,
                "Repeated variables as arguments of a smooth are not permitted",
            ),
        ],
    )
    def test_build_formula_model_rejects_mgcv_smooth_spec_errors(
        self, formula, error_type, match
    ):
        """
        Verify that formula build rejects smooth specs that upstream mgcv rejects.
        """
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "x": [0.0, 1.0, 2.0, 3.0],
                "z": [3.0, 2.0, 1.0, 0.0],
            }
        )
        parsed = parse_gam_formula(formula)
        extracted = extract_formula_terms(parsed)

        with pytest.raises(error_type, match=match):
            build_formula_model(extracted, data=data)
