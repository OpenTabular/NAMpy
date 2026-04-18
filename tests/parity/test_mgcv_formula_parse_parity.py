"""Parser-layer parity checks against mgcv::interpret.gam."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nampy.gam.formula import (
    ParsedParametricTerm,
    ParsedSmoothTerm,
    extract_formula_terms,
    parse_gam_formula,
)
from nampy.gam.specs.build import build_formula_model
from tests._paths import PARITY_DIR, REPO_ROOT

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
    if R_SCRIPT is None:
        pytest.skip("Rscript is required for mgcv parser parity tests.")

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
        return json.loads(output_path.read_text(encoding="utf-8"))


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
    def test_single_formula_parser_matches_mgcv_interpret_gam(self):
        formula = 'y ~ x * z - z + s(w, k=6, bs="cr", by=f, id="g") + offset(o) - 1'
        _assert_parser_parity(formula)

    def test_formula_list_parser_matches_mgcv_interpret_gam(self):
        formula = [
            'y ~ x + s(z, k=7, bs="cr")',
            "~ 1",
            '1 + 2 ~ te(a, b, bs=["cr", "tp"], k=[5, 6], id="shared")',
        ]
        _assert_parser_parity(formula)

    def test_multi_response_formula_list_matches_mgcv_interpret_gam(self):
        formula = [
            "y1 ~ -1",
            "y2 ~ -1",
            '1 + 2 ~ s(x, k=5, bs="cr")',
        ]
        _assert_parser_parity(formula)

    def test_formula_operator_parity_matches_mgcv_interpret_gam(self):
        formula = "y ~ (a + b)/c + z:x + x + offset(o1) - offset(o2)"
        _assert_parser_parity(formula)

    def test_transformed_factor_and_power_parser_matches_mgcv(self):
        formula = 'y ~ (0 + x)**2 + I(z**2) + s(I(w**2), k=5, bs="cr")'
        _assert_parser_parity(formula)

    def test_extract_formula_terms_expands_shared_component_by_linear_predictor(self):
        parsed = parse_gam_formula(
            [
                "y1 ~ -1",
                "y2 ~ -1",
                '1 + 2 ~ s(x, k=5, bs="cr")',
            ]
        )

        extracted = extract_formula_terms(parsed)

        assert len(extracted) == 2
        assert [pred.response_name for pred in extracted] == ["y1", "y2"]
        assert [pred.intercept for pred in extracted] == [False, False]
        assert [len(pred.terms) for pred in extracted] == [1, 1]

        first_term = extracted[0].terms[0]
        second_term = extracted[1].terms[0]

        assert first_term.kind == "s"
        assert second_term.kind == "s"
        assert first_term.features == ("x",)
        assert second_term.features == ("x",)
        assert first_term.raw_label == 's(x, k=5, bs="cr")'
        assert second_term.raw_label == 's(x, k=5, bs="cr")'

    @pytest.mark.parametrize(
        ("formula", "message"),
        [
            (
                "y ~ I(x**2)",
                "Transformed parametric formula terms are parsed exactly",
            ),
            (
                'y ~ s(I(x**2), k=5, bs="cr")',
                "Transformed smooth covariates are parsed exactly",
            ),
            (
                "y ~ x + offset(log(o + 1))",
                "Transformed offset\\(\\.\\.\\.\\) expressions are parsed exactly",
            ),
            (
                "I(y**2) ~ x",
                "Transformed formula responses are parsed exactly",
            ),
        ],
    )
    def test_build_formula_model_raises_not_implemented_for_transforms(
        self, formula, message
    ):
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0],
                "o": [0.5, 1.5, 2.5],
            }
        )

        parsed = parse_gam_formula(formula)
        extracted = extract_formula_terms(parsed)

        with pytest.raises(NotImplementedError, match=message):
            build_formula_model(extracted, data=data)
