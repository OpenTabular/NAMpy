"""Keep example presentation readable without weakening the test suite."""

import ast
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/_ext"))


def test_examples_drop_test_statements_and_preserve_python_suites():
    from example_source import without_assertions

    source = """# Keep this teaching comment.
x = 2
assert (
    x == 2
), "expected two"
if x:
    assert x > 0
np.testing.assert_allclose(x, 2)
result = x
"""
    cleaned = without_assertions(source)
    assert not any(
        isinstance(node, ast.Assert) for node in ast.walk(ast.parse(cleaned))
    )
    assert "testing.assert_" not in cleaned
    assert "# Keep this teaching comment." in cleaned
    namespace = {}
    exec(cleaned, namespace)
    assert namespace["result"] == 2


def test_validation_report_retains_the_failure_behavior():
    from example_source import without_assertions

    cleaned = without_assertions(
        'assert correct, "invalid result"\n', validation_report=True
    )
    exec(cleaned, {"correct": True})
    with pytest.raises(ValueError, match="invalid result"):
        exec(cleaned, {"correct": False})


def rendered(preview, schemas):
    from example_results import result_rst

    source = result_rst(preview, schemas)
    return source, BeautifulSoup(source.split(".. raw:: html\n\n", 1)[1], "html.parser")


def test_scalar_and_matrix_results_have_labels_shapes_and_omission_counts():
    source, soup = rendered(
        {
            "loss": 1.5931084156036377,
            "parameters": {
                "kind": "array",
                "shape": [8, 2],
                "columns": ["mu", "sigma"],
                "values": [[0.0, 0.69315], [0.0, 0.69315]],
                "omitted_by_axis": [6, 0],
            },
            "raw": {
                "kind": "array",
                "shape": [8, 2],
                "values": [[0.0, 0.0]],
                "omitted_by_axis": [7, 0],
            },
        },
        [
            ["result['loss']", "torch.Tensor[torch.float32]", "()"],
            ["result['parameters']", "torch.Tensor[torch.float32]", "(8, 2)"],
        ],
    )
    assert not soup.find_all(["table", "pre"])
    assert ".. code-block::" not in source
    assert "Example output types and shapes" not in source
    assert soup.select_one(".example-result-scalar").get_text() == "1.5931"
    assert "Tensor[float32] · scalar" in soup.get_text()
    assert "8 × 2" in soup.get_text()
    assert "Showing 2 of 8 rows." in soup.get_text()
    assert [node.get_text() for node in soup.select(".example-result-coordinate dt")][
        :2
    ] == ["mu", "sigma"]
    assert soup.select_one("details > summary").get_text().startswith("raw")
    assert not soup.select_one("details").has_attr("open")


def test_nested_samples_keep_truncation_and_escape_user_values():
    _, soup = rendered(
        {
            "kind": "list",
            "length": 8,
            "items": [
                {
                    "kind": "list",
                    "length": 3,
                    "items": [
                        {"term": "<script>bad</script>", "kind": "main"},
                        {
                            "term": {
                                "kind": "tuple",
                                "length": 1,
                                "items": ["nested_feature"],
                                "omitted": 0,
                            },
                            "value": 0.2,
                        },
                    ],
                    "omitted": 1,
                }
            ],
            "omitted": 7,
        },
        [["result", "list", "length 8"]],
    )
    assert "Showing 1 of 8 items." in soup.get_text()
    assert "Showing 2 of 3 items." in soup.get_text()
    assert "main" in soup.get_text()
    assert "nested_feature" in soup.get_text()
    assert not soup.find("script")


def test_covariance_values_remain_omitted():
    _, soup = rendered(
        {"kind": "array", "shape": [12, 12], "values": "omitted; see shape contract"},
        [["result", "numpy.ndarray[float64]", "(12, 12)"]],
    )
    assert "12 × 12" in soup.get_text()
    assert "Values omitted" in soup.get_text()


@pytest.mark.parametrize(
    "page, width, names",
    [
        ("statistical_reference/families/student_lss", 2, ["mu", "sigma"]),
        (
            "neural_reference/families/mvnormdiag",
            6,
            ["loc_0", "loc_1", "loc_2", "scale_0", "scale_1", "scale_2"],
        ),
        ("neural_reference/families/ordinal", 3, ["p0", "p1", "p2"]),
    ],
)
def test_family_preview_columns_match_parameter_contract(tmp_path, page, width, names):
    import json

    from reference_examples import example_rst

    directory = tmp_path / "_examples/reference"
    directory.mkdir(parents=True)
    (directory / "output_previews.json").write_text(
        json.dumps(
            {
                page: {
                    "parameters": {
                        "kind": "array",
                        "shape": [8, width],
                        "values": [[0.5] * width],
                        "omitted_by_axis": [7, 0],
                    },
                }
            }
        )
    )
    (directory / "output_schemas.json").write_text("{}")
    source = example_rst(tmp_path, page, "result = None")
    soup = BeautifulSoup(source.split(".. raw:: html\n\n", 1)[1], "html.parser")
    assert [
        node.get_text() for node in soup.select(".example-result-coordinate dt")
    ] == names
