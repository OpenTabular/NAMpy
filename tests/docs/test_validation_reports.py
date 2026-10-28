"""Exercise report decisions with deliberately invalid future model outputs."""

import ast
import warnings
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def report_functions():
    """Load the actual notebook functions without training its models."""
    from nampy.neural.distributions.registry import FAMILY_REGISTRY
    from nampy.neural.registry import architectures

    notebook = nbformat.read(
        ROOT / "docs/development/notebooks/04_distributional_models.ipynb", 4
    )
    namespace = {
        "np": np,
        "pd": pd,
        "warnings": warnings,
        "warning_rows": [],
        "n": 4,
        "FAMILY_REGISTRY": FAMILY_REGISTRY,
        "architectures": architectures,
    }
    names = {
        "outcome",
        "array_checks",
        "family_checks",
        "task_status",
        "passed_evidence",
        "checked_fit",
    }
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(compile(tree, "validation-notebook", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "family,parameters,target",
    [
        ("normal", [[0.0, -1.0]] * 4, np.zeros(4)),
        ("beta", [[1.0, 0.0]] * 4, np.full(4, 0.5)),
        ("zip", [[1.2, 1.0]] * 4, np.zeros(4)),
        ("categorical", [[0.2, 0.2, 0.2]] * 4, np.array([0, 1, 2, 0])),
        ("quantile", [[1.0, 0.0, 2.0]] * 4, np.zeros(4)),
        ("normal", [[float("nan"), 1.0]] * 4, np.zeros(4)),
    ],
)
def test_invalid_domain_nonfinite_or_unnormalized_outputs_fail(
    family, parameters, target
):
    ns = report_functions()
    natural = np.asarray(parameters)
    checks = ns["family_checks"](family, np.zeros_like(natural), natural, target)
    assert ns["outcome"](checks) == "Failed"


def test_wrong_shapes_and_nonfinite_loss_cannot_pass():
    ns = report_functions()
    assert ns["outcome"](ns["array_checks"](np.ones((4, 1)), (4, 2))) == "Failed"
    assert (
        ns["outcome"]({"finite output": True, "finite loss": np.isfinite(np.inf)})
        == "Failed"
    )
    assert ns["outcome"]({"finite output": True, "reconstruction": False}) == "Failed"
    assert ns["outcome"]({}) == "Failed"


def test_evidence_requires_passed_outcome():
    ns = report_functions()
    for name in ("regression_report", "classification_report", "lss_report"):
        ns[name] = pd.DataFrame([{"architecture": "nam", "status": "Failed"}])
    ns["family_report"] = pd.DataFrame([{"family": "quantile", "status": "Failed"}])
    ns["quantile_status"] = "Failed"
    assert ns["task_status"]("nam", "regression", ns["regression_report"]) == "Failed"
    assert ns["passed_evidence"]("nam") == ""
    ns["regression_report"].loc[0, "status"] = "Passed"
    assert ns["passed_evidence"]("nam") == "R"
    ns["family_report"].loc[0, "status"] = "Passed"
    assert ns["passed_evidence"]("nam") == "R, F"


def test_warning_capture_retains_category_message_and_context_on_failure():
    ns = report_functions()

    class FailingEstimator:
        def fit(self):
            warnings.warn("numerical failure", RuntimeWarning, stacklevel=2)
            raise ValueError("fit failed")

    with pytest.raises(ValueError, match="fit failed"):
        ns["checked_fit"](FailingEstimator(), label="model/task")
    assert ns["warning_rows"] == [
        {
            "model/task": "model/task",
            "category": "RuntimeWarning",
            "message": "numerical failure",
            "expected": False,
        }
    ]
