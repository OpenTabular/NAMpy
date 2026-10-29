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
    from nampy.neural.families.registry import FAMILY_REGISTRY
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
        "family_kwargs",
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


def test_validation_report_architecture_options_cover_live_registry():
    from nampy import models
    from nampy.neural.configuration import (
        FixedDesignTrainingConfig,
        GradientTrainingConfig,
        IGANNTrainingConfig,
        ObjectiveConfig,
        SemiStructuredTrainingConfig,
    )
    from nampy.neural.registry import architectures

    notebook = nbformat.read(
        ROOT / "docs/development/notebooks/04_distributional_models.ipynb", 4
    )
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for statement in ast.parse(cell.source).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ARCHITECTURE_OPTIONS"
                for target in statement.targets
            ):
                options = ast.literal_eval(statement.value)
                assert set(options) == set(architectures())
                for name, spec in architectures().items():
                    spec.config(**options[name])
                namespace = {
                    "models": models,
                    "ARCHITECTURE_OPTIONS": options,
                    "SWEEP_TRAINING": GradientTrainingConfig(max_epochs=1),
                    "MAX_EPOCHS": 1,
                    "RANDOM_STATE": 7,
                    "FixedDesignTrainingConfig": FixedDesignTrainingConfig,
                    "IGANNTrainingConfig": IGANNTrainingConfig,
                    "SemiStructuredTrainingConfig": SemiStructuredTrainingConfig,
                    "ObjectiveConfig": ObjectiveConfig,
                }
                tree = ast.parse(cell.source)
                tree.body = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name in {"estimator_class", "make_estimator"}
                ]
                exec(compile(tree, "architecture-report-setup", "exec"), namespace)
                for suffix in ("Regressor", "Classifier", "LSS"):
                    estimator = namespace["make_estimator"](
                        "semistructured", architectures()["semistructured"], suffix
                    )
                    assert isinstance(estimator.training, SemiStructuredTrainingConfig)
                return
    pytest.fail("Validation report must declare its architecture options.")


def test_validation_report_targets_and_domains_cover_live_family_registry():
    import torch

    from nampy.neural.families.registry import FAMILY_REGISTRY

    notebook = nbformat.read(
        ROOT / "docs/development/notebooks/04_distributional_models.ipynb", 4
    )
    cell = next(
        c
        for c in notebook.cells
        if c.cell_type == "code" and "family_targets =" in c.source
    )
    namespace = report_functions()
    namespace.update(
        n=12,
        x=np.linspace(0.03, 0.97, 12),
        y_lss=np.linspace(-1, 1, 12),
        rng=np.random.default_rng(7),
    )
    exec(
        compile(
            cell.source.split("family_models = {}", 1)[0], "family-report-setup", "exec"
        ),
        namespace,
    )
    assert set(namespace["family_targets"]) == set(FAMILY_REGISTRY)
    for name, spec in FAMILY_REGISTRY.items():
        target = namespace["family_targets"][name]
        objective, _ = spec.instantiate(target, namespace["family_kwargs"](name))
        raw = torch.zeros(12, objective.param_count)
        natural = objective(raw).detach().numpy()
        checks = namespace["family_checks"](name, raw.numpy(), natural, target)
        assert namespace["outcome"](checks) == "Passed", (name, checks)


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
