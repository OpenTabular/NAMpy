"""Inspection, architecture-specific outputs and persistence of neural models."""

from __future__ import annotations

from reference_examples import example_rst, neural_estimator_example
from reference_schema import method_parameter_rows

METHODS = {
    "center_components": (
        "NAMRegressor",
        "AdditivePrediction",
        "same n-row link/response shape",
        "Centered link-scale terms, with reference means transferred into the intercept.",
    ),
    "explain_terms": (
        "NAMRegressor",
        "pandas.DataFrame",
        "one row per retained term/output/bin",
        "Term identity, feature value, mean contribution, count and importance; the example lists columns and dtypes.",
    ),
    "term_importance": (
        "NAMRegressor",
        "pandas.DataFrame",
        "(n_term_outputs, 4)",
        "Columns term, term_type, output, importance, sorted by decreasing mean absolute link contribution.",
    ),
    "interaction_importance": (
        "NAMRegressor",
        "pandas.DataFrame",
        "(n_interaction_outputs, 4)",
        "Interaction-only subset of the term-importance table; an empty table is valid without interactions.",
    ),
    "plot_terms": (
        "NAMRegressor",
        "list[matplotlib.figure.Figure]",
        "one figure per requested page",
        "Numeric single-column term curves; interaction effects use plot_interactions.",
    ),
    "plot": (
        "NAMRegressor",
        "None",
        "no array",
        "Legacy plotting side effect with target scatter/density display; use plot_terms for returned figures.",
    ),
    "plot_interactions": (
        "NAMRegressor",
        "None",
        "no array",
        "Draws pairwise heatmaps or higher-order observed slices for fitted named interactions.",
    ),
    "basis_transform": (
        "GPNAMRegressor",
        "numpy.ndarray[floating]",
        "(n, p_basis)",
        "Architecture fixed-design feature matrix. Supported only when the fitted architecture exposes linear_design; other architectures raise NotImplementedError.",
    ),
    "basis_metadata": (
        "GPNAMRegressor",
        "dict[str, object]",
        "named design-block metadata",
        "Describes the fixed-basis blocks and their fitted dimensions. Unsupported architectures raise NotImplementedError.",
    ),
    "local_term_importance": (
        "SPAMRegressor",
        "list[list[dict[str, object]]]",
        "outer length n; up to top_k dictionaries per observation",
        "Architecture-specific polynomial term expansions for the requested output and top-k budget. Implemented by SPAM; unsupported architectures raise NotImplementedError.",
    ),
    "group_norms": (
        "SNAMRegressor",
        "dict[str, float]",
        "one scalar per reported group",
        "Fitted sparse-term parameter-group norms; supported only by architectures exposing a group-norm hook.",
    ),
    "training_history": (
        "IGANNRegressor",
        "dict[str, object]",
        "one value/series per recorded native metric",
        "Native solver training history, not a universal Lightning log API. Unsupported architectures raise NotImplementedError.",
    ),
    "model_complexity": (
        "NAMRegressor",
        "dict[str, int]",
        "named scalar counts",
        "At least total_parameters and trainable_parameters; architectures can add their own complexity counts.",
    ),
    "save_model": (
        "NAMRegressor",
        "pathlib.Path",
        "scalar path object",
        "Destination path after writing the versioned fitted estimator artifact.",
    ),
    "load_model": (
        "NAMRegressor",
        "estimator instance",
        "scalar object",
        "Restored fitted estimator; prediction schema and learned preprocessing are retained.",
    ),
    "get_params": (
        "NAMRegressor",
        "dict[str, object]",
        "one entry per exposed constructor option",
        "Resolved sklearn-style configuration mapping. deep controls nested parameter expansion where supported.",
    ),
    "set_params": (
        "NAMRegressor",
        "estimator instance",
        "scalar object",
        "Returns self after updating valid configuration parameters; does not execute another fit.",
    ),
}
PARAMETERS = {
    "target": "Zero-based model output index to expand; scalar regression uses zero.",
    "top_k": "Positive maximum number of polynomial terms retained per observation.",
    "deep": "Whether to include supported nested constructor parameters.",
    "parameters": "Constructor-option keyword updates drawn from get_params(); unknown names are rejected.",
    "plot_interactions": "Whether the legacy plot call should also draw fitted interactions.",
}


OPTIONAL_HOOKS = {
    "basis_transform": "linear_design",
    "basis_metadata": "basis_metadata",
    "local_term_importance": "local_term_importance",
    "group_norms": "get_group_norms",
    "training_history": "training_history_",
}


def supported_architectures(method):
    from nampy.neural.registry import architectures

    hook = OPTIONAL_HOOKS[method]
    return [
        name for name, spec in architectures().items() if hasattr(spec.module, hook)
    ]


def generate(root):
    from feature_reference import heading, signature, table, tree, write

    from nampy.models._base import NeuralEstimatorBase

    for name, (cls, kind, shape, meaning) in METHODS.items():
        method = getattr(NeuralEstimatorBase, name)
        text = heading(name) + meaning + "\n\n" + signature(method, "estimator." + name)
        if name in OPTIONAL_HOOKS:
            supported = supported_architectures(name)
            text += (
                "Available architecture hooks: "
                + ", ".join(
                    f":doc:`{architecture} <../{architecture}>`"
                    for architecture in supported
                )
                + ". Unsupported architectures raise ``NotImplementedError``; call after fitting. Native history and metadata retain their native-solver meaning.\n\n"
            )
        rows = method_parameter_rows(method, PARAMETERS)
        text += heading("Parameters", "-") + (
            table(["Parameter", "Type / shape", "Default", "Meaning"], rows)
            if rows
            else "No input parameters.\n\n"
        )
        text += heading("Returns", "-") + table(
            ["Type", "Shape", "Meaning"], [(kind, shape, meaning)]
        )
        if name == "local_term_importance":
            text += "Each inner dictionary contains ``term`` (source-feature tuple), ``order`` (integer interaction degree), and ``contribution`` (float for the selected output). Terms are ranked by absolute contribution within each observation; these are local polynomial terms, not global feature-importance scores.\n\n"
        code = neural_estimator_example(cls)
        if name in {"plot_interactions", "interaction_importance"}:
            code = code.replace("interactions=[]", "interactions=[('x', 'z')]")
        calls = {
            "center_components": "model.center_components(X)",
            "explain_terms": "model.explain_terms(X, max_bins=8)",
            "term_importance": "model.term_importance(X)",
            "interaction_importance": "model.interaction_importance(X)",
            "plot_terms": "model.plot_terms(X, pages=1)",
            "plot": "model.plot(X, y)",
            "plot_interactions": "model.plot_interactions(X)",
            "basis_transform": "model.basis_transform(X)",
            "basis_metadata": "model.basis_metadata()",
            "local_term_importance": "model.local_term_importance(X, target=0, top_k=3)",
            "group_norms": "model.group_norms()",
            "training_history": "model.training_history()",
            "model_complexity": "model.model_complexity()",
            "get_params": "model.get_params(deep=False)",
            "set_params": "model.set_params(lr=0.001)",
        }
        if name in {"save_model", "load_model"}:
            code += """with TemporaryDirectory() as directory:
    path = model.save_model(directory + '/fitted.nampy')
    restored = type(model).load_model(path)
    np.testing.assert_allclose(restored.predict(X), model.predict(X))
"""
            code += f"    result = {'path' if name == 'save_model' else 'restored'}\n"
        else:
            code += "result = " + calls[name] + "\n"
        code += "print(type(result).__name__)\n"
        text += example_rst(root, f"neural_reference/methods/{name}", code)
        text += f".. automethod:: nampy.models._base.NeuralEstimatorBase.{name}\n   :no-index:\n\n"
        text += ":doc:`Model-specific prediction signatures <../architectures/index>` · :doc:`Shared output records </api/contracts>`\n"
        write(root, f"neural_reference/methods/{name}.rst", text)
    write(
        root,
        "neural_reference/methods/index.rst",
        tree(
            "Inspection, plots and persistence",
            list(METHODS),
            "Prediction/probability and distribution-parameter signatures are documented on each estimator page. These shared methods inspect fitted contributions, optional architecture outputs and persistence artifacts. Optional capabilities fail explicitly on unsupported architectures.",
        ),
    )
