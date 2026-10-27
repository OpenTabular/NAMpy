"""Detailed contracts for shared prediction objects and plotting primitives."""

from __future__ import annotations

import inspect

from reference_examples import BASE, MODEL, example_rst
from reference_schema import METHOD_PARAMETER_MEANINGS, parameter_type

SYMBOLS = {
    "nampy.contracts": [
        "FeatureSchema",
        "AdditiveTermMetadata",
        "AdditivePrediction",
        "EnsembleAdditivePrediction",
    ],
    "nampy.explanations": [
        "center_additive_prediction",
        "explain_additive_prediction",
        "term_importance_table",
    ],
    "nampy.plotting": [
        "DistributionalEffects",
        "add_density_shading",
        "combine_distributional_effects",
        "plot_distributional_effects",
        "prepare_distributional_effects",
        "prepared_from_contributions",
        "render_distributional_effects",
        "render_term_plots",
        "render_model_check",
        "render_qq",
        "render_prediction_surface",
    ],
}
MEANINGS = {
    "feature_names": "Ordered source column names. FeatureSchema records d names; distributional rendering uses a subset of the fitted feature keys.",
    "dtypes": "One dtype name per feature, length d, in the same order as feature_names.",
    "features": "Source features on which a term depends; empty tuple means no declared source feature.",
    "parameter": "Distribution parameter identified by this term, or None for a single-output/non-parameter-specific contribution.",
    "kind": "Dependence kind: main, interaction, global or contextual. Reconstruction alone does not prove unary dependence.",
    "response": "Response-scale predictions, shape (n,) or (n,q). Must have the same shape as link in AdditivePrediction.",
    "link": "Additive predictor values, shape (n,) or (n,q). Contributions plus intercept and offset reconstruct this scale.",
    "terms": "Mapping from unique term labels to contribution arrays with n rows. A scalar contribution may use (n,), (n,1), or compatible output width (n,q).",
    "intercept": "Scalar or broadcast-compatible fitted intercept: (), (q,), (n,) or (n,q), subject to the response shape.",
    "backend": "Backend label: gam or neural.",
    "offset": "Optional known predictor offsets, broadcast-compatible with link. None omits the offset.",
    "term_metadata": "Mapping from term label to AdditiveTermMetadata, describing source dependence and distribution-parameter identity.",
    "prediction": "AdditivePrediction whose contributions are summarized or centered; no estimator is refitted.",
    "reference": "Optional AdditivePrediction for the reference population. Its term set and trailing output dimensions must match prediction.",
    "mean": "Mean AdditivePrediction across independently fitted ensemble members.",
    "response_std": "Between-member response standard deviations, matching mean.response shape.",
    "link_std": "Between-member link standard deviations, matching mean.link shape.",
    "term_std": "Between-member standard-deviation arrays per term, aligned with mean.terms.",
    "intercept_std": "Between-member intercept standard deviation, scalar or broadcast-compatible with the mean intercept.",
    "n_estimators": "Positive number of independent ensemble members represented by the result.",
    "feature_values": "Mapping from feature name to observed feature values, one row per effect observation.",
    "point_effects": "Mapping from feature name to point-functional effect curves, shape (n_feature,) or (n_feature,n_members).",
    "conditional_sd": "Mapping from feature name to conditional response SD values, matching the corresponding point-effect curve dimensions.",
    "ax": "Matplotlib Axes modified in place by density shading.",
    "values": "Observed one-dimensional numerical feature values used to calculate density bins.",
    "bins": "Positive number of density bins.",
    "color": "Matplotlib color specification for density bands or the requested surface palette.",
    "min_alpha": "Minimum opacity for nonempty density bins, between zero and one.",
    "max_alpha": "Maximum opacity for the most populated density bin; must be compatible with min_alpha.",
    "label": "Optional legend label for the density shading.",
    "effects": "Prepared DistributionalEffects object; combine_distributional_effects takes a nonempty sequence of compatible objects and stacks member curve columns.",
    "estimator": "Fitted neural distributional estimator providing centered predict_components and its neural family point/spread hooks. Statistical GAMLSS does not currently implement this complete renderer protocol.",
    "kwargs": "Additional preparation/rendering keyword arguments. The combined plotting entry point separates the supported preparation options from renderer options.",
    "frame": "DataFrame containing source features, with n rows matching the contributions.",
    "ylab_format": "Format string for a term axis label, interpolating the term name through {name}.",
    "term_features": "Optional mapping from contribution label to its source-feature tuple; used when labels alone do not identify the source columns.",
    "prepared": "Mapping of prepared numerical plot data from prepared_from_contributions, prepare_plot_gam_data or prepare_vis_gam_data, according to the selected renderer.",
    "data": "Prepared numerical QQ/model-check payload, rather than raw feature data. Use the corresponding diagnostic preparation function.",
    "display_names": "Optional mapping from source feature names to display labels.",
    "effect_bins": "Positive number of numerical bins used to summarize displayed effects.",
    "quantiles": "Lower/upper between-member quantiles for an ensemble display; a length-two tuple of probabilities in [0,1].",
    "ncols": "Positive number of subplot columns.",
    "point_label": "Legend text for point-functional effects.",
    "envelope_label": "Legend text for the conditional-spread envelope.",
    "fit_label": "Legend text for fitted/bin-averaged curves.",
    "ylabel": "Vertical-axis label.",
    "title": "Optional figure title.",
    "xlab": "Optional horizontal-axis label.",
    "ylab": "Optional second-axis label.",
    "zlab": "Optional prediction-axis label.",
    "main": "Optional plot title.",
    "plot_type": "Surface style: persp or contour.",
    "col": "Optional explicit surface colors.",
    "contour_col": "Optional contour-line color.",
    "n_col": "Positive palette size.",
    "theta": "Horizontal perspective viewing angle in degrees.",
    "phi": "Vertical perspective viewing angle in degrees.",
    "shade_col": "Matplotlib color for uncertainty shading.",
    "rug": "Whether to draw rug marks; None uses the prepared-data default.",
    "density": "Whether to draw feature-density shading; None uses the prepared-data default.",
    "density_bins": "Positive bin count; None uses the prepared-data default.",
    "density_color": "Matplotlib density-band color; None uses the prepared-data default.",
    "pages": "Panel pagination request. Zero delegates to the renderer default.",
    **METHOD_PARAMETER_MEANINGS,
}
MEANINGS["data"] = (
    "Prepared numerical QQ/model-check payload from the corresponding diagnostic preparation function, not a raw feature frame."
)
TYPES = {
    "dtypes": "tuple[str, ...]",
    "features": "tuple[str, ...]",
    "parameter": "str | None",
    "kind": "Literal['main', 'interaction', 'global', 'contextual']",
    "response": "numpy.ndarray, shape (n,) or (n,q)",
    "link": "numpy.ndarray, shape (n,) or (n,q)",
    "terms": "Mapping[str, numpy.ndarray]",
    "intercept": "float | numpy.ndarray",
    "backend": "Literal['gam', 'neural']",
    "term_metadata": "dict[str, AdditiveTermMetadata]",
    "prediction": "AdditivePrediction",
    "reference": "AdditivePrediction | None",
    "mean": "AdditivePrediction",
    "response_std": "numpy.ndarray",
    "link_std": "numpy.ndarray",
    "term_std": "dict[str, numpy.ndarray]",
    "intercept_std": "float | numpy.ndarray",
    "feature_values": "Mapping[str, numpy.ndarray]",
    "point_effects": "Mapping[str, numpy.ndarray]",
    "conditional_sd": "Mapping[str, numpy.ndarray]",
    "ax": "matplotlib.axes.Axes",
    "values": "array-like, shape (n,)",
    "effects": "DistributionalEffects | Sequence[DistributionalEffects]",
    "estimator": "fitted estimator",
    "frame": "pandas.DataFrame",
    "term_features": "Mapping[str, tuple[str,...]] | None",
    "prepared": "Mapping[str, object]",
    "data": "Mapping[str, object]",
    "display_names": "Mapping[str, str] | None",
    "density_bins": "int | None",
    "density_color": "Matplotlib color | None",
    "density": "bool | None",
    "title": "str | None",
    "quantiles": "tuple[float, float]",
}
CONTRACT_SETUP = """import numpy as np
import pandas as pd
from nampy.contracts import AdditivePrediction, AdditiveTermMetadata, EnsembleAdditivePrediction, FeatureSchema
from nampy.explanations import center_additive_prediction, explain_additive_prediction, term_importance_table

n = 12
X = pd.DataFrame({'x': np.linspace(0, 1, n)})
term = np.sin(3 * X.x.to_numpy())
link = 0.5 + term
prediction = AdditivePrediction(response=link.copy(), link=link,
                               terms={'x': term}, intercept=0.5, backend='gam',
                               term_metadata={'x': AdditiveTermMetadata(features=('x',))})
prediction.validate_additive_reconstruction()
"""
DIST_SETUP = """import numpy as np
from nampy.plotting import DistributionalEffects

n = 24
x = np.linspace(0, 1, n)
effects = DistributionalEffects(feature_values={'x': x},
                               point_effects={'x': np.sin(3*x)},
                               conditional_sd={'x': 0.1 + 0.2*x})
"""


def example(module, name):
    if module in {"nampy.contracts", "nampy.explanations"}:
        calls = {
            "FeatureSchema": "FeatureSchema.from_data(X)",
            "AdditiveTermMetadata": "AdditiveTermMetadata(features=('x',), kind='main')",
            "AdditivePrediction": "prediction",
            "EnsembleAdditivePrediction": "EnsembleAdditivePrediction(mean=prediction, response_std=np.zeros(n), link_std=np.zeros(n), term_std={'x': np.zeros(n)}, intercept_std=0., n_estimators=2)",
            "center_additive_prediction": "center_additive_prediction(prediction)",
            "explain_additive_prediction": "explain_additive_prediction(X, prediction, max_bins=6)",
            "term_importance_table": "term_importance_table(prediction)",
        }
        return (
            CONTRACT_SETUP + f"result = {calls[name]}\nprint(type(result).__name__)\n"
        )
    if name in {
        "DistributionalEffects",
        "combine_distributional_effects",
        "render_distributional_effects",
    }:
        call = {
            "DistributionalEffects": "effects",
            "combine_distributional_effects": "combine_distributional_effects([effects, effects])",
            "render_distributional_effects": "render_distributional_effects(effects, density_bins=8)",
        }[name]
        return (
            DIST_SETUP
            + f"from nampy.plotting import {name}\nresult = {call}\nprint(type(result).__name__)\n"
        )
    if name == "add_density_shading":
        return (
            DIST_SETUP
            + """import matplotlib.pyplot as plt
from nampy.plotting import add_density_shading

figure, ax = plt.subplots()
ax.plot(x, np.sin(3*x))
result = add_density_shading(ax, x, bins=8)
assert result is None
"""
        )
    if name in {"prepare_distributional_effects", "plot_distributional_effects"}:
        from reference_examples import neural_estimator_example

        return (
            neural_estimator_example("NAMLSS")
            + f"from nampy.plotting import {name}\nresult = {name}(model, X)\nprint(type(result).__name__)\n"
        )
    if name in {"prepared_from_contributions", "render_term_plots"}:
        code = (
            CONTRACT_SETUP
            + "from nampy.plotting import prepared_from_contributions, render_term_plots\nprepared = prepared_from_contributions(X, prediction.terms)\n"
        )
        return (
            code
            + f"result = {'prepared' if name == 'prepared_from_contributions' else 'render_term_plots(prepared)'}\nprint(type(result).__name__)\n"
        )
    prepares = {
        "render_qq": ("prepare_qq_gam_data", "seed=7"),
        "render_model_check": ("prepare_gam_check_data", "seed=7"),
        "render_prediction_surface": (
            "prepare_vis_gam_data",
            "view=['x','z'], n_grid=8",
        ),
    }
    prepare, args = prepares[name]
    return (
        BASE
        + MODEL
        + f"from nampy.gam.diagnostics import {prepare}\nfrom nampy.plotting import {name}\n\nprepared = {prepare}(model, {args})\nresult = {name}(prepared)\nprint(len(result))\n"
    )


def return_description(module, name):
    if name in {
        "FeatureSchema",
        "AdditiveTermMetadata",
        "AdditivePrediction",
        "EnsembleAdditivePrediction",
        "DistributionalEffects",
    }:
        return f"``{name}`` record. Its fields retain the types and dimensions listed in the parameter table; construction validates the declared shape contract."
    if name == "center_additive_prediction":
        return "``AdditivePrediction`` with the same response/link shape as the input; only term centering and intercept allocation change."
    if name == "explain_additive_prediction":
        return "``pandas.DataFrame`` with one row per retained feature/term/output/bin group. Columns include term identity, term type, output index, feature value, contribution, count and importance; exact columns are listed by the executable example below."
    if name == "term_importance_table":
        return "``pandas.DataFrame``, one row per term/output combination, with columns ``term`` (str), ``term_type`` (str), ``output`` (integer) and ``importance`` (float). Sorted by decreasing importance."
    if name == "add_density_shading":
        return "``None``. Adds density-band artists to the supplied Matplotlib Axes in place; it creates no prediction array."
    if name in {"combine_distributional_effects", "prepare_distributional_effects"}:
        return "``DistributionalEffects``. Per-feature point/spread arrays use one row per feature observation and one curve column per represented member. The feature-value arrays preserve the corresponding row order."
    if name == "prepared_from_contributions":
        return "``dict[str, object]`` with ``pd`` (list of term plotting payloads), transformation/shift/selection settings and rendering defaults. Each effect and raw-value array has one entry per corresponding feature observation."
    return "``list[matplotlib.figure.Figure]``, one figure per requested page/panel arrangement. Renderers consume prepared values and do not refit a model; conditional-SD envelopes and coefficient-SE bands retain their separate meanings."


def generate(root):
    import importlib

    from feature_reference import heading, write
    from public_reference import descriptions, literal, table

    for module, names in SYMBOLS.items():
        package = importlib.import_module(module)
        for name in names:
            obj = getattr(package, name)
            rows = []
            docs = {
                **(MEANINGS if module.startswith("nampy.gam.") else SHARED_MEANINGS),
                **descriptions(obj),
            }
            items = list(inspect.signature(obj).parameters.items())
            if module == "nampy.plotting" and name == "plot_distributional_effects":
                from nampy.plotting import (
                    prepare_distributional_effects,
                    render_distributional_effects,
                )

                items = [
                    (key, p)
                    for key, p in items
                    if p.kind is not inspect.Parameter.VAR_KEYWORD
                ]
                present = {key for key, _ in items}
                for delegate in (
                    prepare_distributional_effects,
                    render_distributional_effects,
                ):
                    for key, p in inspect.signature(delegate).parameters.items():
                        if key not in present and key != "effects":
                            items.append((key, p))
                            present.add(key)
                items.append(
                    (
                        "feature_name",
                        inspect.Parameter(
                            "feature_name", inspect.Parameter.KEYWORD_ONLY, default=None
                        ),
                    )
                )
            for key, p in items:
                if key in {"self", "cls"}:
                    continue
                meaning = docs.get(key)
                if module == "nampy.gam.results" and key == "intercept":
                    meaning = "Scalar fitted intercept summary; full coefficient vectors retain any additional general-family intercept directions."
                if not meaning:
                    raise ValueError(
                        f"Missing shared symbol parameter description: {module}.{name}.{key}"
                    )
                if module.startswith("nampy.gam."):
                    typ = (
                        str(p.annotation)
                        .replace("typing.", "")
                        .replace("<class '", "")
                        .replace("'>", "")
                        if p.annotation is not inspect.Parameter.empty
                        else parameter_type(key, p.default, obj=obj)
                    )
                    if key in ARRAY_SHAPES:
                        typ += ", shape " + ARRAY_SHAPES[key]
                    from dataclasses import MISSING, fields, is_dataclass

                    if inspect.isclass(obj) and is_dataclass(obj):
                        field = next(
                            field for field in fields(obj) if field.name == key
                        )
                        if field.default_factory is not MISSING:
                            p = p.replace(default=field.default_factory())
                else:
                    typ = TYPES.get(key) or parameter_type(
                        key, p.default, obj=obj, annotation=p.annotation
                    )
                if (
                    module == "nampy.contracts"
                    and name == "FeatureSchema"
                    and key == "feature_names"
                ):
                    typ = "tuple[str, ...], length d"
                default = (
                    "Required"
                    if p.default is inspect.Parameter.empty
                    else f"``{literal(p.default)}``"
                )
                if p.kind in {p.VAR_KEYWORD, p.VAR_POSITIONAL}:
                    default = "Variable arguments"
                rows.append((f"``{key}``", f"``{typ}``", default, meaning))
            text = heading("Complete parameter contract", "-")
            text += "Dimensions: ``n`` input rows, ``d`` source columns, ``q`` output columns. A shape of ``()`` denotes a scalar.\n\n"
            text += table(["Parameter", "Type / shape", "Default", "Meaning"], rows)
            text += (
                heading("Return value", "-") + return_description(module, name) + "\n\n"
            )
            text += example_rst(
                root, f"api/generated/{module}.{name}", example(module, name)
            )
            write(root, f"_generated/reference/symbols/{module}.{name}.inc", text)


# Formula and fitted-result records are extension inputs/outputs, not estimators.
SYMBOLS["nampy.gam.formula"] = [
    "parse_gam_formula",
    "extract_formula_terms",
    "all_vars1",
    "get_numeric_response_labels",
    "strip_offset_wrapper",
    "ParsedGAMFormula",
    "ParsedPredictorFormula",
    "ParsedFormulaComponent",
    "ParsedParametricTerm",
    "ParsedSmoothTerm",
    "ExtractedPredictor",
    "ExtractedParametricTerm",
    "ExtractedSmoothTerm",
]
SYMBOLS["nampy.gam.results"] = [
    "GAMResult",
    "GAMFitResult",
    "TermFitResult",
    "FitResult",
]
SHARED_MEANINGS = dict(MEANINGS)
MEANINGS.update(
    {
        "formula": "Formula string or supported ordered formula collection. Parsing records syntax; it does not attach data, construct a numerical basis or fit coefficients.",
        "parsed": "ParsedGAMFormula from parse_gam_formula; contains components, predictor terms and offsets before data attachment.",
        "drop_intercept": "Optional scalar or per-component boolean flags. True removes that component intercept; a sequence must match the number of source components.",
        "expr": "Formula expression string; the helper extracts variable names or strips the outer offset wrapper according to the selected function.",
        "formula_or_lhs": "Formula string or left-hand-side expression containing numeric predictor labels for a shared formula component.",
        "variables": "Ordered variable names in this parametric term; a tuple can identify a multi-variable interaction.",
        "raw_label": "Original formula term label before normalization or runtime construction.",
        "factor_labels": "Parametric factor/expression labels retained for later design construction.",
        "kind": "Formula constructor kind (s, te, ti, t2) for Parsed/Extracted smooth terms.",
        "features": "Ordered source covariates for the smooth term, before data columns are resolved.",
        "kwargs": "Parsed constructor-option mapping, such as bs/k/m/by/id; values have not yet been resolved to runtime defaults.",
        "label": "Normalized formula label, or fitted term display label for TermFitResult.",
        "by_variable": "Optional name of the smooth by variable.",
        "response_name": "Response expression/name; None for a one-sided formula.",
        "intercept": "Whether this parsed predictor contains an intercept; numerical result records instead expose the fitted scalar intercept.",
        "terms": "Ordered parsed/extracted terms for the component/predictor, before runtime basis construction.",
        "offset_names": "Tuple of offset expressions parsed from this component. Parsing may retain multiple offsets even where fitting rejects them.",
        "offset_name": "Single offset expression for this extracted predictor, or None.",
        "raw_formula": "Original formula component text.",
        "pf": "Source-style parametric formula bookkeeping text.",
        "pfok": "Source-style flag indicating whether the parametric formula bookkeeping is valid.",
        "fake_formula": "Bookkeeping formula used to retain input-variable references; not a fitted numerical predictor.",
        "fake_names": "Variable names retained by the bookkeeping formula.",
        "pred_names": "Names required by the prediction bookkeeping formula.",
        "pred_formula": "Prediction-side bookkeeping formula, default ~ 1 when no explicit structure is supplied.",
        "lpi": "One-based linear-predictor indices sharing this formula component.",
        "is_base_formula": "Whether this is a base predictor component rather than a shared/additional component.",
        "source_component_indices": "Source component indices aggregated into this predictor, in parser order.",
        "predictors": "One ParsedPredictorFormula per aggregated linear predictor.",
        "components": "Source formula components before predictor aggregation; empty by default in manually constructed records.",
        "nlp": "Number of linear predictors represented by this parsed formula.",
        "raw": "Original formula input retained for inspection; its type follows the caller input.",
        "predictor_index": "Zero-based extracted predictor/component index.",
        "predictor_name": "Human-readable predictor identifier, such as eta1.",
        "compiled_model": "Optional compiled design aggregate, including predictors, terms and penalty metadata. None means compilation has not yet been attached.",
        "fit_core_solution": "Optional numerical solution and transient fitting workspace. None means no solution has been attached.",
        "fit_summary": "Optional GAMFitResult associated with the exact fitted solution. It is invalidated when that solution is replaced.",
        "family_name": "Resolved statistical response family name.",
        "link_name": "Resolved link name or multi-predictor link description.",
        "criterion_name": "Smoothing-selection criterion name, or None when unavailable.",
        "criterion_value": "Scalar fitted criterion value, or None when the route does not report one.",
        "core": "FitResult numerical record referenced by the presentation summary; arrays are not duplicated unnecessarily.",
        "smoothing_params": "Penalty-strength vector in compiled penalty order, including fixed/linked reporting conventions.",
        "side_condition_reports": "Optional per-term reports describing absorbed identifiability side conditions.",
        "term_results": "One TermFitResult per reported term, with coefficient slices and EDF information.",
        "metadata": "Additional compiler/reporting metadata. Contents depend on the term or fitted backend.",
        "term_type": "Reported term kind, such as parametric, smooth or factor-smooth.",
        "basis_name": "Selected basis code, for example cr or tp.",
        "coef_slice": "Half-open (start, stop) coefficient positions for this term in coef_full.",
        "n_coef": "Number of fitted coefficients in this term after constraints/deletions.",
        "edf": "Fitted effective degrees of freedom (term EDF may be None when unavailable).",
        "smoothing_indices": "Indices of penalty strengths associated with this term.",
        "smoothing_ids": "Optional smoothing-sharing identifiers aligned with smoothing_indices.",
        "smoothing_values": "Penalty strengths aligned with this term smoothing indices.",
        "deleted_columns": "Original basis columns removed by side conditions/rank handling.",
        "kept_columns": "Original basis columns retained in the fitted coordinates.",
        "coef_full": "Full fitted coefficient vector including represented intercept directions.",
        "beta": "Fitted non-intercept coefficient vector.",
        "eta": "Training-row additive predictors, before inverse links.",
        "mu": "Training-row family fitted values; source general-family conventions can differ from natural GAMLSS output parameters.",
        "rss": "Residual sum of squares where meaningful for the backend, otherwise None.",
        "deviance": "Family-defined fitted deviance.",
        "trace_H": "Fitted influence-matrix trace.",
        "scale": "Fitted/fixed statistical dispersion or scale, according to family convention.",
        "cov_bayes": "Conditional Bayesian penalized coefficient covariance, or None if omitted/unavailable.",
        "cov_freq": "Frequentist coefficient covariance, or None if omitted/unavailable.",
        "cov_unconditional": "Coefficient covariance corrected for available smoothing uncertainty, or None.",
        "H_coef": "Coefficient-space influence/hat operator used for EDF calculations.",
        "edf2": "Optional per-coefficient uncertainty-adjusted EDF information.",
        "edf_by_term": "Optional EDF vector aligned with reported term order.",
        "penalty_quadratic": "Optional scalar coefficient penalty quadratic form.",
        "loglik": "Optional fitted unpenalized log likelihood.",
        "converged": "Optional coefficient-solver convergence flag.",
        "iter": "Optional coefficient-solver iteration count.",
        "failed_step": "Optional solver step-failure flag.",
        "failure_reason": "Optional solver failure explanation.",
        "inner_trace": "Optional sequence of numerical coefficient-iteration diagnostics.",
        "coef_space": "Coefficient coordinate-space label for coef_full.",
        "cov_bayes_space": "Coordinate-space label of the Bayesian covariance.",
        "cov_freq_space": "Coordinate-space label of the frequentist covariance.",
        "cov_unconditional_space": "Coordinate-space label of corrected covariance.",
        "coef_optimization": "Optional coefficient vector in constrained-solver optimization coordinates.",
        "cov_bayes_optimization": "Optional Bayesian covariance in optimization coordinates.",
        "cov_freq_optimization": "Optional frequentist covariance in optimization coordinates.",
        "positive_coefficient_mask": "Optional boolean vector marking coefficient coordinates transformed to be positive.",
    }
)
FORMULA_SETUP = """from nampy.gam.formula import (
    parse_gam_formula, extract_formula_terms, all_vars1, get_numeric_response_labels,
    strip_offset_wrapper, ParsedParametricTerm, ParsedSmoothTerm,
    ExtractedParametricTerm, ExtractedSmoothTerm,
)

parsed = parse_gam_formula("y ~ z + s(x, bs='cr', k=6) + offset(exposure)")
extracted = extract_formula_terms(parsed)
"""
_original_example = example
_original_return_description = return_description


def example(module, name):
    if module == "nampy.gam.formula":
        expressions = {
            "parse_gam_formula": "parsed",
            "extract_formula_terms": "extracted",
            "all_vars1": "all_vars1('s(x) + z + offset(exposure)')",
            "get_numeric_response_labels": "get_numeric_response_labels('1 + 2 ~ s(x)')",
            "strip_offset_wrapper": "strip_offset_wrapper('offset(exposure)')",
            "ParsedGAMFormula": "parsed",
            "ParsedPredictorFormula": "parsed.predictors[0]",
            "ParsedFormulaComponent": "parsed.components[0]",
            "ParsedParametricTerm": "next(term for term in parsed.predictors[0].terms if isinstance(term, ParsedParametricTerm))",
            "ParsedSmoothTerm": "next(term for term in parsed.predictors[0].terms if isinstance(term, ParsedSmoothTerm))",
            "ExtractedPredictor": "extracted[0]",
            "ExtractedParametricTerm": "next(term for term in extracted[0].terms if isinstance(term, ExtractedParametricTerm))",
            "ExtractedSmoothTerm": "next(term for term in extracted[0].terms if isinstance(term, ExtractedSmoothTerm))",
        }
        return FORMULA_SETUP + f"result = {expressions[name]}\nprint(result)\n"
    if module == "nampy.gam.results":
        expressions = {
            "GAMResult": "model.gam_result_",
            "GAMFitResult": "model.fit_result()",
            "TermFitResult": "model.fit_result().term_results[0]",
            "FitResult": "model.fit_result().core",
        }
        return (
            BASE
            + MODEL
            + f"result = {expressions[name]}\nprint(type(result).__name__)\n"
        )
    return _original_example(module, name)


def return_description(module, name):
    if module == "nampy.gam.formula":
        if name == "all_vars1":
            return "``tuple[str, ...]`` of referenced variable names, in the helper retained order. Length follows the expression."
        if name == "get_numeric_response_labels":
            return "``tuple[int, ...]`` of numeric linear-predictor labels. These are source one-based predictor identifiers, not observation indices."
        if name == "strip_offset_wrapper":
            return "``str`` containing the enclosed offset expression; a string without the outer wrapper is returned stripped of surrounding whitespace."
        if name == "extract_formula_terms":
            return "``list[ExtractedPredictor]``, one record per source component. Each contains extracted term tuples and normalized predictor-sharing indices."
        if name == "parse_gam_formula":
            return "``ParsedGAMFormula`` containing parsed components and predictor views. No design matrix or prediction array is produced at this stage."
        return f"``{name}`` syntax/intention record with the fields listed above. Tuple/list lengths follow the formula structure, not the sample count."
    if module == "nampy.gam.results":
        return f"``{name}`` fitted-state record. Numerical array dimensions and unavailable optional fields are listed in the parameter/field table and the concrete example output below. Access fitted records through the model rather than manually supplying inconsistent arrays."
    return _original_return_description(module, name)


ARRAY_SHAPES = {
    "coef_full": "(p,)",
    "beta": "(p_nonintercept,)",
    "eta": "(n_train,) or (n_train, q)",
    "mu": "(n_train,) or (n_train, q)",
    "cov_bayes": "(p, p)",
    "cov_freq": "(p, p)",
    "cov_unconditional": "(p, p)",
    "H_coef": "(p, p)",
    "edf2": "(p,)",
    "edf_by_term": "(n_terms,)",
    "smoothing_params": "(n_penalties,)",
    "coef_optimization": "(p,)",
    "cov_bayes_optimization": "(p, p)",
    "cov_freq_optimization": "(p, p)",
    "positive_coefficient_mask": "(p,)",
}
