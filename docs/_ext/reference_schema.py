"""Explicit input and output contracts shared by the reference generators."""

from __future__ import annotations

import inspect
import re
from typing import get_type_hints

# These spell out domains that cannot be recovered from a default of None.
PARAMETER_TYPES = {
    "X": "pandas.DataFrame | numpy.ndarray, shape (n, d) | None",
    "X_val": "pandas.DataFrame | numpy.ndarray, shape (n_val, d) | None",
    "y": "array-like, shape (n,) or family-specific (n, q) | None",
    "y_val": "array-like, shape (n_val,) or (n_val, q) | None",
    "data": "pandas.DataFrame, n rows | None",
    "context": "Mapping[str, array-like, shape (n,)] | None",
    "feature": "str | int | Sequence[str | int]",
    "feature_names": "Sequence[str], length d",
    "k": "int",
    "m": "int | Sequence[int | float] | None",
    "basis": "str",
    "by": "str | int | None",
    "label": "str | None",
    "term_id": "str | None",
    "smoothing_id": "str | int | None",
    "sp": "float | array-like, shape (n_penalties,) | None",
    "knots": "Mapping[str, array-like] | array-like | None",
    "xt": "Mapping[str, object] | Sequence[Mapping | None] | str | None",
    "metadata": "Mapping[str, object] | None",
    "pc": "float | array-like | Mapping[str, float] | None",
    "shared_basis_setup": "Mapping[str, object] | None",
    "ord": "int | Sequence[int] | None",
    "mc": "Sequence[bool] | None",
    "link": "str",
    "theta": "float | Sequence[float] | None",
    "phiPen": "float | None",
    "power": "float | None",
    "R": "int | None",
    "K": "int",
    "offset": "float | array-like, shape (n,) or (n, q) | None",
    "offset_val": "float | array-like, shape (n_val,) or (n_val, q) | None",
    "sample_weight": "array-like, shape (n,) | None",
    "sample_weight_val": "array-like, shape (n_val,) | None",
    "cov": "str | numpy.ndarray, shape (p, p) | None",
    "terms": "Iterable[str] | None",
    "exclude": "Iterable[str] | None",
    "block_size": "int | None",
    "dispersion": "float | None",
    "test": "str | None",
    "setseed": "int | None",
    "seed": "int | None",
    "models": "GAM positional arguments",
    "candidate_indices": "Sequence[int] | None",
    "figsize": "tuple[float, float] | None",
    "view": "Sequence[str], length 2 | None",
    "cond": "Mapping[str, object] | None",
    "zlim": "tuple[float, float] | None",
    "col": "Matplotlib color | Sequence[color] | None",
    "contour_col": "Matplotlib color | None",
    "shade_col": "Matplotlib color",
    "xlab": "str | None",
    "ylab": "str | None",
    "zlab": "str | None",
    "main": "str | None",
    "quantile_permutations": "array-like, shape (s_rep, n) | None",
    "response_draws": "array-like, shape (rep, n) | None",
    "formula": "str | Sequence[str] | Mapping[str, str] | None",
    "optimize_smoothing": "bool | None",
    "smoothing_method": "str | None",
    "smoothing_optimizer": "str",
    "smoothing_params": "float | array-like, shape (n_penalties,) | None",
    "min_sp": "float | array-like, shape (n_penalties,) | None",
    "drop_intercept": "bool | Sequence[bool] | None",
    "constraints": "Mapping[str | int, array-like] | None",
    "para_pen": "Mapping[str, object] | None",
    "extra_penalty": "numpy.ndarray, shape (p, p) | None",
    "ar1_start": "array-like[bool], shape (n,) | None",
    "control": "GAMControl | Mapping[str, object] | None",
    "nei": "Mapping[str, object] | None",
    "optim_method": "str | None",
    "sp_log_bounds": "tuple[float, float]",
    "path": "str | pathlib.Path",
    "kwargs": "keyword arguments",
    "_ignored": "compatibility keyword arguments",
    "trainer_kwargs": "Lightning Trainer keyword arguments",
    "family": "str | Mapping[str, object] | BaseFamily | None",
    "distributional_kwargs": "Mapping[str, object] | None",
    "feature_widths": "Mapping[str, int]",
    "feature_preprocessing": "Mapping[str, str] | None",
    "interaction_degree": "int | None",
    "interactions": "Sequence[tuple[str, ...]] | None",
    "norm": "str | None",
    "layer_sizes": "Sequence[int]",
    "activation": "str | torch.nn.Module | type[torch.nn.Module]",
    "embedding_activation": "torch.nn.Module",
    "head_activation": "torch.nn.Module",
    "transformer_activation": "torch.nn.Module",
    "kernel_width": "float | Literal['auto']",
    "kernel_widths": "Mapping[str, float] | None",
    "cg_max_iter": "int | None",
    "rff_random_state": "int | None",
    "reference_layer_sizes": "Sequence[int]",
    "residual_layer_sizes": "Sequence[int]",
    "head_layer_sizes": "Sequence[int]",
    "ranks": "Sequence[int]",
    "interaction_thresholds": "float | Mapping[int, float]",
    "max_terms_per_order": "int | Mapping[int, int] | None",
    "nary": "int | Sequence[int] | None",
    "scaling_strategy": "str | None",
    "optimizer": "str | type[torch.optim.Optimizer] | None",
    "optimizer_kwargs": "Mapping[str, object] | None",
    "dataloader_kwargs": "Mapping[str, object] | None",
    "max_time": "str | datetime.timedelta | Mapping[str, int] | None",
    "warm_start": "bool | str | pathlib.Path",
    "checkpoint_path": "str | pathlib.Path",
    "class_weight": "Mapping[object, float] | Literal['balanced'] | None",
    "sampling_strategy": "Literal['balanced'] | None",
    "loss_fct": "callable | None",
    "n_dim": "int | None",
    "dim": "int | None",
    "num_classes": "int | None",
    "quantiles": "Sequence[float] | None",
    "rob": "float | None",
    "model": "type[torch.nn.Module]",
    "config": "type[dataclass configuration]",
    "estimator": "unfitted neural regressor or classifier",
    "n_jobs": "int | None",
    "regressor": "type | None",
    "classifier": "type | None",
    "lss": "type | None",
    "module_name": "str",
    "architecture": "str",
    "full": "bool",
    "se": "float | bool",
    "scale": "float",
    "sparse": "bool | int",
}
for _transform in [
    "mean",
    "scale",
    "rate",
    "shape",
    "concentration",
    "df",
    "loc",
    "dispersion",
    "prob",
    "zero_prob",
]:
    PARAMETER_TYPES[_transform + "_transform"] = (
        "str | callable(torch.Tensor) -> torch.Tensor"
    )


PARAMETER_TYPES.update(
    {
        "architecture_config": "architecture configuration dataclass | None",
        "objective": "ObjectiveConfig | None",
        "training": "GradientTrainingConfig | FixedDesignTrainingConfig | NativeTrainingConfig | None",
        "preprocessing": "PreprocessingConfig | None",
        "config_class": "type",
        "predictor_names": "tuple[str, ...]",
        "parameters": "dict[str, object]",
    }
)


def numpy_types(obj):
    """Return only declared parameter types, excluding prose and return sections."""
    result = {}
    in_parameters = False
    lines = (inspect.getdoc(obj) or "").splitlines()
    for i, line in enumerate(lines):
        if line in {"Parameters", "Other Parameters"}:
            in_parameters = True
            continue
        if (
            i + 1 < len(lines)
            and re.fullmatch("-{3,}", lines[i + 1])
            and line not in {"Parameters", "Other Parameters"}
        ):
            in_parameters = False
        if in_parameters:
            match = re.match(r"^(\*{0,2}\w+(?:,\s*\w+)*)\s*:\s*(.+)$", line)
            if match:
                for name in match[1].split(","):
                    result[name.strip().lstrip("*")] = re.sub(
                        r",?\s*default\s*=.*$", "", match[2]
                    )
    return result


def parameter_type(
    name,
    default=inspect.Parameter.empty,
    *,
    obj=None,
    annotation=inspect.Parameter.empty,
    declared=None,
):
    if name == "objective__family":
        return "str | None"
    key = name.split("__", 1)[-1]
    class_name = getattr(obj, "__name__", "")
    tensor_classes = {
        "TensorProductSplineTerm",
        "InteractionTensorProductSplineTerm",
        "AlternativeTensorProductSplineTerm",
    }
    if key == "feature" and class_name in {
        "CubicSplineTerm",
        "PSplineTerm1D",
        "DerivativeBSplineTerm1D",
        "ShapeConstrainedBSplineTerm",
    }:
        return "str | int"
    if key == "k" and class_name in tensor_classes | {
        "AdaptiveSmoothTerm",
        "ShapeConstrainedAdaptiveTerm",
        "BivariateShapePSplineTerm",
        "SoapFilmTerm",
    }:
        return "int | Sequence[int]"
    if key == "basis" and class_name in tensor_classes:
        return "str | Sequence[str]"
    if key == "link" and class_name in {
        "GaulssFamily",
        "GammalsFamily",
        "GumblsFamily",
        "ZiplssFamily",
        "GevlssFamily",
        "ShashFamily",
        "TwlssFamily",
    }:
        return "Sequence[str], one per linear predictor"
    if key == "theta":
        if class_name == "vis":
            return "float"
        if class_name == "NegativeBinomialLogFamily":
            return "float"
        if class_name in {"TweedieTwFamily", "BetaRegressionFamily"}:
            return "float | None"
        if class_name in {
            "ScaledTFamily",
            "ZeroInflatedPoissonFamily",
            "CensoredBoxCoxGaussianFamily",
            "OrderedCategoricalFamily",
        }:
            return "Sequence[float] | None"
    if key == "xt" and class_name:
        if class_name in tensor_classes:
            return "Mapping[str, object] | Sequence[Mapping | None] | None"
        if class_name in {
            "FSmoothInteractionTerm",
            "SZSmoothInteractionTerm",
            "AdaptiveSmoothTerm",
            "ShapeConstrainedAdaptiveTerm",
        }:
            return "str | Mapping[str, object] | Sequence | None"
        return "Mapping[str, object] | None"
    if key in PARAMETER_TYPES:
        typ = PARAMETER_TYPES[key]
        if default is inspect.Parameter.empty and key in {"X", "y", "feature_names"}:
            typ = typ.removesuffix(" | None")
        return typ
    if annotation is not inspect.Parameter.empty:
        text = (
            str(annotation)
            .replace("typing.", "")
            .replace("<class '", "")
            .replace("'>", "")
        )
        if text not in {"Any", "object"}:
            return text
    if declared and key in declared and declared[key] not in {"optional", "object"}:
        return declared[key]
    if obj is not None:
        try:
            hints = get_type_hints(obj)
            if key in hints:
                return str(hints[key]).replace("typing.", "")
        except (NameError, TypeError):
            pass
        parsed = numpy_types(obj)
        if key in parsed and parsed[key] not in {"optional", "object"}:
            return parsed[key]
    if default is not inspect.Parameter.empty and default is not None:
        if isinstance(default, bool):
            return "bool"
        if isinstance(default, int):
            return "int"
        if isinstance(default, float):
            return "float"
        if isinstance(default, str):
            return "str"
        if (
            isinstance(default, (tuple, list))
            and default
            and all(isinstance(v, int) for v in default)
        ):
            return "Sequence[int]"
    raise ValueError(
        f"Missing explicit parameter type: {getattr(obj, '__name__', obj)}.{name}"
    )


# n = prediction rows, d = input columns, p = fitted coefficients,
# t = returned terms, r = free smoothing parameters, q = output parameters.
METHOD_RETURNS = {
    "fit": [
        (
            "self",
            "GAM",
            "scalar object",
            "Fitted model; all prediction methods use its committed fitted state.",
        )
    ],
    "predict": [
        (
            "prediction",
            "numpy.ndarray",
            "(n,) / (n, q) / (n, t) / (n, p)",
            "Shape depends on type and family: response/link, terms/iterms, or lpmatrix. Multi-predictor source response columns are family-specific.",
        ),
        (
            "standard_error",
            "numpy.ndarray",
            "same shape as prediction",
            "Returned only as (prediction, standard_error) when return_se=True and the requested prediction route supports standard errors.",
        ),
    ],
    "predict_terms": [
        (
            "output",
            "numpy.ndarray",
            "(n,) or (n, q)",
            "Complete additive link predictor.",
        ),
        (
            "term label keys",
            "numpy.ndarray",
            "(n,)",
            "One vector per term/predictor label. Multi-predictor keys carry eta indices.",
        ),
        (
            "intercept",
            "numpy.ndarray (scalar) | dict[str, float]",
            "scalar(s)",
            "Predictor intercept(s).",
        ),
        (
            "offset",
            "numpy.ndarray | dict | None",
            "(n,) or per-predictor arrays",
            "Known offsets, when present.",
        ),
        (
            "response",
            "numpy.ndarray",
            "family-specific",
            "Included for non-Gaussian families.",
        ),
    ],
    "lpmatrix": [
        (
            "matrix",
            "numpy.ndarray[float64]",
            "(n, p)",
            "Design in fitted coefficient coordinates, excluding offsets.",
        )
    ],
    "summary": [
        (
            "summary",
            "GAMSummary",
            "scalar record",
            "Contains parametric and smooth significance tables, EDF, dispersion and goodness-of-fit fields; the method also prints the summary.",
        )
    ],
    "residuals": [
        (
            "residual",
            "numpy.ndarray[float64]",
            "(n_train,)",
            "Residual type selects the scale; supported kinds depend on the family.",
        )
    ],
    "ar1_standardized_residuals": [
        (
            "residual",
            "numpy.ndarray[float64]",
            "(n_train,)",
            "Response residuals whitened by the known fitted AR(1) transform.",
        )
    ],
    "concurvity": [
        (
            "measure_names",
            "tuple[str, str, str]",
            "length 3",
            "worst, observed, estimate.",
        ),
        ("labels", "list[str]", "length t", "Design-space/term labels."),
        (
            "values (full=True)",
            "numpy.ndarray[float64]",
            "(3, t)",
            "Each term compared with the remaining design jointly.",
        ),
        (
            "values (full=False)",
            "dict[str, numpy.ndarray]",
            "3 arrays, each (t, t)",
            "Pairwise matrices indexed by measure name.",
        ),
    ],
    "k_check": [
        (
            "table",
            "pandas.DataFrame | None",
            "(n_smooths, 4)",
            "Columns k_prime, edf, k_index, p_value. None when there are no smooth terms; unsupported diagnostics may be NaN.",
        )
    ],
    "gam_check": [
        (
            "report",
            "dict[str, object]",
            "named diagnostic fields",
            "Numerical convergence/residual report plus basis-dimension checks. The example below lists the actual returned fields.",
        )
    ],
    "derivative": [
        (
            "derivative",
            "numpy.ndarray[float64]",
            "(n,)",
            "Predictor-scale slope or curvature.",
        ),
        ("se", "numpy.ndarray[float64]", "(n,)", "Conditional pointwise Bayesian SE."),
        (
            "derivative_matrix",
            "numpy.ndarray[float64]",
            "(n, p_term)",
            "Derivative design in the selected term coefficient coordinates.",
        ),
        (
            "term_label / order",
            "str / int",
            "scalar",
            "Identity of the differentiated term and derivative order in SmoothDerivativeResult.",
        ),
    ],
    "vcov": [
        (
            "covariance",
            "numpy.ndarray[float64]",
            "(p, p)",
            "Selected coefficient covariance; symmetry follows fitted coefficient order.",
        )
    ],
    "sp_vcov": [
        (
            "covariance",
            "numpy.ndarray[float64] | None",
            "(r, r), possibly including scale",
            "Covariance in free log-smoothing coordinates. None if required curvature is unavailable.",
        )
    ],
    "gam_vcomp": [
        (
            "vc",
            "numpy.ndarray[float64]",
            "(n_penalties,) or (n_free [+ scale], 3)",
            "Standard deviations alone, or columns SD/lower/upper when supported intervals are available.",
        ),
        ("names", "str | list[str]", "aligned with vc rows", "Component labels."),
        (
            "all / all_names",
            "array-like / list[str] | None",
            "full mapped component order",
            "Expanded linked-penalty components, where applicable.",
        ),
        (
            "rank / rank_hess / conf_lev",
            "int / int / float | None",
            "scalar",
            "Available curvature rank and interval coverage. Entire result is None when components cannot be reported.",
        ),
    ],
    "one_se_rule": [
        (
            "smoothing",
            "numpy.ndarray[float64]",
            "(n_penalties,)",
            "Candidate smoothing strengths; the model is not automatically refitted.",
        )
    ],
    "anova": [
        (
            "table",
            "AnovaGAMSingle | AnovaGAMComparison",
            "scalar record",
            "Within-model term tests or between-model comparison rows, depending on additional models.",
        )
    ],
    "edf1": [
        (
            "edf",
            "numpy.ndarray[float64]",
            "(p,)",
            "Per-coefficient upper-bound EDF, in coefficient order.",
        )
    ],
    "fit_result": [
        (
            "result",
            "GAMFitResult",
            "scalar record",
            "coef_full (p,), eta/mu (n_train,) or (n_train,q), smoothing_params (n_penalties,), covariance fields (p,p) when included; term_results contains one TermFitResult per term.",
        )
    ],
    "parity_snapshot": [
        (
            "snapshot",
            "dict[str, object]",
            "named fitted payload",
            "Serializable coefficients, penalties, predictions, metadata and optionally covariance arrays; see the concrete example schema below.",
        )
    ],
    "save_model": [
        (
            "return",
            "pathlib.Path",
            "scalar path object",
            "Destination path after writing the requested model artifact; use load_model to restore it.",
        )
    ],
    "load_model": [
        (
            "model",
            "GAM",
            "scalar object",
            "Restored fitted model; input/output schemas are retained.",
        )
    ],
    "get_device": [("device", "str", "scalar", "CPU execution designation.")],
}
for _method in ["aic", "bic", "loglik"]:
    METHOD_RETURNS[_method] = [
        (
            "value",
            "float",
            "scalar",
            {
                "aic": "Conditional AIC.",
                "bic": "Effective-df BIC.",
                "loglik": "Unpenalized fitted log likelihood.",
            }[_method],
        )
    ]
for _method in ["plot", "qq", "plot_check", "vis"]:
    METHOD_RETURNS[_method] = [
        (
            "prepared",
            "dict[str, object]",
            "one payload per plot or diagnostic",
            "Numerical grids/effects and associated Matplotlib figures; the example schema below identifies each payload field and its concrete shape.",
        )
    ]
METHOD_RETURNS["plot_derivative"] = [
    (
        "figures",
        "list[matplotlib.figure.Figure]",
        "one entry per rendered panel",
        "Matplotlib figures containing the derivative curve and pointwise band.",
    )
]


def estimator_returns(task):
    rows = [
        (
            "fit",
            "estimator instance",
            "scalar object",
            "Returns self after fitting; no prediction array.",
        ),
        (
            "score",
            "float",
            "scalar",
            "R² for regression, accuracy for classification, family-defined scoring for LSS.",
        ),
        (
            "evaluate",
            "dict[str, float]",
            "one scalar per metric",
            "Metric names depend on objective/family.",
        ),
        (
            "predict_components",
            "AdditivePrediction",
            "scalar record",
            "link and every term have shape (n,K) for K actual predictors; intercept and offset are aligned with these coordinates. Response values and natural parameters are queried separately.",
        ),
        (
            "explain_terms",
            "pandas.DataFrame",
            "one row per displayed contribution",
            "Includes term labels and contribution/importance information; schema depends on requested aggregation.",
        ),
        (
            "term_importance",
            "pandas.DataFrame",
            "one row per term",
            "Aggregated importance, not an uncertainty estimate.",
        ),
        (
            "save_model / load_model",
            "pathlib.Path / estimator instance",
            "scalar path / scalar object",
            "Write and restore the fitted artifact.",
        ),
    ]
    rows.append(
        (
            "plot_terms",
            "list[matplotlib.figure.Figure]",
            "one figure per requested page",
            "Returned numeric main-effect panels; interactions and multi-column terms have separate rendering support.",
        )
    )
    if task == "regression":
        rows.insert(
            1,
            (
                "predict",
                "numpy.ndarray",
                "(n,) or (n, n_outputs)",
                "Response-scale means; scalar targets use one dimension.",
            ),
        )
    elif task == "classification":
        rows[1:1] = [
            (
                "predict",
                "numpy.ndarray",
                "(n,)",
                "Class labels with the fitted class dtype.",
            ),
            (
                "predict_proba",
                "numpy.ndarray[float]",
                "(n, n_classes)",
                "Class probabilities in ``classes_`` order; rows sum to one.",
            ),
        ]
    else:
        rows[1:1] = [
            (
                "predict",
                "numpy.ndarray[float]",
                "(n, q)",
                "Natural family parameters, except class/vector/quantile-specific widths documented on each family page.",
            ),
            (
                "predict(raw=True)",
                "numpy.ndarray[float]",
                "(n, q_raw)",
                "Network/link predictors in the fitted family raw parameter order.",
            ),
            (
                "predict_point",
                "numpy.ndarray",
                "(n,) or (n, response_dimensions)",
                "Family-defined mean, median or most probable class; not always an expectation.",
            ),
            (
                "predict_distribution",
                "nampy.distributions.DistributionPrediction",
                "one scalar conditional distribution per prediction row",
                "Normal, Gamma, Student-t, beta, negative-binomial, lognormal and Weibull: NumPy probability, quantile, sampling, moment and shared quantile-residual operations. Statistical beta-binomial requires explicit trial context; zaga_lss adds exact zeros to Gamma outcomes and beinf_lss adds zero/one atoms to beta proportions. Statistical bccg_lss supplies BCCG predictions with fixed or automatic joint REML and explicit divergent-moment semantics. Neural beta-binomial, ZAGA, BEINF and BCCG fitting are unsupported. Other families and clamped neural NB regimes raise NotImplementedError. Draws condition on fitted parameters.",
            ),
            (
                "distribution_fit_status",
                "nampy.distributions.DistributionFitStatus",
                "scalar record",
                "Backend-owned eligibility evidence: statistical convergence, completed neural training, failure or unknown status. Completed training is not statistical convergence.",
            ),
        ]
    return rows


PARAMETER_TYPES.update(
    {
        "expr": "str",
        "formula_or_lhs": "str",
        "parsed": "ParsedGAMFormula",
        "batch_size": "int | None",
        "parameters": "constructor keyword arguments",
        "reference_X": "pandas.DataFrame | numpy.ndarray, shape (n_ref, d) | None",
        "reference_weight": "array-like, shape (n_ref,) | None",
        "max_bins": "int | None",
        "feature_name": "str | None",
        "feature_names": "Sequence[str] | None",
        "rug": "bool | None",
        "metrics": "Sequence[str] | Mapping[str, callable] | None",
        "y_true": "array-like, shape (n,) or (n, q)",
        "plot_kwargs": "plotting keyword arguments",
        "ar_start": "array-like[bool], shape (n,) | None",
        "start": "array-like, shape (p,) | None",
        "positive_transform": "str",
        "softplus_beta": "float",
        "softplus_threshold": "float",
        "ar1_rho": "float",
        "cat_feature_info": "Mapping[str, object] | None",
        "num_feature_info": "Mapping[str, object] | None",
    }
)
METHOD_PARAMETER_MEANINGS = {
    "X": "Prediction/training feature matrix with n rows and the same d columns, names and preprocessing schema used by the fitted estimator.",
    "y": "Training/score response aligned with X: (n,) for scalar regression or class labels; supported multi-output/family targets use (n,q).",
    "y_true": "Observed response aligned with prediction rows; evaluation does not refit the model.",
    "data": "Optional formula-mode DataFrame containing response and source predictor columns. Supply either the explicit X/y workflow or the documented formula workflow.",
    "sample_weight": "Optional nonnegative row weights, shape (n,). Statistical family-specific weight conventions still apply.",
    "batch_size": "Positive prediction minibatch size; None uses the estimator/data-module default. It changes memory use, not the output dimensions.",
    "center": "Center displayed link-scale contributions against reference_X (or X when omitted), transferring their means to the intercept while preserving predictions.",
    "reference_X": "Reference population for contribution centering. It must have the same feature schema as X; None uses X.",
    "reference_weight": "Optional reference-population weights used for centering; one value per reference_X row.",
    "max_bins": "Maximum numerical bins used to summarize explanation rows; None retains the unbinned representation where supported.",
    "rug": "Include observed values as rug marks; None uses the renderer default.",
    "density": "Draw a distribution/density display below supported feature-effect panels.",
    "density_bins": "Positive number of bins in the marginal feature distribution display.",
    "density_color": "Matplotlib color for the marginal density display.",
    "pages": "Panel pagination: zero uses the renderer default; positive values select the requested panel-page layout.",
    "figsize": "Optional Matplotlib (width, height) in inches.",
    "path": "Destination/source of a fitted model artifact. Parent directories must be writable when saving.",
    "raw": "Return link/network predictors instead of transformed natural distribution parameters when True.",
    "response_scale": "Scale factor used in the distributional-effect visualization; this changes displayed units, not the fitted model.",
    "feature_name": "Optional single feature whose distributional effect should be displayed.",
    "feature_names": "Optional collection of features to display; None uses supported fitted features.",
    "plot_kwargs": "Additional documented plotting arguments forwarded to the distributional renderer.",
    "offset": "Known additive predictor offset, scalar or one value per row; supported distributional families accept per-predictor offsets.",
    "context": "Explicit observation context separate from weights: betabinomial_lss requires {'trials': trials}. Supply again for explicit prediction rows; omitted X reuses fit context. Context-dependent predictions currently require na_action='pass'.",
    "block_size": "Positive statistical prediction row-block size; None uses the engine default.",
    "newdata_guaranteed": "Request the optimized validated prediction path for complete explicit new data; omitted columns are allowed only for excluded terms covered by the contract.",
    "na_action": "Missing-row policy: pass restores missing rows as NaN, omit/exclude retain complete rows, fail raises an error.",
    "type": "Prediction/uncertainty scale, such as response or link. Supported names and widths depend on the estimator and family.",
    "unconditional": "Request supported smoothing-parameter uncertainty corrections for coefficient/prediction uncertainty.",
    "metrics": "Optional metric selection/custom metric mapping. None uses the estimator objective/family defaults; returned values are scalar metric scores.",
    "kwargs": "Additional method-specific options documented by the corresponding statistical engine or renderer.",
    "device": "Loading/execution device, for example cpu. Statistical models execute on the CPU.",
}


METHOD_PARAMETER_MEANINGS.update(
    {
        "X_val": "Optional explicit validation features; provide together with y_val.",
        "y_val": "Optional validation targets, aligned with X_val and the training objective.",
        "offset_val": "Optional known predictor offsets for explicit validation rows.",
        "sample_weight_val": "Optional nonnegative explicit-validation observation weights.",
    }
)
PARAMETER_TYPES.update(
    {
        "X_val": "DataFrame | ndarray | None",
        "y_val": "array-like | None",
        "offset_val": "array-like | None",
        "sample_weight_val": "array-like | None",
    }
)


def method_parameter_rows(method, extra_meanings=None):
    from public_reference import descriptions, literal

    meanings = {
        **METHOD_PARAMETER_MEANINGS,
        **(extra_meanings or {}),
        **descriptions(method),
    }
    rows = []
    for name, parameter in inspect.signature(method).parameters.items():
        if name in {"self", "cls"}:
            continue
        meaning = meanings.get(name)
        if not meaning:
            raise ValueError(
                f"Missing method input contract: {method.__qualname__}.{name}"
            )
        kind = (
            "Required"
            if parameter.default is inspect.Parameter.empty
            else f"``{literal(parameter.default)}``"
        )
        if parameter.kind in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}:
            kind = "Variable arguments"
        rows.append(
            (
                f"``{name}``",
                f"``{parameter_type(name, parameter.default, obj=method, annotation=parameter.annotation)}``",
                kind,
                meaning,
            )
        )
    return rows


def smooth_parameter_meanings(name):
    """Basis-specific semantics beyond the shared runtime term contract."""
    result = {}
    if name in {"tp", "ts"}:
        result.update(
            m="Penalty derivative order. None selects the smallest integer m with 2*m>d+1. For tp, a two-element sequence ending in zero removes the polynomial null space; ts does not support that removal.",
            xt="Optional max.knots (default 2000) and seed (default 1) for deterministic construction-location subsampling. Prediction reuses the retained fitted setup.",
            k="Automatic rank is M+8 in one dimension, M+27 in two and M+100 above two, where M is the polynomial null-space dimension. The rank must accommodate the null space and distinct construction locations.",
        )
    elif name in {"ps", "cp"}:
        result["m"] = (
            "None resolves to [2, 2]. The first integer is basis order (polynomial degree = first + 1); the second is the coefficient difference-penalty order. Both are nonnegative. cp uses cyclic basis/difference construction."
        )
        result["knots"] = (
            "Optional increasing complete knot vector. Automatic knot construction requires k > basis_order + 1; cyclic construction wraps the period. Prediction uses the fitted knots rather than rebuilding them."
        )
    elif name == "bs":
        result["m"] = (
            "None resolves to (3, 2): cubic degree and integrated squared second derivative. A scalar degree uses derivative order max(0, degree-1); a sequence starts with degree followed by distinct derivative penalty orders, each between zero and degree."
        )
        result["knots"] = (
            "Optional increasing knot vector or supported endpoint construction. For a full degree-d knot vector, its length is k+d+1; four-knot input selects the dedicated extended-domain branch."
        )
    elif name == "ds":
        result["m"] = (
            "Scalar penalty derivative order or (order, spectral_shift), default (2, 0). Shift is rounded to half-integers and adjusted to satisfy -d/2 < shift < d/2 and order+shift > d/2 for continuity; adjustments warn."
        )
        result["xt"] = (
            "Optional low-rank construction mapping with max.knots and seed controlling setup-location subsampling; these affect construction, not the prediction row count."
        )
    elif name == "gp":
        result["m"] = (
            "Covariance definition (signed_type, range, power), default (3, -1, 1). Absolute types 1..5 select spherical, power-exponential, or three Matern-style kernels. Negative type uses a stationary constant tail; positive type includes a linear covariate tail. Nonpositive range is inferred from distances; power must lie in (0, 2]."
        )
        result["xt"] = (
            "Optional max.knots/seed low-rank setup controls. Supplied knots and setup subsampling determine the covariance basis retained during prediction."
        )
    elif name in {"ad", "scad"}:
        result["m"] = (
            "Number of basis coefficients used to represent the varying penalty: default 5 in one dimension, or 3 per margin in two dimensions. This is an adaptive-penalty dimension, not the polynomial degree."
        )
        result["xt"] = (
            "Optional marginal basis selection, for example {'bs': 'ps'} (default). Supported alternatives are cp, cr and cc. This is distinct from selecting the outer ad basis."
        )
        result["k"] = (
            "Function-basis dimension: an integer or one value per margin. Automatic values are 40 in one dimension and 15 per margin in two dimensions; the penalty-basis dimension m must remain compatible."
        )
    elif name == "sos":
        result["xt"] = (
            "Optional max.knots (default 2000) and seed (default 1) control low-rank construction-location subsampling on the sphere."
        )
        result["m"] = (
            "Spherical kernel order, accepting integer branches -2 through 4; default 0 uses the second-order Wendelberger construction. Coordinates are latitude then longitude in degrees."
        )
    elif name in {"so", "sf", "sw"}:
        result["xt"] = (
            "Required boundary setup under bnd: list of closed polygon mappings using the formula coordinate names. Optional nmax (default 200) controls grid resolution. A boundary f column supplies known boundary values. so combines boundary/interior, sf retains boundary film, sw retains the zero-boundary interior."
        )
        result["knots"] = (
            "Paired interior coordinate vectors keyed by the two feature names. Each pair is one interior knot, not a Cartesian product. The number of pairs determines the interior basis dimension."
        )
        result["k"] = (
            "Boundary-film basis dimension, scalar or one value per boundary loop. It does not set the number of interior knot coefficients."
        )
    elif name == "mrf":
        result["xt"] = (
            "Mapping with nb (named neighbor lists), penalty (supplied region precision matrix), or polys (named polygons used to derive neighbors). Matrix/graph names must match factor levels; a supplied penalty takes precedence."
        )
        result["k"] = (
            "Retained MRF basis dimension, at most the number of region levels; negative selects the full region basis. Lower dimensions use a reduced-rank representation."
        )
    elif name == "re":
        result["xt"] = (
            "Optional precision specification with S (one matrix or a list) and corresponding rank information. None uses an identity ridge penalty on the full random-effect design."
        )
        result["k"] = (
            "Accepted for interface compatibility; ignored by this runtime constructor. Random-effect design size is determined by the numeric/factor product design."
        )
        result["knots"] = (
            "Accepted for interface compatibility and ignored by this constructor."
        )
        result["fixed"] = (
            "Accepted for interface compatibility and ignored by this constructor; random effects are represented through their penalty."
        )
        result["constraint_mode"] = (
            "Accepted for interface compatibility and ignored; random-effect terms are not centered."
        )
        result["smoothing_id"] = (
            "Must be None: linked smoothing IDs for random-effect terms raise NotImplementedError."
        )
    elif name in {"fs", "sz"}:
        result["xt"] = (
            "Marginal basis name or mapping, for example {'bs': 'cr'}; None uses tp. Remaining options are forwarded only when the chosen marginal basis advertises support. fs requires exactly one factor; sz supports sum-to-zero factor deviations."
        )
        result["m"] = (
            "Optional marginal basis-specific order. It is forwarded only where the selected marginal constructor supports this factor-smooth composition."
        )
        result["k"] = (
            "Dimension of each numerical marginal basis before expanding across factor levels and applying the factor-smooth constraints."
        )
    elif name in {"te", "ti", "t2"}:
        result["m"] = (
            "Optional scalar/order sequence or per-margin specifications, interpreted by each marginal basis. Margins retain their own derivative/covariance order conventions."
        )
        result["k"] = (
            "Scalar basis dimension repeated across margins or one integer per margin. Before constraints, tensor width is the product of marginal basis widths."
        )
        result["xt"] = (
            "Optional per-margin setup mappings, aligned with basis and feature groups; a common mapping is forwarded according to the tensor constructor."
        )
        if name == "ti":
            result["mc"] = (
                "One centering flag per margin; None applies the default marginal constraints that remove main-effect directions. Disabling a constraint changes the interaction space."
            )
    elif name not in {"tp", "ts", "cr", "cs", "cc", "sc"}:
        result["m"] = (
            "SCAM marginal spline order: default 2 gives cubic B-splines. Bivariate bases accept a scalar or a length-two order sequence; shape inequalities are imposed by the coefficient map, not by m."
        )
        if name in {"lmpi", "lipl"}:
            result["xt"] = (
                "Required change point {'xc': float}, inside the covariate range. User-supplied knots are rejected for these local constructions."
            )
    return result


NESTED_CONTROLS = {
    "ndigit": "Estimated reliable objective digits; normalized to integers from 2 through 15.",
    "gradtol": "Positive outer-gradient stopping tolerance.",
    "stepmax": "Maximum permitted NLM step magnitude; zero is reset to 2.",
    "steptol": "Positive relative-step stopping tolerance.",
    "iterlim": "Nonnegative integer limit on NLM iterations.",
    "check_analyticals": "Check supplied analytical derivatives against numerical references where supported.",
    "factr": "Positive relative function-change tolerance multiplier for the optim route.",
    "conv_tol": "Outer Newton convergence tolerance.",
    "max_n_step": "Maximum Newton-direction log smoothing step magnitude.",
    "max_s_step": "Maximum steepest-direction log smoothing step magnitude.",
    "max_half": "Maximum line-search step halvings.",
    "use_svd": "Use the SVD-supported Newton linear solve.",
    "check_analytical": "Enable constrained BFGS analytical-gradient checks.",
    "del": "Finite-difference perturbation used in supported constrained gradient checks.",
    "steptol_bfgs": "Positive constrained BFGS step stopping tolerance.",
    "gradtol_bfgs": "Positive constrained BFGS gradient stopping tolerance.",
}

CORE_EXTRA_MEANINGS = {
    "apply_side_conditions": "Apply cross-term identifiability side conditions during compilation. False retains the requested overlapping columns and can leave an unidentifiable coefficient allocation.",
    "ar1_rho": "Known Gaussian-identity residual AR(1) correlation, strictly between -1 and 1. Zero disables whitening; it is supplied, not estimated.",
    "ar_start": "Boolean training-row vector marking starts of independent AR(1) sequences. None treats the observations as one ordered sequence.",
    "max_step_halving": "Maximum line-search halvings in supported coefficient fitting; defaults to control.mgcv_half.",
    "positive_transform": "Positive coefficient map for SCAM constraints: exp (default) or softplus; it acts on optimizer coordinates, not on response means.",
    "softplus_beta": "Positive slope/scale parameter for the softplus coefficient map; defaults to control.scam_b_notexp.",
    "softplus_threshold": "Positive numerical threshold beyond which the stabilized softplus coefficient map uses its linear branch; defaults to control.scam_threshold_notexp.",
    "start": "Optional starting coefficient vector in the coordinate system required by the selected coefficient solver. Shape must match the compiled coefficients; constrained fits use optimizer coordinates.",
    "trace": "Enable supported solver trace output. When omitted, the selected route may inherit control.trace.",
    "xt": "Additional basis setup in formula-free construction. Formula terms can provide term-specific xt mappings; these carry boundary, graph, low-rank or adaptive-basis options.",
}


def core_parameter_rows():
    from public_reference import GAM_PARAMETERS, literal

    from nampy.gam import GAM
    from nampy.gam.model.api import _GAM_HPARAM_KEYS

    model = GAM()
    meanings = {**GAM_PARAMETERS, **CORE_EXTRA_MEANINGS}
    defaults = {
        name: getattr(model, name) for name in _GAM_HPARAM_KEYS if hasattr(model, name)
    }
    defaults.update(
        control=None,
        para_pen=None,
        constraints=None,
        apply_side_conditions=True,
        trace=False,
    )
    rows = []
    for name in sorted(_GAM_HPARAM_KEYS):
        if name not in defaults or name not in meanings:
            raise ValueError(f"Missing core GAM contract: {name}")
        value = defaults[name]
        rows.append(
            (
                f"``{name}``",
                f"``{parameter_type(name, value)}``",
                f"``{literal(value)}``",
                meanings[name],
            )
        )
    return rows


def neural_family_returns(name, instance):
    width = str(instance.param_count)
    natural_width = width
    point_shape = "(n,)"
    if name == "beinf":
        natural_width = "5"
    if name in {"categorical", "ordinal"}:
        width = "C" if name == "categorical" else "1"
        natural_width = "C"
    elif name == "dirichlet":
        width = natural_width = "D"
        point_shape = "(n, D)"
    elif name == "mvnormdiag":
        width = natural_width = "2*D"
        point_shape = "(n, D)"
    elif name == "quantile":
        width = natural_width = "Q"
    rows = [
        (
            "raw input",
            "torch.Tensor[floating]",
            f"(n, {width})",
            f"Untransformed network predictors for {name}; columns follow param_names: {list(instance.param_names)}.",
        ),
        (
            "forward(raw)",
            "torch.Tensor[floating]",
            f"(n, {natural_width})",
            (
                "Class probabilities in class order."
                if name in {"categorical", "ordinal"}
                else "Natural columns mu, phi, pi0, pi1, pi_interior; four raw predictors." if name=="beinf" else "Transformed parameters in param_names order. dtype/device follow the supplied tensor."
            ),
        ),
        (
            "compute_loss(raw, y)",
            "torch.Tensor[floating]",
            "()",
            "Scalar batch training objective, averaged over rows.",
        ),
        (
            "compute_loss_per_sample(raw, y)",
            "torch.Tensor[floating]",
            "(n,)",
            "One loss contribution per observation.",
        ),
        (
            "predict_point(raw)",
            "torch.Tensor",
            point_shape,
            (
                "Integer class index maximizing probability."
                if name in {"categorical", "ordinal"}
                else "Family-defined floating point prediction; see the conditional mean/point contract above."
            ),
        ),
    ]
    from nampy.neural.objectives import ParameterizedObjective

    if type(instance).predict_spread is ParameterizedObjective.predict_spread:
        rows.append(
            (
                "predict_spread",
                "not a scalar spread API",
                "not applicable",
                "This family retains the base hook, which raises NotImplementedError. For quantile models, gaps between requested quantiles are separate interval summaries, not a predict_spread return value.",
            )
        )
    else:
        rows.append(
            (
                "predict_spread(raw)",
                "torch.Tensor[floating]",
                "(n,)",
                (
                    "Conditional response SD; a nonexistent second moment produces NaN."
                    if name in {"studentt", "loglogistic", "inversegamma"}
                    else "Conditional response standard deviation, distinct from parameter-estimation uncertainty."
                ),
            )
        )
    return rows


FAMILY_NUISANCE_MEANINGS = {
    "scat": {
        "theta": "Length-two natural vector [degrees_of_freedom, scale]. Two positive values fix both. Any negative entry requests estimation from absolute constructor values; None/zero uses source initialization. Working parameters are log(df-min_df) and log(scale)."
    },
    "nb": {
        "theta": "Positive negative-binomial shape, with Var(Y)=mu+mu**2/theta. The class defaults to theta=1 and estimate_theta=False; the nb registry name defaults to estimation. Nonpositive registry values request estimation with an absolute-value/one starting value."
    },
    "negbin": {
        "theta": "Required explicit positive negative-binomial shape in this registry form. Fixed by default; estimate_theta=True requests estimation. This differs from the nb alias."
    },
    "tw": {
        "theta": "Natural Tweedie power convention: a positive theta in (a,b) fixes p=theta. Negative theta starts estimation at p=abs(theta); None/zero estimates power starting at the midpoint of (a,b). Internally, log((p-a)/(b-p)) is optimized."
    },
    "betar": {
        "theta": "Beta precision. Positive fixes precision; None/zero estimates it starting at log precision zero; negative starts estimation at log(abs(theta)). Var(Y)=mu*(1-mu)/(1+precision)."
    },
    "ocat": {
        "theta": "Cutpoint gap vector, length R-2. A supplied vector determines R=len(theta)+2; all positive gaps fix the cutpoints. Any negative gap requests estimation from log(abs(gaps)); zero/None uses estimated working gaps. Response labels must be integers 1..R, with R>=3."
    },
    "zip": {
        "theta": "Optional length-two working intercept/log-slope vector linking presence to log intensity. Any supplied pair fixes both parameters; None estimates both, starting at zero.",
        "b": "Nonnegative lower bound on the presence/log-intensity slope. Negative input is reset to zero.",
    },
    "cnorm": {
        "theta": "Response SD convention: positive fixes SD; None/zero estimates starting at log SD zero. A negative supplied value is retained as the initial working log SD, following source behavior."
    },
    "clog": {
        "theta": "Logistic scale nuisance parameter. Positive fixes scale; None or the family estimation convention requests scale fitting. The detailed source parameterization and censoring encoding below apply."
    },
    "bcg": {
        "theta": "Transformation/scale nuisance parameters for Box-Cox Gaussian censoring. The source positive constructor has a malformed-state restriction; use the explicitly documented putTheta working-state route in the runnable example."
    },
    "gammals": {
        "b": "Lower bound for log dispersion under the source lower-bounded log link; default -7. It is not a lower bound for Gaussian standard deviation."
    },
    "gumbls": {
        "b": "Lower bound for log scale under the source second log link; default -7. Identity log-scale linking follows its separate branch."
    },
    "gaulss": {
        "b": "Lower bound on standard deviation, default 0.01. With second predictor eta, sigma=b+exp(eta) and internal precision tau=1/sigma."
    },
    "shash": {
        "b": "Strictly positive scale floor, default 0.01. Natural sigma=exp(eta_scale)+b."
    },
    "twlss": {
        "b": "Upper bound on natural Tweedie power, default 1.99; require 1<a<b<2."
    },
}

PLOT_PARAMETER_MEANINGS = {
    "residuals": "Draw weighted working partial residuals behind the fitted term curve.",
    "rug": "Draw observed feature locations; None uses the prepared-data default.",
    "se": "True uses the standard band convention; False suppresses bands; a positive number requests that many standard errors where supported.",
    "pages": "Requested figure-page count; zero uses the renderer automatic layout.",
    "select": "Optional zero-based smooth-panel index used by the Python renderer; None includes all plottable panels.",
    "scale": "-1 uses common y-axis limits across compatible curves unless ylim is supplied; other values retain term-specific scaling behavior.",
    "n": "Positive number of grid locations for a univariate curve.",
    "n2": "Positive number of grid locations along each axis of a two-dimensional surface.",
    "n3": "Number of conditioning/slice locations for supported higher-dimensional displays.",
    "theta": "Horizontal perspective viewing angle in degrees.",
    "phi": "Vertical perspective viewing angle in degrees.",
    "jit": "Jitter observed locations when constructing supported rug displays.",
    "xlab": "Optional horizontal-axis label.",
    "ylab": "Optional effect-axis label.",
    "main": "Optional panel title.",
    "ylim": "Optional (lower, upper) effect-axis limits; overrides common automatic limits.",
    "xlim": "Optional (lower, upper) covariate-axis limits.",
    "too_far": "Mask surface-grid points too far from observed covariate support; zero disables distance masking.",
    "shade_col": "Matplotlib color used to fill uncertainty bands.",
    "shift": "Additive display shift applied before trans; fitted predictions remain unchanged.",
    "trans": "Optional numerical display transformation callable, applied after shift; None leaves the displayed scale unchanged.",
    "se_with_mean": "Include available mean/intercept uncertainty in ordinary smooth bands. Shape-constrained routes retain their documented restrictions.",
    "unconditional": "Request supported smoothing-parameter uncertainty corrections for plot bands.",
    "by_resids": "Apply the supported by-variable convention to partial residual displays.",
    "scheme": "Basis-specific rendering scheme, scalar or one per smooth; the meaning depends on curve, tensor, MRF or spherical geometry.",
    "figsize": "Optional Matplotlib figure size in inches, as (width, height).",
}


def plot_parameter_rows():
    from public_reference import literal

    from nampy.gam.diagnostics import plot_gam

    rows = []
    overrides = {
        "select": "int | None",
        "se": "bool | float",
        "scale": "int | float",
        "trans": "callable | None",
        "ylim": "tuple[float, float] | None",
        "xlim": "tuple[float, float] | None",
        "scheme": "int | Sequence[int]",
    }
    for name, p in inspect.signature(plot_gam).parameters.items():
        if name == "model":
            continue
        typ = overrides.get(name) or parameter_type(
            name, p.default, annotation=p.annotation
        )
        rows.append(
            (
                f"``{name}``",
                f"``{typ}``",
                f"``{literal(p.default)}``",
                PLOT_PARAMETER_MEANINGS[name],
            )
        )
    return rows


def statistical_family_returns(name, family):
    nlp = int(getattr(family, "n_linear_predictors", 1))
    general = getattr(family, "family_class", "") == "general"
    raw = f"(n, {nlp})" if general else "(n,)"
    if name == "multinom":
        raw = "(n, K)"
    rows = [
        (
            "constructor",
            "BaseFamily subclass",
            "scalar object",
            "Independent mutable response-family state, not predictions.",
        ),
        (
            "GAM.predict(type='link')",
            "numpy.ndarray[float64]",
            raw,
            "Additive predictor columns, before family response transformations.",
        ),
    ]
    if name == "ocat":
        shape = "(n, R)"
        meaning = "Category probabilities for integer labels 1..R, in that order. This is not a scalar expected category."
    elif name == "multinom":
        shape = "(n, K+1)"
        meaning = "Class probabilities p0,...,pK; link predictors have K non-reference log-odds columns."
    elif name == "ziplss":
        shape = "(n,)"
        meaning = "Expected hurdle-process counts. Natural parameter prediction is a separate two-column interface."
    elif name in {"gaulss", "gammals", "gumbls", "gevlss", "shash", "twlss"}:
        shape = raw
        meaning = {
            "gaulss": "Source response columns (mu, tau), where tau=1/sigma. Natural output uses sigma instead.",
            "gammals": "Source response columns (mu, log(phi)), where Var(Y)=phi*mu**2. Natural GAMLSS output is (mu, sigma), with sigma=sqrt(phi), the coefficient of variation.",
            "gumbls": "Source response columns (location, log_scale); natural output exponentiates scale.",
            "gevlss": "Source response columns (mu, log_sigma, xi); natural output uses sigma.",
            "shash": "Source response columns (mu, log_sigma, epsilon, log_delta); natural output uses sigma and delta.",
            "twlss": "Source response columns (mu, theta, rho); natural output maps power p and dispersion phi.",
        }[name]
    elif name == "bcg":
        shape = "(n,)"
        meaning = "Gaussian location on the Box-Cox transformed scale, without an inverse Box-Cox mean correction."
    elif name == "zip":
        shape = "(n,)"
        meaning = "Expected hurdle count p*lambda/(1-exp(-lambda)); stored fitted/link values remain on log-intensity scale."
    else:
        shape = "(n,)"
        meaning = (
            "Family response mean/location in its documented response/link convention."
        )
    rows.append(
        ("GAM.predict(type='response')", "numpy.ndarray[float64]", shape, meaning)
    )
    if general:
        columns = getattr(family, "distribution_parameter_names", None) or getattr(
            family, "parameter_names", ()
        )
        width = "K+1" if name == "multinom" else str(len(columns))
        meaning = (
            "Natural columns: "
            + ", ".join(columns)
            + ". Used by GAMLSS.predict and distribution_parameters_from_eta."
        )
        if name == "gammals":
            meaning += " sigma is the positive coefficient of variation: sigma=exp(log_dispersion/2), Var(Y)=(mu*sigma)**2; it is not a Gaussian SD or Gamma rate."
        if name == "gaulss":
            meaning += " sigma is the positive response standard deviation."
        rows.append(
            (
                "Natural parameter interface",
                "numpy.ndarray[float64]",
                f"(n, {width})",
                meaning,
            )
        )
    return rows
