"""Generate estimator and capability references from the installed public API."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def literal(value):
    if inspect.isclass(value) or inspect.isfunction(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value).replace("\n", " ")


def descriptions(obj):
    """Read parameter descriptions from NumPy-style source docstrings."""
    result = {}
    current = []
    for line in (inspect.getdoc(obj) or "").splitlines():
        match = re.match(r"^(\*{0,2}\w+(?:,\s*\w+)*)(?:\s*:\s*.+)?$", line)
        if match:
            current = [name.strip().lstrip("*") for name in match[1].split(",")]
            for name in current:
                result[name] = ""
        elif line.startswith(" ") and current:
            for name in current:
                result[name] += " " + line.strip()
        elif line.strip():
            current = []
    return {key: value.strip() for key, value in result.items()}


def table(headers, rows):
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :class: api-reference-table",
        "",
    ]
    for row in [headers, *rows]:
        for index, value in enumerate(row):
            text = str(value).replace("\n", " ")
            lines.append(("   * - " if index == 0 else "     - ") + text)
    return "\n".join(lines) + "\n\n"


DEPENDENCE = {
    "namformer": "Feature-specific marginal heads plus named interactions and an all-feature global_transformer term.",
    "natt": "Numerical terms are feature-specific; categorical attention readouts can depend on other categorical inputs.",
    "spam": "Low-rank polynomial blocks; interpret their configured feature subsets, not independent unary networks.",
    "nbm_spam": "Shared feature concepts combined by polynomial heads; block interpretation depends on configured degrees.",
}


PARAMETER_MEANINGS = {
    "feature_widths": "Explicit positive first-layer widths indexed by source-feature name; override adaptive width selection.",
    "num_basis_functions": "Positive upper bound on the first-layer width when NAM adaptive_width is enabled.",
    "units_multiplier": "Positive multiplier of fitted feature cardinality in adaptive first-layer width selection.",
    "regularize_interactions": "Apply the NAM output/parameter regularizers to interaction subnetworks as well as main effects.",
    "aggregation": "How outputs of the configured members or terms are combined.",
    "archipelago_baseline": "Reference configuration used to score Archipelago interaction candidates.",
    "basis_l1_regularization": "L1 penalty strength on learned basis/projection coefficients.",
    "boost_rate": "Multiplier applied to each native boosting update.",
    "cg_max_iter": "Maximum conjugate-gradient iterations; None uses the solver default.",
    "cg_rtol": "Relative convergence tolerance for the conjugate-gradient solver.",
    "clip_predictions": "Positive numerical limit used by native IGANN classification updates.",
    "device": "Device used by the architecture-native solver.",
    "early_stopping": "Number of non-improving native boosting stages allowed.",
    "elm_alpha": "Nonnegative ridge penalty for extreme-learning-machine stage coefficients.",
    "elm_random_state": "Seed for the fixed random extreme-learning-machine features.",
    "elm_scale": "Scale of fixed random extreme-learning-machine hidden weights.",
    "execution_mode": "SIAN execution layout, including block-masked versus independent subnetworks.",
    "feature_dropout": "Dropout probability applied to contribution outputs; masking semantics depend on architecture.",
    "feature_output_bias": "Whether a feature head includes its own output bias.",
    "group_lasso_include_interactions": "Include interaction subnetworks in the group penalty and group-norm report.",
    "group_lasso_lambda": "Nonnegative coefficient of the sum of term-group parameter norms.",
    "heredity_fraction": "SIAN heredity filter for admitting higher-order interaction candidates.",
    "identify": "Apply the spline-layer identification/centering convention.",
    "ignore_unary": "Omit the direct unary block from the polynomial representation.",
    "init_reg": "Regularization strength for the initial native linear fit; binary IGANN requires a positive value.",
    "interaction_degree": "Maximum order of automatically enumerated explicit source-feature interactions.",
    "interaction_detector": "SIAN candidate-selection method; explicit interactions bypass discovery.",
    "interaction_thresholds": "Interaction acceptance threshold, globally or indexed by order.",
    "interactions": "Explicit tuples of source-feature names; each term receives those inputs jointly.",
    "intercept": "Include a trainable predictor intercept.",
    "kernel_width": "Kernel length scale; auto follows the fitted-input scale rule described in the GPNAM guide.",
    "kernel_widths": "Optional feature-wise kernel length scales, overriding the common width.",
    "l1_regularization": "Coefficient of the architecture L1 parameter penalty.",
    "learn_knots": "Allow spline knot locations to be learned rather than fixed.",
    "lower_order_correction": "Remove lower-order contributions from polynomial products.",
    "lr_schedule": "Learning-rate schedule used by gradient training; see the shared training contract.",
    "max_candidates": "Maximum number of interaction candidates considered during discovery.",
    "max_interaction_order": "Highest interaction order considered by SIAN discovery.",
    "max_terms_per_order": "Optional cap on accepted interaction terms at each order.",
    "n_estimators": "Maximum number of stages in the architecture-native boosting fit.",
    "n_hid": "Number of random hidden units in each extreme-learning-machine stage.",
    "num_estimators": "Number of jointly trained additive-tree members.",
    "optimizer": "Gradient optimizer selection; see the shared training contract for accepted names.",
    "orthogonal": "Constrain polynomial projection matrices with an orthogonal parameterization.",
    "proximal": "Use the architecture-supported proximal regularization update.",
    "ranks": "Low-rank widths by polynomial head; in SPAM ranks[i] corresponds to degree i+2.",
    "reference_batch_size": "Batch size for the interaction-detector reference network.",
    "reference_device": "Device for the interaction-detector reference network.",
    "reference_epochs": "Maximum reference-network epochs before interaction discovery.",
    "reference_layer_sizes": "Hidden widths of the interaction-detector reference network.",
    "reference_lr": "Learning rate of the interaction-detector reference network.",
    "reference_weight_decay": "Weight decay for the interaction-detector reference network.",
    "reg_order": "Norm order used by the polynomial regularization penalty; it is not polynomial degree.",
    "regularization_scale": "Multiplier of the polynomial model parameter regularizer.",
    "residual_layer_sizes": "Hidden widths of the optional SIAN residual reference network.",
    "residual_network": "Include the optional SIAN residual network.",
    "rff_num_feat": "Number of random Fourier features used to approximate each kernel block.",
    "rff_random_state": "Random seed for Fourier frequency and phase construction.",
    "rff_scheme": "Sampling scheme used to construct random Fourier features.",
    "ridge": "Nonnegative ridge penalty in the fixed linear-design solve.",
    "selection_batch_size": "Batch size when scoring candidate interactions.",
    "selection_max_pairs": "Maximum sample-pair budget used for interaction scoring.",
    "selection_max_samples": "Maximum observations used during interaction selection.",
    "selection_output_index": "Output dimension examined by the interaction detector.",
    "smoothing": "Coefficient of the spline roughness penalty.",
    "solver": "Architecture-native solver choice; cg uses a regularized fixed-design solve for supported GPNAM regression.",
    "spam_dropout": "Dropout probability in the polynomial heads.",
    "threshold_mode": "Interpret interaction thresholds as relative fractions or according to the configured detector policy.",
    "tree_depth": "Number of binary decision levels in each soft tree.",
    "tree_lamda": "Coefficient of the soft-tree split regularization penalty.",
    "tree_temperature": "Controls the softness of tree routing during training.",
    "use_geometric_mean": "Normalize polynomial products using the configured geometric-mean representation.",
    "use_hard_routing_in_eval": "Replace soft branch probabilities with discrete routing during evaluation.",
}


def estimator_text(cls, spec, objective):
    from pretab.preprocessor import Preprocessor

    from nampy.models._base import NeuralEstimatorBase
    from nampy.neural.configs.nam_config import DefaultNAMConfig
    from nampy.neural.configs.nbm_config import DefaultNBMConfig

    estimator = cls()
    params = estimator.get_params(deep=False)
    docs = {
        **descriptions(DefaultNAMConfig),
        **descriptions(DefaultNBMConfig),
        **descriptions(Preprocessor),
        **PARAMETER_MEANINGS,
        **descriptions(spec.config),
    }
    model_page = spec.name
    if model_page == "ensemble_treenam":
        model_page = "ensemble_treenam"
    lines = [
        f"Architecture: ``{spec.name}``. Task: **{objective}**.\n",
        f":doc:`Architecture, restrictions, and interpretation </models/{model_page}>`.\n",
        DEPENDENCE.get(
            spec.name,
            "Main-effect terms depend on their named source feature; explicit interaction terms use their configured feature subset.",
        ),
        "\nConstructor parameters\n~~~~~~~~~~~~~~~~~~~~~~\n",
        "Defaults below are resolved from the actual estimator, including configuration factories and registry overrides. Unknown names raise ``TypeError``.\n",
    ]
    if spec.input_requirements:
        lines.insert(
            3,
            "\n"
            + table(
                ["Input contract", "Requirement"], list(spec.input_requirements.items())
            ),
        )
    if spec.fixed_family is not None:
        lines.insert(
            3, f"This architecture requires ``family={spec.fixed_family!r}``.\n"
        )
    rows = []
    for name, default in sorted(params.items()):
        description = docs.get(name, docs.get(name.split("__", 1)[-1], ""))
        if name == "family":
            description = "Neural response family alias; see /api/utils for support and parameter domains."
        elif name == "distributional_kwargs":
            description = "Family constructor options, configured here rather than in fit(); quantile models use quantiles=[...]."
        if not description:
            description = f"Configuration option of type {type(default).__name__}; see the architecture reference for its role and restrictions."
        rows.append((f"``{name}``", f"``{literal(default)}``", description))
    lines.append(table(["Parameter", "Effective default", "Meaning"], rows))
    if spec.name == "igann":
        lines.append(
            "This LSS task uses gradient training with solver='auto'; the native solver supports squared-error regression and binary classification only.\n"
            if objective == "distributional"
            else "The default solver='auto' uses native boosting for squared-error regression and binary classification; n_estimators limits stages. Multiclass classification uses gradient training, as does solver='gradient'.\n"
        )
    elif spec.name == "gpnam":
        lines.append(
            "Default regression uses the fixed-design conjugate-gradient solver (solver='cg'); optimizer and scheduler controls apply when solver='gradient'.\n"
            if objective == "regression"
            else "Classification and LSS use gradient training; the fixed-design conjugate-gradient route is restricted to default-loss regression.\n"
        )
    lines += [
        "Training controls\n~~~~~~~~~~~~~~~~~\n",
        "The common training controls below apply to gradient-trained estimators. Native solvers such as IGANN use their architecture-specific stage controls.\n",
        table(
            ["Fit parameter", "Default", "Meaning"],
            [
                (
                    f"``{name}``",
                    "Optional keyword arguments"
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD
                    else "Required"
                    if parameter.default is inspect.Parameter.empty
                    else f"``{literal(parameter.default)}``",
                    descriptions(NeuralEstimatorBase.fit).get(
                        name,
                        "See the shared training contract for this fit control.",
                    ),
                )
                for name, parameter in inspect.signature(
                    NeuralEstimatorBase.fit
                ).parameters.items()
                if name != "self"
            ],
        ),
        ":doc:`Full training contract </api/neural_training>`.\n",
        "Predictions and explanations\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
    ]
    if objective == "regression":
        shape = "``(n_samples,)`` for a scalar target; ``(n_samples, n_outputs)`` for multi-output regression. Identity response scale; score is R²."
    elif objective == "classification":
        shape = "``predict`` returns class labels; ``predict_proba`` returns ``(n_samples, n_classes)`` probabilities in ``classes_`` order. Component extraction is binary-only and describes logits, not additive probabilities."
    else:
        shape = "``predict`` returns natural distribution parameters; ``predict(raw=True)`` returns additive predictors. For fixed scalar families, natural columns follow ``family_.param_names``. Categorical and ordinal outputs instead have one probability per class; multivariate families depend on configured dimension, and quantiles have one column per requested level. Raw width follows ``family_.param_count``. ``predict_point`` returns the family-defined point prediction."
    if spec.fixed_family == "quantile":
        shape = "``predict`` returns one conditional quantile per configured level. Training minimizes pinball loss; ``evaluate`` labels it ``pinball_loss`` (summed over levels, averaged over rows). These quantiles do not specify a complete probability density. ``predict_components`` retains the quantile dimension, which the single-column ``plot_terms`` renderer skips."
    lines += [
        shape + "\n",
        "``predict_components`` returns :class:`nampy.contracts.AdditivePrediction`; named contributions plus intercept and offset reconstruct its **link**. Metadata records known source dependence. Reconstruction does not establish identifiability or stability. Centering preserves predictions.\n",
        "Fitted attributes\n~~~~~~~~~~~~~~~~~\n",
        "``feature_names_in_`` and ``n_features_in_`` record the input schema. ``model`` contains the fitted objective/module; ``data_module`` owns preprocessing. Classifiers expose ``classes_``; LSS estimators expose ``family_`` and ``distributional_kwargs_``. ``model_complexity()`` returns fitted parameter counts.\n",
        "Methods\n~~~~~~~\n",
    ]
    methods = [
        "predict",
        "score",
        "evaluate",
        "predict_components",
        "explain_terms",
        "term_importance",
        "plot_terms",
        "save_model",
        "load_model",
    ]
    if objective == "classification":
        methods += ["predict_proba"]
    if objective == "distributional":
        methods += ["predict_point", "plot_distributional_effects"]
    if spec.name == "snam":
        methods += ["group_norms"]
    for method in methods:
        if hasattr(cls, method):
            lines.append(
                f".. automethod:: nampy.models.{cls.__name__}.{method}\n   :no-index:\n\n"
            )
    lines += [
        "Complete example\n~~~~~~~~~~~~~~~~\n",
        ".. code-block:: python\n\n",
        "   import numpy as np\n   import pandas as pd\n",
        f"   from nampy.models import {cls.__name__}\n\n",
        '   rng = np.random.default_rng(7)\n   X = pd.DataFrame({"x": np.linspace(-1, 1, 240)})\n',
    ]
    if objective == "classification":
        lines.append('   y = rng.binomial(1, 1 / (1 + np.exp(-3 * X["x"])))\n')
    else:
        lines.append('   y = np.sin(3 * X["x"]) + rng.normal(0, 0.15, len(X))\n')
    lines += [
        f"   model = {cls.__name__}(n_estimators=50)\n"
        if spec.name == "igann" and objective != "distributional"
        else f"   model = {cls.__name__}()\n",
        "   model.fit(X, y, random_state=7)\n"
        if (spec.name == "igann" and objective != "distributional")
        or (spec.name == "gpnam" and objective == "regression")
        else "   model.fit(X, y, max_epochs=50, random_state=7)\n",
        "   prediction = model.predict(X)\n",
        "   components = model.predict_components(X)\n",
        '   model.save_model("fitted.nampy")\n',
        f'   restored = {cls.__name__}.load_model("fitted.nampy")\n\n',
        "These rows illustrate the interface; use held-out data to evaluate predictive quality. Persistence files use pickle and must come from a trusted source.\n",
    ]
    return "\n".join(lines)


GAM_PARAMETERS = {
    "formula": "Formula string; GAMLSS also accepts a mapping from parameter names to formulas. None creates one smooth per input column.",
    "family": "Statistical response family. Regressor accepts single-predictor regression families; classifier accepts binomial; GAMLSS resolves location-scale aliases.",
    "k": "Basis dimension in formula-free mode; formula terms can supply their own dimensions.",
    "basis": "Smooth basis code for formula-free terms, such as tp or cr.",
    "fit_intercept": "Include a predictor intercept in formula-free mode.",
    "optimize_smoothing": "Select smoothing parameters automatically when True; differs from the raw GAM default of False.",
    "smoothing_method": "Selection criterion, e.g. fixed, gcv, ubre, ml, reml, or laml, subject to family/structure support.",
    "smoothing_optimizer": "Outer selection route, e.g. outer_newton, bfgs, efs, optim, or nlm; see the runtime-checked matrix.",
    "smoothing_params": "Fixed penalty strengths when automatic optimization is disabled.",
    "select": "Add supported null-space selection penalties; transformed-coefficient restrictions apply.",
    "knots": "Optional user-supplied knot information for supported bases.",
    "min_sp": "Lower bounds on smoothing parameters, where supported by the selection route.",
    "drop_intercept": "Optional per-predictor intercept policy used by the compiled formula model.",
    "covariance": "Prediction covariance convention: bayes or frequentist as supported by the statistical model.",
    "score_gamma": "Complexity multiplier used in smoothing-selection score calculations.",
    "max_irls_iter": "Maximum penalized iteratively reweighted least-squares iterations.",
    "irls_tol": "Convergence tolerance for coefficient fitting.",
    "sp_log_bounds": "Lower and upper bounds in log smoothing-parameter coordinates.",
    "scale": "Known statistical dispersion/scale when positive; zero requests the family-specific estimation policy.",
    "control": "Optional GAM solver control settings from gam_control().",
    "nei": "Neighborhood specification for supported NCV/QNCV routes.",
    "coefficient_optimizer": "Inner coefficient solver, including newton or supported SCAM bfgs.",
    "optim_method": "SciPy/R-optim-compatible method choice for the optim outer route.",
}


def statistical_text(cls):
    name = cls.__name__
    rows = [
        (f"``{parameter}``", f"``{literal(value)}``", GAM_PARAMETERS[parameter])
        for parameter, value in cls().get_params(deep=False).items()
    ]
    text = ":doc:`Statistical GAM workflow </tutorials/first_gam>` and :doc:`runtime-checked optimizer routes </development/optimizer_matrix>`.\n\n"
    text += "Constructor parameters\n~~~~~~~~~~~~~~~~~~~~~~\n\n" + table(
        ["Parameter", "Default", "Meaning"], rows
    )
    text += "Predictions and explanations\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    if name == "GAMLSS":
        text += "``predict`` returns natural parameters in ``parameter_names_`` order; ``predict(raw=True)`` returns the raw parameter predictors. ``predict_point`` returns the family-defined mean. Component arrays have shape ``(n_samples, n_parameters)`` and identify their parameter in structured metadata.\n\n"
    elif name == "GAMClassifier":
        text += "Binary classification: ``predict`` returns labels and ``predict_proba`` returns ``(n_samples, 2)`` probabilities in ``classes_`` order. Components are contributions to the link predictor rather than additive probability changes.\n\n"
    else:
        text += "``predict`` returns one response-scale mean per row. ``standard_errors`` returns pointwise prediction uncertainty in the requested scale; it is not future-outcome spread or a simultaneous confidence band.\n\n"
    text += "``gam_`` is the fitted statistical model. ``schema_`` records the validated input structure when the adapter can record a schema. Call explanation methods on the wrapper and use ``gam_`` for the complete statistical inference and diagnostic surface.\n\nMethods\n~~~~~~~\n\n"
    for method in [
        "fit",
        "predict",
        "predict_proba",
        "predict_point",
        "score",
        "evaluate",
        "standard_errors",
        "lpmatrix",
        "predict_components",
        "explain_terms",
        "term_importance",
        "summary",
        "plot",
        "save_model",
        "load_model",
    ]:
        if hasattr(cls, method):
            text += f".. automethod:: nampy.models.{name}.{method}\n   :no-index:\n\n"
    text += "Complete examples\n~~~~~~~~~~~~~~~~~\n\nSee the :doc:`complete fitting examples </tutorials/modeling_workflows>` for executable data setup, fitted predictions, component tables, and plots.\n"
    return text


def generate(root):
    from nampy import models
    from nampy.neural.registry import architectures

    output = root / "docs/_generated/reference"
    output.mkdir(parents=True, exist_ok=True)
    generated = root / "docs/api/generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name in ["GAMRegressor", "GAMClassifier", "GAMLSS"]:
        (output / f"{name}.inc").write_text(statistical_text(getattr(models, name)))
    matrix = []
    for name, spec in architectures().items():
        matrix.append(
            (
                f":doc:`{spec.estimator_prefix} </models/{name}>`",
                "; ".join(
                    label
                    for capability, label in [
                        ("regression", "regression"),
                        ("classification", "binary and multiclass prediction"),
                        ("distributional", "LSS"),
                    ]
                    if spec.supports(capability)
                ),
                "; ".join(
                    label
                    for capability, label in [
                        ("regression", "Regression components"),
                        ("classification", "Binary-classification components"),
                        ("distributional", "LSS link components"),
                    ]
                    if spec.supports(capability)
                )
                if spec.supports("additive_components")
                else "No additive explanation declared.",
                DEPENDENCE.get(
                    name, "Named features or explicitly configured subsets."
                ),
                "Native boosting for regression/binary classification; gradient training for multiclass, LSS, or solver='gradient'"
                if name == "igann"
                else "Default CG regression; gradient classification/LSS or solver='gradient'"
                if name == "gpnam"
                else "Gradient training",
                "Conditional quantiles and pinball loss; no complete probability density."
                if spec.fixed_family == "quantile"
                else "Family-dependent LSS parameters; conditional spread where the family defines it."
                if spec.supports("distributional")
                else "Response prediction; no LSS interface or automatic calibrated intervals.",
            )
        )
        for task, suffix in [
            ("regression", "Regressor"),
            ("classification", "Classifier"),
            ("distributional", "LSS"),
        ]:
            if not spec.supports(task):
                continue
            cls = getattr(models, spec.estimator_prefix + suffix)
            (output / f"{cls.__name__}.inc").write_text(
                estimator_text(cls, spec, task).rstrip() + "\n"
            )
            filename = generated / f"nampy.models.{cls.__name__}.rst"
            title = f"nampy.models.{cls.__name__}"
            text = (
                title
                + "\n"
                + "=" * len(title)
                + "\n\n.. currentmodule:: nampy.models\n\n"
                + f".. autoclass:: {cls.__name__}\n\n.. include:: /_generated/reference/{cls.__name__}.inc\n"
            )
            if (
                not filename.exists()
                or filename.read_text(encoding="utf-8-sig") != text
            ):
                filename.write_text(text)
    generate_families(root)
    (output / "model_capabilities.inc").write_text(
        table(
            [
                "Model",
                "Prediction tasks",
                "Explanation support",
                "Feature dependence",
                "Fitting",
                "Uncertainty",
            ],
            matrix,
        ).rstrip()
        + "\n"
    )


# These meanings complement registry-derived aliases, parameter order, and
# actual transform configuration. Statistical GAMLSS uses different families.
FAMILY_MEANINGS = {
    "normal": (
        "Real scalar",
        "mean",
        "standard deviation",
        "Normal negative log likelihood",
    ),
    "robustnormal": (
        "Real scalar",
        "mean",
        "No scalar predict_spread hook",
        "Normal negative log likelihood",
    ),
    "studentt": (
        "Real scalar",
        "location",
        "Student-t SD where df > 2; undefined otherwise",
        "Student-t negative log likelihood",
    ),
    "poisson": (
        "Nonnegative integer",
        "rate",
        "square root of rate",
        "Poisson negative log likelihood",
    ),
    "negativebinom": (
        "Nonnegative integer",
        "mean",
        "SD from mean and dispersion",
        "Negative-binomial negative log likelihood",
    ),
    "gamma": (
        "Positive scalar",
        "shape / rate",
        "sqrt(shape) / rate",
        "Gamma negative log likelihood",
    ),
    "inversegamma": (
        "Positive scalar",
        "rate / (shape - 1), numerically safeguarded",
        "SD where shape > 2; undefined otherwise",
        "Inverse-gamma negative log likelihood",
    ),
    "beta": (
        "Unit interval; numerical target clipping applies",
        "alpha / (alpha + beta)",
        "Beta standard deviation",
        "Beta negative log likelihood",
    ),
    "lognormal": (
        "Positive scalar",
        "exp(loc + scale squared / 2)",
        "Lognormal standard deviation",
        "Lognormal negative log likelihood",
    ),
    "loglogistic": (
        "Positive scalar",
        "scale (median)",
        "SD where shape > 2; undefined otherwise",
        "Log-logistic negative log likelihood",
    ),
    "weibull": (
        "Positive scalar",
        "scale times Gamma(1 + 1/shape)",
        "Weibull standard deviation",
        "Weibull negative log likelihood",
    ),
    "zip": (
        "Nonnegative integer",
        "(1 - zero_prob) times rate",
        "Mixture standard deviation",
        "Zero-inflated Poisson negative log likelihood",
    ),
    "zinb": (
        "Nonnegative integer",
        "(1 - zero_prob) times mean",
        "Mixture standard deviation",
        "Zero-inflated negative-binomial negative log likelihood",
    ),
    "hurdlepoisson": (
        "Nonnegative integer",
        "Positive-process mean weighted by nonzero probability",
        "Not implemented by the family",
        "Hurdle Poisson negative log likelihood",
    ),
    "hurdlenegativebinom": (
        "Nonnegative integer",
        "Truncated positive-process mean weighted by nonzero probability",
        "Not implemented by the family",
        "Hurdle negative-binomial negative log likelihood",
    ),
    "tweedie": (
        "Nonnegative scalar",
        "mean",
        "sqrt(dispersion times mean to variance_power)",
        "Compound Poisson-Gamma negative log likelihood; truncated series",
    ),
    "categorical": (
        "Integer class index or one-hot target",
        "Most likely class",
        "Not a scalar spread",
        "Categorical cross-entropy",
    ),
    "ordinal": (
        "Ordered integer class index",
        "Most likely class",
        "Not a scalar spread",
        "Cumulative-logit negative log likelihood",
    ),
    "dirichlet": (
        "Positive composition summing to one",
        "Normalized concentration vector",
        "Not implemented by the family",
        "Dirichlet negative log likelihood",
    ),
    "mvnormdiag": (
        "Real vector with configured n_dim",
        "Location vector",
        "Not implemented by the family",
        "Independent multivariate normal negative log likelihood",
    ),
    "quantile": (
        "Real scalar",
        "Median when requested, otherwise average requested quantiles",
        "Use gaps between requested quantiles",
        "Pinball loss; no probability density",
    ),
}


def generate_families(root):
    from nampy.neural.distributions.registry import FAMILY_REGISTRY

    lines = [
        "Neural family contracts\n-----------------------\n\n",
        "Aliases, parameter order, and default transforms below are read from the installed registry and family instances.\n",
        "Estimator-specific overrides are shown on estimator pages. The positive transform is softplus plus epsilon; probabilities use the family-specific simplex or unit-interval transform. These are neural parameters, distinct from statistical GAM/GAMLSS parameterizations.\n",
        "Quantile, categorical, and ordinal outputs have special widths; inspect the fitted family rather than assuming a common location-scale layout.\n\n",
    ]
    for name, spec in FAMILY_REGISTRY.items():
        y = (
            np.ones((6, 3)) / 3
            if name in {"dirichlet", "mvnormdiag"}
            else np.array([0, 1, 2, 0, 1, 2])
        )
        instance, _ = spec.instantiate(y)
        support, point, spread, loss = FAMILY_MEANINGS[name]
        transforms = {
            key: literal(value)
            for key, value in vars(instance).items()
            if key.endswith("_transform")
        }
        parameters = ", ".join(instance.param_names)
        if name in {"dirichlet", "mvnormdiag", "categorical"}:
            parameters += " (illustrated with three dimensions/classes)"
        if name == "ordinal":
            parameters += (
                " (raw predictor); transformed prediction has one probability per class"
            )
        rows = [
            ("Alias", f"``{name}``"),
            ("Target support", support),
            ("Parameter order", parameters),
            (
                "Default transforms",
                ", ".join(f"{key}: {value}" for key, value in transforms.items())
                or (
                    "First raw value is free; later quantiles add cumulative softplus increments when enforce_monotonic=True. QNAM uses ordered term outputs and disables this second transform."
                    if name == "quantile"
                    else "Family-specific probability transformation; inspect the family contract."
                ),
            ),
            ("Point prediction", point),
            ("Spread", spread),
            ("Loss", loss),
        ]
        title = f"{spec.distribution.__name__} ({name})"
        lines.extend(
            [
                title + "\n" + "~" * len(title) + "\n\n",
                table(["Property", "Meaning"], rows),
            ]
        )
    (root / "docs/_generated/reference/neural_families.inc").write_text(
        "\n".join(lines).rstrip() + "\n"
    )


def materialize_defaults(app, what, name, obj, options, signature, return_annotation):
    if inspect.isclass(obj) and is_dataclass(obj):
        try:
            instance = obj()
            parameters = [
                p.replace(default=getattr(instance, p.name))
                if p.name in {f.name for f in fields(obj)}
                else p
                for p in inspect.signature(obj).parameters.values()
            ]
            return str(
                inspect.signature(obj).replace(parameters=parameters)
            ), return_annotation
        except (TypeError, ValueError):
            return None
    return None


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.connect("builder-inited", lambda app: generate(Path(app.srcdir).parent))
    app.connect("autodoc-process-signature", materialize_defaults)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
