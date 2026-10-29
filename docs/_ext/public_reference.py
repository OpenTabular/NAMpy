"""Generate estimator and capability references from the installed public API."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

from reference_schema import (
    estimator_returns,
    method_parameter_rows,
    numpy_types,
    parameter_type,
)


def literal(value):
    if value is inspect.Parameter.empty:
        return "required"
    if inspect.isclass(value) or inspect.isfunction(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value).replace("\n", " ")


def constructor_defaults(cls):
    """Effective factory defaults, or signature defaults for required inputs."""
    parameters = inspect.signature(cls).parameters
    if any(p.default is inspect.Parameter.empty
           and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
           for p in parameters.values()):
        return {name: p.default for name, p in parameters.items()}
    return cls().get_params(deep=False)


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
    "semistructured": "Joint fixed-penalty structured effects and a learned global neural representation, residualized against all structured columns on training rows. New-data predictions use the saved training projection. See :doc:`/user_guide/semistructured` for the scalar-response, categorical-input and classification scope.",
    "namformer": "Feature-specific marginal heads plus named interactions and an all-feature global_transformer term.",
    "natt": "Numerical terms are feature-specific; categorical attention readouts can depend on other categorical inputs.",
    "spam": "Low-rank polynomial blocks; interpret their configured feature subsets, not independent unary networks.",
    "nbm_spam": "Shared feature concepts combined by polynomial heads; block interpretation depends on configured degrees.",
}


PARAMETER_MEANINGS = {
    "fisher_diagnostics": "False disables fit-time Fisher diagnostics; True uses FisherDiagnostics defaults; an explicit FisherDiagnostics config controls estimators, probe budgets and recording. Postfit fisher_information remains available after supported likelihood fits.",
    "structured_formula": "Right-hand-side statistical formula for the structured block on preprocessed training features; None selects all numeric linear effects.",
    "neural_features": "Distinct numeric source features supplied jointly to the learned representation; empty selects all.",
    "hidden_dims": "Hidden widths of the learned Tanh representation; empty selects a linear representation.",
    "representation_dim": "Number of learned neural representation columns before projection.",
    "neural_weight_decay": "Adam weight decay on neural parameters only; fixed spline roughness penalties are supplied separately.",
    "smoothing_params": "Fixed smoothing parameters for the compiled structured spline penalties.",
    "epochs": "Maximum full-cohort native optimization steps.",
    "k": "Default structured smooth basis dimension, overridden by explicit formula terms.",
    "basis": "Default structured smooth basis type, overridden by explicit formula terms.",
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
    "feature_dropout": "Probability of dropping a whole term per observation, sharing the mask across output columns; disabled during evaluation.",
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


PARAMETER_MEANINGS.update(
    {
        "architecture_config": "Architecture dataclass instance containing representation and regularization choices. No flat architecture overrides are accepted.",
        "objective": "ObjectiveConfig containing the loss, distribution family and task-specific weighting choices.",
        "training": "Route-specific execution configuration. None resolves the supported architecture/objective default using the observed class count.",
        "preprocessing": "PreprocessingConfig containing explicit PreTab transformation options; unspecified options use registry and PreTab defaults.",
        "objective__family": "Neural distribution family name; None uses the architecture default. Distributional tasks only.",
        "objective__distributional_kwargs": "Constructor options for the selected neural distribution family, including parameter transforms and quantile levels.",
        "objective__loss_fct": "Optional regression loss callable. It must support the selected target shape and per-observation values when weights are supplied.",
        "objective__class_weight": "Classification class-to-weight mapping or balanced weighting, multiplied by observation weights.",
        "objective__sampling_strategy": "Optional balanced classification batch sampling, independent of loss weighting.",
    }
)
PARAMETER_MEANINGS.update(
    {
        "training__" + name: meaning
        for name, meaning in {
            "random_state": "Seed for splitting, architecture initialization and training-data order.",
            "val_size": "Held-out fraction when explicit validation rows are absent. Exact fixed-design regression uses every training row.",
            "batch_size": "Number of observations per training batch.",
            "shuffle": "Shuffle training rows between epochs; validation rows are not shuffled.",
            "dataloader_kwargs": "Additional Torch DataLoader options, such as num_workers.",
            "max_epochs": "Maximum gradient-training epochs; it is not a native boosting-stage limit.",
            "max_steps": "Maximum optimizer steps; -1 leaves the step count unlimited.",
            "max_time": "Optional Lightning wall-clock training limit.",
            "patience": "Validation checks without improvement before epoch-based stopping.",
            "early_stopping_steps": "Optional patience measured in optimizer steps instead of epoch checks.",
            "monitor": "Logged scalar metric used for stopping and best-checkpoint selection.",
            "mode": "Whether smaller (min) or larger (max) monitored values are preferred.",
            "lr": "Gradient optimizer learning rate.",
            "lr_patience": "Plateau-scheduler checks without improvement before reducing the learning rate.",
            "lr_factor": "Multiplicative learning-rate reduction for the plateau scheduler.",
            "lr_schedule": "Gradient learning-rate schedule: plateau, inverse_sqrt, cosine, warmup_cosine or none.",
            "lr_warmup_steps": "Optimizer steps used for learning-rate warmup.",
            "lr_decay_steps": "Step-decay interval or cosine horizon; zero uses the supported inferred horizon.",
            "lr_decay_factor": "Multiplicative factor applied at each configured step-decay interval.",
            "weight_decay": "Optimizer weight decay, separate from explicit architecture loss penalties.",
            "optimizer": "Gradient optimizer name or compatible optimizer class.",
            "optimizer_kwargs": "Additional optimizer-specific constructor arguments.",
            "warm_start": "Reuse weights on subsequent fits, starting fresh on the first fit; a checkpoint path can supply initial weights.",
            "average_checkpoints": "Select the arithmetic mean of recent epoch-end weights instead of the single best checkpoint.",
            "n_last_checkpoints": "Number of recent epoch-end states retained for checkpoint averaging.",
            "pretrain_epochs": "Masked reconstruction epochs before supervised fitting; unsupported architectures reject this operation.",
            "pretrain_max_steps": "Maximum masked reconstruction optimizer steps; -1 leaves this count unlimited.",
            "pretraining_ratio": "Fraction of input features masked for supported reconstruction pretraining.",
            "pretraining_noise": "Noise fraction used by the supported reconstruction objective.",
            "pretraining_feature_mask": "Apply the supported source-feature mask during reconstruction pretraining.",
            "checkpoint_path": "Directory retaining per-fit checkpoints; None uses a temporary directory cleaned after fitting.",
            "trainer_kwargs": "Additional Lightning Trainer options. Controls owned by this configuration must not be duplicated here.",
            "ridge": "Nonnegative fixed-design ridge penalty, excluding the intercept.",
            "cg_rtol": "Relative residual tolerance for the fixed-design conjugate-gradient solve.",
            "cg_max_iter": "Optional conjugate-gradient iteration limit; None uses the solver default.",
            "boost_rate": "IGANN native boosting step multiplier.",
            "init_reg": "IGANN native initialization regularization strength.",
            "elm_alpha": "IGANN native ELM ridge regularization strength.",
            "early_stopping": "IGANN native validation patience measured in boosting stages.",
            "sparse": "IGANN native best-subset feature budget; zero disables sparse selection.",
            "device": "Device used by the architecture-native fitting algorithm.",
            "clip_predictions": "IGANN native prediction clipping bound used by the upstream update equations.",
        }.items()
    }
)


def constructor_parameter_group(name, architecture):
    group = name.partition("__")[0]
    return {
        "architecture_config": "Architecture",
        "objective": "Objective",
        "training": "Training defaults",
        "preprocessing": "Preprocessing",
        "fisher_diagnostics": "Inspection",
    }[group]


def estimator_text(cls, spec, objective):
    if objective == "survival":
        return survival_estimator_text(cls, neural=True)
    from pretab.preprocessor import Preprocessor

    from nampy.neural.configs.nam_config import DefaultNAMConfig
    from nampy.neural.configs.nbm_config import DefaultNBMConfig

    estimator = cls()
    params = estimator.get_params()
    from dataclasses import fields

    route = estimator._resolved_training([0, 1])
    params.update(
        {
            f"training__{field.name}": getattr(route, field.name)
            for field in fields(route)
        }
    )
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
    groups = {
        key: []
        for key in (
            "Architecture",
            "Objective",
            "Interactions",
            "Regularization",
            "Preprocessing",
            "Training defaults",
            "Inspection",
        )
    }
    rows = []
    declared = {
        **numpy_types(DefaultNAMConfig),
        **numpy_types(DefaultNBMConfig),
        **numpy_types(Preprocessor),
        **numpy_types(spec.config),
    }
    for name, default in sorted(params.items()):
        description = docs.get(name, docs.get(name.split("__", 1)[-1], ""))
        if name == "family":
            description = "Neural response family alias; see /api/utils for support and parameter domains."
        elif spec.name == "spline_nam" and name == "n_knots":
            description = "Number of knots in the learnable spline feature layer. Use preprocessing__n_knots to configure the separate preprocessing knot count."
        elif name == "distributional_kwargs":
            description = "Family constructor options, configured here rather than in fit(); quantile models use quantiles=[...]."
        if not description:
            description = f"Configuration option of type {type(default).__name__}; see the architecture reference for its role and restrictions."
        if spec.name == "semistructured":
            description = {
                "training__batch_size": "Inference/preparation batch size; native optimization uses the complete training cohort.",
                "training__shuffle": "DataLoader ordering setting; the native full-cohort objective has no epoch minibatch order.",
                "training__dataloader_kwargs": "Shared data preparation/loader options; native full-cohort optimization does not iterate a DataLoader.",
                "training__lr": "Adam learning rate for joint structured and neural optimization.",
                "training__patience": "Full-cohort validation steps without improvement before stopping.",
            }.get(name, description)
        group = constructor_parameter_group(name, spec.name)
        if (
            group == "Architecture"
            and name in inspect.signature(spec.config).parameters
            and name not in {"family", "distributional_kwargs"}
        ):
            description = f":doc:`Architecture parameter definition </neural_reference/{spec.name}>`."
        groups[group].append(
            (
                f"``{name}``",
                f"``{literal(default)}``",
                description,
                f"``{parameter_type(name, default, obj=(type(route) if name.startswith('training__') else spec.config), declared=declared)}``",
            )
        )
    for group, rows in groups.items():
        if rows:
            lines.append(
                f".. rubric:: {group}\n\n"
                + table(
                    ["Parameter", "Effective default", "Meaning", "Type / input shape"],
                    rows,
                )
            )
    if spec.name == "igann":
        lines.append(
            "This LSS task uses gradient training by default; the native solver supports squared-error regression and binary classification only.\n"
            if objective == "distributional"
            else "The default training route uses native boosting for squared-error regression and binary classification; n_estimators limits stages. Multiclass classification uses gradient training, as does GradientTrainingConfig.\n"
        )
    elif spec.name == "semistructured":
        lines.append("SemiStructuredTrainingConfig uses native CPU float64 full-cohort Adam with fixed structured penalties and a training-fitted weighted projection. Scalar regression, classification and scalar-response LSS use the existing objective interface. Scalar numeric inputs, fitted one-hot categorical blocks and explicit predictor-width offsets are supported. Vector features/responses, formula offsets, Lightning training, automatic smoothing, joint REML and neural coefficient intervals are unsupported.\n")
    elif spec.name == "gpnam":
        lines.append(
            "Default regression uses the fixed-design conjugate-gradient solver (FixedDesignTrainingConfig); optimizer and scheduler controls apply with GradientTrainingConfig.\n"
            if objective == "regression"
            else "Classification and LSS use gradient training; the fixed-design conjugate-gradient route is restricted to default-loss regression.\n"
        )
    lines += [
        "Training controls\n~~~~~~~~~~~~~~~~~\n",
        ":doc:`Full shared fit contract </api/neural_training>`: input and validation data, batching, device, optimizer, scheduler, callbacks and stopping. Native solvers such as IGANN use their architecture-specific stage controls described above. All training controls belong to constructor records; fit accepts data, offsets and weights.\n",
        "Predictions and explanations\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
    ]
    if objective == "regression":
        shape = "``(n_samples,)`` for a scalar target; ``(n_samples, n_outputs)`` for multi-output regression. Identity response scale; score is R²."
    elif objective == "classification":
        shape = "``predict`` returns class labels; ``predict_proba`` returns ``(n_samples, n_classes)`` probabilities in ``classes_`` order. Component extraction describes all actual logits; probabilities are queried separately. ``score`` returns mean classification accuracy (higher is better), with optional sample weights."
    else:
        shape = "``predict`` returns the family-defined point prediction; ``predict_parameters`` returns natural distribution parameters and ``predict_link`` returns additive predictors. Quantile-loss families instead return the explicitly fitted quantile vector and do not provide a full probability law or natural parameter matrix. For fixed scalar families, natural columns follow ``parameter_names_``. Categorical and ordinal outputs instead have one probability per class; multivariate families depend on configured dimension, and quantiles have one column per requested level. Raw width follows ``family_.param_count``. ``predict_point`` returns the family-defined point prediction. ``score`` returns negative mean family loss (higher is better); evaluation loss is lower-is-better."
    if spec.name == "nodegam":
        lines.append(
            "NODE-GAM contributions require an interpretable selector state. The short runnable example explicitly completes annealing (anneal_steps=1) in a single-layer unary model. Validate additive reconstruction before interpreting contributions from partially trained or more complex selector configurations.\n"
        )
    if spec.fixed_family == "quantile":
        shape = "``predict`` returns one conditional quantile per configured level. Training minimizes pinball loss; ``evaluate`` labels it ``pinball_loss`` (summed over levels, averaged over rows). These quantiles do not specify a complete probability density. ``predict_components`` retains the quantile dimension, which the single-column ``plot_terms`` renderer skips."
    lines += [
        shape + "\n",
        table(
            ["Method/output", "Python type", "Shape", "Meaning"],
            estimator_returns(objective),
        ),
        "``predict_components`` returns :class:`nampy.contracts.AdditivePrediction`; named contributions plus intercept and offset reconstruct its **link**. Metadata records known source dependence. Reconstruction does not establish identifiability or stability. Centering preserves predictions.\n",
        "Fitted attributes\n~~~~~~~~~~~~~~~~~\n",
        "``feature_names_in_`` and ``n_features_in_`` record the input schema. ``model`` contains the fitted objective/module; ``fitted_preprocessing_`` owns the fitted preprocessing/schema. Classifiers expose ``classes_``; LSS estimators expose ``family_`` and ``distributional_kwargs_``. ``model_complexity()`` returns fitted parameter counts.\n",
        "Methods\n~~~~~~~\n",
    ]
    methods = [
        "capabilities",
        "predict_link",
        "fisher_information",
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
        methods += [
            "predict_parameters",
            "predict_point",
            "predict_quantiles",
            "predict_distribution",
            "distribution_fit_status",
            "plot_distributional_effects",
        ]
    if spec.name == "snam":
        methods += ["group_norms"]
    shared = {
        "predict": "/api/contracts",
        "predict_proba": "/api/contracts",
        "predict_parameters": "/api/contracts",
        "predict_point": "/api/contracts",
        "predict_quantiles": "/user_guide/calibration",
        "predict_distribution": "/user_guide/gamlss",
        "distribution_fit_status": "/user_guide/gamlss",
        "score": "/api/neural_training",
        "evaluate": "/api/neural_training",
        "predict_components": "/api/contracts",
        "plot_distributional_effects": "/api/plotting",
    }
    rows = []
    for method in methods:
        if hasattr(cls, method):
            destination = shared.get(method, "/neural_reference/methods/" + method)
            rows.append((f"``{method}``", f":doc:`Contract <{destination}>`"))
    lines.append(table(["Method", "Canonical description"], rows))
    return "\n".join(lines)


GAM_PARAMETERS = {
    "fisher_diagnostics": PARAMETER_MEANINGS["fisher_diagnostics"],
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
    "para_pen": "Parametric term penalties: matrices, ranks, fixed/estimated smoothing and optional shared one-hot L maps. Single-formula models only.",
    "constraints": "Homogeneous raw-coefficient constraints selected by smooth label or one-based index; supported cr/cs/cc/ps/cp compositions only.",
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


def survival_estimator_text(cls, *, neural):
    import inspect

    meanings = {
        "cuts": "Finite increasing boundaries starting at zero for piecewise hazards; omit for Weibull/lognormal.",
        "formula": "Static covariate RHS for piecewise hazards, or parameter-name mapping of RHS formulas for Weibull/lognormal.",
        "likelihood": "piecewise_exponential, weibull or lognormal. Statistical parametric fitting supports fixed smoothing and automatic joint REML.",
        "temporal_formula": "Optional RHS temporal-effect formula using survival_time and source covariates; piecewise smooth-baseline route only.",
        "baseline": "Statistical piecewise baseline: factor (default) or smooth, evaluated at right interval endpoints with k basis functions; parametric models require the default.",
        "architecture_config": "Registered representation settings; initially LinReg and NAM.",
        "objective": "SurvivalObjectiveConfig(likelihood=..., cuts=...); cuts apply only to piecewise hazards. Right censoring with optional entry.",
        "training": "GradientTrainingConfig only; subject batches and global subject-weight normalization.",
        "preprocessing": "Train-only unsupervised preprocessing: none/standardization and one-hot/int.",
    }
    rows = []
    for name, parameter in inspect.signature(cls).parameters.items():
        default = "required" if parameter.default is inspect.Parameter.empty else literal(parameter.default)
        rows.append((f"``{name}``", f"``{default}``", meanings.get(name, GAM_PARAMETERS.get(name, name))))
    text = ":doc:`Survival workflow and limitations </user_guide/survival>`.\n\n"
    text += "Constructor parameters\n~~~~~~~~~~~~~~~~~~~~~~\n\n" + table(["Parameter", "Default", "Meaning"], rows)
    text += "Full shared fit contract\n~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    text += "``fit(X, SurvivalTarget, sample_weight=..., offset=...)`` keeps event status separate from weights. Piecewise offsets are known scalar log-rate offsets per subject; parametric offsets have shape (n_subjects,2), one per parameter predictor. New prediction subjects need explicit offsets if the fit used them. Targets contain time, event (0/1), optional entry and unique subject IDs. Entry conditions the likelihood. Piecewise models also accept LongitudinalSurvivalTarget with contiguous start/stop records, one terminal event, and constant per-subject case weights across records. score divides by total subject weight. Shared predict_trajectory evaluates explicitly supplied future covariate paths; ordinary predict_distribution keeps its constant-covariate interpretation.\n\n"
    if neural:
        text += "Static fits can split subjects automatically; longitudinal fits require explicit disjoint whole-subject X_val/y_val. No automatic record split or longitudinal warm start is supported. Preprocessing fits only training records; likelihood normalization uses independent subjects, not record counts. Piecewise predictor/component arrays are (n_subjects,n_intervals); Weibull/lognormal arrays are (n_subjects,2). Weibull links are log scale/log shape; lognormal links are location/log SD. Native/fixed routes, custom samplers, distributed training, and supervised preprocessing are unsupported.\n\n"
    else:
        text += "An existing Poisson GAM fits expanded exposures with log-exposure offsets. The fitted Poisson object and its native criterion remain unchanged. ``survival_log_likelihood_`` removes the event log-exposure constant; score is mean weighted event-time log likelihood. Components return SurvivalComponents with subject/interval mapping and an AdditivePrediction on log-hazard rows.\n\n"
        text += "Censored Weibull/lognormal use the generic joint adapter with analytic derivatives through order four, fixed smoothing and automatic joint REML. Parameter-name RHS formulas can model both parameters; a single RHS models the first with an intercept-only second. Parametric components return AdditivePrediction with shape (n_subjects,2). Native optimizer algorithms and stopping rules are preserved.\n\n"
        from nampy.gam.families.gamlss.censored import (
            CensoredLogNormal,
            CensoredWeibull,
        )

        text += table(["Censored likelihood", "Validated automatic REML optimizers"], [
            (family.name, ", ".join(sorted(family.supported_smoothing_optimizers)))
            for family in (CensoredWeibull, CensoredLogNormal)
        ])
    text += "Predictions and supported operations\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    text += "``predict`` and ``predict_survival`` require explicit times and return (n_subjects,n_times). ``predict_link`` returns interval log hazards or latent-parameter predictors; ``predict_hazard`` returns rates, never exposure-adjusted Poisson means. ``predict_distribution`` returns SurvivalPrediction supporting log survival, cumulative hazard, quantiles, median and RMST. Piecewise RMST is exact; Weibull/lognormal RMST uses controlled adaptive quadrature. RMST supports scalar or subject-specific conditioning landmarks. Pre-landmark curve cells are NaN. Unreached piecewise quantiles are NaN; piecewise tails beyond cuts are unavailable. Parametric tails are model-based extrapolation.\n\n"
    if not neural:
        text += "baseline='smooth' uses the existing penalized Poisson GAM with a right-endpoint time spline. Integration remains piecewise constant. At least k distinct interval endpoints and positive exposure in every interval are required; unlike independent interval intercepts, coupled smooth baselines can have zero events in an interval. Fitted endpoints must still converge. Neural models retain their existing interval-specific heads; matched spline-baseline smoothing is not supplied. temporal_formula accepts identified ti(survival_time, covariate, ...) terms with a smooth statistical baseline; exposure integration and neural interval-head semantics are unchanged.\n\n"
    text += "Shared survival_residuals preserves entry conditioning and Cox-Snell censoring flags. survival_curve computes entry-aware KM/NA descriptive estimates. concordance_index requires an explicit risk direction. brier_score requires an explicit training CensoringDistribution; its result provides finite-grid integrated Brier errors. Longitudinal targets reduce outcomes and case weights to one per subject; predictions and risk values must follow subject_ids order and use explicitly supplied complete paths. IPCW and concordance currently reject delayed entry. These functions consume the shared predictions from either backend and do not alter estimator score semantics.\n\n"
    text += "prepare_survival_effects evaluates reference predictors plus selected centered terms before survival transformation, retaining offsets, interval baselines, categorical/interaction coordinates and conditioning metadata. These contrasts do not form an additive decomposition of probabilities. prepare_survival_curve_interval supports statistical pointwise or finite-grid simultaneous joint coefficient inference with Bayesian or available smoothing-corrected covariance. Simultaneous curve coverage is joint across all valid subjects/times, on the log cumulative-hazard scale. prepare_survival_effect_interval uses paired reference-plus-term draws; simultaneous coverage includes all selected terms/subjects/times jointly on the survival-difference scale. Critical values use R type-8 quantiles of maximum standardized deviations. Single neural fits reject coefficient/effect intervals. combine_survival_predictions averages response-level survival and labels empirical between-fit ranges separately. No function infers replication or subject alignment.\n\n"
    text += "prepare_survival_hazard_interval and prepare_survival_rmst_interval reuse joint draws for statistical pointwise or finite-grid simultaneous intervals. Hazard bands calibrate log hazard jointly across valid subjects/times; parametric time-zero limits remain unavailable. RMST integrates each draw from fixed landmarks to explicit restriction horizons, with exact piecewise or controlled parametric integration. Bounds respect remaining-lifetime support; simultaneous scope is all valid subjects/horizons, in time units. Earlier cells are NaN. Separate functional calls do not claim joint coverage across different quantities. Single neural fits reject hazard/RMST coefficient intervals.\n\n"
    text += "bootstrap_survival performs ordinary uniform whole-subject pairs refits with fresh estimators/preprocessing on both engines. Its pointwise percentile intervals reuse shared survival/hazard/RMST operations and carry subject_bootstrap source metadata. Neural routes require fixed external validation data with disjoint explicit source IDs, reject warm starts/execution objects, and distinguish fixed-seed sampling uncertainty from combined independent-seed training/sampling variation. Failures are recorded or raised without replacement; intervals refuse failed attempts. Predictions remain on a fixed cohort; the source estimator is not refitted. For longitudinal targets resampling selects complete histories, with fresh occurrence IDs and retained source-subject maps. prediction_trajectory supplies fixed future paths; effects can prepare path-specific contrasts with fixed reference trajectories. Bootstrap, coefficient and between-fit sources are never relabeled as one another. Finite-grid simultaneous bootstrap bands reuse shared survival/hazard/RMST inference calculations. Paired effect contrasts are declared before refitting through effects, use a fixed reference cohort, and support pointwise percentiles or joint calibration over all selected terms/subjects/times. Coverage and uncertainty sources remain explicit; missing term layouts and degenerate draws disagreeing with the fitted estimate reject inference. BCa/nested-studentized/basic, continuous-time coverage and reference-population resampling remain unsupported.\n\n"
    text += "Post-fit conformal lower event-time bounds are available through nampy.calibration on both backends. Conservative observed-time and administrative-cutoff/known-selection routes have different assumptions and labels; these do not calibrate entire survival curves. See /user_guide/calibration.\n\n"
    text += "Unsupported routes\n~~~~~~~~~~~~~~~~~~\n\nCensored EFS/optim/magic/nlm.fd, separate censored ML/LAML/GCV/UBRE/NCV criteria, neural/longitudinal Cox (the separate CoxGAM supports static fixed and eligible automatic smoothing), parametric time-varying covariates, recurrent events, competing risks, left/interval censoring, zero-event saturated interval-factor boundary fits, delayed-entry IPCW/concordance, conditional censoring models, dynamic AUC, Uno concordance, randomized censor-aware PIT, continuous-time bands, single-fit neural coefficient/effect/hazard/RMST intervals and simultaneous inference from replicated fits remain unavailable. Survival coefficient simulation also rejects transformed coefficients and nonidentity/wider prediction-coordinate maps. Censored EFS has not passed the independent REML-profile endpoint gate. Parametric fitting requires two distinct positive-weight exact events. Neural survival retains gradient training only. Save/load uses the current trusted fitted-state artifact without migration shims.\n"
    return text


def cox_estimator_text(cls):
    meanings = {
        "formula": "Right-hand-side predictor formula; the global intercept is removed after conventional contrast coding.",
        "k": "Default spline dimension for formula-free effects.",
        "basis": "Existing statistical spline basis code.",
        "smoothing_params": "Fixed penalties or starting values for automatic REML smoothing.",
        "optimize_smoothing": "Enable native general-family automatic REML; fixed smoothing remains the default.",
        "smoothing_optimizer": "outer_newton or bfgs; stratified automatic Cox currently requires bfgs.",
        "control": "Existing GAM control settings for automatic smoothing; max_iter owns the coefficient iteration limit.",
        "knots": "Fitted spline knot specifications, reused at prediction.",
        "max_iter": "Maximum coefficient iterations on the selected fixed or automatic route.",
        "tol": "Relative step convergence tolerance for the coefficient solver.",
    }
    rows = [(f"``{name}``", f"``{literal(value)}``", meanings[name])
            for name, value in cls().get_params(deep=False).items()]
    return (":doc:`Cox scope and survival workflow </user_guide/survival>`.\n\n"
        + "Constructor parameters\n~~~~~~~~~~~~~~~~~~~~~~\n\n"
        + table(["Parameter", "Default", "Meaning"], rows)
        + "fit(X, SurvivalTarget, strata=..., offset=...) supports static right-censored records with fixed spline penalties. Risk sets include all tied event/censor records before the event contribution, following upstream mgcv coxlpl/coxpp conventions. predict_distribution returns shared step-cumulative-hazard predictions on common requested-stratum follow-up support; predict(..., times=...) returns survival. RMST integrates those steps exactly. predict_components describes log relative hazard, not survival. score is per-subject partial likelihood, not normalized event-time density; do not rank it against full survival likelihoods.\n\n"
        + "Automatic REML reuses the existing general-family solver with exact risk-set derivatives through fourth order. Newton/BFGS are supported for one stratum; stratified automatic fitting requires BFGS because strict Newton coefficient convergence has not passed its endpoint gate. predict_standard_error follows mgcv coxpp/coxpred, including baseline and coefficient terms. predict_interval returns approximate pointwise Wald survival limits. diagnostics returns source Schoenfeld and unscaled drift-corrected cumulative-score data, plus baseline-consistent martingale/deviance residuals. bootstrap_survival re-estimates coefficients, baseline and eligible smoothing; stratified calls require aligned fit_context and prediction_context strata. RMST and finite-grid simultaneous uncertainty are available through bootstrap. Delayed entry, longitudinal/neural Cox, case weights, alternate ties, continuous hazard density, analytic simultaneous/conditional curve intervals and coefficient-only substitutions remain unsupported. Version-4 trusted fitted-state save/load preserves supported predictions/inference.\n")


def calibrated_quantile_text(cls):
    meanings = {
        "formula": "Additive formula; RHS form uses y passed to fit.",
        "level": "Conditional quantile level strictly between zero and one.",
        "bandwidth": "Positive ELF bandwidth in response units (qgam co); None estimates the source SHASH bandwidth.",
        "smoothing_params": "Nonnegative spline penalties, or initial penalties when automatic smoothing is enabled.",
        "k": "Default smooth basis dimension.", "basis": "Default smooth basis type.",
        "fit_intercept": "Include the predictor intercept.",
        "log_scale_bounds": "Finite search bounds for log loss scale; a boundary minimum raises.",
        "max_iter": "Maximum coefficient score-function evaluations at each loss scale.",
        "tol": "Coefficient solver tolerance, certified by an exact Newton score step.",
        "calibration_tol": "Absolute bounded scalar calibration tolerance.",
        "optimize_smoothing": "Select REML penalties independently at every calibration scale.",
        "smoothing_optimizer": "Exact outer REML optimizer (bfgs or outer_newton).",
    }
    rows = [(f"``{name}``", f"``{literal(parameter.default)}``", meanings[name])
            for name, parameter in inspect.signature(cls).parameters.items()]
    return (":doc:`Calibrated quantile inference </user_guide/quantile_inference>`.\n\n"
        + table(["Parameter", "Default", "Meaning"], rows)
        + "Unweighted independent rows, identity link, supplied or adaptive ELF bandwidth and spline penalties. The qgam 2.0.1 calFast criterion selects a scalar loss scale by matching posterior and shrinkage-sandwich quantile-function variances. predict_interval returns approximate pointwise conditional-quantile intervals, distinct from future-response conformal intervals. predict_components reconstructs the conditional quantile. score is negative mean pinball loss.\n\n"
        + "Fitted state includes ``loss_scale_``, ``posterior_covariance_``, ``sandwich_covariance_``, ``calibration_loss_`` and the scalar ``calibration_profile_``. Failed refits preserve the successful fitted model; set_params invalidates it. Trusted save/load preserves supported predictions and intervals.\n\n"
        + "fit_levels shares Gaussian/SHASH preparation across independently calibrated levels; quantiles can cross and their inference is not joint. Covariate-dependent loss scales, weighted/bootstrap calibration, simultaneous bands, noncrossing multiple quantiles and likelihood/AIC inference are unsupported. The existing QuantileGAM fitting route is unchanged.\n")


def statistical_text(cls):
    if cls.__name__ == "CalibratedQuantileGAM":
        return calibrated_quantile_text(cls)
    if cls.__name__ == "CoxGAM":
        return cox_estimator_text(cls)
    if cls.__name__ == "SurvivalGAM":
        return survival_estimator_text(cls, neural=False)
    name = cls.__name__
    rows = [
        (
            f"``{parameter}``",
            f"``{literal(value)}``",
            GAM_PARAMETERS[parameter],
            f"``{parameter_type(parameter, value)}``",
        )
        for parameter, value in cls().get_params(deep=False).items()
    ]
    text = ":doc:`Statistical GAM workflow </tutorials/first_gam>` and :doc:`Compatibility </compatibility>`.\n\n"
    text += "Constructor parameters\n~~~~~~~~~~~~~~~~~~~~~~\n\n" + table(
        ["Parameter", "Default", "Meaning", "Type / input shape"], rows
    )
    text += "Predictions and explanations\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    if name == "GAMLSS":
        text += "``predict`` returns the family-defined point prediction. ``predict_parameters`` returns natural parameters in ``parameter_names_`` order; ``predict_link`` returns additive predictors in ``predictor_names_`` order. ``predict_point`` returns the family-defined point prediction (mean, median or most probable class). Component arrays have shape ``(n_samples, n_parameters)`` and identify their parameter in structured metadata.\n\n"
    elif name == "GAMClassifier":
        text += "Binary classification: ``predict`` returns labels and ``predict_proba`` returns ``(n_samples, 2)`` probabilities in ``classes_`` order. Components are contributions to the link predictor rather than additive probability changes.\n\n"
    else:
        text += "``predict`` returns one response-scale mean per row. ``standard_errors`` returns pointwise prediction uncertainty in the requested scale; it is not future-outcome spread or a simultaneous confidence band.\n\n"
    text += table(
        ["Method/output", "Python type", "Shape", "Meaning"],
        estimator_returns(
            "distributional"
            if name == "GAMLSS"
            else "classification"
            if name == "GAMClassifier"
            else "regression"
        ),
    )
    text += "``gam_`` is the fitted statistical model. ``schema_`` records the validated input structure when the adapter can record a schema. Call explanation methods on the wrapper and use ``gam_`` for the complete statistical inference and diagnostic surface.\n\nMethods\n~~~~~~~\n\n"
    for method in [
        "fit",
        "capabilities",
        "predict_link",
        "predict_parameters",
        "predict_quantiles",
        "fisher_information",
        "predict",
        "predict_proba",
        "predict_point",
        "predict_distribution",
        "distribution_fit_status",
        "score",
        "evaluate",
        "standard_errors",
        "lpmatrix",
        "predict_components",
        "explain_terms",
        "term_importance",
        "summary",
        "plot",
        "qq",
        "plot_check",
        "vis",
        "plot_derivative",
        "save_model",
        "load_model",
    ]:
        if hasattr(cls, method):
            rows = method_parameter_rows(getattr(cls, method), GAM_PARAMETERS)
            if rows:
                text += f".. rubric:: {method} parameters\n\n" + table(
                    ["Parameter", "Type / shape", "Default", "Meaning"], rows
                )
            if method == "plot":
                from reference_schema import plot_parameter_rows

                text += (
                    ".. rubric:: Expanded statistical plotting keywords\n\n"
                    + table(
                        ["Keyword", "Type", "Default", "Meaning"], plot_parameter_rows()
                    )
                )
            text += f".. automethod:: nampy.models.{name}.{method}\n   :no-index:\n\n"
    text += "Complete examples\n~~~~~~~~~~~~~~~~~\n\nSee the :doc:`homepage fitting examples </index>` for executable data setup, fitted predictions, component tables, and plots.\n"
    return text


def write_if_changed(path, content):
    """Preserve include mtimes so an unchanged registry does not rebuild every page."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate(root):
    from nampy import models
    from nampy.neural.registry import architectures

    output = root / "docs/_generated/reference"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["GAMRegressor", "GAMClassifier", "GAMLSS", "SurvivalGAM", "CoxGAM", "CalibratedQuantileGAM"]:
        write_if_changed(
            output / f"{name}.inc", statistical_text(getattr(models, name))
        )
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
                        ("survival", "piecewise-exponential survival"),
                    ]
                    if spec.supports(capability)
                ),
                "; ".join(
                    label
                    for capability, label in [
                        ("regression", "Regression components"),
                        ("classification", "Classification logit components"),
                        ("distributional", "LSS link components"),
                        ("survival", "interval log-hazard components"),
                    ]
                    if spec.supports(capability)
                )
                if spec.supports("additive_components")
                else "No additive explanation declared.",
                DEPENDENCE.get(
                    name, "Named features or explicitly configured subsets."
                ),
                "Native boosting for regression/binary classification; gradient training for multiclass, LSS, or explicit GradientTrainingConfig"
                if name == "igann"
                else "Default CG regression; gradient classification/LSS or explicit GradientTrainingConfig"
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
            ("survival", "Survival"),
        ]:
            if not spec.supports(task):
                continue
            cls = getattr(models, spec.estimator_prefix + suffix)
            write_if_changed(
                output / f"{cls.__name__}.inc",
                estimator_text(cls, spec, task).rstrip() + "\n",
            )
    generate_families(root)
    write_if_changed(
        output / "model_capabilities.inc",
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
        + "\n",
    )


# These meanings complement registry-derived aliases, parameter order, and
# actual transform configuration. Statistical GAMLSS uses different families.
FAMILY_MEANINGS = {
    "bct": ("Positive scalar", "Existence-checked shared BCT mean", "Existence-checked shared BCT SD", "Normalized Student-t Box-Cox likelihood; first-order CPU normalizer gradients"),
    "bcpe": ("Positive scalar", "Existence-checked shared BCPE mean", "Existence-checked shared BCPE SD", "Normalized power-exponential Box-Cox likelihood; first-order CPU normalizer gradients"),
    "lognormal_observation": ("Positive ObservationTarget with censoring and fixed truncation", "Conditional median", "Conditional SD unavailable", "Censored lognormal likelihood with fixed truncation normalizer"),
    "weibull_observation": ("Positive ObservationTarget with censoring and fixed truncation", "Conditional median", "Conditional SD unavailable", "Censored Weibull likelihood with fixed truncation normalizer"),
    "bccg": ("Positive scalar", "Existence-checked shared BCCG mean", "Existence-checked shared BCCG SD", "Normalized BCCG log likelihood"),
    "beinf": ("Proportions including zero and one", "pi1 + pi_interior*mu", "Normalized mixture SD", "Normalized beta density and exact atom log masses"),
    "zaga": ("Nonnegative scalar with exact zeros", "(1-pi0)*mu", "Normalized Gamma/atom mixture SD", "Normalized Gamma density and exact zero mass"),
    "betabinomial": ("Integer successes; fixed trials>=2", "trials*mu", "Beta-binomial SD", "Normalized beta-binomial log mass"),
    "normal_observation": ("ObservationTarget exact/left/right/interval observations", "Conditional truncated-normal mean", "Conditional truncated-normal SD", "Censored normal likelihood with fixed truncation normalizer"),
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
        "Robustly transformed Normal training loss when rob is not None; ordinary Normal NLL when rob=None",
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
        "No predict_spread hook; use quantile gaps as interval summaries",
        "Pinball loss; no probability density",
    ),
}


def generate_families(root):
    """Legacy include is an index; family pages own the full contract."""
    path = root / "docs/_generated/reference/neural_families.inc"
    path.parent.mkdir(parents=True, exist_ok=True)
    from nampy.neural.families.registry import FAMILY_REGISTRY

    write_if_changed(
        path,
        table(
            ["Family", "Canonical contract"],
            [
                (name, f":doc:`{name} </neural_reference/families/{name}>`")
                for name in FAMILY_REGISTRY
            ],
        ),
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
