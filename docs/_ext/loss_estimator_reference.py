"""Method contracts for the fixed-penalty scalar statistical estimators."""

import inspect

INPUTS = {
    "path": (
        "str | pathlib.Path",
        "NAMpy estimator artifact path. Load only trusted pickle artifacts.",
    ),
    "X": (
        "pandas.DataFrame | array-like (n,d) | None",
        "Training/new rows. Formula mode requires a DataFrame containing source features and formula-offset columns; formula-free models also accept numeric arrays in fitted column order.",
    ),
    "data": (
        "pandas.DataFrame | array-like (n,d) | None",
        "Fit-only alias for X; supply exactly one of X and data.",
    ),
    "y": (
        "array-like (n,) or (n,1) | None",
        "Finite scalar response. At fit time None reads the formula response column; formula-free fitting requires y. score always requires the true response.",
    ),
    "sample_weight": (
        "array-like (n,) | str | None",
        "Finite nonnegative weights with positive sum; None means ones. fit additionally accepts a formula-data column name; score accepts an array, not a column name.",
    ),
    "offset": (
        "array-like (n,) | None",
        "Known additive offset, default zeros, added to evaluated formula/smooth-owned offsets. Separate fit-time offsets are not reused by predict on new data.",
    ),
    "term_features": (
        "mapping[str,str] | None",
        "Map term labels to numeric DataFrame columns used for plotting. None infers one-feature terms; multivariate terms require an explicit mapping.",
    ),
    "rug": ("bool", "Add observed input locations beneath term curves when True."),
    "figsize": (
        "tuple[float,float] | None",
        "Figure width and height in inches; None uses renderer defaults.",
    ),
}
RETURNS = {
    "save_model": (
        "path", "pathlib.Path", "scalar path",
        "Written estimator artifact containing the fitted design and loss solution.",
    ),
    "load_model": (
        "estimator", "ExpectileGAM | QuantileGAM", "scalar estimator",
        "Restored estimator of the requested class; incompatible formats and classes raise errors.",
    ),
    "fit": (
        "self",
        "estimator instance",
        "scalar object",
        "Stores the fitted design and loss solution. Missing/nonfinite input, invalid weights, or incompatible shapes raise errors. Nonconvergence emits ConvergenceWarning and leaves diagnostics in ``fit_result_``.",
    ),
    "predict": (
        "prediction",
        "numpy.ndarray[float64]",
        "(n,)",
        "Conditional expectile/quantile on the response scale, including supplied/evaluated offsets. No standard-error tuple or interval is returned.",
    ),
    "score": (
        "score",
        "float",
        "()",
        "Negative weighted mean unpenalized validation loss; higher is better. Does not return R² or training penalized objective.",
    ),
    "predict_components": (
        "components",
        "AdditivePrediction",
        "response/link (n,); term arrays (n,1)",
        "Identity response/link scale. Intercept plus named terms plus offset reconstructs prediction. ``coef_`` and all terms use the same identified design coordinates.",
    ),
    "lpmatrix": (
        "matrix",
        "numpy.ndarray[float64]",
        "(n,p)",
        "Full coefficient design including an intercept when fitted. ``matrix @ coef_`` excludes known offsets; add the evaluated new-data offset to reconstruct predict.",
    ),
    "plot_terms": (
        "figures",
        "list[matplotlib.figure.Figure]",
        "one per rendered page",
        "Observed-row term curves, with optional rug. These are contributions, not partial-dependence curves or inferential bands. Formula offsets are evaluated; fit-only offsets are not reused.",
    ),
}


def method_contracts(cls):
    from feature_reference import heading, table
    from public_reference import literal

    text = heading("Methods", "-")
    text += "Prediction, scoring, plotting and design evaluation require a fitted estimator and raise NotFittedError beforehand. Persistence can also save and restore an unfitted configuration. New-data arrays are supported only by formula-free models; formula models require DataFrames. Inputs must be finite and match the fitted feature schema.\n\n"
    for name, output in RETURNS.items():
        method = getattr(cls, name)
        anchor = cls.__name__.lower() + "-" + name.replace("_", "-")
        text += f".. _{anchor}:\n\n" + heading(name, "~")
        text += f".. code-block:: python\n\n   model.{name}{inspect.signature(method).replace(parameters=[p for p in inspect.signature(method).parameters.values() if p.name != 'self'])}\n\n"
        rows = []
        for key, param in inspect.signature(method).parameters.items():
            if key == "self":
                continue
            kind, meaning = INPUTS[key]
            if key == "y" and name == "score":
                kind = "array-like (n,) or (n,1)"
            if key == "sample_weight" and name == "score":
                kind = "array-like (n,) | None"
            if key == "X" and name != "fit":
                kind = kind.removesuffix(" | None")
            rows.append(
                (
                    f"``{key}``",
                    kind,
                    "Required"
                    if param.default is inspect.Parameter.empty
                    else f"``{literal(param.default)}``",
                    meaning,
                )
            )
        text += table(["Parameter", "Type / shape", "Default", "Meaning"], rows)
        text += table(["Output", "Type", "Shape", "Meaning"], [output])
        if name == "score":
            loss = (
                "level*r**2 for r>=0 and (1-level)*r**2 otherwise"
                if cls.__name__ == "ExpectileGAM"
                else "max(level*r, (level-1)*r), including when bandwidth>0"
            )
            text += f"For residual ``r=y-predict(X)``, the per-row loss is ``{loss}``. The result is ``-sum(w*loss)/sum(w)``. The smoothing penalty is excluded. score has no offset keyword; formula offsets are evaluated from X, while a separate fit-only offset is omitted. ``GridSearchCV.fit(X, y)`` needs y explicitly for scoring even when fit can read a formula response column.\n\n"
    text += heading("Fitted attributes and convergence", "-") + table(
        ["Attribute", "Type / shape", "Meaning"],
        [
            (
                "``coef_``",
                "numpy.ndarray (p,)",
                "Full identified coefficient vector, including any intercept.",
            ),
            (
                "``design_``",
                "CompiledDesign",
                "Fitted basis/preprocessing, term ownership and known-offset evaluation.",
            ),
            (
                "``smoothing_params_``",
                "numpy.ndarray (s,)",
                "Resolved fixed nonnegative penalty strengths.",
            ),
            ("``penalty_``", "numpy.ndarray (p,p)", "Full fixed coefficient penalty S."),
            ("``n_features_in_``", "int", "Fitted source schema width."),
            (
                "``fit_result_``",
                "LossFit",
                "coefficients (p,), fitted (n,), objective (float), iterations (int), converged (bool), optimality (float). fitted is ``B@coef_``, excluding offsets. objective is summed training loss plus beta.T@S@beta/2.",
            ),
        ],
    )
    text += "Check ``fit_result_.converged``, ``iterations`` and ``optimality`` before interpreting a solution. :doc:`asymmetric_fitting` defines each solver stopping rule. No likelihood EDF, Gaussian covariance or calibrated intervals are attached to these loss fits.\n\n"
    return text


def shared_solver_page(root):
    from feature_reference import heading, table, write

    text = heading("Expectile and quantile fitting conventions")
    text += "Both estimators compile an identified design and minimize a sum of weighted observation losses plus ``beta.T @ S @ beta / 2``. Fixed penalty strengths are supplied by the caller; scaling all weights changes effective regularization. Validation score instead normalizes by the sum of weights and excludes the penalty.\n\n"
    text += table(
        ["Route", "Training loss", "Stopping and diagnostics"],
        [
            (
                "ExpectileGAM: LAWS",
                "Asymmetric squared error, weighted by level above zero and 1-level below zero.",
                "Backtracked weighted least-squares updates; infinity-norm gradient tolerance scaled by 1+norm(X.T@(w*y), infinity).",
            ),
            (
                "QuantileGAM: bandwidth=0",
                "Exact convex pinball loss.",
                "Sparse epigraph LP/QP. tol governs primal/dual accuracy; optimality records the maximum KKT feasibility, stationarity and complementarity residual.",
            ),
            (
                "QuantileGAM: bandwidth>0",
                "Smooth ELF: (level-1)*r + bandwidth*log(1+exp(r/bandwidth)), learning rate fixed at one.",
                "Gradient-based smooth optimization; inspect the optimizer convergence flag and gradient diagnostic. Validation score remains exact pinball loss.",
            ),
        ],
    )
    text += "Separate levels can cross. Smoothing is fixed or chosen through validation. These APIs do not implement qgam learning-rate calibration or calibrated uncertainty. Upstream comparison scope is documented in :doc:`/development/neighbor_parity`.\n\n:doc:`ExpectileGAM` · :doc:`QuantileGAM`\n"
    write(root, "statistical_reference/estimators/asymmetric_fitting.rst", text)
