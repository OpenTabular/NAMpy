"""Authored navigation and canonical ownership of reference contracts.

Registry-derived membership is checked here so a newly registered feature cannot
silently disappear from the grouped indexes.
"""

from __future__ import annotations

import inspect
import json

from smooth_contracts import SHAPE_NAMES, smooth_inputs

SMOOTH_NAMES = {
    "tp": "Thin-plate regression spline",
    "ts": "Shrinkage thin-plate spline",
    "cr": "Cubic regression spline",
    "cs": "Shrinkage cubic spline",
    "cc": "Cyclic cubic spline",
    "ps": "P-spline",
    "cp": "Cyclic P-spline",
    "bs": "Derivative-penalized B-spline",
    "ds": "Duchon spline",
    "gp": "Gaussian-process smooth",
    "sos": "Spherical spline",
    "ad": "Adaptive spline",
    "so": "Soap-film smooth",
    "sf": "Soap boundary film",
    "sw": "Soap interior smooth",
    "sc": "Unconstrained SC B-spline",
    "scad": "Adaptive SC B-spline",
    "re": "Random effect",
    "mrf": "Markov random field",
    "fs": "Factor-specific smooth",
    "sz": "Sum-to-zero factor smooth",
    "te": "Full tensor product",
    "ti": "Tensor interaction",
    "t2": "Range/null-space tensor product",
}


SMOOTH_NAMES.update(SHAPE_NAMES)


def smooth_title(name, description):
    title = SMOOTH_NAMES.get(
        name,
        description.split(".")[0]
        .split(":")[0]
        .split(";")[0]
        .removesuffix(" construction"),
    )
    return f"{title} (``{name}``)"


CONTROL_GROUPS = {
    "coefficient_fitting": (
        "Coefficient fitting",
        ["epsilon", "maxit", "irls_reg", "rank_tol", "scale_est"],
    ),
    "smoothing_optimization": (
        "Smoothing optimization",
        [
            "mgcv_tol",
            "mgcv_half",
            "nlm",
            "optim",
            "newton",
            "efs_lspmax",
            "efs_tol",
            "edge_correct",
        ],
    ),
    "shape_fitting": (
        "Shape-constrained fitting",
        [
            "scam_max_half",
            "scam_devtol_fit",
            "scam_steptol_fit",
            "scam_print_warn",
            "scam_b_notexp",
            "scam_threshold_notexp",
            "scam_bfgs",
        ],
    ),
    "execution": (
        "Output and execution",
        ["nthreads", "ncv_threads", "trace", "keep_data"],
    ),
    "basis_penalty": (
        "Basis and penalty construction",
        ["id_links_bases", "scale_penalty"],
    ),
}


def control_anchor(name):
    """Section IDs must follow docutils normalization, including underscores."""
    return "control-" + name.replace("_", "-")


FAMILY_GROUPS = {
    "Continuous responses": {
        "gaussian",
        "normal",
        "scat",
        "studentt",
        "robustnormal",
        "gaulss",
        "gumbls",
        "gevlss",
        "shash",
        "student_lss",
        "quasi",
    },
    "Positive responses": {
        "gamma",
        "inverse_gaussian",
        "inversegamma",
        "lognormal",
        "weibull",
        "loglogistic",
        "gammals",
    },
    "Nonnegative semicontinuous responses": {"tw", "twlss", "tweedie"},
    "Proportions": {"beta", "betar", "beta_lss"},
    "Counts": {
        "poisson",
        "nb",
        "negbin",
        "negativebinom",
        "nb_lss",
        "zip",
        "zinb",
        "ziplss",
        "hurdlepoisson",
        "hurdlenegativebinom",
        "quasipoisson",
    },
    "Categorical and ordered responses": {
        "binomial",
        "quasibinomial",
        "ocat",
        "ordinal",
        "categorical",
        "multinom",
    },
    "Censored responses": {"cnorm", "clog", "cpois", "bcg"},
    "Multivariate responses": {"dirichlet", "mvnormdiag"},
    "Quantiles": {"quantile"},
}

# Domains and ordering are separate from constructor-option meanings.
NEURAL_PARAMETERS = {
    "zero_prob_transform": "Transform of the structural-zero or hurdle-zero probability; default maps raw values into (0,1).",
    "name": "Display label for this distribution; does not change its mathematical family.",
    "eps": "Positive numerical floor used by positivity transforms and stable logarithms; it is not response noise.",
    "target_eps": "Positive clipping margin for targets at support boundaries. Clipping changes the evaluated boundary target.",
    "mean_transform": "Transform of the conditional mean column; default identity for an unrestricted mean and positive for a positive mean.",
    "loc_transform": "Transform of the location column. For LogNormal this is the mean of log(Y), not the response mean.",
    "scale_transform": "Transform of the positive scale column; default positive means softplus(raw)+eps.",
    "shape_transform": "Transform of the positive shape column; default positive means softplus(raw)+eps.",
    "rate_transform": "Transform of the positive rate (inverse scale) column; default positive means softplus(raw)+eps.",
    "dispersion_transform": "Transform of the positive dispersion column; its variance interpretation is given below.",
    "df_transform": "Positive transform of degrees of freedom before adding min_df.",
    "min_df": "Positive lower offset for Student-t degrees of freedom. Mean requires df>1 and variance requires df>2.",
    "rob": "RobustNormal softplus log-likelihood tuning constant; None disables the robustness transform and recovers Gaussian negative log likelihood.",
    "num_classes": "Number of classes C, at least two; None is inferred from integer training labels. Output class order is ``classes_``.",
    "n_dim": "Response dimension D, positive integer; None is inferred from the two-dimensional training target.",
    "dim": "Alias for the positive response dimension D; must agree with n_dim if both are supplied.",
    "concentration_transform": "Transform applied to every positive Dirichlet concentration alpha_i.",
    "prob_transform": "Transform from class logits to normalized class probabilities; default probabilities is row-wise softmax.",
    "quantiles": "Strictly increasing probability levels in (0,1); None selects 0.25, 0.5, 0.75.",
    "enforce_monotonic": "If True, uses the first raw column as the initial quantile and cumulative softplus increments for later levels. This preserves nondecreasing quantiles without sorting raw columns.",
    "variance_power": "Fixed Tweedie power p in (1,2), giving Var(Y)=dispersion*mean**p.",
    "series_max_terms": "Positive maximum number of terms in the compound-Poisson likelihood series; controls truncation cost and accuracy.",
}


def neural_parameter_meanings(name):
    meanings = dict(NEURAL_PARAMETERS)
    if name == "beta":
        meanings["shape_transform"] = (
            "Transform of Beta alpha, the first positive concentration parameter."
        )
        meanings["scale_transform"] = (
            "Transform of Beta beta, the second positive concentration parameter; this is not a standard deviation."
        )
    return meanings


def grouped_families(root, backend, names):
    from feature_reference import FAMILY_SUPPORT, heading, table, write
    from public_reference import FAMILY_MEANINGS

    from nampy.gam.families.registry import make_gam_family
    from nampy.neural.distributions.registry import FAMILY_REGISTRY

    text = heading(
        "Response families"
        if backend == "statistical"
        else "Neural response distributions"
    )
    text += "Choose by response support, then compare parameterization and fitting interface. Column meanings on the individual page take precedence over a similarly named family in another backend.\n\n"
    text += "The parameter column describes natural distribution parameters, not link predictors or lower-level response output columns. Family pages distinguish these interfaces. Fixed nuisance parameters are listed in each constructor contract.\n\n"
    text += "For likelihoods, links, response support and joint parameter predictors, start with :doc:`/statistical_reference/families/theory`. Individual pages develop the theory of their response law before its API contracts.\n\n"
    covered = set()
    for title, members in FAMILY_GROUPS.items():
        selected = sorted(set(names) & members)
        if not selected:
            continue
        rows = []
        for name in selected:
            covered.add(name)
            if backend == "statistical":
                options = {
                    "name": name,
                    **(
                        {"R": 3}
                        if name == "ocat"
                        else {"theta": 2.0}
                        if name == "negbin"
                        else {}
                    ),
                }
                family = make_gam_family(options)
                params = (
                    getattr(family, "distribution_parameter_names", None)
                    or getattr(family, "parameter_names", None)
                    or ("mu",)
                )
                support = FAMILY_SUPPORT.get(
                    name,
                    {
                        "student_lss": "Real scalar, Student-t location/scale with fixed degrees of freedom.",
                        "beta_lss": "Proportion strictly between zero and one, mean/precision predictors.",
                        "nb_lss": "Nonnegative integer counts, mean/inverse-shape dispersion predictors.",
                    }.get(name, "Multi-parameter response family; see contract."),
                )
                interface = (
                    "GAMLSS / core GAM"
                    if getattr(family, "family_class", "") == "general"
                    else "GAMRegressor / core GAM"
                )
                if name == "multinom":
                    params = ("p0, ..., pK",)
                if name == "binomial":
                    interface = "GAMClassifier / core GAM"
            else:
                family = FAMILY_REGISTRY[name].distribution
                import numpy as np

                sample = (
                    np.ones((6, 3)) / 3
                    if name in {"dirichlet", "mvnormdiag"}
                    else np.array([0, 1, 2, 0, 1, 2])
                )
                instance, _ = FAMILY_REGISTRY[name].instantiate(sample)
                params = instance.param_names
                support = FAMILY_MEANINGS[name][0]
                interface = "LSS estimators supporting this response family"
            rows.append(
                (
                    f":doc:`{name} <{name}>`",
                    support,
                    ", ".join(f"``{p}``" for p in params),
                    interface,
                )
            )
        text += heading(title, "-") + table(
            [
                "Family",
                "Response support / interpretation",
                "Natural distribution parameters (column order)",
                "Interface",
            ],
            rows,
        )
    if covered != set(names):
        raise ValueError(f"Family index needs a support group: {set(names) - covered}")
    text += ".. toctree::\n   :hidden:\n\n"
    if backend == "statistical":
        text += "   Mathematical foundations <theory>\n"
    text += "".join(f"   {name}\n" for name in names)
    write(root, f"{backend}_reference/families/index.rst", text)


def organize_reference(root):
    from dataclasses import fields

    from feature_reference import (
        FAMILY_SUPPORT,
        SMOOTH_MEANINGS,
        heading,
        table,
        tree,
        write,
    )

    from nampy.gam import gam_control
    from nampy.gam.basis_registry import basis_descriptors
    from nampy.gam.families.registry import GENERAL_FAMILIES
    from nampy.neural.distributions.registry import FAMILY_REGISTRY

    descriptors = sorted(basis_descriptors(), key=lambda d: d.name)
    groups = {
        key: []
        for key in [
            "Ordinary one-dimensional smooths",
            "Cyclic smooths",
            "Multivariate, spatial and spherical smooths",
            "Group effects and structured penalties",
            "Tensor products",
            "Shape-constrained smooths",
        ]
    }
    for d in descriptors:
        name = d.name
        group = (
            "Tensor products"
            if name in {"te", "ti", "t2"}
            else "Cyclic smooths"
            if name in {"cc", "cp"}
            else "Group effects and structured penalties"
            if name in {"re", "mrf", "fs", "sz"}
            else "Multivariate, spatial and spherical smooths"
            if name in {"tp", "ts", "ds", "gp", "sos", "so", "sf", "sw"}
            else "Shape-constrained smooths"
            if name in SHAPE_NAMES
            else "Ordinary one-dimensional smooths"
        )
        groups[group].append(
            (
                f":doc:`{name} <{name}>`",
                SMOOTH_NAMES.get(name, SMOOTH_MEANINGS[name].split(".")[0]),
                smooth_inputs(d),
                SMOOTH_MEANINGS[name],
            )
        )
    text = (
        heading("Smooth bases and tensor products")
        + "Choose the modeling purpose first. Start with :doc:`theory` for basis expansions, penalties, null spaces and EDF. Each detail page explains its specific construction, then documents the formula interface and links separately to its runtime constructor.\n\n"
    )
    for title, rows in groups.items():
        text += heading(title, "-") + table(
            ["Code", "Name", "Inputs", "Main distinction"], rows
        )
    text += ".. toctree::\n   :hidden:\n\n   Mathematical foundations <theory>\n" + "".join(
        f"   {d.name}\n" for d in descriptors
    )
    write(root, "statistical_reference/smooths/index.rst", text)
    canonical_classes = {}
    for descriptor in descriptors:
        if descriptor.runtime_class:
            canonical_classes.setdefault(
                descriptor.runtime_class.__name__, descriptor.name
            )
    canonical_classes.update(FSmoothInteractionTerm="fs", SZSmoothInteractionTerm="sz")
    for name in canonical_classes:
        write(
            root,
            f"api/generated/nampy.gam.smooths.{name}.rst",
            ":orphan:\n\n"
            + heading(name)
            + f":doc:`Canonical runtime constructor </extension_reference/smooths/{name}>`.\n",
        )
    # Generate these explicitly: an old autosummary folder is absent in clean checkouts.
    for name in ["BaseSmoothTerm", "ByState", "make_smooth_term", "register_smooth"]:
        directive = (
            "autoclass" if name in {"BaseSmoothTerm", "ByState"} else "autofunction"
        )
        write(
            root,
            f"api/generated/nampy.gam.smooths.{name}.rst",
            ":orphan:\n\n"
            + heading(name)
            + f".. {directive}:: nampy.gam.smooths.{name}\n",
        )
    write(
        root,
        "extension_reference/smooths/index.rst",
        tree(
            "Runtime smooth terms",
            [*sorted(canonical_classes)],
            "Constructor options operate on basis setup and fitted coefficient coordinates. They are distinct from the formula language.",
        ),
    )

    from control_reference import generate as generate_controls

    generate_controls(root)
    defaults = gam_control()
    all_fields = {f.name: f for f in fields(defaults)}

    grouped_families(
        root,
        "statistical",
        [
            *FAMILY_SUPPORT,
            *(
                n
                for n, s in GENERAL_FAMILIES.items()
                if s.reference_group == "distributional_extensions"
            ),
        ],
    )
    grouped_families(root, "neural", list(FAMILY_REGISTRY))
    from feature_reference import OPTIMIZER_MEANINGS, SELECTION_MEANINGS

    selection = (
        heading("Smoothness selection")
        + "The criterion defines the objective; the optimizer determines how smoothing parameters are updated. Support depends on both choices and the family/coefficient route.\n\n"
        + "Start with :doc:`theory` for coefficient fitting, the outer selection loop, predictive risk and marginal likelihood.\n\n"
        + ".. toctree::\n   :maxdepth: 1\n\n   Mathematical foundations <theory>\n\n"
    )
    for title, choices in [
        ("Criteria", SELECTION_MEANINGS),
        ("Optimizers", OPTIMIZER_MEANINGS),
    ]:
        selection += (
            heading(title, "-")
            + ".. toctree::\n   :maxdepth: 1\n\n"
            + "".join(f"   {n}\n" for n in choices)
            + "\n"
        )
    selection += heading("Special optimizer routes", "-") + table(
        ["Route", "Restriction"],
        [
            (
                "``magic``",
                "Gaussian identity-link GCV/UBRE performance iteration only; selected automatically for supported performance-iteration routes.",
            ),
            (
                "``nlm.fd``",
                "Finite-difference NLM exposed for SCAM transformed-coefficient fits only; ordinary mgcv-style GAM routes reject it.",
            ),
        ],
    )
    write(root, "statistical_reference/selection/index.rst", selection)
    write(
        root,
        "extension_reference/index.rst",
        tree(
            "Extension APIs",
            [
                "smooths/index",
                "design",
                "term_construction",
                "Neural estimator extension </neural_reference/extensions/index>",
                "Neural implementation modules </neural_reference/internals>",
                "Statistical implementation modules </api/gam_advanced>",
            ],
            "Implementation interfaces for building terms, registering architectures and extending family or fitting behavior. Ordinary estimator use starts in the statistical or neural model reference.",
            related_links=False,
        ),
    )
    write(
        root,
        "extension_reference/design.rst",
        tree(
            "Compiled design and offsets",
            [
                f"/statistical_reference/extensions/{name}"
                for name in [
                    "compile_design",
                    "CompiledDesign",
                    "DesignEvaluation",
                    "DesignTerm",
                ]
            ],
            "Shared design compilation and new-data evaluation keep formula bases, coefficient coordinates and offsets consistent across statistical objectives.",
        ),
    )
    write(
        root,
        "extension_reference/term_construction.rst",
        tree(
            "Term construction helpers",
            [
                f"/api/generated/nampy.gam.smooths.{name}"
                for name in [
                    "BaseSmoothTerm",
                    "ByState",
                    "make_smooth_term",
                    "register_smooth",
                ]
            ],
            "Implement a runtime term or register its constructor. Formula users select the basis code through s(), te(), ti() or t2().",
        ),
    )
    inventory = []
    for d in descriptors:
        inventory.append(
            {
                "kind": "smooth",
                "symbol": d.name,
                "formula": d.name
                if d.name in {"te", "ti", "t2"}
                else f"s(..., bs={d.name!r})",
                "page": f"statistical_reference/smooths/{d.name}",
                "runtime_page": f"extension_reference/smooths/{d.runtime_class.__name__ if d.runtime_class else {'fs': 'FSmoothInteractionTerm', 'sz': 'SZSmoothInteractionTerm'}[d.name]}",
                "example": f"statistical_reference/smooths/{d.name}",
                "tests": ["tests/parity", "tests/docs/test_feature_reference.py"],
            }
        )
    for slug, (_, names) in CONTROL_GROUPS.items():
        for name in names:
            inventory.append(
                {
                    "kind": "control",
                    "symbol": name,
                    "page": f"statistical_reference/controls/{slug}",
                    "anchor": f"{control_anchor(name)}",
                    "public_import": "nampy.gam.gam_control",
                    "signature": f"gam_control({name}=...)",
                    "parameters": [
                        {
                            "name": name,
                            "default": repr(getattr(defaults, name)),
                            "type": str(all_fields[name].type),
                        }
                    ],
                    "input_contract": f"statistical_reference/controls/{slug}#{control_anchor(name)}",
                    "return_schema": "_examples/reference/output_schemas.json#statistical_reference/controls/"
                    + slug,
                    "restrictions": f"statistical_reference/controls/{slug}#{control_anchor(name)}",
                    "example": f"statistical_reference/controls/{slug}",
                    "tests": ["tests/docs/test_feature_reference.py"],
                }
            )
    from feature_reference import reference_targets
    from neural_method_reference import METHODS
    from public_reference import literal
    from reference_symbols import SYMBOLS

    from nampy import models
    from nampy.gam.model.api import GAM
    from nampy.neural.registry import architectures

    targets = reference_targets()

    def record(symbol, obj, page, example, kind):
        params = []
        try:
            signature = inspect.signature(obj)
        except (ValueError, TypeError):
            signature = None
        defaults = (
            obj().get_params(deep=False)
            if kind == "estimator" and hasattr(obj, "get_params")
            else None
        )
        if kind == "architecture":
            instance = obj()
            defaults = {
                field.name: getattr(instance, field.name) for field in fields(instance)
            }
        if defaults is None and signature:
            defaults = {
                n: p.default
                for n, p in signature.parameters.items()
                if n not in {"self", "cls"}
            }
        for n, default in (defaults or {}).items():
            params.append({"name": n, "default": literal(default)})
        rendered_signature = str(signature)
        if signature:
            for parameter in signature.parameters.values():
                if (
                    callable(parameter.default)
                    and parameter.default is not inspect.Parameter.empty
                ):
                    rendered_signature = rendered_signature.replace(
                        repr(parameter.default), literal(parameter.default)
                    )
        inventory.append(
            {
                "kind": kind,
                "symbol": symbol,
                "public_import": getattr(obj, "__module__", "")
                + "."
                + getattr(obj, "__name__", symbol),
                "signature": rendered_signature,
                "parameters": params,
                "page": page,
                "example": example,
                "input_contract": page,
                "return_schema": "_examples/reference/output_schemas.json#" + example,
                "restrictions": page,
                "tests": ["tests/docs/test_feature_reference.py"],
            }
        )

    record("GAM", GAM, "statistical_reference/GAM", "statistical_reference/GAM", "core")
    inventory[-1]["public_import"] = "nampy.gam.GAM"
    import importlib

    for module, names in SYMBOLS.items():
        for name in names:
            page = "api/generated/" + module + "." + name
            record(
                name,
                getattr(importlib.import_module(module), name),
                page,
                page,
                "shared",
            )
    from nampy.gam import design, extensions

    for module in [design, extensions]:
        for name in module.__all__:
            page = "statistical_reference/extensions/" + name
            record(name, getattr(module, name), page, page, "statistical_extension")
    for name in models.__all__:
        record(
            name,
            getattr(models, name),
            targets[name],
            targets[name],
            "estimator"
            if name.endswith(("Regressor", "Classifier", "LSS"))
            and name not in {"NeuralRegressor", "NeuralClassifier", "NeuralLSS"}
            else "extension",
        )
        inventory[-1]["public_import"] = "nampy.models." + name
    for name in FAMILY_SUPPORT.keys() | GENERAL_FAMILIES.keys():
        if (
            name not in FAMILY_SUPPORT
            and GENERAL_FAMILIES[name].reference_group != "distributional_extensions"
        ):
            continue
        options = {
            "name": name,
            **(
                {"R": 3}
                if name == "ocat"
                else {"theta": 2.0}
                if name == "negbin"
                else {}
            ),
        }
        from nampy.gam.families.registry import make_gam_family

        cls = type(make_gam_family(options))
        page = "statistical_reference/families/" + name
        record(name, cls, page, page, "statistical_family")
    for name, spec in FAMILY_REGISTRY.items():
        page = "neural_reference/families/" + name
        record(name, spec.distribution, page, page, "neural_family")
    for name, spec in architectures().items():
        page = "neural_reference/" + name
        record(spec.config.__name__, spec.config, page, page, "architecture")
    for name, obj in inspect.getmembers(GAM, predicate=callable):
        if name.startswith("_"):
            continue
        page = "statistical_reference/methods/" + name
        record("GAM." + name, obj, page, page, "method")
    from nampy.models._base import NeuralEstimatorBase

    for name in METHODS:
        page = "neural_reference/methods/" + name
        record(name, getattr(NeuralEstimatorBase, name), page, page, "neural_method")
    for d in descriptors:
        entry = next(
            row
            for row in inventory
            if row["kind"] == "smooth" and row["symbol"] == d.name
        )
        if d.runtime_class:
            entry["public_import"] = (
                d.runtime_class.__module__ + "." + d.runtime_class.__name__
            )
            entry["signature"] = str(inspect.signature(d.runtime_class))
        entry.update(
            input_contract=entry["page"],
            return_schema="_examples/reference/output_schemas.json#" + entry["example"],
            restrictions=entry["page"],
        )
    inventory.sort(key=lambda entry: (entry["kind"], entry["symbol"]))
    write(
        root,
        "_data/public_reference_inventory.json",
        json.dumps(inventory, indent=2),
    )
