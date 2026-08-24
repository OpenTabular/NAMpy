"""Internal predictor-spec construction helpers for the GAM facade."""

from __future__ import annotations

import copy

from ..data import knots_for_feature
from ..formula import extract_formula_terms, parse_gam_formula
from . import LinearPredictorSpec, TermSpec, build_smooth_spec
from .build import FormulaBuildResult, build_formula_model


def make_predictor_specs(model, feature_names, *, knots=None):
    # The array (non-formula) construction path builds one main-effect smooth
    # per feature column. Tensor terms are a formula-only surface: use
    # te(...)/ti(...) in a formula instead of removed constructor specs.
    terms = []

    basis = str(model.basis).lower()
    main_terms = []

    for name in feature_names:
        term_knots = knots_for_feature(model, name, knots=knots)

        if basis in {"cr", "cs", "cc"}:
            main_terms.append(
                TermSpec(
                    kind="smooth",
                    features=(str(name),),
                    by_variable=None,
                    smooth_spec=build_smooth_spec(
                        special="s",
                        bs=basis,
                        k=model.k,
                        sp=None,
                        fx=False,
                        select=bool(model.select),
                        knots=term_knots,
                    ),
                    smoothing_id=None,
                    label=name,
                    metadata={},
                )
            )
        elif basis in {"bs", "ds", "ps", "cp"}:
            main_terms.append(
                TermSpec(
                    kind="smooth",
                    features=(str(name),),
                    by_variable=None,
                    smooth_spec=build_smooth_spec(
                        special="s",
                        bs=basis,
                        k=model.k,
                        m=None,
                        sp=None,
                        fx=False,
                        select=bool(model.select),
                        knots=term_knots,
                    ),
                    smoothing_id=None,
                    label=name,
                    metadata={},
                )
            )
        elif basis in {"tp", "ts"}:
            main_terms.append(
                TermSpec(
                    kind="smooth",
                    features=(str(name),),
                    by_variable=None,
                    smooth_spec=build_smooth_spec(
                        special="s",
                        bs=basis,
                        k=model.k,
                        m=None,
                        sp=None,
                        fx=False,
                        select=bool(model.select),
                        knots=term_knots,
                        xt=None,
                    ),
                    smoothing_id=None,
                    label=name,
                    metadata={},
                )
            )
        elif basis == "re":
            main_terms.append(
                TermSpec(
                    kind="smooth",
                    features=(str(name),),
                    by_variable=None,
                    smooth_spec=build_smooth_spec(
                        special="s",
                        bs="re",
                        sp=None,
                        xt=None,
                    ),
                    smoothing_id=None,
                    label=name,
                    metadata={},
                )
            )
        else:
            raise NotImplementedError(
                "Automatic main-effect construction currently supports "
                "{'bs','cr','cs','cc','cp','ds','ps','tp','ts','re'}, "
                f"got {model.basis!r}."
            )

    terms.extend(main_terms)

    n_predictors = int(getattr(model.family, "n_linear_predictors", 1))
    parameter_names = tuple(getattr(model.family, "parameter_names", ()) or ())
    if parameter_names and len(parameter_names) != n_predictors:
        raise ValueError(
            f"Family {model.family.name!r} declares {len(parameter_names)} parameter "
            f"name(s) for {n_predictors} predictor(s)."
        )

    return [
        LinearPredictorSpec(
            name=("eta" if n_predictors == 1 else f"eta{i + 1}"),
            terms=copy.deepcopy(terms),
            has_intercept=bool(model.fit_intercept),
            parameter_name=(parameter_names[i] if parameter_names else None),
        )
        for i in range(n_predictors)
    ]


def prepare_formula_inputs(
    model, data, formula, y=None, knots=None, drop_intercept=None
):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed, drop_intercept=drop_intercept)
    build_result: FormulaBuildResult = build_formula_model(
        extracted,
        data=data,
        y=y,
        knots=knots,
        default_k=model.k,
        default_basis=model.basis,
        default_select=model.select,
    )
    # Specs remain component-owned; this metadata lets the compiler create
    # overlapping logical predictor index arrays without duplicating blocks.
    for spec, lpi in zip(
        build_result.predictor_specs, build_result.component_lpi, strict=True
    ):
        spec.metadata["lpi"] = tuple(int(v) for v in lpi)
        spec.metadata["n_linear_predictors"] = int(
            build_result.n_linear_predictors
        )
    return (
        parsed,
        build_result.predictor_specs,
        build_result.X,
        build_result.feature_names,
        build_result.response,
        build_result.used_columns,
        build_result.offsets,
        build_result.preprocess_state,
    )
