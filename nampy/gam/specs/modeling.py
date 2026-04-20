"""Internal predictor-spec construction helpers for the GAM facade."""

from __future__ import annotations

import warnings

import numpy as np

from ..data import knots_for_feature, knots_for_features
from ..formula import extract_formula_terms, parse_gam_formula
from . import LinearPredictorSpec, TermSpec, build_smooth_spec
from .build import FormulaBuildResult, build_formula_model


def make_tensor_term(model, spec, *, knots=None):
    if hasattr(spec, "fit") and callable(spec.fit):
        return spec

    if (
        isinstance(spec, (tuple, list))
        and len(spec) >= 2
        and not isinstance(spec, dict)
    ):
        features = list(spec)
        return TermSpec(
            kind="smooth",
            features=tuple(str(f) for f in features),
            by_variable=None,
            smooth_spec=build_smooth_spec(
                special="te",
                bs=model.basis,
                k=model.k,
                knots=knots_for_features(model, features, knots=knots),
                select=bool(model.select),
            ),
            smoothing_id=None,
            label=f"te({', '.join(map(str, features))})",
            metadata={},
        )

    if not isinstance(spec, dict):
        raise TypeError(
            "Each tensor term specification must be either a term object, "
            "a tuple/list of features, or a dict."
        )

    features = spec.get("features", None)
    if features is None:
        raise ValueError(
            "Tensor term dicts must contain a 'features' entry, "
            "e.g. {'features': ('x1', 'x2')}"
        )

    kind = str(spec.get("kind", "te")).lower()
    k = spec.get("k", model.k)
    basis = spec.get("basis", model.basis)
    label = spec.get("label", f"{kind}({', '.join(map(str, features))})")
    smoothing_id = spec.get("smoothing_id", None)
    metadata = spec.get("metadata", None)
    by = spec.get("by", None)
    mc = spec.get("mc", None)
    full = bool(spec.get("full", False))
    ord_ = spec.get("ord", None)
    fixed = bool(spec.get("fixed", spec.get("fx", False)))
    select = bool(spec.get("select", model.select))
    sp = spec.get("sp", None)
    term_knots = spec.get("knots", knots_for_features(model, features, knots=knots))

    if kind == "ti" and mc is not None:
        mc_vals = [bool(v) for v in ([mc] if np.isscalar(mc) else mc)]
        if len(mc_vals) == 0:
            raise ValueError("mc must not be empty.")
        if not any(mc_vals):
            warnings.warn(
                f"{label}: all ti() marginal constraints are turned off (mc={mc_vals}). "
                "This leaves the term without identifiability constraints unless the rest "
                "of the model provides them.",
                stacklevel=2,
            )

    if kind == "t2":
        return TermSpec(
            kind="smooth",
            features=tuple(str(f) for f in features),
            by_variable=by,
            smooth_spec=build_smooth_spec(
                special=kind,
                bs=basis,
                k=k,
                full=full,
                ord_=ord_,
                fx=fixed,
                select=select,
                sp=sp,
                knots=term_knots,
            ),
            smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
            label=label,
            metadata=dict(metadata or {}),
        )

    return TermSpec(
        kind="smooth",
        features=tuple(str(f) for f in features),
        by_variable=by,
        smooth_spec=build_smooth_spec(
            special=kind,
            bs=basis,
            k=k,
            mc=mc,
            fx=fixed,
            select=select,
            sp=sp,
            knots=term_knots,
        ),
        smoothing_id=(None if smoothing_id is None else str(smoothing_id)),
        label=label,
        metadata=dict(metadata or {}),
    )


def make_predictor_specs(model, feature_names, *, knots=None):
    terms = []

    if model.main_effects:
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
            elif basis == "ps":
                main_terms.append(
                    TermSpec(
                        kind="smooth",
                        features=(str(name),),
                        by_variable=None,
                        smooth_spec=build_smooth_spec(
                            special="s",
                            bs="ps",
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
            elif basis == "gp":
                main_terms.append(
                    TermSpec(
                        kind="smooth",
                        features=(str(name),),
                        by_variable=None,
                        smooth_spec=build_smooth_spec(
                            special="s",
                            bs="gp",
                            k=model.k,
                            m=None,
                            sp=None,
                            fx=False,
                            select=bool(model.select),
                            pc=None,
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
                    "{'cr','cs','cc','ps','tp','ts','gp','re'}, "
                    f"got {model.basis!r}."
                )

        terms.extend(main_terms)

    has_full_te = False
    has_t2 = False
    if model.tensor_terms is not None:
        for spec in model.tensor_terms:
            term = make_tensor_term(model, spec, knots=knots)
            if getattr(term, "term_type", None) == "tensor_smooth":
                has_full_te = True
            if getattr(term, "term_type", None) == "tensor_anova":
                has_t2 = True
            terms.append(term)

    if model.main_effects and has_full_te:
        warnings.warn(
            "Model contains both main-effect smooths and full te() tensor-product terms. "
            "This is identifiable via side conditions, but is typically less stable and less "
            "interpretable than a ti() ANOVA-style decomposition.",
            stacklevel=2,
        )
    if model.main_effects and has_t2:
        warnings.warn(
            "Model contains both separate main-effect smooths and t2() terms. "
            "t2() already contains ANOVA-style lower-order components, so this combination "
            "can create strong overlap in the current framework.",
            stacklevel=2,
        )

    return [LinearPredictorSpec(name="eta", terms=terms)]


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
