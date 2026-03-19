from dataclasses import replace

import numpy as np

from ..specs import LinearPredictorSpec, ParametricTermSpec, SmoothTermSpec
from .parser import ParsedGAMFormula, ParsedParametricTerm, ParsedSmoothTerm


def apply_drop_intercept(parsed: ParsedGAMFormula, drop_intercept=None):
    if drop_intercept is None:
        return parsed

    n_pred = len(parsed.predictors)
    if np.isscalar(drop_intercept):
        flags = [bool(drop_intercept)] * n_pred
    else:
        flags = [bool(v) for v in drop_intercept]
        if len(flags) != n_pred:
            raise ValueError(
                f"drop_intercept must have length {n_pred} for a list-of-formula model, "
                f"got {len(flags)}."
            )

    predictors = []
    for pf, flag in zip(parsed.predictors, flags):
        predictors.append(
            replace(
                pf,
                intercept=(False if flag else pf.intercept),
            )
        )

    return ParsedGAMFormula(
        response_name=parsed.response_name,
        predictors=predictors,
        raw=parsed.raw,
    )


def _coerce_fx(fx):
    if isinstance(fx, (list, tuple)):
        raise NotImplementedError(
            "Vector-valued fx is not yet supported by the current formula compiler."
        )
    return bool(fx)


def _knots_for_features(knots, features):
    if knots is None:
        return None
    if isinstance(knots, dict):
        vals = [knots.get(str(f), None) for f in features]
        return None if all(v is None for v in vals) else vals[0] if len(features) == 1 else vals
    return knots


def compile_predictor_specs_from_formula(
    parsed: ParsedGAMFormula,
    *,
    default_k=10,
    default_basis="cr",
    default_select=False,
    knots=None,
):
    predictor_specs = []

    for i, pf in enumerate(parsed.predictors):
        terms = []

        for term in pf.terms:
            if isinstance(term, ParsedParametricTerm):
                terms.append(
                    ParametricTermSpec(
                        name=term.raw_label,
                        raw_label=term.raw_label,
                        label=term.raw_label,
                        metadata={
                            "formula_term": term.raw_label,
                            "source_variables": list(term.variables),
                            "parametric_interaction": len(term.variables) > 1,
                        },
                    )
                )
                continue

            if not isinstance(term, ParsedSmoothTerm):
                raise TypeError(f"Unknown parsed term type: {type(term)}")

            kind = term.kind
            features = list(term.features)
            kw = dict(term.kwargs)

            k = kw.pop("k", default_k)
            basis = kw.pop("bs", kw.pop("basis", default_basis))
            by = kw.pop("by", None)
            smoothing_id = kw.pop("id", kw.pop("smoothing_id", None))
            fixed = _coerce_fx(kw.pop("fx", False))
            select = bool(kw.pop("select", default_select))

            m = kw.pop("m", None)
            xt = kw.pop("xt", None)
            sp = kw.pop("sp", None)
            pc = kw.pop("pc", None)

            mc = kw.pop("mc", None)
            full = kw.pop("full", False)
            ord_ = kw.pop("ord", None)

            if kw:
                raise NotImplementedError(
                    f"Unsupported smooth arguments in {term.raw_label!r}: {sorted(kw)}"
                )

            term_knots = _knots_for_features(knots, features)

            terms.append(
                SmoothTermSpec(
                    special=kind,
                    bs=basis,
                    features=features,
                    k=k,
                    fx=fixed,
                    select=select,
                    m=m,
                    by=by,
                    xt=xt,
                    id=smoothing_id,
                    sp=sp,
                    pc=pc,
                    knots=term_knots,
                    mc=mc,
                    full=full,
                    ord=ord_,
                    label=term.raw_label,
                    metadata={"formula_term": term.raw_label},
                )
            )

        predictor_specs.append(
            LinearPredictorSpec(
                name=f"eta{i+1}",
                terms=terms,
                offset_name=pf.offset_name,
                metadata={
                    "raw_formula": pf.raw_formula,
                    "offset_name": pf.offset_name,
                    "response_name": pf.response_name,
                    "intercept": bool(pf.intercept),
                },
            )
        )

    return predictor_specs
