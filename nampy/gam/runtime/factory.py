"""
Stage 2 of the GAM fit pipeline: runtime term materialization.

Converts declarative TermSpec objects into fitted runtime term instances.
Each runtime term owns all basis-specific mathematics for its smooth family:
basis construction, penalty definition, term-local constraints, by-variable
handling, and new-data transforms.

This module dispatches on ``TermSpec.kind``, ``basis_options["special"]``, and
``basis_options["bs"]`` to select the correct runtime class.  It must not
implement basis construction itself — it only routes to the canonical classes
in ``gam/smooths/*``.
"""

from __future__ import annotations

from typing import Any

from ..specs import LinearPredictorSpec, TermSpec

from ..smooths.registry import make_smooth_term
from ..terms.linear import LinearTerm
from ..smooths.univariate.cubic_regression import SplineTerm1D
from ..smooths.univariate.pspline import PSplineTerm1D
from ..smooths.univariate.thin_plate import ThinPlateSplineTerm  # imported for registration / direct access
from ..smooths.univariate.gp import GPSmoothTerm
from ..smooths.tensor.te import TensorProductSplineTerm  # imported for registration
from ..smooths.tensor.ti import InteractionTensorProductSplineTerm  # imported for registration
from ..smooths.tensor.t2 import TensorANOVASplineTerm  # imported for registration

from ..smooths.categorical.factor_smooth import FSmoothInteractionTerm, SZSmoothInteractionTerm
from ..smooths.categorical.mrf import MarkovRandomFieldTerm
from ..smooths.categorical.random_effect import RandomEffectTerm


def _as_list_or_repeat(value: Any, n: int):
    if isinstance(value, str):
        return [value] * n
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) != n:
            raise ValueError(f"Expected length {n}, got {len(out)}.")
        return out
    return [value] * n


def instantiate_term(term_like: TermSpec | Any):
    if hasattr(term_like, "fit") and callable(term_like.fit):
        return term_like

    if not isinstance(term_like, TermSpec):
        raise TypeError(
            "instantiate_term expects TermSpec (or already-materialized runtime term)."
        )

    if term_like.kind == "parametric":
        if len(term_like.features) != 1:
            raise ValueError("Parametric TermSpec requires exactly one feature.")
        return LinearTerm(
            feature=term_like.features[0],
            label=term_like.label,
            term_id=term_like.term_id,
            metadata=dict(term_like.metadata or {}),
        )
    if term_like.kind != "smooth":
        raise TypeError(f"Unsupported TermSpec.kind={term_like.kind!r}")

    opts = dict(term_like.basis_options or {})
    special = str(opts.get("special", "s")).lower()
    features = list(term_like.features)
    metadata = dict(term_like.metadata or {})
    metadata["term_spec"] = {
        "kind": term_like.kind,
        "features": list(term_like.features),
        "by_variable": term_like.by_variable,
        "basis_options": dict(opts),
        "smoothing_id": term_like.smoothing_id,
        "label": term_like.label,
    }
    bs = str(opts.get("bs", "cr")).lower()
    k = opts.get("k", -1)
    fx = bool(opts.get("fx", False))
    select = bool(opts.get("select", False))
    m = opts.get("m", None)
    xt = opts.get("xt", None)
    sp = opts.get("sp", None)
    pc = opts.get("pc", None)
    knots = opts.get("knots", None)
    constraint_mode = str(opts.get("constraint_mode", "auto"))
    shared_basis_setup = opts.get("shared_basis_setup", None)
    mc = opts.get("mc", None)
    full = bool(opts.get("full", False))
    ord_ = opts.get("ord", None)
    by = term_like.by_variable
    smoothing_id = term_like.smoothing_id
    label = term_like.label

    if special == "s":
        if bs in {"cr", "cs", "cc"}:
            if len(features) != 1:
                raise NotImplementedError(
                    f"Current runtime only materializes 1D s(..., bs={bs!r}) terms."
                )

            return SplineTerm1D(
                feature=features[0],
                k=k,
                basis=bs,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                select=select,
                fixed=fx,
                constraint_mode=constraint_mode,
                shared_basis_setup=shared_basis_setup,
                pc=pc,
                knots=knots,
                metadata=metadata,
            )

        if bs == "ps":
            if len(features) != 1:
                raise NotImplementedError(
                    "Current runtime only materializes 1D s(..., bs='ps') terms."
                )

            return PSplineTerm1D(
                feature=features[0],
                k=k,
                basis=bs,
                m=m,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                select=select,
                fixed=fx,
                constraint_mode=constraint_mode,
                pc=pc,
                knots=knots,
                metadata=metadata,
            )

        if bs in {"tp", "ts"}:
            return make_smooth_term(
                bs,
                feature=features,
                k=k,
                basis=bs,
                m=m,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                select=select,
                fixed=fx,
                constraint_mode=constraint_mode,
                pc=pc,
                knots=knots,
                xt=xt,
                metadata=metadata,
            )

        if bs == "gp":
            return GPSmoothTerm(
                feature=features,
                k=k,
                basis=bs,
                m=m,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                select=select,
                fixed=fx,
                constraint_mode=constraint_mode,
                pc=pc,
                knots=knots,
                xt=xt,
                metadata=metadata,
            )

        if bs == "mrf":
            if select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='mrf' smooths."
                )
            return MarkovRandomFieldTerm(
                feature=features,
                k=k,
                basis=bs,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                xt=xt,
                knots=knots,
                metadata=metadata,
            )

        if bs == "re":
            if select:
                raise NotImplementedError(
                    "select=True is not implemented for bs='re' terms."
                )
            return RandomEffectTerm(
                feature=features,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                xt=xt,
                metadata=metadata,
            )

        if bs == "fs":
            if select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='fs' factor smooths."
                )
            return FSmoothInteractionTerm(
                feature=features,
                k=k,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                xt=xt,
                fixed=fx,
                knots=knots,
                metadata=metadata,
            )

        if bs == "sz":
            if select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='sz' factor smooths."
                )
            return SZSmoothInteractionTerm(
                feature=features,
                k=k,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=sp,
                xt=xt,
                fixed=fx,
                knots=knots,
                metadata=metadata,
            )

        raise NotImplementedError(
            f"Current runtime materializes bs in "
            f"{{'cr','cs','cc','ps','tp','ts','gp','mrf','re','fs','sz'}} for s(...). "
            f"Received bs={bs!r}."
        )

    if special in {"te", "ti", "t2"}:
        raw_bs = opts.get("bs", "cr")
        basis = _as_list_or_repeat(raw_bs, len(features))
        basis = [str(b).lower() for b in basis]
        if any(b != "cr" for b in basis):
            raise NotImplementedError(
                f"Current runtime only materializes bs='cr' marginals for {special}(...). "
                f"Received basis={basis!r}."
            )

        kwargs = dict(
            feature=features,
            k=k,
            basis=basis,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            select=select,
            fixed=fx,
            knots=knots,
            metadata=metadata,
        )

        if special == "ti":
            kwargs["mc"] = mc
        elif special == "t2":
            kwargs["full"] = full
            kwargs["ord"] = ord_

        return make_smooth_term(special, **kwargs)

    raise ValueError(f"Unknown smooth special {special!r}.")


def instantiate_predictor_terms(predictor_specs):
    out = []
    for pred in predictor_specs:
        for t in pred.terms:
            if not isinstance(t, TermSpec):
                raise TypeError(
                    f"Predictor terms must be canonical TermSpec, got {type(t)}."
                )
        out.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=[instantiate_term(t) for t in pred.terms],
                has_intercept=bool(getattr(pred, "has_intercept", False)),
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=dict(pred.metadata),
            )
        )
    return out

__all__ = [
    "instantiate_term",
    "instantiate_predictor_terms",
]
