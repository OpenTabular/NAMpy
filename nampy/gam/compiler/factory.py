"""Compiler-owned term instantiation from declarative specs."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..basis_registry import get_basis_descriptor
from ..smooths.categorical.fs import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
)
from ..smooths.categorical.re import RandomEffectTerm
from ..smooths.parametric import LinearTerm
from ..smooths.registry import make_basis_term, make_smooth_term
from ..smooths.shape.bivariate import BivariateShapePSplineTerm
from ..smooths.shape.scop import ShapeConstrainedPSplineTerm
from ..specs import LinearPredictorSpec, PenaltyGroupSpec, TermSpec
from ..specs.smooth import (
    AlternativeTensorProductSmoothSpec,
    FactorSmoothInteractionSpec,
    RandomEffectSmoothSpec,
    ShapeConstrainedSmoothSpec,
    SumToZeroFactorSmoothSpec,
    TensorInteractionSmoothSpec,
    TensorProductSmoothSpec,
    tensor_basis_list,
)


def _tensor_feature_groups(features, d):
    features = [str(f) for f in features]
    if d is None:
        return list(features)
    dims = [int(v) for v in (d if isinstance(d, (list, tuple)) else [d])]
    if sum(dims) != len(features):
        raise ValueError(
            f"Tensor marginal dimensions {dims!r} do not sum to {len(features)} features."
        )
    out = []
    pos = 0
    for di in dims:
        group = features[pos : pos + di]
        out.append(group[0] if di == 1 else list(group))
        pos += di
    return out


def _tensor_group_knots(knots, features, groups):
    if knots is None or not isinstance(knots, (list, tuple)):
        return knots
    if len(knots) != len(features) or len(groups) == len(features):
        return knots
    out = []
    pos = 0
    for group in groups:
        width = len(group) if isinstance(group, list) else 1
        part = list(knots[pos : pos + width])
        out.append(part[0] if width == 1 else part)
        pos += width
    return out


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

    features = list(term_like.features)
    metadata = dict(term_like.metadata or {})
    smooth_spec = term_like.smooth_spec
    if smooth_spec is None:
        raise ValueError(f"Smooth TermSpec {term_like.label!r} is missing smooth_spec.")
    metadata["term_spec"] = {
        "kind": term_like.kind,
        "features": list(term_like.features),
        "by_variable": term_like.by_variable,
        "basis_options": smooth_spec.to_basis_options(),
        "smoothing_id": term_like.smoothing_id,
        "label": term_like.label,
        "smooth_spec_type": type(smooth_spec).__name__,
    }
    by = term_like.by_variable
    smoothing_id = term_like.smoothing_id
    label = term_like.label

    basis_name = str(getattr(smooth_spec, "bs", "")).lower()
    descriptor = get_basis_descriptor(basis_name)
    if descriptor is not None and descriptor.direct_runtime:
        return make_basis_term(
            basis_name,
            feature=features,
            k=smooth_spec.k,
            m=getattr(smooth_spec, "m", None),
            xt=getattr(smooth_spec, "xt", None),
            pc=getattr(smooth_spec, "pc", None),
            knots=smooth_spec.knots,
            shared_basis_setup=getattr(smooth_spec, "shared_basis_setup", None),
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            fixed=smooth_spec.fx,
            constraint_mode=getattr(smooth_spec, "constraint_mode", "auto"),
            metadata=metadata,
        )

    if isinstance(smooth_spec, ShapeConstrainedSmoothSpec):
        if len(features) == 2:
            return BivariateShapePSplineTerm(
                feature=features,
                k=smooth_spec.k,
                basis=str(smooth_spec.bs).lower(),
                m=smooth_spec.m,
                label=label,
                term_id=term_like.term_id,
                smoothing_id=smoothing_id,
                by=by,
                sp=smooth_spec.sp,
                select=smooth_spec.select,
                fixed=smooth_spec.fx,
                knots=smooth_spec.knots,
                metadata=metadata,
            )
        if len(features) != 1:
            raise NotImplementedError("SCOP smooths support one or two features.")
        return ShapeConstrainedPSplineTerm(
            feature=features[0],
            k=smooth_spec.k,
            basis=str(smooth_spec.bs).lower(),
            m=smooth_spec.m,
            xt=smooth_spec.xt,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            fixed=smooth_spec.fx,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(smooth_spec, RandomEffectSmoothSpec):
        return RandomEffectTerm(
            feature=features,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            xt=smooth_spec.xt,
            metadata=metadata,
        )

    if isinstance(smooth_spec, FactorSmoothInteractionSpec):
        return FSmoothInteractionTerm(
            feature=features,
            k=smooth_spec.k,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            m=smooth_spec.m,
            xt=smooth_spec.xt,
            fixed=smooth_spec.fx,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(smooth_spec, SumToZeroFactorSmoothSpec):
        return SZSmoothInteractionTerm(
            feature=features,
            k=smooth_spec.k,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            m=smooth_spec.m,
            xt=smooth_spec.xt,
            fixed=smooth_spec.fx,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(
        smooth_spec,
        (
            TensorProductSmoothSpec,
            TensorInteractionSmoothSpec,
            AlternativeTensorProductSmoothSpec,
        ),
    ):
        groups = _tensor_feature_groups(features, getattr(smooth_spec, "d", None))
        basis = [str(b).lower() for b in tensor_basis_list(smooth_spec, len(groups))]
        kwargs = {
            "feature": groups,
            "k": smooth_spec.k,
            "basis": basis,
            "m": smooth_spec.m,
            "xt": smooth_spec.xt,
            "label": label,
            "term_id": term_like.term_id,
            "smoothing_id": smoothing_id,
            "by": by,
            "sp": smooth_spec.sp,
            "select": smooth_spec.select,
            "fixed": smooth_spec.fx,
            "knots": _tensor_group_knots(smooth_spec.knots, features, groups),
            "pc": smooth_spec.pc,
            "metadata": metadata,
        }

        if isinstance(smooth_spec, TensorInteractionSmoothSpec):
            kwargs["mc"] = smooth_spec.mc
        elif isinstance(smooth_spec, AlternativeTensorProductSmoothSpec):
            kwargs["full"] = smooth_spec.full
            kwargs["ord"] = smooth_spec.ord

        return make_smooth_term(smooth_spec.special, **kwargs)

    raise ValueError(f"Unknown smooth spec type {type(smooth_spec).__name__!r}.")


def _expected_penalty_group_size(runtime_term):
    smoothing_id = getattr(runtime_term, "smoothing_id", None)
    if smoothing_id is None:
        return None

    if bool(getattr(runtime_term, "fixed", False)):
        return 0

    if hasattr(runtime_term, "expected_linked_penalty_count"):
        value = runtime_term.expected_linked_penalty_count
        return None if value is None else int(value)

    fixed_flags = getattr(runtime_term, "fixed_flags", None)
    if fixed_flags is not None:
        n_penalties = int(np.sum(~np.asarray(fixed_flags, dtype=bool)))
    elif getattr(runtime_term, "n_main_penalties", None) is not None:
        n_penalties = int(runtime_term.n_main_penalties)
    else:
        term_type = str(getattr(runtime_term, "term_type", "smooth"))
        if term_type in {"tensor_smooth", "tensor_interaction"}:
            n_penalties = len(list(getattr(runtime_term, "feature", ()) or ()))
        else:
            n_penalties = 1

    if bool(getattr(runtime_term, "select", False)):
        n_penalties += 1

    return int(n_penalties)


def _build_penalty_group_specs(terms):
    groups = {}
    for term in terms:
        smoothing_id = getattr(term, "smoothing_id", None)
        if smoothing_id is None:
            continue

        key = str(smoothing_id)
        expected_count = _expected_penalty_group_size(term)
        if key not in groups:
            groups[key] = PenaltyGroupSpec(
                smoothing_id=key,
                term_ids=(
                    []
                    if getattr(term, "term_id", None) is None
                    else [str(term.term_id)]
                ),
                labels=[str(getattr(term, "label", key))],
                sp_count=expected_count,
            )
            continue

        group = groups[key]
        if getattr(term, "term_id", None) is not None:
            group.term_ids.append(str(term.term_id))
        group.labels.append(str(getattr(term, "label", key)))
        if (
            group.sp_count is not None
            and expected_count is not None
            and group.sp_count != expected_count
        ):
            raise ValueError(
                f"Linked smoothing id {key!r} expects {group.sp_count} smoothing "
                f"parameters from earlier terms, got {expected_count}."
            )
        if group.sp_count is None:
            group.sp_count = expected_count

    return list(groups.values())


def instantiate_predictor_terms(predictor_specs):
    out = []
    for pred in predictor_specs:
        for t in pred.terms:
            if not isinstance(t, TermSpec):
                raise TypeError(
                    f"Predictor terms must be canonical TermSpec, got {type(t)}."
                )
        runtime_terms = [instantiate_term(t) for t in pred.terms]
        metadata = dict(pred.metadata)
        metadata["penalty_group_specs"] = _build_penalty_group_specs(runtime_terms)
        out.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=runtime_terms,
                has_intercept=bool(getattr(pred, "has_intercept", False)),
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=metadata,
            )
        )
    return out


__all__ = [
    "instantiate_term",
    "instantiate_predictor_terms",
]
