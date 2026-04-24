"""Compiler-owned term instantiation from declarative specs."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..smooths.categorical.fs import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
)
from ..smooths.categorical.mrf import MarkovRandomFieldTerm
from ..smooths.categorical.re import RandomEffectTerm
from ..smooths.registry import make_smooth_term
from ..smooths.univariate.cr import CubicSplineTerm
from ..smooths.univariate.gp import GPSmoothTerm
from ..smooths.univariate.ps import PSplineTerm1D
from ..specs import LinearPredictorSpec, PenaltyGroupSpec, TermSpec
from ..specs.smooth import (
    CubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
    CyclicCubicRegressionSmoothSpec,
    FactorSmoothInteractionSpec,
    GPSmoothSpec,
    MarkovRandomFieldSmoothSpec,
    PSplineSmoothSpec,
    RandomEffectSmoothSpec,
    SumToZeroFactorSmoothSpec,
    TensorANOVASmoothSpec,
    TensorInteractionSmoothSpec,
    TensorProductSmoothSpec,
    ThinPlateShrinkageSmoothSpec,
    ThinPlateSmoothSpec,
    tensor_basis_list,
)
from ..terms.linear import LinearTerm


def _as_list_or_repeat(value: Any, n: int):
    if isinstance(value, str):
        return [value] * n
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) != n:
            raise ValueError(f"Expected length {n}, got {len(out)}.")
        return out
    return [value] * n


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

    if isinstance(
        smooth_spec,
        (
            CubicRegressionSmoothSpec,
            CubicShrinkageSmoothSpec,
            CyclicCubicRegressionSmoothSpec,
        ),
    ):
        bs = str(smooth_spec.bs).lower()
        if len(features) != 1:
            raise NotImplementedError(
                f"Current runtime only materializes 1D s(..., bs={bs!r}) terms."
            )
        return CubicSplineTerm(
            feature=features[0],
            k=smooth_spec.k,
            basis=bs,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            fixed=smooth_spec.fx,
            constraint_mode=smooth_spec.constraint_mode,
            shared_basis_setup=smooth_spec.shared_basis_setup,
            pc=smooth_spec.pc,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(smooth_spec, PSplineSmoothSpec):
        if len(features) != 1:
            raise NotImplementedError(
                "Current runtime only materializes 1D s(..., bs='ps') terms."
            )
        return PSplineTerm1D(
            feature=features[0],
            k=smooth_spec.k,
            basis="ps",
            m=smooth_spec.m,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            fixed=smooth_spec.fx,
            constraint_mode=smooth_spec.constraint_mode,
            pc=smooth_spec.pc,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(smooth_spec, (ThinPlateSmoothSpec, ThinPlateShrinkageSmoothSpec)):
        return make_smooth_term(
            str(smooth_spec.bs).lower(),
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
            constraint_mode=smooth_spec.constraint_mode,
            pc=smooth_spec.pc,
            knots=smooth_spec.knots,
            xt=smooth_spec.xt,
            metadata=metadata,
        )

    if isinstance(smooth_spec, GPSmoothSpec):
        return GPSmoothTerm(
            feature=features,
            k=smooth_spec.k,
            basis="gp",
            m=smooth_spec.m,
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            fixed=smooth_spec.fx,
            constraint_mode=smooth_spec.constraint_mode,
            pc=smooth_spec.pc,
            knots=smooth_spec.knots,
            xt=smooth_spec.xt,
            metadata=metadata,
        )

    if isinstance(smooth_spec, MarkovRandomFieldSmoothSpec):
        return MarkovRandomFieldTerm(
            feature=features,
            k=smooth_spec.k,
            basis="mrf",
            label=label,
            term_id=term_like.term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=smooth_spec.sp,
            select=smooth_spec.select,
            xt=smooth_spec.xt,
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
            xt=smooth_spec.xt,
            fixed=smooth_spec.fx,
            knots=smooth_spec.knots,
            metadata=metadata,
        )

    if isinstance(
        smooth_spec,
        (TensorProductSmoothSpec, TensorInteractionSmoothSpec, TensorANOVASmoothSpec),
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
            "metadata": metadata,
        }

        if isinstance(smooth_spec, TensorInteractionSmoothSpec):
            kwargs["mc"] = smooth_spec.mc
        elif isinstance(smooth_spec, TensorANOVASmoothSpec):
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

    fixed_flags = getattr(runtime_term, "fixed_flags", None)
    if fixed_flags is not None:
        n_penalties = int(np.sum(~np.asarray(fixed_flags, dtype=bool)))
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
