from __future__ import annotations

from typing import Any

from ..specs import LinearPredictorSpec, ParametricTermSpec, SmoothTermSpec

from .registry import make_smooth_term
from .univariate.cubic_regression import LinearTerm, SplineTerm1D
from .univariate.pspline import PSplineTerm1D
from .univariate.thin_plate import ThinPlateSplineTerm  # imported for registration / direct access
from .univariate.gp import GPSmoothTerm
from .tensor.te import TensorProductSplineTerm  # imported for registration
from .tensor.ti import InteractionTensorProductSplineTerm  # imported for registration
from .tensor.t2 import TensorANOVASplineTerm  # imported for registration

from .categorical.factor_smooth import FSmoothInteractionTerm, SZSmoothInteractionTerm
from .categorical.mrf import MarkovRandomFieldTerm
from .categorical.random_effect import RandomEffectTerm


def _as_list_or_repeat(value: Any, n: int):
    if isinstance(value, str):
        return [value] * n
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) != n:
            raise ValueError(f"Expected length {n}, got {len(out)}.")
        return out
    return [value] * n


def _spec_metadata(spec: SmoothTermSpec) -> dict[str, Any]:
    out = dict(spec.metadata)
    out["smooth_spec"] = spec.to_metadata_dict()
    return out


def materialize_term(term_like):
    if hasattr(term_like, "fit") and callable(term_like.fit):
        return term_like

    if isinstance(term_like, ParametricTermSpec):
        return LinearTerm(
            feature=term_like.name,
            label=term_like.label,
            metadata=dict(term_like.metadata),
        )

    if not isinstance(term_like, SmoothTermSpec):
        raise TypeError(f"Unsupported term/spec object: {type(term_like)}")

    special = term_like.special
    features = list(term_like.features)
    metadata = _spec_metadata(term_like)

    if special == "s":
        bs = str(term_like.bs).lower()

        if bs in {"cr", "cs", "cc"}:
            if len(features) != 1:
                raise NotImplementedError(
                    f"Current runtime only materializes 1D s(..., bs={bs!r}) terms."
                )

            return SplineTerm1D(
                feature=features[0],
                k=term_like.k,
                basis=bs,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                select=bool(term_like.select),
                fixed=bool(term_like.fx),
                constraint_mode=term_like.constraint_mode,
                shared_basis_setup=term_like.shared_basis_setup,
                pc=term_like.pc,
                knots=term_like.knots,
                metadata=metadata,
            )

        if bs == "ps":
            if len(features) != 1:
                raise NotImplementedError(
                    "Current runtime only materializes 1D s(..., bs='ps') terms."
                )

            return PSplineTerm1D(
                feature=features[0],
                k=term_like.k,
                basis=bs,
                m=term_like.m,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                select=bool(term_like.select),
                fixed=bool(term_like.fx),
                constraint_mode=term_like.constraint_mode,
                pc=term_like.pc,
                knots=term_like.knots,
                metadata=metadata,
            )

        if bs in {"tp", "ts"}:
            return make_smooth_term(
                bs,
                feature=features,
                k=term_like.k,
                basis=bs,
                m=term_like.m,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                select=bool(term_like.select),
                fixed=bool(term_like.fx),
                constraint_mode=term_like.constraint_mode,
                pc=term_like.pc,
                knots=term_like.knots,
                xt=term_like.xt,
                metadata=metadata,
            )

        if bs == "gp":
            return GPSmoothTerm(
                feature=features,
                k=term_like.k,
                basis=bs,
                m=term_like.m,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                select=bool(term_like.select),
                fixed=bool(term_like.fx),
                constraint_mode=term_like.constraint_mode,
                pc=term_like.pc,
                knots=term_like.knots,
                xt=term_like.xt,
                metadata=metadata,
            )

        if bs == "mrf":
            if term_like.select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='mrf' smooths."
                )
            return MarkovRandomFieldTerm(
                feature=features,
                k=term_like.k,
                basis=bs,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                xt=term_like.xt,
                knots=term_like.knots,
                metadata=metadata,
            )

        if bs == "re":
            if term_like.select:
                raise NotImplementedError(
                    "select=True is not implemented for bs='re' terms."
                )
            return RandomEffectTerm(
                feature=features,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                xt=term_like.xt,
                metadata=metadata,
            )

        if bs == "fs":
            if term_like.select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='fs' factor smooths."
                )
            return FSmoothInteractionTerm(
                feature=features,
                k=term_like.k,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                xt=term_like.xt,
                fixed=bool(term_like.fx),
                knots=term_like.knots,
                metadata=metadata,
            )

        if bs == "sz":
            if term_like.select:
                raise NotImplementedError(
                    "select=True is not yet implemented for bs='sz' factor smooths."
                )
            return SZSmoothInteractionTerm(
                feature=features,
                k=term_like.k,
                label=term_like.label,
                smoothing_id=term_like.id,
                by=term_like.by,
                sp=term_like.sp,
                xt=term_like.xt,
                fixed=bool(term_like.fx),
                knots=term_like.knots,
                metadata=metadata,
            )

        raise NotImplementedError(
            f"Current runtime materializes bs in "
            f"{{'cr','cs','cc','ps','tp','ts','gp','mrf','re','fs','sz'}} for s(...). "
            f"Received bs={bs!r}."
        )

    if special in {"te", "ti", "t2"}:
        if term_like.select:
            raise NotImplementedError(
                "select=True is not yet implemented for te()/ti()/t2() tensor smooths."
            )
        basis = _as_list_or_repeat(term_like.bs, len(features))
        if any(str(bs).lower() != "cr" for bs in basis):
            raise NotImplementedError(
                f"Current runtime only materializes bs='cr' marginals for {special}(...). "
                f"Received basis={basis!r}."
            )

        kwargs = dict(
            feature=features,
            k=term_like.k,
            basis=basis,
            label=term_like.label,
            smoothing_id=term_like.id,
            by=term_like.by,
            sp=term_like.sp,
            fixed=bool(term_like.fx),
            knots=term_like.knots,
            metadata=metadata,
        )

        if special == "ti":
            kwargs["mc"] = term_like.mc
        elif special == "t2":
            kwargs["full"] = bool(term_like.full)
            kwargs["ord"] = term_like.ord

        return make_smooth_term(special, **kwargs)

    raise ValueError(f"Unknown smooth special {special!r}.")


def materialize_predictor_specs(predictor_specs):
    out = []
    for pred in predictor_specs:
        out.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=[materialize_term(t) for t in pred.terms],
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=dict(pred.metadata),
            )
        )
    return out
