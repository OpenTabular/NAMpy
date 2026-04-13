import warnings
from dataclasses import replace

import numpy as np

from ...splines.cubic import CubicSplines
from ..specs import LinearPredictorSpec, TermSpec, replace_smooth_spec
from ..specs.smooth import (
    CubicRegressionSmoothSpec,
    CubicShrinkageSmoothSpec,
)


def _resolve_feature_column(X, feature_names, feature):
    feature = str(feature)
    if feature not in feature_names:
        raise KeyError(
            f"Feature {feature!r} not found in feature_names={feature_names}."
        )
    idx = feature_names.index(feature)
    return np.asarray(X[:, idx], dtype=np.float64).ravel()


def _eligible_id_pool_term(term):
    if not isinstance(term, TermSpec):
        return False
    if term.kind != "smooth":
        return False
    return (
        term.smoothing_id is not None
        and term.smooth_spec is not None
        and str(term.smooth_spec.special) == "s"
        and len(term.features) == 1
        and isinstance(
            term.smooth_spec,
            (CubicRegressionSmoothSpec, CubicShrinkageSmoothSpec),
        )
    )


def attach_shared_basis_metadata(predictor_specs, X, feature_names):
    X = np.asarray(X)
    feature_names = list(feature_names)

    all_by_id = {}
    eligible_by_id = {}

    for pi, pred in enumerate(predictor_specs):
        for ti, term in enumerate(pred.terms):
            if (
                isinstance(term, TermSpec)
                and term.kind == "smooth"
                and term.smoothing_id is not None
            ):
                key = str(term.smoothing_id)
                all_by_id.setdefault(key, []).append((pi, ti, term))

                if _eligible_id_pool_term(term):
                    eligible_by_id.setdefault(key, []).append((pi, ti, term))

    replacements = {}

    for id_key, eligible_items in eligible_by_id.items():
        all_items = all_by_id.get(id_key, [])
        if len(all_items) > 1 and len(all_items) != len(eligible_items):
            unsupported = [term.label for _, _, term in all_items]
            raise NotImplementedError(
                f"Linked id={id_key!r} currently requires full mgcv-parity support "
                f"across every linked term. NAMpy's old partial pooling subset for "
                f"compatible 1D cubic `s()` terms was removed. Terms: {unsupported}"
            )
        if len(eligible_items) < 2:
            continue

        first_term = eligible_items[0][2]
        if first_term.smooth_spec is None:
            raise ValueError(f"Smooth term {first_term.label!r} is missing smooth_spec.")
        first_fx = bool(first_term.smooth_spec.fx)

        # fx mismatch: mixing fixed and penalized terms is semantically incompatible
        # because fixed terms have no smoothing parameter to share.
        fx_incompatible = []
        for _pi, _ti, term in eligible_items:
            if term.smooth_spec is None:
                raise ValueError(f"Smooth term {term.label!r} is missing smooth_spec.")
            if bool(term.smooth_spec.fx) != first_fx:
                fx_incompatible.append(
                    f"{term.label!r} has fx={term.smooth_spec.fx}, expected fx={first_fx}"
                )
        if fx_incompatible:
            raise NotImplementedError(
                f"Linked id={id_key!r}: all terms must have the same fx value. "
                f"Problems: {fx_incompatible}"
            )

        # k harmonisation: mgcv resolves k mismatches by using the first term's k
        # for the shared basis, matching the representative-term convention in mgcv.
        all_k = [
            int(term.smooth_spec.k) for _, _, term in eligible_items if term.smooth_spec
        ]
        canonical_k = all_k[0]
        if len(set(all_k)) > 1:
            warnings.warn(
                f"Linked id={id_key!r}: terms have differing k values "
                f"{sorted(set(all_k))}; harmonizing all to k={canonical_k} "
                f"(first term's k), matching mgcv behaviour.",
                stacklevel=2,
            )

        pooled_columns = []
        for _pi, _ti, term in eligible_items:
            x_col = _resolve_feature_column(X, feature_names, term.features[0])
            pooled_columns.append(x_col)

        if len(pooled_columns) < 2:
            continue

        pooled_x = np.concatenate(pooled_columns)

        # Build a temporary spline from pooled data to extract canonical knots and
        # penalty matrices.  Storing the k×k penalty (not the full n×k basis) keeps
        # the shared_setup dict O(k²) rather than O(n).
        pooled_spl = CubicSplines(pooled_x, canonical_k)
        shared_setup = {
            "mode": "pooled_cr_1d",
            "id": id_key,
            "k": canonical_k,
            "fx": first_fx,
            "n_linked_terms": len(eligible_items),
            "features": [term.features[0] for _, _, term in eligible_items],
            "pooled_knots": pooled_spl.knots.tolist(),
            "pooled_raw_penalty": pooled_spl.raw_penalty.tolist(),
            "pooled_center_mat": pooled_spl.center_mat.tolist(),
            "pooled_penalty": pooled_spl.penalty.tolist(),
        }

        for pi, ti, term in eligible_items:
            if term.smooth_spec is None:
                raise ValueError(f"Smooth term {term.label!r} is missing smooth_spec.")
            replacements[(pi, ti)] = replace(
                term,
                smooth_spec=replace_smooth_spec(
                    term.smooth_spec,
                    k=canonical_k,
                    shared_basis_setup=shared_setup,
                ),
            )

    out_specs = []
    for pi, pred in enumerate(predictor_specs):
        new_terms = []
        for ti, term in enumerate(pred.terms):
            new_terms.append(replacements.get((pi, ti), term))

        out_specs.append(
            LinearPredictorSpec(
                name=pred.name,
                terms=new_terms,
                has_intercept=bool(getattr(pred, "has_intercept", False)),
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=dict(pred.metadata),
            )
        )

    return out_specs


__all__ = ["attach_shared_basis_metadata"]
