from dataclasses import replace
import warnings

import numpy as np

from ..specs import LinearPredictorSpec, TermSpec


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
    opts = dict(term.basis_options or {})
    return (
        term.smoothing_id is not None
        and str(opts.get("special", "s")) == "s"
        and len(term.features) == 1
        and str(opts.get("bs", "cr")).lower() in {"cr", "cs"}
    )


def attach_shared_basis_metadata(predictor_specs, X, feature_names):
    X = np.asarray(X)
    feature_names = list(feature_names)

    all_by_id = {}
    eligible_by_id = {}

    for pi, pred in enumerate(predictor_specs):
        for ti, term in enumerate(pred.terms):
            if isinstance(term, TermSpec) and term.kind == "smooth" and term.smoothing_id is not None:
                key = str(term.smoothing_id)
                all_by_id.setdefault(key, []).append((pi, ti, term))

                if _eligible_id_pool_term(term):
                    eligible_by_id.setdefault(key, []).append((pi, ti, term))

    replacements = {}

    for id_key, eligible_items in eligible_by_id.items():
        all_items = all_by_id.get(id_key, [])
        if len(eligible_items) < 2:
            continue

        first_term = eligible_items[0][2]
        first_opts = dict(first_term.basis_options or {})
        first_k = int(first_opts.get("k", 10))
        first_fx = bool(first_opts.get("fx", False))
        incompatible = []
        pooled_columns = []

        for pi, ti, term in eligible_items:
            opts = dict(term.basis_options or {})
            if int(opts.get("k", 10)) != first_k:
                incompatible.append(
                    f"{term.label!r} has k={opts.get('k')}, expected k={first_k}"
                )
                continue
            if bool(opts.get("fx", False)) != first_fx:
                incompatible.append(
                    f"{term.label!r} has fx={opts.get('fx')}, expected fx={first_fx}"
                )
                continue

            x_col = _resolve_feature_column(X, feature_names, term.features[0])
            pooled_columns.append(x_col)

        if incompatible:
            raise NotImplementedError(
                f"Linked id={id_key!r} currently supports only common-k/common-fx "
                f"1D cr smooths. Problems: {incompatible}"
            )

        if len(pooled_columns) < 2:
            continue

        pooled_x = np.concatenate(pooled_columns)

        if len(all_items) > len(eligible_items):
            skipped = [
                term.label
                for _, _, term in all_items
                if not _eligible_id_pool_term(term)
            ]
            warnings.warn(
                f"id={id_key!r} is used by terms outside the current shared-basis "
                f"pooling subset. Shared basis setup was applied only to compatible "
                f"1D cr s() terms; skipped terms: {skipped}"
            )

        shared_setup = {
            "mode": "pooled_cr_1d",
            "id": id_key,
            "k": first_k,
            "fx": first_fx,
            "n_linked_terms": len(eligible_items),
            "features": [term.features[0] for _, _, term in eligible_items],
            "pooled_x": pooled_x.tolist(),
        }

        for pi, ti, term in eligible_items:
            opts = dict(term.basis_options or {})
            opts["shared_basis_setup"] = shared_setup
            replacements[(pi, ti)] = replace(term, basis_options=opts)

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
