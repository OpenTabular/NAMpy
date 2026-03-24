# gsm/smooths/ids.py
from dataclasses import replace
import warnings

import numpy as np

from ..specs import LinearPredictorSpec, SmoothTermSpec


def _resolve_feature_column(X, feature_names, feature):
    feature = str(feature)
    if feature not in feature_names:
        raise KeyError(
            f"Feature {feature!r} not found in feature_names={feature_names}."
        )
    idx = feature_names.index(feature)
    return np.asarray(X[:, idx], dtype=np.float64).ravel()


def _eligible_id_pool_term(term):
    return (
        isinstance(term, SmoothTermSpec)
        and term.id is not None
        and term.special == "s"
        and len(term.features) == 1
        and str(term.bs).lower() in {"cr", "cs"}
    )


def attach_shared_basis_metadata(predictor_specs, X, feature_names):
    """
    Attach constructor-time shared-basis setup metadata to compatible smooth.specs.

    Current scope
    -------------
    Implements a conservative subset of mgcv-style id pooling:
    - only 1D s(..., bs="cr", id=...)
    - requires common k and fx within each linked id group
    - pools observed covariate data across linked terms for basis setup
    - leaves smoothing-parameter linkage unchanged (already handled by id/smoothing_id)

    Returns
    -------
    New predictor_specs list with compatible SmoothTermSpec objects replaced by
    clones carrying `shared_basis_setup`.
    """
    X = np.asarray(X, dtype=np.float64)
    feature_names = list(feature_names)

    all_by_id = {}
    eligible_by_id = {}

    for pi, pred in enumerate(predictor_specs):
        for ti, term in enumerate(pred.terms):
            if isinstance(term, SmoothTermSpec) and term.id is not None:
                key = str(term.id)
                all_by_id.setdefault(key, []).append((pi, ti, term))

                if _eligible_id_pool_term(term):
                    eligible_by_id.setdefault(key, []).append((pi, ti, term))

    replacements = {}

    for id_key, eligible_items in eligible_by_id.items():
        all_items = all_by_id.get(id_key, [])

        if len(eligible_items) < 2:
            continue

        first_term = eligible_items[0][2]
        first_k = int(first_term.k)
        first_fx = bool(first_term.fx)

        incompatible = []
        pooled_columns = []

        for pi, ti, term in eligible_items:
            if int(term.k) != first_k:
                incompatible.append(
                    f"{term.label!r} has k={term.k}, expected k={first_k}"
                )
                continue
            if bool(term.fx) != first_fx:
                incompatible.append(
                    f"{term.label!r} has fx={term.fx}, expected fx={first_fx}"
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
            replacements[(pi, ti)] = replace(
                term,
                shared_basis_setup=shared_setup,
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
                parameter_name=pred.parameter_name,
                offset_name=pred.offset_name,
                metadata=dict(pred.metadata),
            )
        )

    return out_specs
