from dataclasses import replace

import numpy as np

from ..specs import LinearPredictorSpec, TermSpec, replace_smooth_spec


def _resolve_feature_column(X, feature_names, feature):
    feature = str(feature)
    if feature not in feature_names:
        raise KeyError(
            f"Feature {feature!r} not found in feature_names={feature_names}."
        )
    idx = feature_names.index(feature)
    return np.asarray(X[:, idx], dtype=object).ravel()


def _clone_linked_smooth_spec(base_term, term):
    if base_term.smooth_spec is None or term.smooth_spec is None:
        raise ValueError("Linked smooth terms must carry smooth specifications.")
    if len(base_term.features) != len(term.features):
        raise ValueError("`id` linked smooths must have same number of arguments.")

    clone = base_term.smooth_spec
    overrides = {}
    if hasattr(clone, "xt") and hasattr(term.smooth_spec, "xt"):
        overrides["xt"] = term.smooth_spec.xt
    if hasattr(clone, "knots") and hasattr(term.smooth_spec, "knots"):
        overrides["knots"] = term.smooth_spec.knots
    if overrides:
        clone = replace_smooth_spec(clone, **overrides)
    return clone


def _term_shared_basis_setup(id_key, term, pooled_feature_values, group_terms):
    return {
        "mode": "linked_id",
        "id": str(id_key),
        "pooled_feature_names": [str(f) for f in term.features],
        "pooled_feature_values": [
            np.asarray(col, dtype=object).copy() for col in pooled_feature_values
        ],
        "n_linked_terms": len(group_terms),
        "linked_term_labels": [str(group_term.label) for group_term in group_terms],
        "reference_basis_options": (
            None
            if group_terms[0].smooth_spec is None
            else group_terms[0].smooth_spec.to_basis_options()
        ),
    }


def attach_shared_basis_metadata(predictor_specs, X, feature_names):
    X = np.asarray(X, dtype=object)
    feature_names = list(feature_names)

    items_by_id = {}
    for pi, pred in enumerate(predictor_specs):
        for ti, term in enumerate(pred.terms):
            if (
                isinstance(term, TermSpec)
                and term.kind == "smooth"
                and term.smoothing_id is not None
            ):
                items_by_id.setdefault((pi, str(term.smoothing_id)), []).append(
                    (pi, ti, term)
                )

    replacements = {}

    for (_pi_key, id_key), items in items_by_id.items():
        if len(items) < 2:
            continue

        base_term = items[0][2]
        if base_term.smooth_spec is None:
            raise ValueError(f"Smooth term {base_term.label!r} is missing smooth_spec.")

        group_terms = [term for _, _, term in items]
        has_tensor_pc = any(
            term.smooth_spec is not None
            and str(term.smooth_spec.special).lower() in {"te", "ti", "t2"}
            and getattr(term.smooth_spec, "pc", None) is not None
            for term in group_terms
        )
        feature_tuples = {
            tuple(str(f) for f in term.features) for term in group_terms
        }
        if has_tensor_pc and len(feature_tuples) > 1:
            raise NotImplementedError(
                "pc= is not supported across id-linked te()/ti() terms with "
                "different feature sets; mgcv 1.9-4 fails while constructing "
                "the shared tensor basis."
            )
        has_scalar_pc = any(
            term.smooth_spec is not None
            and str(term.smooth_spec.special).lower() == "s"
            and getattr(term.smooth_spec, "pc", None) is not None
            for term in group_terms
        )
        if has_scalar_pc and len(feature_tuples) > 1:
            raise NotImplementedError(
                "pc= is not supported across id-linked s() terms with different "
                "feature sets; mgcv 1.9-4 fails while constructing the shared basis."
            )
        has_mrf = any(
            term.smooth_spec is not None
            and (
                str(getattr(term.smooth_spec, "bs", "")).lower() == "mrf"
                or (
                    isinstance(getattr(term.smooth_spec, "bs", None), (list, tuple))
                    and "mrf"
                    in {
                        str(value).lower()
                        for value in getattr(term.smooth_spec, "bs", ())
                    }
                )
            )
            for term in group_terms
        )
        if has_mrf and len(feature_tuples) > 1:
            raise NotImplementedError(
                "id= is not supported across MRF terms with different feature "
                "sets; mgcv 1.9-4 loses the factor topology while pooling them."
            )
        for term in group_terms[1:]:
            _clone_linked_smooth_spec(base_term, term)

        pooled_feature_values = []
        for pos in range(len(base_term.features)):
            pooled_feature_values.append(
                np.concatenate(
                    [
                        _resolve_feature_column(X, feature_names, term.features[pos])
                        for _, _, term in items
                    ]
                )
            )

        for pi, ti, term in items:
            cloned_spec = _clone_linked_smooth_spec(base_term, term)
            shared_basis_setup = _term_shared_basis_setup(
                id_key=id_key,
                term=term,
                pooled_feature_values=pooled_feature_values,
                group_terms=group_terms,
            )
            replacements[(pi, ti)] = replace(
                term,
                smooth_spec=cloned_spec,
                metadata={
                    **dict(term.metadata or {}),
                    "shared_basis_setup": shared_basis_setup,
                },
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
