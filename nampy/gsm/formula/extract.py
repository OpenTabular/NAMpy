import numpy as np
import pandas as pd

from ..specs import ParametricTermSpec, SmoothTermSpec


def _collect_used_columns_from_predictor_specs(predictor_specs):
    used = set()

    for pred in predictor_specs:
        for term in pred.terms:
            if isinstance(term, ParametricTermSpec):
                src = term.metadata.get("source_variables", None)
                if src is not None:
                    used.update(str(v) for v in src)
                else:
                    used.add(term.name)
                continue

            if isinstance(term, SmoothTermSpec):
                used.update(term.features)
                if isinstance(term.by, str):
                    used.add(term.by)
                continue

            feat = getattr(term, "feature", None)
            if feat is not None:
                if isinstance(feat, (list, tuple)):
                    used.update(str(v) for v in feat)
                else:
                    used.add(str(feat))

            by = getattr(term, "by", None)
            if isinstance(by, str):
                used.add(by)

    return used


def _dataframe_to_feature_matrix(X_df: pd.DataFrame):
    """
    Convert a DataFrame to a model feature matrix while preserving non-numeric
    columns for smooth constructors like bs="re".
    """
    non_numeric = [
        c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
    ]

    if len(non_numeric) == 0:
        X_np = X_df.to_numpy(dtype=np.float64)
        if not np.isfinite(X_np).all():
            raise ValueError("Referenced numeric columns contain NaN or Inf.")
        return X_np

    for c in X_df.columns:
        s = X_df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = np.asarray(s, dtype=np.float64)
            if not np.isfinite(vals).all():
                raise ValueError(f"Referenced numeric column {c!r} contains NaN or Inf.")
        else:
            if s.isna().any():
                raise ValueError(
                    f"Referenced non-numeric column {c!r} contains missing values, "
                    "which are not currently supported in fitting."
                )

    return X_df.to_numpy(dtype=object)


def extract_formula_data(data, parsed, y=None, predictor_specs=None):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Formula-based fitting currently requires `data` to be a pandas DataFrame."
        )

    if predictor_specs is not None:
        used = _collect_used_columns_from_predictor_specs(predictor_specs)
        offset_names = [
            pred.offset_name for pred in predictor_specs if pred.offset_name is not None
        ]
    else:
        used = set()
        offset_names = []
        for pf in parsed.predictors:
            if pf.offset_name is not None:
                offset_names.append(pf.offset_name)

            for term in pf.terms:
                if hasattr(term, "variables"):
                    used.update(term.variables)
                elif hasattr(term, "features"):
                    used.update(term.features)
                    by_val = term.kwargs.get("by", None) if hasattr(term, "kwargs") else None
                    if isinstance(by_val, str):
                        used.add(by_val)

    offset_names = sorted(set(offset_names))
    if len(offset_names) > 1:
        raise NotImplementedError(
            "Current fitting core supports one active linear predictor only, so "
            "multiple distinct predictor-specific offsets are not yet supported."
        )
    offset_name = offset_names[0] if offset_names else None

    used_cols = [c for c in data.columns if c in used]
    missing = sorted(used.difference(set(data.columns)))
    if missing:
        raise KeyError(
            f"Formula references columns not present in `data`: {missing}"
        )

    X_df = data[used_cols]
    X_np = _dataframe_to_feature_matrix(X_df)
    feature_names = list(X_df.columns)

    if y is None:
        if parsed.response_name is None:
            raise ValueError(
                "Formula does not specify a response, so `y` must be supplied explicitly."
            )
        if parsed.response_name not in data.columns:
            raise KeyError(
                f"Response column {parsed.response_name!r} not found in `data`."
            )
        y_out = np.asarray(data[parsed.response_name]).ravel()
    else:
        y_out = np.asarray(y).ravel()

    offset_out = None
    if offset_name is not None:
        if offset_name not in data.columns:
            raise KeyError(
                f"Offset column {offset_name!r} not found in `data`."
            )
        if not pd.api.types.is_numeric_dtype(data[offset_name]):
            raise NotImplementedError(
                "Current formula-based GAM fitting supports numeric offsets only. "
                f"Offset column {offset_name!r} is non-numeric."
            )
        offset_out = np.asarray(data[offset_name], dtype=np.float64).ravel()

    return X_np, feature_names, y_out, used_cols, offset_out
