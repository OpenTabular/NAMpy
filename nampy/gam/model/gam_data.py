# gam/model/gam_data.py
"""Data coercion helpers for the GAM class."""
import numpy as np
import pandas as pd


class _GAMDataMixin:
    def _coerce_X(self, X):
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_np = self._dataframe_to_feature_matrix(X)
        else:
            X_np = np.asarray(X)
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            if X_np.ndim != 2:
                raise ValueError("X must be a 2D array or DataFrame.")

            feature_names = [f"x{i}" for i in range(X_np.shape[1])]

            # Preserve mixed/object arrays if the caller supplied them.
            if X_np.dtype != object:
                X_num = np.asarray(X_np, dtype=np.float64)
                if not np.isfinite(X_num).all():
                    raise ValueError("X contains NaN or Inf")
                X_np = X_num

        return X_np, feature_names

    def _coerce_optional_offset(self, offset, n_rows, *, name="offset"):
        if offset is None:
            return None
        out = np.asarray(offset, dtype=np.float64).ravel()
        if out.shape != (int(n_rows),):
            raise ValueError(
                f"{name} must have shape ({int(n_rows)},), got {out.shape}."
            )
        if not np.isfinite(out).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return out

    def _combine_offsets(self, *offsets):
        arrs = [np.asarray(o, dtype=np.float64) for o in offsets if o is not None]
        if not arrs:
            return None
        out = np.zeros_like(arrs[0], dtype=np.float64)
        for arr in arrs:
            if arr.shape != out.shape:
                raise ValueError(
                    f"Offset arrays must all have the same shape, got {out.shape} and {arr.shape}."
                )
            out = out + arr
        return out

    def _knots_for_feature(self, feature_name, *, knots=None):
        knots = self.knots if knots is None else knots
        if knots is None:
            return None
        if isinstance(knots, dict):
            return knots.get(str(feature_name), None)
        return knots

    def _knots_for_features(self, feature_names, *, knots=None):
        knots = self.knots if knots is None else knots
        if knots is None:
            return None
        if isinstance(knots, dict):
            vals = [knots.get(str(f), None) for f in feature_names]
            return None if all(v is None for v in vals) else vals
        return knots

    def _dataframe_to_feature_matrix(
        self, X_df: pd.DataFrame, *, allow_missing_non_numeric=False
    ):
        non_numeric = [
            c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])
        ]

        if len(non_numeric) == 0:
            X_np = X_df.to_numpy(dtype=np.float64)
            if not np.isfinite(X_np).all():
                raise ValueError("X contains NaN or Inf")
            return X_np

        for c in X_df.columns:
            s = X_df[c]
            if pd.api.types.is_numeric_dtype(s):
                vals = np.asarray(s, dtype=np.float64)
                if not np.isfinite(vals).all():
                    raise ValueError(f"Numeric column {c!r} contains NaN or Inf.")
            else:
                if s.isna().any() and not bool(allow_missing_non_numeric):
                    raise ValueError(
                        f"Non-numeric column {c!r} contains missing values, "
                        "which are not currently supported in fitting."
                    )

        return X_df.to_numpy(dtype=object)

    def _coerce_feature_matrix(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        return X

    def _coerce_offset(self, offset, n_rows):
        from ..fit.offsets import coerce_offset_array

        return coerce_offset_array(offset, n_rows)

    def _prediction_offset(self, X, offset):
        from ..fit.offsets import resolve_prediction_offset

        return resolve_prediction_offset(self, X, offset)

    def _coerce_formula_predict_inputs(self, X):
        from ..formula.preprocess import apply_formula_preprocess_to_new_data

        if not self.formula_mode_:
            X_np, feature_names = self._coerce_X(X)
            return X_np, feature_names, None

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Prediction for formula-based GAMs currently requires a pandas DataFrame."
            )

        X_work = apply_formula_preprocess_to_new_data(X, self.formula_preprocess_state_)

        missing = [c for c in self.formula_used_columns_ if c not in X_work.columns]
        if missing:
            raise KeyError(f"Prediction data is missing formula columns: {missing}")

        X_df = X_work[self.formula_used_columns_]
        X_np = self._dataframe_to_feature_matrix(X_df, allow_missing_non_numeric=True)

        offset = None
        if self.formula_offset_name_ is not None:
            if self.formula_offset_name_ not in X_work.columns:
                raise KeyError(
                    f"Prediction data is missing formula offset column: {self.formula_offset_name_!r}"
                )
            if not pd.api.types.is_numeric_dtype(X_work[self.formula_offset_name_]):
                raise NotImplementedError(
                    "Current formula-based prediction supports numeric offsets only. "
                    f"Offset column {self.formula_offset_name_!r} is non-numeric."
                )
            offset = X_work[self.formula_offset_name_].to_numpy(dtype=np.float64)

        return X_np, list(X_df.columns), offset
