"""Compatibility adjustments to fitted external preprocessing pipelines."""

import re

import numpy as np
from pretab.transformers.ple.ple import PLETransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class _FittedPiecewiseLinearEncoder(TransformerMixin, BaseEstimator):
    """Evaluate fitted PreTab PLE intervals independently of query batches.

    PreTab changes its final threshold to the query maximum when that maximum
    is smaller. Keep the learned thresholds fixed while preserving its interval
    boundaries and raw-value tail convention. This vectorized implementation
    uses the fitted interval thresholds, without evaluating condition strings.
    """

    def __init__(self, knots):
        self.knots = knots
        self.n_features_in_ = len(knots)

    def fit(self, X, y=None):
        raise RuntimeError("A fitted PLE adapter cannot be refitted.")

    @classmethod
    def from_fitted(cls, encoder):
        knots = tuple(
            tuple(
                sorted(
                    {
                        float(value)
                        for condition in conditions
                        for value in re.findall(encoder.pattern, condition)
                    }
                )
            )
            for conditions in encoder.conditions_
        )
        if any(not values or not np.isfinite(values).all() for values in knots):
            raise ValueError("Fitted PLE intervals require finite split thresholds.")
        return cls(knots)

    def transform(self, X):
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if values.shape[1] != len(self.knots):
            raise ValueError("PLE input width differs from its fitted feature width.")
        encoded = []
        for feature, knots in zip(values.T, self.knots, strict=True):
            cuts = np.asarray(knots)
            bins = np.searchsorted(cuts, feature, side="left")
            columns = np.arange(len(cuts) + 1)
            result = (columns[None, :] < bins[:, None]).astype(np.float32)
            interior = (feature > cuts[0]) & (feature < cuts[-1])
            coordinate = feature.copy()
            position = bins[interior]
            coordinate[interior] = (feature[interior] - cuts[position - 1]) / (
                cuts[position] - cuts[position - 1]
            )
            result[np.arange(len(feature)), bins] = coordinate
            encoded.append(result)
        return np.hstack(encoded)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features


def stabilize_fitted_preprocessing(preprocessor):
    """Keep fitted feature vocabularies and PLE thresholds stable at inference."""

    column_transformer = getattr(preprocessor, "column_transformer", None)
    for _name, transformer, _columns in getattr(
        column_transformer, "transformers_", ()
    ):
        named_steps = getattr(transformer, "named_steps", {})
        encoder = named_steps.get("onehot")
        if encoder is not None and hasattr(encoder, "handle_unknown"):
            encoder.set_params(handle_unknown="ignore")
        for name, step in named_steps.items():
            if isinstance(step, PLETransformer):
                transformer.set_params(
                    **{name: _FittedPiecewiseLinearEncoder.from_fitted(step)}
                )
