import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class QuantilePreprocessor(BaseEstimator, TransformerMixin):
    """
    Quantile preprocessing with optional noise injection, mirroring NodeGAMLSS logic.
    """

    def __init__(
        self,
        output_distribution="normal",
        n_quantiles=2000,
        quantile_noise=0.0,
        random_state=101,
    ):
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.quantile_noise = quantile_noise
        self.random_state = random_state
        self.transformer = None

    def fit(self, X, y=None):
        X = self._ensure_array(X)
        X_fit = X
        if self.quantile_noise and self.quantile_noise > 0:
            rng = np.random.RandomState(self.random_state)
            stds = np.std(X_fit, axis=0, keepdims=True)
            noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
            X_fit = X_fit + noise_std * rng.randn(*X_fit.shape)

        n_quantiles = min(self.n_quantiles, X_fit.shape[0]) if X_fit.shape[0] > 1 else 1
        self.transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=self.output_distribution,
            random_state=self.random_state,
            copy=False,
        )
        self.transformer.fit(X_fit)
        return self

    def transform(self, X):
        if self.transformer is None:
            raise RuntimeError("QuantilePreprocessor is not fitted yet.")
        X = self._ensure_array(X)
        return self.transformer.transform(X)

    def _ensure_array(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(X, dtype=np.float32)
