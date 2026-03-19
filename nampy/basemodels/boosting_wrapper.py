from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BoostingBackendAdapter(ABC):
    """Backend adapter contract for boosting-style estimators."""

    @abstractmethod
    def fit(
        self,
        x,
        y,
        X_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
    ):
        """Fit the backend model."""

    @abstractmethod
    def predict(self, x):
        """Predict point estimates."""

    def predict_proba(self, x):
        raise AttributeError(f"{self.__class__.__name__} does not support predict_proba.")

    def pred_dist(self, x, max_iter: Optional[int] = None):
        raise AttributeError(f"{self.__class__.__name__} does not support pred_dist.")

    @property
    def feature_importances_(self):
        return None

    def raw_params(self, x):
        raise AttributeError(f"{self.__class__.__name__} does not expose raw_params.")


class BoostingSurvivalBackendAdapter(ABC):
    """Backend adapter contract for survival boosting estimators."""

    @abstractmethod
    def fit(self, x, t, e, X_val=None, T_val=None, E_val=None):
        """Fit the survival backend model."""

    @abstractmethod
    def predict(self, x):
        """Predict survival target estimates."""

    @abstractmethod
    def pred_dist(self, x, max_iter: Optional[int] = None):
        """Predict output distribution."""

    def raw_params(self, x):
        raise AttributeError(f"{self.__class__.__name__} does not expose raw_params.")

    @property
    def feature_importances_(self):
        return None


class GenericBoostingWrapper:
    """Task-agnostic wrapper around a boosting backend adapter."""

    def __init__(self, backend: BoostingBackendAdapter):
        self.backend = backend

    def fit(
        self,
        x,
        y,
        X_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
    ):
        self.backend.fit(
            x,
            y,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
        )
        return self

    def predict(self, x):
        return self.backend.predict(x)

    def predict_proba(self, x):
        return self.backend.predict_proba(x)

    def pred_dist(self, x, max_iter: Optional[int] = None):
        return self.backend.pred_dist(x, max_iter=max_iter)

    def raw_params(self, x):
        return self.backend.raw_params(x)

    @property
    def feature_importances_(self):
        return self.backend.feature_importances_


class GenericBoostingSurvivalWrapper:
    """Wrapper around a survival boosting backend adapter."""

    def __init__(self, backend: BoostingSurvivalBackendAdapter):
        self.backend = backend

    def fit(self, x, t, e, X_val=None, T_val=None, E_val=None):
        self.backend.fit(x, t, e, X_val=X_val, T_val=T_val, E_val=E_val)
        return self

    def predict(self, x):
        return self.backend.predict(x)

    def pred_dist(self, x, max_iter: Optional[int] = None):
        return self.backend.pred_dist(x, max_iter=max_iter)

    def raw_params(self, x):
        return self.backend.raw_params(x)

    @property
    def feature_importances_(self):
        return self.backend.feature_importances_
