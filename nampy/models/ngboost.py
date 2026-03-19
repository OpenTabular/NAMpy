import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from ..arch_utils.ngboost_utils import y_from_censored
from ..basemodels.ngboost import NGBSurvival as NGBSurvivalBase
from ..basemodels.ngboost import NGBoost as NGBoostBase
from .boosting_base import BaseBoostingEstimator, BaseBoostingRegressorClassifier
from ..configs.ngboost_config import DefaultNGBoostConfig


class _BaseNGBoostEstimator(BaseBoostingRegressorClassifier):
    def __init__(self, task, **kwargs):
        super().__init__(task=task, config_cls=DefaultNGBoostConfig, **kwargs)

    def pred_dist(self, x, max_iter=None):
        self._check_fitted()
        matrix = self._transform_matrix(x)
        return self.model.pred_dist(matrix, max_iter=max_iter)

    def predict_feature_vals(self, x):
        self._check_fitted()
        matrix = self._transform_matrix(x)
        return {
            "output": self.model.predict(matrix),
            "raw_params": self.model.estimator.pred_param(matrix),
        }

    def _check_fitted(self):
        super()._check_fitted()

    @property
    def feature_importances_(self):
        self._check_fitted()
        importances = self.model.feature_importances_
        if importances is None:
            return None
        return importances


class NGBoostRegressor(_BaseNGBoostEstimator):
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)

    def fit(
        self,
        X,
        y,
        val_size=0.2,
        X_val=None,
        y_val=None,
        random_state=101,
        sample_weight=None,
        val_sample_weight=None,
    ):
        X_train, X_val_df, y_train, y_val_arr = self._split_validation(
            X, y, X_val=X_val, y_val=y_val, val_size=val_size, random_state=random_state
        )
        self._fit_preprocessor(X_train, y_train, X_val_df, y_val_arr)

        train_matrix = self._transform_matrix(X_train)
        val_matrix = self._transform_matrix(X_val_df)

        self.model = NGBoostBase(task="regression", config=self.config, **self.config_kwargs)
        self.model.fit(
            train_matrix,
            np.asarray(y_train),
            X_val=val_matrix,
            y_val=np.asarray(y_val_arr),
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
        )
        return self

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(self._transform_matrix(X))

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        return {"Mean Squared Error": mean_squared_error(y_true, preds)}


class NGBoostClassifier(_BaseNGBoostEstimator):
    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)

    def fit(
        self,
        X,
        y,
        val_size=0.2,
        X_val=None,
        y_val=None,
        random_state=101,
        sample_weight=None,
        val_sample_weight=None,
    ):
        X_train, X_val_df, y_train, y_val_arr = self._split_validation(
            X, y, X_val=X_val, y_val=y_val, val_size=val_size, random_state=random_state
        )
        self._fit_preprocessor(X_train, y_train, X_val_df, y_val_arr)

        train_matrix = self._transform_matrix(X_train)
        val_matrix = self._transform_matrix(X_val_df)
        num_classes = len(np.unique(np.asarray(y_train)))

        self.model = NGBoostBase(
            task="classification",
            num_classes=num_classes,
            config=self.config,
            **self.config_kwargs,
        )
        self.model.fit(
            train_matrix,
            np.asarray(y_train),
            X_val=val_matrix,
            y_val=np.asarray(y_val_arr),
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
        )
        return self

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(self._transform_matrix(X))

    def predict_proba(self, X):
        self._check_fitted()
        return self.model.predict_proba(self._transform_matrix(X))

    def evaluate(self, X, y_true):
        preds = self.predict(X)
        return {"Accuracy": accuracy_score(y_true, preds)}


class NGBSurvival(BaseBoostingEstimator):
    def __init__(self, **kwargs):
        super().__init__(config_cls=DefaultNGBoostConfig, **kwargs)

    def fit(
        self,
        X,
        T,
        E,
        val_size=0.2,
        X_val=None,
        T_val=None,
        E_val=None,
        random_state=101,
    ):
        if (X_val is None) ^ (T_val is None) or (X_val is None) ^ (E_val is None):
            raise ValueError("X_val, T_val, and E_val must be provided together.")

        X_df = self._ensure_dataframe(X)
        T = np.asarray(T)
        E = np.asarray(E)

        if X_val is None:
            indices = np.arange(len(X_df))
            train_idx, val_idx = train_test_split(
                indices, test_size=val_size, random_state=random_state
            )
            X_train = X_df.iloc[train_idx].reset_index(drop=True)
            X_val_df = X_df.iloc[val_idx].reset_index(drop=True)
            T_train, T_val_arr = T[train_idx], T[val_idx]
            E_train, E_val_arr = E[train_idx], E[val_idx]
        else:
            X_train = X_df
            X_val_df = self._ensure_dataframe(X_val)
            T_train, T_val_arr = T, np.asarray(T_val)
            E_train, E_val_arr = E, np.asarray(E_val)

        self._fit_preprocessor(X_train, T_train, X_val_df, T_val_arr)
        train_matrix = self._transform_matrix(X_train)
        val_matrix = self._transform_matrix(X_val_df)

        self.model = NGBSurvivalBase(config=self.config, **self.config_kwargs)
        self.model.fit(
            train_matrix,
            T_train,
            E_train,
            X_val=val_matrix,
            T_val=T_val_arr,
            E_val=E_val_arr,
        )
        return self

    def _check_fitted(self):
        super()._check_fitted()

    def predict(self, X):
        self._check_fitted()
        return self.model.predict(self._transform_matrix(X))

    def pred_dist(self, X, max_iter=None):
        self._check_fitted()
        return self.model.pred_dist(self._transform_matrix(X), max_iter=max_iter)

    def predict_survival_function(self, X, times):
        dist = self.pred_dist(X)
        times = np.asarray(times, dtype=float)
        return np.asarray([dist.sf(times_i) for times_i in times]).T

    def evaluate(self, X, T, E):
        self._check_fitted()
        matrix = self._transform_matrix(X)
        params = self.model.estimator.pred_param(matrix)
        manifold = self.model.estimator.Manifold(params.T)
        return {"NLL": manifold.total_score(y_from_censored(np.asarray(T), np.asarray(E)))}

    @property
    def feature_importances_(self):
        self._check_fitted()
        return self.model.feature_importances_
