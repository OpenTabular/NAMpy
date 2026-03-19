import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from pretab.preprocessor import Preprocessor


PREPROCESSOR_ARG_NAMES = [
    "n_bins",
    "numerical_preprocessing",
    "categorical_preprocessing",
    "use_decision_tree_bins",
    "binning_strategy",
    "task",
    "cat_cutoff",
    "treat_all_integers_as_numerical",
    "degree",
    "n_knots",
    "scaling_strategy",
    "feature_preprocessing",
]


class BaseBoostingEstimator(BaseEstimator):
    """Shared sklearn-style preprocessing/params utilities for boosting models."""

    def __init__(self, config_cls, **kwargs):
        self.config_kwargs = {
            key: value for key, value in kwargs.items() if key not in PREPROCESSOR_ARG_NAMES
        }
        self.config = config_cls(**self.config_kwargs)
        preprocessor_kwargs = {
            key: value for key, value in kwargs.items() if key in PREPROCESSOR_ARG_NAMES
        }
        if preprocessor_kwargs.get("categorical_preprocessing") in ("one_hot", "one-hot"):
            preprocessor_kwargs["categorical_preprocessing"] = "one-hot"
        if preprocessor_kwargs.get("numerical_preprocessing") == "normalization":
            preprocessor_kwargs["numerical_preprocessing"] = "minmax"
        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.num_feature_info = None
        self.cat_feature_info = None

    def get_params(self, deep=True):
        params = dict(self.config_kwargs)
        if deep:
            params.update(
                {
                    "preprocessor__" + key: value
                    for key, value in self.preprocessor.get_params().items()
                }
            )
        return params

    def set_params(self, **parameters):
        config_updates = {}
        preprocessor_params = {}
        for key, value in parameters.items():
            if key.startswith("preprocessor__"):
                preprocessor_params[key.split("__", 1)[1]] = value
            elif key in self.config_kwargs:
                config_updates[key] = value
            else:
                raise ValueError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}. "
                    f"Valid parameters: {sorted(self.config_kwargs.keys())}."
                )
        self.config_kwargs.update(config_updates)
        for key, value in config_updates.items():
            setattr(self.config, key, value)
        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)
        return self

    def _ensure_dataframe(self, x):
        if isinstance(x, pd.DataFrame):
            return x.copy()
        return pd.DataFrame(x)

    def _fit_preprocessor(self, X_train, y_train, X_val, y_val):
        combined_x = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        combined_y = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)
        self.preprocessor.fit(combined_x, combined_y)
        num_info, cat_info, _ = self.preprocessor.get_feature_info(verbose=False)
        self.num_feature_info = num_info
        self.cat_feature_info = cat_info
        self.feature_names_in_ = list(X_train.columns)

    def _transform_matrix(self, x):
        processed = self.preprocessor.transform(self._ensure_dataframe(x))
        arrays = []
        feature_names = []
        n_rows = None
        for info_dict, prefix in (
            (self.num_feature_info or {}, "num_"),
            (self.cat_feature_info or {}, "cat_"),
        ):
            for feature_name in info_dict:
                key = prefix + feature_name
                if key not in processed:
                    continue
                values = np.asarray(processed[key])
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                values = values.astype(np.float64, copy=False)
                arrays.append(values)
                n_rows = values.shape[0]
                if values.shape[1] == 1:
                    feature_names.append(feature_name)
                else:
                    feature_names.extend(
                        [f"{feature_name}[{idx}]" for idx in range(values.shape[1])]
                    )
        self.feature_names_out_ = feature_names
        if not arrays:
            if n_rows is None:
                n_rows = len(x)
            return np.empty((n_rows, 0), dtype=np.float64)
        return np.concatenate(arrays, axis=1)

    def _check_fitted(self):
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")


class BaseBoostingRegressorClassifier(BaseBoostingEstimator):
    def __init__(self, task, config_cls, **kwargs):
        super().__init__(config_cls=config_cls, **kwargs)
        self.task = task

    def _split_validation(self, x, y, X_val=None, y_val=None, val_size=0.2, random_state=101):
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together; got only one.")
        x = self._ensure_dataframe(x)
        y = np.asarray(y)
        if X_val is not None:
            return x, y, self._ensure_dataframe(X_val), np.asarray(y_val)
        stratify = y if self.task == "classification" and len(np.unique(y)) > 1 else None
        return train_test_split(
            x,
            y,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify,
        )
