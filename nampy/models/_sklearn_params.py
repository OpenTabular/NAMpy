from __future__ import annotations

import inspect
from typing import Any

from pretab.preprocessor import Preprocessor


def _preprocessor_defaults() -> dict[str, Any]:
    parameters = inspect.signature(Preprocessor.__init__).parameters
    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _normalize_preprocessor_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    if normalized.get("categorical_preprocessing") in ("one_hot", "one-hot"):
        normalized["categorical_preprocessing"] = "one-hot"
    if normalized.get("numerical_preprocessing") == "normalization":
        normalized["numerical_preprocessing"] = "minmax"
    return normalized


class NeuralEstimatorParameterMixin:
    """Own config and preprocessing parameters for NAMpy sklearn wrappers."""

    def _initialize_estimator_parameters(self, config_class, kwargs):
        config_names = set(getattr(config_class, "__dataclass_fields__", {}))
        preprocessor_defaults = _preprocessor_defaults()
        preprocessor_names = set(preprocessor_defaults)

        flat_kwargs = {}
        nested_preprocessor_kwargs = {}
        for name, value in kwargs.items():
            if name.startswith("preprocessor__"):
                nested_preprocessor_kwargs[name.split("__", 1)[1]] = value
            else:
                flat_kwargs[name] = value

        unknown_flat = set(flat_kwargs) - config_names - preprocessor_names
        unknown_nested = set(nested_preprocessor_kwargs) - preprocessor_names
        if unknown_flat or unknown_nested:
            unknown = sorted(unknown_flat | unknown_nested)
            valid = sorted(config_names | preprocessor_names)
            raise TypeError(
                f"Unexpected parameter(s) {unknown} for {self.__class__.__name__}. "
                f"Valid parameters are {valid}."
            )

        config_inputs = {
            name: value for name, value in flat_kwargs.items() if name in config_names
        }
        explicit_preprocessor = {
            name: value
            for name, value in flat_kwargs.items()
            if name in preprocessor_names
        }
        explicit_preprocessor.update(nested_preprocessor_kwargs)
        explicit_preprocessor = _normalize_preprocessor_params(explicit_preprocessor)

        self._config_param_names = tuple(sorted(config_names))
        self._preprocessor_param_names = tuple(sorted(preprocessor_names))
        self.config = config_class(**config_inputs)
        self.config_kwargs = {
            name: getattr(self.config, name) for name in self._config_param_names
        }
        self._preprocessor_kwargs = dict(preprocessor_defaults)
        self._preprocessor_kwargs.update(explicit_preprocessor)
        self._provided_preprocessor_kwargs = explicit_preprocessor
        self._rebuild_preprocessor()

    def _rebuild_preprocessor(self):
        self.preprocessor = Preprocessor(**self._preprocessor_kwargs)

    def get_params(self, deep=True):
        params = dict(self.config_kwargs)
        for name, value in self._preprocessor_kwargs.items():
            if name not in params:
                params[name] = value
            elif params[name] != value:
                params[f"preprocessor__{name}"] = value

        if deep:
            params.update(
                {
                    f"preprocessor__{name}": value
                    for name, value in self._preprocessor_kwargs.items()
                }
            )
        return params

    def set_params(self, **parameters):
        config_updates = {}
        preprocessor_updates = {}

        for name, value in parameters.items():
            if name.startswith("preprocessor__"):
                preprocessor_name = name.split("__", 1)[1]
                if preprocessor_name not in self._preprocessor_param_names:
                    raise ValueError(
                        f"Invalid parameter {name!r} for {self.__class__.__name__}."
                    )
                preprocessor_updates[preprocessor_name] = value
                continue

            owns_config = name in self._config_param_names
            owns_preprocessor = name in self._preprocessor_param_names
            if not owns_config and not owns_preprocessor:
                valid = sorted(
                    set(self._config_param_names) | set(self._preprocessor_param_names)
                )
                raise ValueError(
                    f"Invalid parameter {name!r} for {self.__class__.__name__}. "
                    f"Valid parameters are {valid}."
                )
            if owns_config:
                config_updates[name] = value
            if owns_preprocessor:
                preprocessor_updates[name] = value

        for name, value in config_updates.items():
            setattr(self.config, name, value)
            self.config_kwargs[name] = value

        if preprocessor_updates:
            self._preprocessor_kwargs.update(
                _normalize_preprocessor_params(preprocessor_updates)
            )
            self._rebuild_preprocessor()

        return self
