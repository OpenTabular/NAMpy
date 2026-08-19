# lss.py
import warnings

import numpy as np
import pandas as pd
import torch

from ..neural.distributions.registry import (
    FAMILY_REGISTRY,
    family_name_for_instance,
    resolve_family,
)
from ._base import NeuralEstimatorBase, TrainingPlan


class NeuralLSS(NeuralEstimatorBase):
    def __init__(
        self,
        model,
        config,
        family="normal",
        distributional_kwargs=None,
        **kwargs,
    ):
        self.family = family
        self.distributional_kwargs = distributional_kwargs
        self.family_ = None
        self.distributional_kwargs_ = None
        self._initialize_estimator_parameters(config, kwargs)
        self.model = None
        self.data_module = None

        # Raise a warning if task is set to 'classification'
        if self._provided_preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. Be aware of your preferred distribution, that this might lead to unsatisfactory results.",
                UserWarning,
                stacklevel=2,
            )

        self.base_model = model

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["family"] = self.family
        params["distributional_kwargs"] = self.distributional_kwargs
        return params

    def set_params(self, **parameters):
        if "family" in parameters:
            self.family = parameters.pop("family")
            self.family_ = None
        if "distributional_kwargs" in parameters:
            self.distributional_kwargs = parameters.pop("distributional_kwargs")
            self.distributional_kwargs_ = None
        return super().set_params(**parameters)

    def __sklearn_tags__(self):
        # Distribution-parameter output: neither a classifier nor a regressor.
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def fit(self, X, y, **fit_params):
        """
        Train the configured distributional regression model.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        **fit_params
            Training parameters accepted by :meth:`NeuralEstimatorBase.fit`.
            ``family`` and ``distributional_kwargs`` are estimator constructor
            parameters so sklearn can clone and tune them.

        Returns
        -------
        self : object
            The fitted model.
        """
        moved = {"family", "distributional_kwargs"} & set(fit_params)
        if moved:
            names = ", ".join(sorted(moved))
            raise TypeError(
                f"{names} must be configured on the LSS estimator constructor, "
                "not passed to fit()."
            )

        distributional_kwargs = dict(self.distributional_kwargs or {})

        # Infer distributional dimensions for families that require it, when not provided.
        fam = str(self.family).lower()

        if fam == "dirichlet":
            y_arr = np.asarray(y)
            if "n_dim" not in distributional_kwargs:
                if y_arr.ndim != 2 or y_arr.shape[1] < 2:
                    raise ValueError(
                        "Dirichlet family requires y with shape (n_samples, K), K>=2."
                    )
                distributional_kwargs["n_dim"] = int(y_arr.shape[1])

        if fam == "categorical":
            y_arr = np.asarray(y)
            if "num_classes" not in distributional_kwargs:
                if y_arr.ndim == 2 and y_arr.shape[1] > 1:
                    distributional_kwargs["num_classes"] = int(y_arr.shape[1])
                else:
                    distributional_kwargs["num_classes"] = int(
                        len(np.unique(y_arr.reshape(-1)))
                    )

        if fam in FAMILY_REGISTRY:
            self.family_ = resolve_family(fam).distribution(**distributional_kwargs)
            self.distributional_kwargs_ = distributional_kwargs
        else:
            raise ValueError(f"Unsupported family: {self.family}")

        return super().fit(X, y, **fit_params)

    def _build_training_plan(self, y, y_val):
        if getattr(self, "_fit_loss_fct", None) is not None:
            raise ValueError(
                "Custom loss_fct is only supported for regression tasks; "
                "LSS losses come from the distribution family."
            )
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        plan = TrainingPlan(
            datamodule_regression=True,
            taskmodel_kwargs={
                "num_classes": self.family_.param_count,
                "family": self.family_,
                "lss": True,
            },
        )
        return y, y_val, plan

    def predict(self, X, raw=False):
        predictions = self._predict(X)["output"]

        if not raw:
            return self.model.family(predictions).cpu().numpy()

        # Convert predictions to NumPy array and return
        else:
            return predictions.cpu().numpy()

    def score(self, X, y):
        """Return the negative mean NLL of ``y`` under the predicted parameters.

        Higher is better, so sklearn model-selection utilities can rank fits.
        """
        raw_pred = self._predict(X)["output"]
        family = (
            self.model.family
            if getattr(self.model, "family", None) is not None
            else self.family_
        )
        with torch.no_grad():
            target_dtype = getattr(family, "target_dtype", torch.float32)
            y_tensor = torch.as_tensor(y, dtype=target_dtype, device=raw_pred.device)
            if y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor[:, 0]
            nll = family.compute_loss(raw_pred, y_tensor)
        return -float(nll.detach().cpu().item())

    def _plot_series_labels(self, n_series: int):
        param_names = getattr(getattr(self, "family_", None), "param_names", None)
        if param_names is not None:
            return list(param_names)[:n_series]
        return [f"Param {i + 1}" for i in range(n_series)]

    def predict_components(self, X):
        """Per-term contributions on the raw parameter scale.

        ``link`` holds the raw multi-column network output (one column per
        distribution parameter), ``response`` the family-transformed
        parameters, and each term a matching multi-column contribution;
        additivity holds on the raw (link) scale.
        """
        from ..contracts import AdditivePrediction

        pred_dict = self._predict(X)
        raw = pred_dict["output"]
        family = (
            self.model.family
            if getattr(self.model, "family", None) is not None
            else self.family_
        )
        with torch.no_grad():
            response = family(raw).cpu().numpy()
        terms, intercept = self._split_output_components(pred_dict)
        return AdditivePrediction(
            response=response,
            link=raw.cpu().numpy(),
            terms=terms,
            intercept=intercept,
            backend="neural",
        )

    def evaluate(self, X, y_true, metrics=None, distribution_family=None):
        """
        Evaluate the fitted distributional model.

        NLL is computed from raw network outputs with ``family.compute_loss``.
        Other metrics use transformed parameters from ``family(raw_predictions)``.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Input samples.
        y_true : array-like
            True targets. Can be shape (n,), (n,1), or (n,K) for multivariate families
            (e.g. Dirichlet).
        metrics : dict, optional
            Mapping metric_name -> callable(y_true, transformed_predictions).
            If None, uses `get_default_metrics(...)`.
        distribution_family : str, optional
            Family name override. If None, inferred from ``self.family_``.

        Returns
        -------
        scores : dict
            Metric values, including "NLL" when a family is available.
        """
        # Basic fitted check
        if self.model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")
        if getattr(self, "family_", None) is None:
            raise ValueError(
                "No distribution family found. Fit the model with a valid family first."
            )

        # Infer family name if not provided
        if distribution_family is None:
            distribution_family = family_name_for_instance(self.family_)

        distribution_family = str(distribution_family).lower()

        # Default metrics if none provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # ------------------------------------------------------------------
        # Single forward pass for raw predictions, then transform once
        # ------------------------------------------------------------------
        pred_dict = self._predict(X)  # returns dict of torch tensors
        raw_pred = pred_dict["output"]  # torch.Tensor on model device

        # Compute NLL from raw outputs
        scores = {}
        fam_for_loss = (
            self.model.family
            if hasattr(self.model, "family") and self.model.family is not None
            else self.family_
        )

        with torch.no_grad():
            target_dtype = getattr(fam_for_loss, "target_dtype", torch.float32)
            y_tensor = torch.as_tensor(
                y_true, dtype=target_dtype, device=raw_pred.device
            )

            # Keep multi-output LSS targets intact; only squeeze [N,1] -> [N]
            if y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
                y_for_loss = y_tensor[:, 0]
            else:
                y_for_loss = y_tensor

            nll = fam_for_loss.compute_loss(raw_pred, y_for_loss)
            scores["NLL"] = float(nll.detach().cpu().item())

            # Transformed predictions for all other metrics
            transformed_pred = fam_for_loss(raw_pred)

        predictions_transformed = transformed_pred.detach().cpu().numpy()

        # ------------------------------------------------------------------
        # Compute user/default metrics on transformed predictions
        # ------------------------------------------------------------------
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = float(metric_func(y_true, predictions_transformed))

        return scores

    def get_default_metrics(self, distribution_family):
        """Return metrics declared by the resolved distribution family."""
        from ..neural.distributions.default_metrics import default_metrics_for

        metadata = resolve_family(distribution_family)
        return default_metrics_for(metadata.metric_profile, self.family_)
