# regressor.py
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import RegressorTags

from ..contracts import AdditivePrediction
from ..neural.objectives import RegressionObjective
from ._base import NeuralEstimatorBase, TrainingPlan


class NeuralRegressor(NeuralEstimatorBase):
    _estimator_type = "regressor"

    def __init__(self, model, config, **kwargs):
        self._initialize_estimator_parameters(config, kwargs)
        self.model = None
        self.data_module = None

        # Raise a warning if task is set to 'classification'
        if self._provided_preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. The Regressor is designed for regression tasks.",
                UserWarning,
                stacklevel=2,
            )

        self.base_model = model

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def _build_training_plan(self, y, y_val):
        y = np.asarray(y)
        if y.ndim == 1:
            n_outputs = 1
        elif y.ndim == 2 and y.shape[1] >= 1:
            n_outputs = int(y.shape[1])
        else:
            raise ValueError(
                "Regression targets must have shape (n_samples,) or "
                f"(n_samples, n_outputs); received shape {y.shape}."
            )
        self.n_outputs_ = n_outputs

        if y_val is not None:
            y_val = np.asarray(y_val)
            if y_val.ndim == 1:
                validation_outputs = 1
            elif y_val.ndim == 2 and y_val.shape[1] >= 1:
                validation_outputs = int(y_val.shape[1])
            else:
                raise ValueError(
                    "Validation targets must have shape (n_samples,) or "
                    "(n_samples, n_outputs)."
                )
            if validation_outputs != n_outputs:
                raise ValueError(
                    "Training and validation targets must have the same output width."
                )

        objective = RegressionObjective(
            n_outputs, loss_fct=getattr(self, "_fit_loss_fct", None)
        )

        plan = TrainingPlan(objective=objective)
        return y, y_val, plan

    def predict(self, X, *, batch_size=None):
        output = self._predict(X, batch_size=batch_size)["output"]
        return self.model.objective.transform(output).squeeze(-1).cpu().numpy()

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    def predict_components(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
        batch_size=None,
    ) -> AdditivePrediction:
        """Return per-term contributions as a backend-neutral result.

        The intercept reported as 0.0 means the constant is absorbed into
        the feature networks unless the architecture emits an explicit
        ``"intercept"`` entry.
        """
        pred_dict = self._predict(X, batch_size=batch_size)
        link = pred_dict["output"].squeeze(-1).cpu().numpy()
        terms, intercept = self._split_output_components(pred_dict)
        prediction = AdditivePrediction(
            response=link,
            link=link,
            terms=terms,
            intercept=intercept,
            backend="neural",
        )
        return self._maybe_center_components(
            prediction,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
        )

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are the metric functions.

        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        if metrics is None:
            metrics = {"Mean Squared Error": mean_squared_error}

        predictions = self.predict(X)

        return {
            metric_name: metric_func(y_true, predictions)
            for metric_name, metric_func in metrics.items()
        }
