# classifier.py
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags

from .._contracts import AdditivePrediction
from ._base import NeuralEstimatorBase, TrainingPlan


class NeuralClassifier(NeuralEstimatorBase):
    _estimator_type = "classifier"

    def __init__(self, model, config, **kwargs):
        requested_task = kwargs.get("task")
        kwargs.setdefault("task", "classification")
        self._initialize_estimator_parameters(config, kwargs)
        self.model = None
        self.data_module = None

        if requested_task == "regression":
            warnings.warn(
                "The task is set to 'regression'. The Classifier is designed for classification tasks.",
                UserWarning,
                stacklevel=2,
            )

        self.base_model = model

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def _build_training_plan(self, y, y_val):
        if getattr(self, "_fit_loss_fct", None) is not None:
            raise ValueError(
                "Custom loss_fct is only supported for regression tasks."
            )
        y = np.asarray(y).ravel()
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) < 2:
            raise ValueError(
                "Classification targets must contain at least 2 classes; "
                f"got {len(self.classes_)}."
            )
        y_encoded = self._label_encoder.transform(y)

        y_val_encoded = None
        if y_val is not None:
            # transform() raises on labels unseen during fit.
            y_val_encoded = self._label_encoder.transform(np.asarray(y_val).ravel())

        plan = TrainingPlan(
            datamodule_regression=False,
            taskmodel_kwargs={
                "num_classes": len(self.classes_),
                "task": "classification",
            },
            stratify=y_encoded,
        )
        return y_encoded, y_val_encoded, plan

    def predict(self, X):
        output = self._predict(X)["output"]
        if output.shape[1] == 1:
            probabilities = torch.sigmoid(output)
            indices = (probabilities > 0.5).long().squeeze(-1).cpu().numpy()
        else:
            probabilities = torch.softmax(output, dim=1)
            indices = torch.argmax(probabilities, dim=1).cpu().numpy()
        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities; columns follow ``self.classes_``.
        """
        output = self._predict(X)["output"]
        if output.shape[1] == 1:
            p1 = torch.sigmoid(output)
            probabilities = torch.cat([1.0 - p1, p1], dim=1)
        else:
            probabilities = torch.softmax(output, dim=1)
        return probabilities.cpu().numpy()

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        return float(
            accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        )

    def predict_components(self, X) -> AdditivePrediction:
        """Return per-term logit-scale contributions (binary tasks only)."""
        pred_dict = self._predict(X)
        output = pred_dict["output"]
        if output.shape[1] != 1:
            raise NotImplementedError(
                "predict_components supports binary classifiers only; "
                "multiclass outputs have one logit column per class."
            )
        link = output.squeeze(-1).cpu().numpy()
        response = torch.sigmoid(output).squeeze(-1).cpu().numpy()
        terms, intercept = self._split_output_components(pred_dict)
        return AdditivePrediction(
            response=response,
            link=link,
            terms=terms,
            intercept=intercept,
            backend="neural",
        )

    def _plot_series_labels(self, n_series: int):
        if n_series <= 1:
            return None
        return [f"Class {i + 1}" for i in range(n_series)]

    def evaluate(self, X, y_true, metrics=None):
        """
        Evaluate the model on the given data using specified metrics.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples to predict.
        y_true : array-like of shape (n_samples,)
            The true class labels against which to evaluate the predictions.
        metrics : dict
            A dictionary where keys are metric names and values are tuples containing the metric function
            and a boolean indicating whether the metric requires probability scores (True) or class labels (False).

        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.

        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        if metrics is None:
            metrics = {"Accuracy": (accuracy_score, False)}

        scores = {}

        if any(use_proba for _, use_proba in metrics.values()):
            probabilities = self.predict_proba(X)

        if any(not use_proba for _, use_proba in metrics.values()):
            predictions = self.predict(X)

        for metric_name, (metric_func, use_proba) in metrics.items():
            if use_proba:
                try:
                    scores[metric_name] = metric_func(y_true, probabilities)
                except ValueError:
                    # Some binary metrics (e.g. roc_auc_score) expect p(class=1)
                    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                        scores[metric_name] = metric_func(y_true, probabilities[:, 1])
                    else:
                        raise
            else:
                scores[metric_name] = metric_func(y_true, predictions)

        return scores
