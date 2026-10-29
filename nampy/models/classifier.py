# classifier.py

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags

from ..contracts import AdditivePrediction
from ..neural.objectives import classification_objective
from ..neural.prediction import split_output_components
from ..neural.training import TrainingPlan
from ._neural import NeuralEstimatorBase


class NeuralClassifier(NeuralEstimatorBase):
    """Shared estimator interface for neural classification models."""

    _estimator_type = "classifier"

    def __init__(
        self,
        model,
        config_class,
        *,
        architecture_config=None,
        objective=None,
        training=None,
        preprocessing=None,
    ):
        self._initialize_estimator_parameters(
            config_class,
            architecture_config=architecture_config,
            objective=objective,
            training=training,
            preprocessing=preprocessing,
        )
        self.base_model = model
        self.model = None
        self.data_module = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def _build_training_plan(self, y, y_val):
        if self.objective.loss_fct is not None:
            raise ValueError("Custom loss_fct is only supported for regression tasks.")
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
            objective=classification_objective(len(self.classes_)),
            stratify=y_encoded,
        )
        return y_encoded, y_val_encoded, plan

    def predict(self, X, *, batch_size=None, offset=None):
        probabilities = self.predict_proba(X, batch_size=batch_size, offset=offset)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X, *, batch_size=None, offset=None):
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
        output = self._predict(X, batch_size=batch_size, offset=offset)["output"]
        return self.model.objective.transform(output).cpu().numpy()

    def score(self, X, y, sample_weight=None, *, offset=None):
        """Return the mean accuracy on the given test data and labels."""
        return float(
            accuracy_score(
                y, self.predict(X, offset=offset), sample_weight=sample_weight
            )
        )

    def predict_components(
        self,
        X,
        *,
        center: bool = False,
        reference_X=None,
        reference_weight=None,
        reference_offset=None,
        batch_size=None,
        offset=None,
    ) -> AdditivePrediction:
        """Return per-term logit-scale contributions (binary tasks only)."""
        pred_dict = self._predict(X, batch_size=batch_size, offset=offset)
        output = pred_dict["output"]
        link = output.cpu().numpy()
        terms, intercept = split_output_components(pred_dict)
        prediction = AdditivePrediction(
            link=link,
            terms=terms,
            intercept=intercept,
            offset=0.0 if offset is None else offset,
            backend="neural",
        )
        return self._maybe_center_components(
            prediction,
            center=center,
            reference_X=reference_X,
            reference_weight=reference_weight,
            reference_offset=reference_offset,
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
