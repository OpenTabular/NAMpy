"""Joint Torch training of compiled GAM terms plus a neural additive model.

NOT an mgcv fit: the compiled-term stage reuses the exact mgcv-parity basis
construction, constraints, and penalty matrices, but the smoothing
parameters are FIXED and the joint coefficients are optimized with Torch.
Results will not and should not match ``GAM``/mgcv; the joint estimators
never enter the parity suites. The recommended way to choose the smoothing
parameters is ``gam_source=``: fit a plain ``GAM`` first (REML-selected
lambdas) and lift its compiled terms read-only.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import ClassifierTags, RegressorTags

from ..models._base import NeuralEstimatorBase
from ..models._data import prepare_predict_features
from ..neural.training.engine import TrainingPlan
from .compiled_terms import CompiledGAMTerms
from .net import GAM_DESIGN_KEY, GAMNet


class _GAMNetBase(NeuralEstimatorBase):
    """Shared machinery for the joint GAM-plus-net estimators.

    Parameters
    ----------
    gam_formula : str
        mgcv-style formula (with response) for the compiled-term stage,
        e.g. ``"y ~ s(x0, k=8)"``. Ignored when ``gam_source`` is given.
    neural_model_class : type
        A NAMpy architecture class (e.g. ``nampy.neural.modules.LinReg``)
        for the neural additive part.
    neural_config_class : type
        The architecture's config dataclass.
    lam : array-like
        Fixed smoothing parameters, one per penalty group. Required unless
        ``gam_source`` supplies fitted values.
    gam_source : GAM, optional
        A fitted :class:`nampy.gam.GAM` to lift the compiled model and
        REML-selected smoothing parameters from (read-only; recommended).
    """

    def __init__(
        self,
        gam_formula=None,
        neural_model_class=None,
        neural_config_class=None,
        *,
        lam=None,
        gam_source=None,
        default_k=10,
        default_basis="tp",
        **kwargs,
    ):
        self.gam_formula = gam_formula
        self.neural_model_class = neural_model_class
        self.lam = lam
        self.gam_source = gam_source
        self.default_k = default_k
        self.default_basis = default_basis
        self._initialize_estimator_parameters(neural_config_class, kwargs)
        self.model = None
        self.data_module = None
        self.base_model = GAMNet
        self.gam_terms_ = None
        self.neural_features_ = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _compile_terms(self, data):
        if self.gam_source is not None:
            gam_terms = CompiledGAMTerms.from_fitted_gam(self.gam_source)
            design_train = gam_terms.design(data)
        else:
            if self.gam_formula is None or self.lam is None:
                raise ValueError(
                    "Either gam_source or both gam_formula and lam are "
                    "required."
                )
            gam_terms = CompiledGAMTerms.from_formula(
                self.gam_formula,
                data,
                lam=self.lam,
                default_k=self.default_k,
                default_basis=self.default_basis,
            )
            design_train = gam_terms.design(None)
        return gam_terms, design_train

    def _resolve_targets(self, gam_terms, data, y, val_data, y_val):
        if y is None:
            if gam_terms.response is None:
                raise ValueError("y is required when the formula has no response.")
            y = gam_terms.response
        if val_data is not None and y_val is None:
            response_name = gam_terms.response_name
            if response_name is None or response_name not in val_data.columns:
                raise ValueError(
                    "y_val is required when val_data does not contain the "
                    f"formula response column {response_name!r}."
                )
            y_val = np.asarray(val_data[response_name])
        return np.asarray(y), y_val

    def fit(
        self,
        data,
        y=None,
        *,
        neural_features,
        val_data=None,
        y_val=None,
        **fit_kwargs,
    ):
        """Compile the GAM terms on ``data`` and train jointly.

        Parameters
        ----------
        data : pandas.DataFrame
            Table holding the formula columns (and response, when the
            formula names one) plus the neural feature columns.
        y : array-like, optional
            Targets; may be omitted when the formula carries the response.
        neural_features : sequence of str
            Columns of ``data`` fed to the neural part.
        val_data : pandas.DataFrame, optional
            Explicit validation rows (same columns as ``data``); the
            compiled design for these rows rides the validation
            passthrough channel.
        y_val : array-like, optional
            Validation targets; defaults to the formula response column of
            ``val_data``.
        **fit_kwargs
            Standard :meth:`NeuralEstimatorBase.fit` arguments (epochs,
            batch size, trainer flags, ...). Use ``val_data``, not
            ``X_val``.
        """
        if fit_kwargs.get("X_val") is not None:
            raise ValueError(
                "Use val_data= (a DataFrame with the formula columns) "
                "instead of X_val/y_val for the joint estimators."
            )

        neural_features = list(neural_features)
        if not neural_features:
            raise ValueError("neural_features must name at least one column.")
        missing = [name for name in neural_features if name not in data.columns]
        if missing:
            raise ValueError(f"neural_features not found in data: {missing}")

        gam_terms, design_train = self._compile_terms(data)
        y, y_val = self._resolve_targets(gam_terms, data, y, val_data, y_val)

        self.gam_terms_ = gam_terms
        self.neural_features_ = neural_features
        self._gam_payload_for_fit = gam_terms
        self._design_for_fit = design_train.astype(np.float32)
        self._design_val_for_fit = None
        if val_data is not None:
            self._design_val_for_fit = gam_terms.design(val_data).astype(
                np.float32
            )
            fit_kwargs["X_val"] = val_data[neural_features]
            fit_kwargs["y_val"] = y_val
        try:
            return super().fit(data[neural_features], y, **fit_kwargs)
        finally:
            del self._gam_payload_for_fit
            del self._design_for_fit
            del self._design_val_for_fit

    def _joint_plan_parts(self):
        taskmodel_kwargs = {
            "base_model_class": self.neural_model_class,
            "gam_payload": self._gam_payload_for_fit,
        }
        passthrough = {GAM_DESIGN_KEY: self._design_for_fit}
        passthrough_val = None
        if self._design_val_for_fit is not None:
            passthrough_val = {GAM_DESIGN_KEY: self._design_val_for_fit}
        return taskmodel_kwargs, passthrough, passthrough_val

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, X):
        """Inference on a full DataFrame (formula columns + neural columns)."""
        from sklearn.exceptions import NotFittedError

        if self.model is None or self.data_module is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        X_neural = prepare_predict_features(self, X[self.neural_features_])
        cat_tensors, num_tensors = self.data_module.preprocess_test_data(X_neural)

        design = torch.tensor(self.gam_terms_.design(X), dtype=torch.float32)
        num_tensors = dict(num_tensors)
        num_tensors[GAM_DESIGN_KEY] = design

        device = next(self.model.parameters()).device
        cat_tensors = {k: v.to(device) for k, v in cat_tensors.items()}
        num_tensors = {k: v.to(device) for k, v in num_tensors.items()}

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model(
                    num_features=num_tensors, cat_features=cat_tensors
                )
        finally:
            self.model.train(was_training)


class GAMNetRegressor(_GAMNetBase):
    """Jointly train compiled mgcv-parity GAM terms with a neural net.

    Single-output regression under a fixed-lambda quadratic smoothness
    penalty. NOT an mgcv fit; see the module docstring.
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def _build_training_plan(self, y, y_val):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(
                "GAMNetRegressor supports 1-d regression targets only."
            )
        self.n_outputs_ = 1

        taskmodel_kwargs, passthrough, passthrough_val = self._joint_plan_parts()
        taskmodel_kwargs.update({"num_classes": 1, "task": "regression"})
        plan = TrainingPlan(
            datamodule_regression=True,
            taskmodel_kwargs=taskmodel_kwargs,
            passthrough=passthrough,
            passthrough_val=passthrough_val,
        )
        return y, y_val, plan

    def predict(self, X):
        return self._predict(X)["output"].squeeze(-1).cpu().numpy()

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))


class GAMNetClassifier(_GAMNetBase):
    """Binary classification with compiled GAM terms inside the net.

    The combined logits (compiled terms + neural terms) train through
    BCE-with-logits; ``classes_`` records the original labels. NOT an mgcv
    fit; see the module docstring.
    """

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    def _build_training_plan(self, y, y_val):
        y = np.asarray(y).ravel()
        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "GAMNetClassifier supports binary targets only; "
                f"got {len(self.classes_)} classes."
            )
        y_encoded = self._label_encoder.transform(y)
        y_val_encoded = None
        if y_val is not None:
            y_val_encoded = self._label_encoder.transform(
                np.asarray(y_val).ravel()
            )

        taskmodel_kwargs, passthrough, passthrough_val = self._joint_plan_parts()
        taskmodel_kwargs.update({"num_classes": 2, "task": "classification"})
        plan = TrainingPlan(
            datamodule_regression=False,
            taskmodel_kwargs=taskmodel_kwargs,
            stratify=y_encoded,
            passthrough=passthrough,
            passthrough_val=passthrough_val,
        )
        return y_encoded, y_val_encoded, plan

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities; columns follow ``self.classes_``."""
        output = self._predict(X)["output"]
        p1 = torch.sigmoid(output).squeeze(-1).cpu().numpy()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Predicted class labels (original label dtype)."""
        p1 = self.predict_proba(X)[:, 1]
        return self.classes_[(p1 >= 0.5).astype(int)]

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels."""
        return float(
            accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        )
