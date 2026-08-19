"""Joint Torch training of compiled GAM terms plus a neural additive model.

EXPERIMENTAL, and NOT an mgcv fit: the compiled-term stage reuses the exact
mgcv-parity basis construction, constraints, and penalty matrices, but the
smoothing parameters are FIXED and the joint coefficients are optimized with
Torch. Results will not and should not match ``GAM``/mgcv; never compare
them in parity suites.

v1 scope: single-output regression; fixed ``lam`` (supplied or lifted from a
fitted GAM via ``gam_source``); no explicit validation set.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.utils import RegressorTags

from ..models._base import NeuralEstimatorBase
from ..models._data import prepare_predict_features
from ..neural.training.engine import TrainingPlan
from .compiled_terms import CompiledGAMTerms
from .net import GAM_DESIGN_KEY, HybridAdditiveNet


class HybridJointRegressor(NeuralEstimatorBase):
    """Jointly train compiled mgcv-parity GAM terms with a neural net.

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
        fitted smoothing parameters from (read-only).
    """

    _estimator_type = "regressor"

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
        self.base_model = HybridAdditiveNet
        self.gam_terms_ = None
        self.neural_features_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags

    def fit(self, data, y=None, *, neural_features, **fit_kwargs):
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
        **fit_kwargs
            Standard :meth:`NeuralEstimatorBase.fit` arguments (epochs,
            batch size, trainer flags, ...). ``X_val`` is not supported.
        """
        if fit_kwargs.get("X_val") is not None or fit_kwargs.get("y_val") is not None:
            raise NotImplementedError(
                "HybridJointRegressor does not support an explicit "
                "validation set yet; use the automatic split."
            )

        neural_features = list(neural_features)
        if not neural_features:
            raise ValueError("neural_features must name at least one column.")
        missing = [name for name in neural_features if name not in data.columns]
        if missing:
            raise ValueError(f"neural_features not found in data: {missing}")

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

        if y is None:
            if gam_terms.response is None:
                raise ValueError(
                    "y is required when the formula has no response."
                )
            y = gam_terms.response
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(
                "HybridJointRegressor supports 1-d regression targets only."
            )

        self.gam_terms_ = gam_terms
        self.neural_features_ = neural_features
        self._gam_payload_for_fit = gam_terms
        self._design_for_fit = design_train.astype(np.float32)
        try:
            return super().fit(data[neural_features], y, **fit_kwargs)
        finally:
            del self._gam_payload_for_fit
            del self._design_for_fit

    def _build_training_plan(self, y, y_val):
        y = np.asarray(y)
        self.n_outputs_ = 1
        plan = TrainingPlan(
            datamodule_regression=True,
            taskmodel_kwargs={
                "num_classes": 1,
                "task": "regression",
                "base_model_class": self.neural_model_class,
                "gam_payload": self._gam_payload_for_fit,
            },
            passthrough={GAM_DESIGN_KEY: self._design_for_fit},
        )
        return y, y_val, plan

    def _predict(self, X):
        """Inference on a full DataFrame (formula columns + neural columns)."""
        from sklearn.exceptions import NotFittedError

        if self.model is None or self.data_module is None:
            raise NotFittedError(
                "This HybridJointRegressor instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        X_neural = prepare_predict_features(self, X[self.neural_features_])
        cat_tensors, num_tensors = self.data_module.preprocess_test_data(X_neural)

        design = torch.tensor(
            self.gam_terms_.design(X), dtype=torch.float32
        )
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

    def predict(self, X):
        return self._predict(X)["output"].squeeze(-1).cpu().numpy()

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))
