# sklearn_lss.py
import warnings

import numpy as np
import pandas as pd
import properscoring as ps
import torch
from sklearn.metrics import accuracy_score, mean_squared_error

from ..neural.distributions.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    DirichletDistribution,
    GammaDistribution,
    HurdleNegativeBinomialDistribution,
    HurdlePoissonDistribution,
    InverseGammaDistribution,
    LogLogisticDistribution,
    LogNormalDistribution,
    MultivariateNormalDiagDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    OrdinalCumulativeLogitDistribution,
    PoissonDistribution,
    Quantile,
    RobustNormalDistribution,
    StudentTDistribution,
    TweedieDistribution,
    WeibullDistribution,
    ZeroInflatedNegativeBinomialDistribution,
    ZeroInflatedPoissonDistribution,
)
from ..neural.distributions.metrics import (
    beta_mean_mse,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from ..neural.training.engine import TrainingPlan
from ._sklearn_base import NeuralEstimatorBase

_DISTRIBUTION_CLASSES = {
    "normal": NormalDistribution,
    "poisson": PoissonDistribution,
    "gamma": GammaDistribution,
    "beta": BetaDistribution,
    "dirichlet": DirichletDistribution,
    "studentt": StudentTDistribution,
    "negativebinom": NegativeBinomialDistribution,
    "inversegamma": InverseGammaDistribution,
    "categorical": CategoricalDistribution,
    "quantile": Quantile,
    "robustnormal": RobustNormalDistribution,
    "lognormal": LogNormalDistribution,
    "weibull": WeibullDistribution,
    "loglogistic": LogLogisticDistribution,
    "zip": ZeroInflatedPoissonDistribution,
    "zinb": ZeroInflatedNegativeBinomialDistribution,
    "hurdlepoisson": HurdlePoissonDistribution,
    "hurdlenegativebinom": HurdleNegativeBinomialDistribution,
    "tweedie": TweedieDistribution,
    "ordinal": OrdinalCumulativeLogitDistribution,
    "mvnormdiag": MultivariateNormalDiagDistribution,
}


class SklearnBaseLSS(NeuralEstimatorBase):
    def __init__(self, model, config, **kwargs):
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

    def __sklearn_tags__(self):
        # Distribution-parameter output: neither a classifier nor a regressor.
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def fit(
        self,
        X,
        y,
        family,
        val_size: float = 0.2,
        X_val=None,
        y_val=None,
        max_epochs: int = 100,
        random_state: int = 101,
        batch_size: int = 128,
        shuffle: bool = True,
        patience: int = 15,
        monitor: str = "val_loss",
        mode: str = "min",
        lr: float = 1e-4,
        lr_patience: int = 10,
        factor: float = 0.1,
        weight_decay: float = 1e-06,
        checkpoint_path="model_checkpoints",
        distributional_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        """
        Trains the distributional regression model using the provided training data.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values.
        family : str
            The name of the distribution family to use for the loss function.
        distributional_kwargs : dict, default=None
            Any arguments that are specific for a certain distribution.

        The remaining parameters match :meth:`NeuralEstimatorBase.fit`.

        Returns
        -------
        self : object
            The fitted model.
        """
        if distributional_kwargs is None:
            distributional_kwargs = {}

        # Infer distributional dimensions for families that require it, when not provided.
        fam = str(family).lower()

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

        if fam in _DISTRIBUTION_CLASSES:
            self.family = _DISTRIBUTION_CLASSES[fam](**distributional_kwargs)
        else:
            raise ValueError("Unsupported family: {}".format(family))

        return super().fit(
            X,
            y,
            val_size=val_size,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            random_state=random_state,
            batch_size=batch_size,
            shuffle=shuffle,
            patience=patience,
            monitor=monitor,
            mode=mode,
            lr=lr,
            lr_patience=lr_patience,
            factor=factor,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

    def _build_training_plan(self, y, y_val):
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        plan = TrainingPlan(
            datamodule_regression=True,
            taskmodel_kwargs={
                "num_classes": self.family.param_count,
                "family": self.family,
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
            else self.family
        )
        with torch.no_grad():
            target_dtype = getattr(family, "target_dtype", torch.float32)
            y_tensor = torch.as_tensor(y, dtype=target_dtype, device=raw_pred.device)
            if y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor[:, 0]
            nll = family.compute_loss(raw_pred, y_tensor)
        return -float(nll.detach().cpu().item())

    def _plot_series_labels(self, n_series: int):
        param_names = getattr(getattr(self, "family", None), "param_names", None)
        if param_names is not None:
            return list(param_names)[:n_series]
        return [f"Param {i + 1}" for i in range(n_series)]

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
            Family name override. If None, inferred from `self.family`.

        Returns
        -------
        scores : dict
            Metric values, including "NLL" when a family is available.
        """
        # Basic fitted check
        if self.model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")
        if getattr(self, "family", None) is None:
            raise ValueError(
                "No distribution family found. Fit the model with a valid family first."
            )

        # Infer family name if not provided
        if distribution_family is None:
            fam_obj = self.family
            cls_name = fam_obj.__class__.__name__.lower()
            # Normalize class names to your public family keys
            family_map = {
                "normaldistribution": "normal",
                "poissondistribution": "poisson",
                "gammadistribution": "gamma",
                "betadistribution": "beta",
                "dirichletdistribution": "dirichlet",
                "studenttdistribution": "studentt",
                "negativebinomialdistribution": "negativebinom",
                "inversegammadistribution": "inversegamma",
                "categoricaldistribution": "categorical",
                "quantile": "quantile",
                "robustnormaldistribution": "robustnormal",
            }
            distribution_family = family_map.get(
                cls_name, cls_name.replace("distribution", "")
            )

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
            else self.family
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
        """
        Provide sensible default metrics for each supported distribution family.

        Metrics use transformed distribution parameters returned by
        ``self.family(raw_predictions)``. For example, normal and robust-normal
        families return ``[mean, scale]``; count families return their rate or
        mean/dispersion parameters; and categorical families return class
        probabilities.

        Parameters
        ----------
        distribution_family : str
            Family identifier.

        Returns
        -------
        metrics : dict
            Mapping of metric_name -> callable(y_true, transformed_predictions)
        """
        family = str(distribution_family).lower()

        def _y_1d(y):
            y = np.asarray(y)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            return y.reshape(-1) if y.ndim == 1 else y

        def _categorical_labels(y):
            y = np.asarray(y)
            # Accept labels [N], [N,1], or one-hot/probs [N,K]
            if y.ndim == 2 and y.shape[1] == 1:
                return y[:, 0].astype(int)
            if y.ndim == 2:
                return np.argmax(y, axis=1).astype(int)
            return y.reshape(-1).astype(int)

        def _normal_crps(y, pred):
            # pred = [mean, scale]
            y = _y_1d(y).astype(float)
            pred = np.asarray(pred, dtype=float)
            mu = pred[:, 0]
            scale = np.clip(pred[:, 1], 1e-9, None)  # std, not variance
            return float(
                np.mean(
                    [
                        ps.crps_gaussian(y[i], mu=mu[i], sig=scale[i])
                        for i in range(len(y))
                    ]
                )
            )

        def _normal_mse(y, pred):
            y = _y_1d(y).astype(float)
            pred = np.asarray(pred, dtype=float)
            return float(mean_squared_error(y, pred[:, 0]))

        def _normal_mae(y, pred):
            y = _y_1d(y).astype(float)
            pred = np.asarray(pred, dtype=float)
            return float(np.mean(np.abs(y - pred[:, 0])))

        def _quantile_pinball(y, pred):
            # pred shape [N, Q], uses family.quantiles if available
            y = _y_1d(y).astype(float)
            pred = np.asarray(pred, dtype=float)
            if pred.ndim != 2:
                raise ValueError(
                    "Quantile predictions must be 2D (n_samples, n_quantiles)."
                )

            quantiles = getattr(self.family, "quantiles", None)
            if quantiles is None:
                raise ValueError(
                    "Quantile default metric requires `self.family.quantiles`."
                )
            q = np.asarray(quantiles, dtype=float)
            if pred.shape[1] != len(q):
                raise ValueError(
                    f"Predictions have {pred.shape[1]} quantiles but family.quantiles has {len(q)} entries."
                )

            y2 = y[:, None]
            e = y2 - pred
            loss = np.maximum((q[None, :] - 1.0) * e, q[None, :] * e)
            return float(np.mean(np.sum(loss, axis=1)))

        def _quantile_median_mae(y, pred):
            y = _y_1d(y).astype(float)
            pred = np.asarray(pred, dtype=float)
            quantiles = getattr(self.family, "quantiles", None)
            if quantiles is None:
                # fallback: use center column
                median_pred = pred[:, pred.shape[1] // 2]
                return float(np.mean(np.abs(y - median_pred)))

            q = list(map(float, quantiles))
            if 0.5 in q:
                idx = q.index(0.5)
            else:
                idx = int(np.argmin(np.abs(np.asarray(q) - 0.5)))
            return float(np.mean(np.abs(y - pred[:, idx])))

        default_metrics = {
            "normal": {
                "MSE": _normal_mse,
                "MAE": _normal_mae,
                "CRPS": _normal_crps,
            },
            "robustnormal": {
                "MSE": _normal_mse,
                "MAE": _normal_mae,
                "CRPS": _normal_crps,
            },
            "poisson": {
                # poisson_deviance accepts [rate] or 1D mean/rate
                "Poisson Deviance": poisson_deviance,
            },
            "gamma": {
                # gamma_deviance accepts transformed [shape, rate] directly
                "Gamma Deviance": gamma_deviance,
            },
            "beta": {
                "Beta Mean MSE": beta_mean_mse,
            },
            "dirichlet": {
                "Dirichlet Error": dirichlet_error,
            },
            "studentt": {
                # student_t_loss expects transformed [df, loc, scale]
                "Student-T NLL": student_t_loss,
            },
            "negativebinom": {
                # negative_binomial_deviance accepts transformed [mean,dispersion] directly
                "Negative Binomial Deviance": negative_binomial_deviance,
            },
            "inversegamma": {
                # inverse_gamma_loss expects transformed [shape, rate]
                "Inverse Gamma NLL": inverse_gamma_loss,
            },
            "categorical": {
                "Accuracy": lambda y, p: float(
                    accuracy_score(
                        _categorical_labels(y), np.argmax(np.asarray(p), axis=1)
                    )
                ),
            },
            "quantile": {
                "Pinball Loss": _quantile_pinball,
                "Median MAE": _quantile_median_mae,
            },
        }

        return default_metrics.get(family, {})
