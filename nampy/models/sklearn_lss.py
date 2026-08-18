# sklearn_lss.py
import warnings

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import properscoring as ps
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, mean_squared_error

from ..neural.data.datamodule import NAMpyDataModule
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
from ..neural.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)
from ..neural.training.lightning_wrapper import TaskModel
from ._sklearn_base import NeuralEstimatorBase
from ._sklearn_data import prepare_fit_features, prepare_predict_features


class SklearnBaseLSS(NeuralEstimatorBase):
    def __init__(self, model, config, **kwargs):
        self._initialize_estimator_parameters(config, kwargs)
        self.model = None

        # Raise a warning if task is set to 'classification'
        if self._provided_preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. Be aware of your preferred distribution, that this might lead to unsatisfactory results.",
                UserWarning,
                stacklevel=2,
            )

        self.base_model = model

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
        Trains the distributional regression model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
        family : str
            The name of the distribution family to use for the loss function. Examples include 'normal' for regression tasks.
        val_size : float, default=0.2
            The proportion of the dataset to include in the validation split if `X_val` is None. Ignored if `X_val` is provided.
        X_val : DataFrame or array-like, shape (n_samples, n_features), optional
            The validation input samples. If provided, `X` and `y` are not split and this data is used for validation.
        y_val : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The validation target values. Required if `X_val` is provided.
        max_epochs : int, default=100
            Maximum number of epochs for training.
        random_state : int, default=101
            Controls the shuffling applied to the data before applying the split.
        batch_size : int, default=128
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=15
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-4
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=1e-06
            Weight decay (L2 penalty) coefficient.
        distributional_kwargs : dict, default=None
            Any arguments that are specific for a certain distribution.
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted model.
        """
        distribution_classes = {
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

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if fam in distribution_classes:
            self.family = distribution_classes[fam](**distributional_kwargs)
        else:
            raise ValueError("Unsupported family: {}".format(family))

        X = prepare_fit_features(self, X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values

        self.data_module = NAMpyDataModule(
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            shuffle=shuffle,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
            regression=True,
            **dataloader_kwargs,
        )

        self.data_module.setup_data(
            X, y, X_val=X_val, y_val=y_val, val_size=val_size, random_state=random_state
        )

        self.model = TaskModel(
            model_class=self.base_model,
            num_classes=self.family.param_count,
            family=self.family,
            config=self.config,
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
            lss=True,
        )

        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # Adjust according to your validation metric
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        # Initialize the trainer and train the model
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            **trainer_kwargs,
        )
        trainer.fit(self.model, self.data_module)

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X, raw=False):
        predictions = self._predict(X)["output"]

        if not raw:
            return self.model.family(predictions).cpu().numpy()

        # Convert predictions to NumPy array and return
        else:
            return predictions.cpu().numpy()

    def predict_feature_vals(self, X):
        return self._predict(X)

    def _predict(self, X):
        """
        Predicts target values for the given input samples.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The input samples for which to predict target values.

        Returns
        -------
        predictions : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The predicted target values.
        """
        # Ensure model and data module are initialized
        if self.model is None or self.data_module is None:
            raise ValueError("The model or data module has not been fitted yet.")

        X = prepare_predict_features(self, X)

        # Preprocess the data using the data module
        cat_tensor_dict, num_tensor_dict = self.data_module.preprocess_test_data(X)

        # Move tensors to appropriate device
        device = next(self.model.parameters()).device
        cat_tensor_dict = {
            key: tensor.to(device) for key, tensor in cat_tensor_dict.items()
        }
        num_tensor_dict = {
            key: tensor.to(device) for key, tensor in num_tensor_dict.items()
        }

        # Set model to evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            predictions = self.model(
                num_features=num_tensor_dict, cat_features=cat_tensor_dict
            )

        return predictions

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

    def _infer_distributional_kwargs(self, family, y, distributional_kwargs=None):
        """
        Infer required constructor kwargs for families whose output dimension depends
        on the target shape / label support.

        Supported:
        - dirichlet -> n_dim
        - categorical -> num_classes
        - ordinal -> num_classes
        - mvnormdiag -> n_dim
        """
        kwargs = dict(distributional_kwargs or {})
        fam = self._normalize_family_name(family)
        y_arr = np.asarray(y)

        if fam == "dirichlet":
            if "n_dim" not in kwargs:
                if y_arr.ndim != 2 or y_arr.shape[1] < 2:
                    raise ValueError(
                        "Dirichlet family requires y with shape (n_samples, K), K>=2."
                    )
                kwargs["n_dim"] = int(y_arr.shape[1])

        elif fam == "categorical":
            if "num_classes" not in kwargs:
                if y_arr.ndim == 2 and y_arr.shape[1] > 1:
                    # one-hot / class-probability targets
                    kwargs["num_classes"] = int(y_arr.shape[1])
                else:
                    y_flat = y_arr.reshape(-1)
                    if not np.allclose(y_flat, np.round(y_flat)):
                        raise ValueError(
                            "Categorical family requires integer class labels or "
                            "one-hot encoded targets."
                        )
                    unique = np.unique(y_flat.astype(int))
                    if unique.size < 2:
                        raise ValueError(
                            "Categorical family requires at least 2 classes."
                        )
                    if not np.array_equal(unique, np.arange(unique.size)):
                        raise ValueError(
                            "Categorical family with label targets requires zero-based, "
                            "contiguous labels {0, ..., K-1}. "
                            f"Got labels {unique.tolist()}."
                        )
                    kwargs["num_classes"] = int(unique.size)

        elif fam == "ordinal":
            if "num_classes" not in kwargs:
                if y_arr.ndim == 2 and y_arr.shape[1] > 1:
                    # one-hot ordinal targets
                    kwargs["num_classes"] = int(y_arr.shape[1])
                else:
                    y_flat = y_arr.reshape(-1)
                    if not np.allclose(y_flat, np.round(y_flat)):
                        raise ValueError(
                            "Ordinal family requires integer ordered labels or "
                            "one-hot encoded targets."
                        )
                    unique = np.unique(y_flat.astype(int))
                    if unique.size < 2:
                        raise ValueError(
                            "Ordinal family requires at least 2 ordered classes."
                        )
                    if not np.array_equal(unique, np.arange(unique.size)):
                        raise ValueError(
                            "Ordinal family with label targets requires zero-based, "
                            "contiguous labels {0, ..., K-1}. "
                            f"Got labels {unique.tolist()}."
                        )
                    kwargs["num_classes"] = int(unique.size)

        elif fam == "mvnormdiag":
            if "n_dim" not in kwargs and "dim" not in kwargs:
                if y_arr.ndim != 2 or y_arr.shape[1] < 2:
                    raise ValueError(
                        "mvnormdiag family requires y with shape (n_samples, K), K>=2."
                    )
                kwargs["n_dim"] = int(y_arr.shape[1])

        return kwargs

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

    def _plot_single_feature_effects(
        self, x_plot, predictions, y_true, ax, feature_name=None, num_bins=30
    ):
        """
        Plot the effect of a single feature for LSS regression, with separate lines for each parameter.

        Parameters
        ----------
        x_plot : np.ndarray
            The feature values for plotting.
        predictions : np.ndarray
            The predicted values (shape (n, k) for distributional parameters).
        y_true : np.ndarray
            The true target values (for scatter plot).
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        feature_name : str, optional
            The name of the feature for labels.
        num_bins : int, optional
            Number of bins for density shading, by default 30.
        """
        n_params = predictions.shape[1] if predictions.ndim > 1 else 1

        # Reduce multi-output targets (e.g. Dirichlet simplex) to a 1D signal for display
        y_1d = (
            y_true.mean(axis=1) if np.asarray(y_true).ndim > 1 else np.asarray(y_true)
        )
        y_range = (y_1d.min() - 1, y_1d.max() + 1)

        plot_density_shading(ax, x_plot, y_range, num_bins)

        # Plot shape functions for each distributional parameter
        for i in range(n_params):
            contribs = predictions[:, i] if predictions.ndim > 1 else predictions
            label = (
                self.family.param_names[i]
                if hasattr(self, "family")
                else f"Param {i + 1}"
            )
            ax.plot(x_plot, contribs, label=label)

        y_true_centered = y_1d - np.mean(y_1d)
        ax.scatter(
            x_plot, y_true_centered, color="gray", alpha=0.3, s=2, label="True Values"
        )

        ax.set_title(
            f"Shape Function: {feature_name}" if feature_name else "Shape Function"
        )
        ax.set_xlabel(feature_name or "Feature")
        ax.set_ylabel("Contribution")
        ax.legend()

    def _plot_interaction_effects(
        self,
        interaction_name,
        interaction_preds,
        X_train_scaled=None,
        num_bins=30,
    ):
        """
        Plot the interaction effect between two features as a heatmap.

        Parameters
        ----------
        interaction_name : str
            The name of the interaction in "feature1:feature2" format.
        interaction_preds : np.ndarray or torch.Tensor
            Predicted interaction contributions from the model, shape (n_samples,) or
            (n_samples, n_params).
        X_train_scaled : pd.DataFrame, optional
            Input data used to extract raw feature values for axis labels.
        num_bins : int, optional
            Number of bins for the heatmap grid, by default 30.
        """
        feature1, feature2 = interaction_name.split(":")

        if hasattr(interaction_preds, "detach"):
            interaction_preds = interaction_preds.detach().cpu().numpy()
        else:
            interaction_preds = np.asarray(interaction_preds)
        if interaction_preds.ndim == 1:
            interaction_preds = interaction_preds[:, np.newaxis]
        n_params = interaction_preds.shape[1]

        x1_vals = (
            X_train_scaled[feature1].values
            if X_train_scaled is not None
            else np.arange(len(interaction_preds))
        )
        x2_vals = (
            X_train_scaled[feature2].values
            if X_train_scaled is not None
            else np.arange(len(interaction_preds))
        )

        fig, axes = create_subplot_grid(n_params)

        x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
        x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
        x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
        x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

        for param_idx, ax in enumerate(axes[:n_params]):
            contribs = interaction_preds[:, param_idx]

            grid_sum = np.zeros((num_bins - 1, num_bins - 1))
            grid_count = np.zeros((num_bins - 1, num_bins - 1), dtype=int)
            np.add.at(grid_sum, (x1_bin_idx, x2_bin_idx), contribs)
            np.add.at(grid_count, (x1_bin_idx, x2_bin_idx), 1)
            grid = np.where(
                grid_count > 0, grid_sum / np.maximum(grid_count, 1), np.nan
            )

            im = ax.imshow(
                grid.T,
                origin="lower",
                aspect="auto",
                extent=[x1_bins[0], x1_bins[-1], x2_bins[0], x2_bins[-1]],
                cmap="RdBu_r",
            )
            plt.colorbar(im, ax=ax, label="Contribution")
            title = f"Interaction: {feature1} × {feature2}"
            if n_params > 1:
                param_label = (
                    self.family.param_names[param_idx]
                    if hasattr(self, "family") and hasattr(self.family, "param_names")
                    else f"Param {param_idx + 1}"
                )
                title += f" ({param_label})"
            ax.set_title(title)
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

        for ax in axes[n_params:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot(self, X, y_true, feature_name=None, plot_interactions=False):
        """
        Plot feature effects in a unified grid layout.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        y_true : np.ndarray
            True target values for comparison.
        feature_name : str, optional
            Specific feature to plot. If None, plots all numerical features.
        plot_interactions : bool, optional
            Whether to also plot pairwise feature interactions, by default False.
        """
        X_prepared, num_feature_names = prepare_plot_data(
            X, self.data_module.num_feature_info, self.data_module.cat_feature_info
        )

        if feature_name is not None and feature_name not in num_feature_names:
            raise ValueError(
                f"Feature '{feature_name}' not found. Available: {num_feature_names}"
            )

        features_to_plot = [feature_name] if feature_name else num_feature_names
        predictions = self._predict(X_prepared)

        # Filter to features with predictions
        features_to_plot = [f for f in features_to_plot if f in predictions]
        if not features_to_plot:
            raise ValueError("No features found with predictions to plot.")

        # Create grid and plot
        fig, axes = create_subplot_grid(len(features_to_plot))

        for ax, fname in zip(axes, features_to_plot, strict=False):
            self._plot_single_feature_effects(
                X_prepared[fname].values,
                predictions[fname],
                y_true,
                ax,
                feature_name=fname,
            )

        # Hide unused subplots
        for ax in axes[len(features_to_plot) :]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

        # Plot interactions if requested
        if plot_interactions:
            for interaction_name in predictions.keys():
                if ":" in interaction_name:
                    self._plot_interaction_effects(
                        interaction_name,
                        predictions[interaction_name],
                        X_train_scaled=X_prepared,
                    )
