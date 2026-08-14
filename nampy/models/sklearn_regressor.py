# sklearn_regressor.py
import warnings

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from ..basemodels.lightning_wrapper import TaskModel
from ..data_utils.datamodule import NAMpyDataModule
from ..utils.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)
from ._sklearn_data import prepare_fit_features, prepare_predict_features
from ._sklearn_params import NeuralEstimatorParameterMixin


class SklearnBaseRegressor(NeuralEstimatorParameterMixin, BaseEstimator):
    def __init__(self, model, config, **kwargs):
        self._initialize_estimator_parameters(config, kwargs)
        self.model = None

        # Raise a warning if task is set to 'classification'
        if self._provided_preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. The Regressor is designed for regression tasks.",
                UserWarning,
                stacklevel=2,
            )

        self.base_model = model

    def fit(
        self,
        X,
        y,
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
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        """
        Trains the regression model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (real numbers).
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
        checkpoint_path : str, default="model_checkpoints"
            Path where the checkpoints are being saved.
        dataloader_kwargs: dict, default={}
            The kwargs for the pytorch dataloader class.
        **trainer_kwargs : Additional keyword arguments for PyTorch Lightning's Trainer class.


        Returns
        -------
        self : object
            The fitted regressor.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        X = prepare_fit_features(self, X)
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
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together.")
        if X_val is not None:
            X_val = prepare_predict_features(self, X_val)
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
            config=self.config,
            num_classes=n_outputs,
            task="regression",
            cat_feature_info=self.data_module.cat_feature_info,
            num_feature_info=self.data_module.num_feature_info,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
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

    def predict(self, X):
        return self._predict(X)["output"].squeeze(-1).cpu().numpy()

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

        # Convert predictions to NumPy array and return
        return predictions

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


        Notes
        -----
        This method uses the `predict` method to generate predictions and computes each metric.


        Examples
        --------
        >>> from sklearn.metrics import mean_squared_error, r2_score
        >>> from sklearn.model_selection import train_test_split
        >>> from NAMpy.models import NAMpyRegressor
        >>> metrics = {
        ...     'Mean Squared Error': mean_squared_error,
        ...     'R2 Score': r2_score
        ... }
        >>> # Assuming 'X_test' and 'y_test' are your test dataset and labels
        >>> # Evaluate using the specified metrics
        >>> results = regressor.evaluate(X_test, y_test, metrics=metrics)


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.
        """
        if metrics is None:
            metrics = {"Mean Squared Error": mean_squared_error}

        # Generate predictions using the trained model
        predictions = self.predict(X)

        # Initialize dictionary to store results
        scores = {}

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def _plot_single_feature_effects(
        self, x_plot, predictions, y_true, ax, feature_name=None, num_bins=30
    ):
        """
        Plot the effect of a single feature on a given axes.

        Parameters
        ----------
        x_plot : np.ndarray
            The feature values for plotting.
        predictions : np.ndarray
            The predicted contributions from the model.
        y_true : np.ndarray
            The true target values (for scatter plot).
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        feature_name : str, optional
            The name of the feature for labels.
        num_bins : int, optional
            Number of bins for density shading, by default 30.
        """
        contribs = predictions.flatten()
        y_true_centered = y_true - np.mean(y_true)
        y_range = (y_true_centered.min() - 1, y_true_centered.max() + 1)

        plot_density_shading(ax, x_plot, y_range, num_bins)
        ax.plot(x_plot, contribs, color="black", label="Shape Function")
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
            (n_samples, n_outputs).
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
        n_outputs = interaction_preds.shape[1]

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

        fig, axes = create_subplot_grid(n_outputs)

        x1_bins = np.linspace(x1_vals.min(), x1_vals.max(), num_bins)
        x2_bins = np.linspace(x2_vals.min(), x2_vals.max(), num_bins)
        x1_bin_idx = np.clip(np.digitize(x1_vals, x1_bins) - 1, 0, num_bins - 2)
        x2_bin_idx = np.clip(np.digitize(x2_vals, x2_bins) - 1, 0, num_bins - 2)

        for out_idx, ax in enumerate(axes[:n_outputs]):
            contribs = interaction_preds[:, out_idx]

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
            if n_outputs > 1:
                title += f" (Output {out_idx + 1})"
            ax.set_title(title)
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

        for ax in axes[n_outputs:]:
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
