import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import warnings
from ..basemodels.lightning_wrapper import TaskModel
from ..data_utils.datamodule import NAMpyDataModule
from ..preprocessing import Preprocessor
import matplotlib.pyplot as plt
import numpy as np


class SklearnBaseRegressor(BaseEstimator):
    def __init__(self, model, config, **kwargs):
        preprocessor_arg_names = [
            "n_bins",
            "numerical_preprocessing",
            "categorical_preprocessing",
            "use_decision_tree_bins",
            "binning_strategy",
            "task",
            "cat_cutoff",
            "treat_all_integers_as_numerical",
            "knots",
            "degree",
        ]

        self.config_kwargs = {
            k: v for k, v in kwargs.items() if k not in preprocessor_arg_names
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in preprocessor_arg_names
        }

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. The Regressor is designed for regression tasks.",
                UserWarning,
            )

        self.base_model = model

    def get_params(self, deep=True):
        """
        Get parameters for this estimator. Overrides the BaseEstimator method.

        Parameters
        ----------
        deep : bool, default=True
            If True, returns the parameters for this estimator and contained sub-objects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = self.config_kwargs  # Parameters used to initialize DefaultConfig

        # If deep=True, include parameters from nested components like preprocessor
        if deep:
            # Assuming Preprocessor has a get_params method
            preprocessor_params = {
                "preprocessor__" + key: value
                for key, value in self.preprocessor.get_params().items()
            }
            params.update(preprocessor_params)

        return params

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator. Overrides the BaseEstimator method.

        Parameters
        ----------
        **parameters : dict
            Estimator parameters to be set.

        Returns
        -------
        self : object
            The instance with updated parameters.
        """
        # Update config_kwargs with provided parameters
        valid_config_keys = self.config_kwargs.keys()
        config_updates = {k: v for k, v in parameters.items() if k in valid_config_keys}
        self.config_kwargs.update(config_updates)

        # Update the config object
        for key, value in config_updates.items():
            setattr(self.config, key, value)

        # Handle preprocessor parameters (prefixed with 'preprocessor__')
        preprocessor_params = {
            k.split("__")[1]: v
            for k, v in parameters.items()
            if k.startswith("preprocessor__")
        }
        if preprocessor_params:
            # Assuming Preprocessor has a set_params method
            self.preprocessor.set_params(**preprocessor_params)

        return self

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
        dataloader_kwargs={},
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
        batch_size : int, default=64
            Number of samples per gradient update.
        shuffle : bool, default=True
            Whether to shuffle the training data before each epoch.
        patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before early stopping.
        monitor : str, default="val_loss"
            The metric to monitor for early stopping.
        mode : str, default="min"
            Whether the monitored metric should be minimized (`min`) or maximized (`max`).
        lr : float, default=1e-3
            Learning rate for the optimizer.
        lr_patience : int, default=10
            Number of epochs with no improvement on the validation loss to wait before reducing the learning rate.
        factor : float, default=0.1
            Factor by which the learning rate will be reduced.
        weight_decay : float, default=0.025
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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
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

        self.data_module.preprocess_data(
            X, y, X_val, y_val, val_size=val_size, random_state=random_state
        )

        self.model = TaskModel(
            model_class=self.base_model,
            config=self.config,
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
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict(self, X):
        return self._predict(X)["output"].cpu().numpy()

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

    def _plot_single_feature_effects(self, x_plot, predictions, y_true, num_bins=30):
        """
        Internal function to plot the effect of a single feature, including shading based on data density and a scatter plot of true values.

        Parameters
        ----------
        x_plot : np.ndarray
            The simulated feature values for plotting.
        predictions : np.ndarray
            The predicted values for the feature from the model.
        y_true : np.ndarray
            The true values for the target variable (for scatter plot).
        num_bins : int, optional
            The number of bins to use for density shading, by default 30.
        """
        # Get the predictions from the model
        contribs = predictions.flatten()
        y_true_plot = y_true - np.mean(y_true)

        # Create density-based shading
        counts, bin_edges = np.histogram(x_plot, bins=num_bins)
        norm_counts = counts / counts.max()  # Normalize to range [0, 1]

        plt.figure(figsize=(8, 6))
        for i in range(num_bins):
            plt.bar(
                bin_edges[i],
                (y_true_plot.min() - 1, y_true_plot.max() + 1),
                width=bin_edges[i + 1] - bin_edges[i],
                color=plt.cm.Reds(norm_counts[i]),
                alpha=0.6,
            )

        # Plot the shape function (predicted contributions)
        plt.plot(
            x_plot,
            contribs,
            color="black",
            label="Shape Function",
        )

        # Add scatter plot of the true values
        plt.scatter(
            x_plot, y_true_plot, color="gray", alpha=0.3, s=2, label="True Values"
        )

        plt.title(f"Shape Function for {x_plot}")
        plt.xlabel("Feature")
        plt.ylabel("Contribution")
        plt.legend()
        plt.show()

    def plot(self, X, y_true, plot_interactions=False):
        """
        Main function to plot feature effects. Calls single feature effect plotting and, if requested, interaction effect plotting.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for generating predictions.
        y_true : np.ndarray
            True target values for comparison in the scatter plot.
        plot_interactions : bool, optional
            Whether to also plot pairwise feature interactions, by default False.
        """
        # Simulate the data
        X_simulated = pd.DataFrame(X)

        # Sort each column in ascending order (only for numerical columns)
        for feature_name in X_simulated.columns:
            if pd.api.types.is_numeric_dtype(X_simulated[feature_name]):
                X_simulated[feature_name] = (
                    X_simulated[feature_name].sort_values().values
                )

        # Generate predictions using the model
        predictions = self._predict(X_simulated)

        # Plot single feature effects for numerical features only
        for feature_name in X_simulated.columns:
            if feature_name in predictions and pd.api.types.is_numeric_dtype(
                X_simulated[feature_name]
            ):
                x_plot = X_simulated[feature_name].values  # Use simulated data directly
                self._plot_single_feature_effects(
                    x_plot, predictions[feature_name], y_true, num_bins=30
                )

        # Plot pairwise interaction effects (if requested)
        if plot_interactions:
            for interaction_name in predictions.keys():
                if ":" in interaction_name:  # Pairwise interaction check
                    feature1, feature2 = interaction_name.split(":")
                    self._plot_interaction_effects(
                        interaction_name,
                        predictions[feature1],
                        predictions[feature2],
                        X_train_scaled=X_simulated,
                    )
