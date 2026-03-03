import warnings

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import properscoring as ps
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

from pretab.preprocessor import Preprocessor

from ..basemodels.lightning_wrapper import TaskModel
from ..data_utils.datamodule import NAMpyDataModule
from ..utils.distributional_metrics import (
    beta_brier_score,
    dirichlet_error,
    gamma_deviance,
    inverse_gamma_loss,
    negative_binomial_deviance,
    poisson_deviance,
    student_t_loss,
)
from ..utils.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    DirichletDistribution,
    GammaDistribution,
    InverseGammaDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    Quantile,
    RobustNormalDistribution,
    StudentTDistribution,
)
from ..utils.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)


class SklearnBaseLSS(BaseEstimator):
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
            "degree",
            "n_knots",
            "scaling_strategy",
            "feature_preprocessing",
        ]

        self.config_kwargs = {
            k: v for k, v in kwargs.items() if k not in preprocessor_arg_names
        }
        self.config = config(**self.config_kwargs)

        preprocessor_kwargs = {
            k: v for k, v in kwargs.items() if k in preprocessor_arg_names
        }
        if "knots" in kwargs and "n_knots" not in preprocessor_kwargs:
            preprocessor_kwargs["n_knots"] = kwargs["knots"]
        if preprocessor_kwargs.get("categorical_preprocessing") in (
            "one_hot",
            "one-hot",
        ):
            preprocessor_kwargs["categorical_preprocessing"] = "one-hot"
        if preprocessor_kwargs.get("numerical_preprocessing") == "normalization":
            preprocessor_kwargs["numerical_preprocessing"] = "minmax"

        self.preprocessor = Preprocessor(**preprocessor_kwargs)
        self.model = None

        # Raise a warning if task is set to 'classification'
        if preprocessor_kwargs.get("task") == "classification":
            warnings.warn(
                "The task is set to 'classification'. Be aware of your preferred distribution, that this might lead to unsatisfactory results.",
                UserWarning,
                stacklevel=2,
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
        if "knots" in preprocessor_params and "n_knots" not in preprocessor_params:
            preprocessor_params["n_knots"] = preprocessor_params.pop("knots")
        if preprocessor_params:
            self.preprocessor.set_params(**preprocessor_params)

        return self

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
        }

        if distributional_kwargs is None:
            distributional_kwargs = {}

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if family in distribution_classes:
            self.family = distribution_classes[family](**distributional_kwargs)
        else:
            raise ValueError("Unsupported family: {}".format(family))

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
        distribution_family : str, optional
            Specifies the distribution family the model is predicting for. If None, it will attempt to infer based
            on the model's settings.


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.


        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Infer distribution family from model settings if not provided
        if distribution_family is None:
            distribution_family = getattr(self.model, "distribution_family", "normal")

        # Setup default metrics if none are provided
        if metrics is None:
            metrics = self.get_default_metrics(distribution_family)

        # Make predictions (raw=True for distribution parameter outputs)
        predictions = self.predict(X, raw=True)

        # Initialize dictionary to store results
        scores = {}

        # Compute NLL using the distribution family's compute_loss method
        if self.family is not None:
            import torch

            pred_tensor = torch.tensor(predictions, dtype=torch.float32)
            y_tensor = torch.tensor(y_true, dtype=torch.float32)
            nll = self.family.compute_loss(pred_tensor, y_tensor)
            scores["NLL"] = nll.item()

        # Get transformed predictions for other metrics
        predictions_transformed = self.predict(X, raw=False)

        # Compute each metric
        for metric_name, metric_func in metrics.items():
            scores[metric_name] = metric_func(y_true, predictions_transformed)

        return scores

    def get_default_metrics(self, distribution_family):
        """
        Provides default metrics based on the distribution family.

        Parameters
        ----------
        distribution_family : str
            The distribution family for which to provide default metrics.


        Returns
        -------
        metrics : dict
            A dictionary of default metric functions.
        """
        default_metrics = {
            "normal": {
                "MSE": lambda y, pred: mean_squared_error(y, pred[:, 0]),
                "CRPS": lambda y, pred: np.mean(
                    [
                        ps.crps_gaussian(y[i], mu=pred[i, 0], sig=np.sqrt(pred[i, 1]))
                        for i in range(len(y))
                    ]
                ),
            },
            "poisson": {"Poisson Deviance": poisson_deviance},
            "gamma": {"Gamma Deviance": gamma_deviance},
            "beta": {"Brier Score": beta_brier_score},
            "dirichlet": {"Dirichlet Error": dirichlet_error},
            "studentt": {"Student-T Loss": student_t_loss},
            "negativebinom": {"Negative Binomial Deviance": negative_binomial_deviance},
            "inversegamma": {"Inverse Gamma Loss": inverse_gamma_loss},
            "categorical": {"Accuracy": accuracy_score},
        }
        return default_metrics.get(distribution_family, {})

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
        y_range = (y_true.min() - 1, y_true.max() + 1)

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

        y_true_centered = y_true - np.mean(y_true)
        ax.scatter(
            x_plot, y_true_centered, color="gray", alpha=0.3, s=2, label="True Values"
        )

        ax.set_title(
            f"Shape Function: {feature_name}" if feature_name else "Shape Function"
        )
        ax.set_xlabel(feature_name or "Feature")
        ax.set_ylabel("Contribution")
        ax.legend()

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

        for ax, fname in zip(axes, features_to_plot):
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
                    feature1, feature2 = interaction_name.split(":")
                    self._plot_interaction_effects(
                        interaction_name,
                        predictions[feature1],
                        predictions[feature2],
                        X_train_scaled=X_prepared,
                    )
