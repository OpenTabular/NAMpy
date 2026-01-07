import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    StandardScaler,
    QuantileTransformer,
    PolynomialFeatures,
    SplineTransformer,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .ple_encoding import PLE
from .prepro_utils import ContinuousOrdinalEncoder, CustomBinner, OneHotFromOrdinal
from .quantile_preprocessing import QuantilePreprocessor


class Preprocessor:
    """
    A comprehensive preprocessor for structured data, capable of handling both numerical and categorical features.
    It supports various preprocessing strategies for numerical data, including binning, one-hot encoding,
    standardization, and normalization. Categorical features can be transformed using continuous ordinal encoding.
    Additionally, it allows for the use of decision tree-derived bin edges for numerical feature binning.

    The class is designed to work seamlessly with pandas DataFrames, facilitating easy integration into
    machine learning pipelines.

    Parameters
    ----------
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', 'normalization', 'quantile',
        'polynomial', 'splines', and 'none'.
    categorical_preprocessing : str, default="int"
        The preprocessing strategy for categorical features. Valid options are
        'int', 'one_hot', and 'leave_one_out'.
    global_transform : str, default="none"
        Optional global transform applied after categorical encoding and before
        per-feature transforms. Valid options are 'none' and 'quantile'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    task : str, default="regression"
        Indicates the type of machine learning task ('regression' or 'classification'). This can
        influence certain preprocessing behaviors, especially when using decision tree-based binning.
    cat_cutoff : float or int, default=0.03
        Indicates the cutoff after which integer values are treated as categorical.
        If float, it's treated as a percentage. If int, it's the maximum number of
        unique values for a column to be considered categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, all integer columns will be treated as numerical, regardless
        of their unique value count or proportion.
    degree : int, default=3
        The degree of the polynomial features to be used in preprocessing.
    knots : int, default=12
        The number of knots to be used in spline transformations.

    Attributes
    ----------
    column_transformer : ColumnTransformer
        An instance of sklearn's ColumnTransformer that holds the
        configured preprocessing pipelines for different feature types.
    fitted : bool
        Indicates whether the preprocessor has been fitted to the data.
    """

    def __init__(
        self,
        n_bins=50,
        numerical_preprocessing="ple",
        categorical_preprocessing="int",
        global_transform="none",
        use_decision_tree_bins=False,
        binning_strategy="uniform",
        task="regression",
        cat_cutoff=0.03,
        treat_all_integers_as_numerical=False,
        degree=3,
        knots=12,
        quantile_preprocessing="feature",
        quantile_noise=0.0,
        quantile_output_distribution="normal",
        quantile_n_quantiles=2000,
    ):
        self.n_bins = n_bins
        self.numerical_preprocessing = numerical_preprocessing.lower()
        if self.numerical_preprocessing not in [
            "ple",
            "binning",
            "one_hot",
            "standardization",
            "normalization",
            "quantile",
            "polynomial",
            "splines",
            "none",
        ]:
            raise ValueError(
                "Invalid numerical_preprocessing value. Supported values are 'ple', 'binning', 'one_hot', 'standardization', 'quantile', 'polynomial', 'splines', 'none' and 'normalization'."
            )

        self.categorical_preprocessing = categorical_preprocessing.lower()
        if self.categorical_preprocessing not in [
            "int",
            "one_hot",
            "one-hot",
            "leave_one_out",
        ]:
            raise ValueError(
                "Invalid categorical_preprocessing value. Supported values are 'int', 'one_hot', and 'leave_one_out'."
            )

        self.global_transform = global_transform.lower()
        if self.global_transform not in ["none", "quantile"]:
            raise ValueError(
                "Invalid global_transform. Supported values are 'none' and 'quantile'."
            )

        self.use_decision_tree_bins = use_decision_tree_bins
        self.column_transformer = None
        self.fitted = False
        self.binning_strategy = binning_strategy
        self.task = task
        self.cat_cutoff = cat_cutoff
        self.treat_all_integers_as_numerical = treat_all_integers_as_numerical
        self.degree = degree
        self.n_knots = knots
        self.quantile_preprocessing = quantile_preprocessing
        self.quantile_noise = quantile_noise
        self.quantile_output_distribution = quantile_output_distribution
        self.quantile_n_quantiles = quantile_n_quantiles
        self.quantile_preprocessor = None
        self.global_transformer = None
        self.categorical_encoder = None
        self.quantile_numerical_features = []
        if self.quantile_preprocessing not in ["feature", "global"]:
            raise ValueError(
                "Invalid quantile_preprocessing. Supported values are 'feature' and 'global'."
            )
        if self.quantile_output_distribution not in ["normal", "uniform"]:
            raise ValueError(
                "Invalid quantile_output_distribution. Supported values are 'normal' and 'uniform'."
            )

    def set_params(self, **params):
        for key, value in params.items():
            if key == "knots":
                self.n_knots = value
                continue
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {
            "n_bins": self.n_bins,
            "numerical_preprocessing": self.numerical_preprocessing,
            "categorical_preprocessing": self.categorical_preprocessing,
            "global_transform": self.global_transform,
            "use_decision_tree_bins": self.use_decision_tree_bins,
            "binning_strategy": self.binning_strategy,
            "task": self.task,
            "cat_cutoff": self.cat_cutoff,
            "treat_all_integers_as_numerical": self.treat_all_integers_as_numerical,
            "degree": self.degree,
            "knots": self.n_knots,
            "quantile_preprocessing": self.quantile_preprocessing,
            "quantile_noise": self.quantile_noise,
            "quantile_output_distribution": self.quantile_output_distribution,
            "quantile_n_quantiles": self.quantile_n_quantiles,
        }

    def _detect_column_types(self, X):
        """
        Identifies and separates the features in the dataset into numerical and categorical types based on the data type
        and the proportion of unique values.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset, where the features are columns in a DataFrame or keys in a dict.


        Returns
        -------
            tuple: A tuple containing two lists, the first with the names of numerical features and the second with the names of categorical features.
        """
        categorical_features = []
        numerical_features = []

        if isinstance(X, dict):
            X = pd.DataFrame(X)

        for col in X.columns:
            num_unique_values = X[col].nunique()
            total_samples = len(X[col])

            if self.treat_all_integers_as_numerical and X[col].dtype.kind == "i":
                numerical_features.append(col)
            else:
                if isinstance(self.cat_cutoff, float):
                    cutoff_condition = (
                        num_unique_values / total_samples
                    ) < self.cat_cutoff
                elif isinstance(self.cat_cutoff, int):
                    cutoff_condition = num_unique_values < self.cat_cutoff
                else:
                    raise ValueError(
                        "cat_cutoff should be either a float or an integer."
                    )

                if X[col].dtype.kind not in "iufc" or (
                    X[col].dtype.kind == "i" and cutoff_condition
                ):
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)

        return numerical_features, categorical_features

    def _fit_categorical_encoder(self, X, y, categorical_features):
        if not categorical_features:
            return X, categorical_features, []
        if self.categorical_preprocessing != "leave_one_out":
            return X, categorical_features, []
        if y is None:
            raise ValueError(
                "categorical_preprocessing='leave_one_out' requires y during fit."
            )
        try:
            from category_encoders import LeaveOneOutEncoder
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "category_encoders is required for leave_one_out encoding. "
                "Install it via `pip install category_encoders`."
            ) from exc

        self.categorical_encoder = LeaveOneOutEncoder(cols=categorical_features)
        self.categorical_encoder.fit(X, y)
        X_encoded = self.categorical_encoder.transform(X)
        new_numerical = list(categorical_features)
        return X_encoded, [], new_numerical

    def _apply_categorical_encoder(self, X):
        if self.categorical_encoder is None:
            return X
        return self.categorical_encoder.transform(X)

    def _fit_global_transformer(self, X):
        if self.global_transform != "quantile":
            return X
        self.global_transformer = QuantilePreprocessor(
            output_distribution=self.quantile_output_distribution,
            quantile_noise=self.quantile_noise,
            n_quantiles=self.quantile_n_quantiles,
            random_state=101,
        )
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        self.global_transformer.fit(X_values)
        transformed = self.global_transformer.transform(X_values)
        return pd.DataFrame(transformed, columns=X.columns)

    def _apply_global_transformer(self, X):
        if self.global_transformer is None:
            return X
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        transformed = self.global_transformer.transform(X_values)
        return pd.DataFrame(transformed, columns=X.columns)

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the data by identifying feature types and configuring the appropriate transformations for each feature.
        It sets up a column transformer with a pipeline of transformations for numerical and categorical features based on the specified preprocessing strategy.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset to fit the preprocessor on.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True for determining optimal bin edges using decision trees.


        Returns
        -------
            self: The fitted Preprocessor instance.
        """
        if isinstance(X, dict):
            X = pd.DataFrame(X)

        numerical_features, categorical_features = self._detect_column_types(X)
        if (
            self.global_transform == "quantile"
            and categorical_features
            and self.categorical_preprocessing != "leave_one_out"
        ):
            raise ValueError(
                "global_transform='quantile' requires categorical_preprocessing='leave_one_out' "
                "or no categorical features."
            )
        X, categorical_features, encoded_numerical = self._fit_categorical_encoder(
            X, y, categorical_features
        )
        numerical_features = list(numerical_features) + list(encoded_numerical)
        transformers = []

        if self.global_transform == "quantile":
            X = self._fit_global_transformer(X)

        if (
            self.numerical_preprocessing == "quantile"
            and self.quantile_preprocessing == "global"
            and numerical_features
        ):
            self.quantile_preprocessor = QuantilePreprocessor(
                output_distribution=self.quantile_output_distribution,
                quantile_noise=self.quantile_noise,
                n_quantiles=self.quantile_n_quantiles,
                random_state=101,
            )
            self.quantile_preprocessor.fit(X[numerical_features])
            self.quantile_numerical_features = list(numerical_features)

        if numerical_features:
            for feature in numerical_features:
                numeric_transformer_steps = [
                    ("imputer", SimpleImputer(strategy="mean"))
                ]

                if self.numerical_preprocessing in ["binning", "one_hot"]:
                    bins = (
                        self._get_decision_tree_bins(X[[feature]], y, [feature])
                        if self.use_decision_tree_bins
                        else self.n_bins
                    )
                    if isinstance(bins, int):
                        numeric_transformer_steps.extend(
                            [
                                (
                                    "discretizer",
                                    KBinsDiscretizer(
                                        n_bins=(
                                            bins
                                            if isinstance(bins, int)
                                            else len(bins) - 1
                                        ),
                                        encode="ordinal",
                                        strategy=self.binning_strategy,
                                        subsample=200_000 if len(X) > 200_000 else None,
                                    ),
                                ),
                            ]
                        )
                    else:
                        numeric_transformer_steps.extend(
                            [
                                (
                                    "discretizer",
                                    CustomBinner(bins=bins),
                                ),
                            ]
                        )

                    if self.numerical_preprocessing == "one_hot":
                        numeric_transformer_steps.extend(
                            [
                                ("onehot_from_ordinal", OneHotFromOrdinal()),
                            ]
                        )

                elif self.numerical_preprocessing == "standardization":
                    numeric_transformer_steps.append(("scaler", StandardScaler()))

                elif self.numerical_preprocessing == "normalization":
                    numeric_transformer_steps.append(("normalizer", MinMaxScaler()))

                elif self.numerical_preprocessing == "quantile":
                    if self.quantile_preprocessing != "global":
                        numeric_transformer_steps.append(
                            (
                                "quantile",
                                QuantileTransformer(
                                    n_quantiles=self.quantile_n_quantiles,
                                    random_state=101,
                                    output_distribution=self.quantile_output_distribution,
                                ),
                            )
                        )

                elif self.numerical_preprocessing == "polynomial":
                    numeric_transformer_steps.append(
                        (
                            "polynomial",
                            PolynomialFeatures(self.degree, include_bias=False),
                        )
                    )

                elif self.numerical_preprocessing == "splines":
                    numeric_transformer_steps.append(
                        (
                            "splines",
                            SplineTransformer(
                                degree=self.degree,
                                n_knots=self.n_knots,
                                include_bias=False,
                            ),
                        ),
                    )

                elif self.numerical_preprocessing == "ple":
                    numeric_transformer_steps.append(("normalizer", MinMaxScaler()))
                    numeric_transformer_steps.append(
                        ("ple", PLE(n_bins=self.n_bins, task=self.task))
                    )
                elif self.numerical_preprocessing == "none":
                    pass

                numeric_transformer = Pipeline(numeric_transformer_steps)

                transformers.append((f"num_{feature}", numeric_transformer, [feature]))

        if categorical_features:
            for feature in categorical_features:
                # Create a pipeline for each categorical feature
                if self.categorical_preprocessing == "int":
                    categorical_transformer = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "continuous_ordinal",
                                ContinuousOrdinalEncoder(),
                            ),
                        ]
                    )
                elif self.categorical_preprocessing in ["one_hot", "one-hot"]:
                    categorical_transformer = Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant", fill_value="missing"
                                ),
                            ),
                            ("continuous_ordinal", ContinuousOrdinalEncoder()),
                            ("one_hot", OneHotFromOrdinal()),
                        ]
                    )
                else:
                    raise ValueError(
                        "categorical_preprocessing should be either 'int' or 'one_hot'."
                    )
                # Append the transformer for the current categorical feature
                transformers.append(
                    (f"cat_{feature}", categorical_transformer, [feature])
                )

        self.column_transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )
        X_fit = X
        if self.quantile_preprocessor is not None and self.quantile_numerical_features:
            X_fit = X.copy()
            X_fit[self.quantile_numerical_features] = self.quantile_preprocessor.transform(
                X_fit[self.quantile_numerical_features]
            )
        self.column_transformer.fit(X_fit, y)

        self.fitted = True

    def _get_decision_tree_bins(self, X, y, numerical_features):
        """
        Uses decision tree models to determine optimal bin edges for numerical feature binning. This method is used when `use_decision_tree_bins` is True.

        Parameters
        ----------
            X (DataFrame): The input dataset containing only the numerical features for which the bin edges are to be determined.
            y (array-like): The target variable for training the decision tree models.
            numerical_features (list of str): The names of the numerical features for which the bin edges are to be determined.


        Returns
        -------
            list: A list of arrays, where each array contains the bin edges determined by the decision tree for a numerical feature.
        """
        bins = []
        for feature in numerical_features:
            tree_model = (
                DecisionTreeClassifier(max_depth=3)
                if y.dtype.kind in "bi"
                else DecisionTreeRegressor(max_depth=3)
            )
            tree_model.fit(X[[feature]], y)
            thresholds = tree_model.tree_.threshold[tree_model.tree_.feature != -2]
            bin_edges = np.sort(np.unique(thresholds))

            bins.append(
                np.concatenate(([X[feature].min()], bin_edges, [X[feature].max()]))
            )
        return bins

    def transform(self, X):
        """
        Transforms the input data using the preconfigured column transformer and converts the output into a dictionary
        format with keys corresponding to transformed feature names and values as arrays of transformed data.

        This method converts the sparse or dense matrix returned by the column transformer into a more accessible
        dictionary format, where each key-value pair represents a feature and its transformed data.
        Transforms the input data using the preconfigured column transformer and converts the output into a dictionary
        format with keys corresponding to transformed feature names and values as arrays of transformed data.

        This method converts the sparse or dense matrix returned by the column transformer into a more accessible
        dictionary format, where each key-value pair represents a feature and its transformed data.

        Parameters
        ----------
            X (DataFrame): The input data to be transformed.
            X (DataFrame): The input data to be transformed.


        Returns
        -------
            dict: A dictionary where keys are the names of the features (as per the transformations defined in the
            column transformer) and the values are numpy arrays of the transformed data.
        """
        if not self.fitted:
            raise NotFittedError(
                "The preprocessor must be fitted before transforming new data. Use .fit or .fit_transform"
            )
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        X = self._apply_categorical_encoder(X)
        if self.global_transformer is not None:
            X = self._apply_global_transformer(X)
        if self.quantile_preprocessor is not None and self.quantile_numerical_features:
            X = X.copy()
            X[self.quantile_numerical_features] = self.quantile_preprocessor.transform(
                X[self.quantile_numerical_features]
            )
        transformed_X = self.column_transformer.transform(X)

        # Now let's convert this into a dictionary of arrays, one per column
        transformed_dict = self._split_transformed_output(X, transformed_X)
        return transformed_dict

    def _split_transformed_output(self, X, transformed_X):
        """
        Splits the transformed data array into a dictionary where keys correspond to the original column names or
        feature groups and values are the transformed data for those columns.

        This helper method is utilized within `transform` to segregate the transformed data based on the
        specification in the column transformer, assigning each transformed section to its corresponding feature name.

        Parameters
        ----------
            X (DataFrame): The original input data, used for determining shapes and transformations.
            transformed_X (numpy array): The transformed data as a numpy array, outputted by the column transformer.


        Returns
        -------
            dict: A dictionary mapping each transformation's name to its respective numpy array of transformed data.
            The type of each array (int or float) is determined based on the type of transformation applied.
        """
        start = 0
        transformed_dict = {}
        for (
            name,
            transformer,
            columns,
        ) in self.column_transformer.transformers_:
            if transformer != "drop":
                end = start + transformer.transform(X[[columns[0]]]).shape[1]
                dtype = int if "cat" in name else float
                transformed_dict[name] = transformed_X[:, start:end].astype(dtype)
                start = end

        return transformed_dict

    def fit_transform(self, X, y=None):
        """
        Fits the preprocessor to the data and then transforms the data using the fitted preprocessing pipelines. This is a convenience method that combines `fit` and `transform`.

        Parameters
        ----------
            X (DataFrame or dict): The input dataset to fit the preprocessor on and then transform.
            y (array-like, optional): The target variable. Required if `use_decision_tree_bins` is True.


        Returns
        -------
            dict: A dictionary with the transformed data, where keys are the base feature names and values are the transformed features as arrays.
        """
        self.fit(X, y)
        self.fitted = True
        return self.transform(X)

    def get_feature_info(self):
        """
        Retrieves information about how features are encoded within the model's preprocessor.
        This method identifies the type of encoding applied to each feature, categorizing them into binned or ordinal
        encodings and other types of encodings (e.g., one-hot encoding after discretization).

        This method should only be called after the preprocessor has been fitted, as it relies on the structure and
        configuration of the `column_transformer` attribute.

        Raises
        ------
        RuntimeError
            If the `column_transformer` is not yet fitted, indicating that the preprocessor must be
            fitted before invoking this method.

        Returns
        -------
        tuple of (dict, dict)
            - num_feature_info: A dictionary where each key corresponds to a numerical feature, and its value is another dictionary
              containing encoding type, dimensionality, and unique values.
            - cat_feature_info: A dictionary where each key corresponds to a categorical feature, and its value is another dictionary
              containing encoding type, dimensionality, and unique values.
        """
        num_feature_info = {}
        cat_feature_info = {}

        if not self.column_transformer:
            raise RuntimeError("The preprocessor has not been fitted yet.")

        for (
            name,
            transformer_pipeline,
            columns,
        ) in self.column_transformer.transformers_:
            steps = [step[0] for step in transformer_pipeline.steps]

            for feature_name in columns:
                feature_details = {}

                # Handle features processed with discretization
                if "discretizer" in steps:
                    step = transformer_pipeline.named_steps["discretizer"]
                    n_bins = step.n_bins_[0] if hasattr(step, "n_bins_") else None
                    feature_details["encoding"] = "discretization"
                    feature_details["dimension"] = n_bins
                    feature_details["unique_vals"] = n_bins

                    # Check if discretization is followed by one-hot encoding
                    if "onehot_from_ordinal" in steps:
                        feature_details["encoding"] = "discretization + one-hot"
                        feature_details["dimension"] = (
                            n_bins  # Number of bins before one-hot
                        )
                        feature_details["unique_vals"] = n_bins
                        cat_feature_info[feature_name] = feature_details
                    else:
                        num_feature_info[feature_name] = feature_details

                # Handle one-hot encoding after continuous ordinal encoding
                elif "continuous_ordinal" in steps:
                    step = transformer_pipeline.named_steps["continuous_ordinal"]
                    n_categories = len(step.mapping_[columns.index(feature_name)])
                    feature_details["encoding"] = "ordinal"
                    feature_details["dimension"] = 1
                    feature_details["unique_vals"] = n_categories

                    # Now check if it also went through one-hot encoding
                    if "one_hot" in steps:
                        step = transformer_pipeline.named_steps["one_hot"]
                        transformed_feature = step.transform(
                            np.zeros((1, len(columns)))
                        )
                        feature_details["encoding"] = "one-hot"
                        feature_details["dimension"] = transformed_feature.shape[
                            1
                        ]  # Dimension after one-hot encoding
                        feature_details["unique_vals"] = transformed_feature.shape[
                            1
                        ]  # Unique categories correspond to one-hot encoding dimension

                    cat_feature_info[feature_name] = feature_details

                # Handle other numerical feature encodings
                else:
                    last_step = transformer_pipeline.steps[-1][1]
                    step_names = [step[0] for step in transformer_pipeline.steps]
                    step_descriptions = " -> ".join(step_names)
                    feature_details["encoding"] = step_descriptions

                    if hasattr(last_step, "transform"):
                        transformed_feature = last_step.transform(
                            np.zeros((1, len(columns)))
                        )
                        feature_details["dimension"] = transformed_feature.shape[1]
                        feature_details["unique_vals"] = (
                            None  # Numerical features don't have unique categories
                        )
                        num_feature_info[feature_name] = feature_details

        return cat_feature_info, num_feature_info
