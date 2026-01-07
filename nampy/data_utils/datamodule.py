import torch
import pandas as pd
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .dataset import NAMpyDataset


class NAMpyPredictDataset(Dataset):
    """
    Dataset for inference-only pipelines that do not have labels.
    """

    def __init__(self, cat_features_list, num_features_list, cat_keys=None, num_keys=None):
        self.cat_features_list = cat_features_list
        self.num_features_list = num_features_list
        self.cat_keys = cat_keys or []
        self.num_keys = num_keys or []

    def __len__(self):
        if self.cat_features_list:
            return len(self.cat_features_list[0])
        if self.num_features_list:
            return len(self.num_features_list[0])
        return 0

    def __getitem__(self, idx):
        cat_features = {
            key: feature_tensor[idx]
            for key, feature_tensor in zip(self.cat_keys, self.cat_features_list)
        }
        num_features = {
            key: torch.as_tensor(feature_tensor[idx]).clone().detach().to(torch.float32)
            for key, feature_tensor in zip(self.num_keys, self.num_features_list)
        }
        return cat_features, num_features


class NAMpyDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning data module for managing training and validation data loaders in a structured way.

    This class simplifies the process of batch-wise data loading for training and validation datasets during
    the training loop, and is particularly useful when working with PyTorch Lightning's training framework.

    Parameters:
        preprocessor: object
            An instance of your preprocessor class.
        batch_size: int
            Size of batches for the DataLoader.
        shuffle: bool
            Whether to shuffle the training data in the DataLoader.
        X_val: DataFrame or None, optional
            Validation features. If None, uses train-test split.
        y_val: array-like or None, optional
            Validation labels. If None, uses train-test split.
        val_size: float, optional
            Proportion of data to include in the validation split if `X_val` and `y_val` are None.
        random_state: int, optional
            Random seed for reproducibility in data splitting.
        regression: bool, optional
            Whether the problem is regression (True) or classification (False).
    """

    def __init__(
        self,
        preprocessor,
        batch_size,
        shuffle,
        regression,
        X_val=None,
        y_val=None,
        val_size=0.2,
        random_state=101,
        **dataloader_kwargs,
    ):
        """
        Initialize the data module with the specified preprocessor, batch size, shuffle option,
        and optional validation data settings.

        Args:
            preprocessor (object): An instance of the preprocessor class for data preprocessing.
            batch_size (int): Size of batches for the DataLoader.
            shuffle (bool): Whether to shuffle the training data in the DataLoader.
            X_val (DataFrame or None, optional): Validation features. If None, uses train-test split.
            y_val (array-like or None, optional): Validation labels. If None, uses train-test split.
            val_size (float, optional): Proportion of data to include in the validation split if `X_val` and `y_val` are None.
            random_state (int, optional): Random seed for reproducibility in data splitting.
            regression (bool, optional): Whether the problem is regression (True) or classification (False).
        """
        super().__init__()
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cat_feature_info = None
        self.num_feature_info = None
        self.X_val = X_val
        self.y_val = y_val
        self.val_size = val_size
        self.random_state = random_state
        self.regression = regression
        if self.regression:
            self.labels_dtype = torch.float32
        else:
            self.labels_dtype = torch.long

        # Initialize placeholders for data
        self.X_train = None
        self.y_train = None
        self.test_preprocessor_fitted = False
        self.dataloader_kwargs = dataloader_kwargs
        self.cat_keys = None
        self.num_keys = None
        self.test_cat_tensors = None
        self.test_num_tensors = None
        self.test_labels = None
        self.predict_cat_tensors = None
        self.predict_num_tensors = None

    def preprocess_data(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        val_size=0.2,
        random_state=101,
    ):
        """
        Preprocesses the training and validation data.

        Parameters
        ----------
        X_train : DataFrame or array-like, shape (n_samples_train, n_features)
            Training feature set.
        y_train : array-like, shape (n_samples_train,)
            Training target values.
        X_val : DataFrame or array-like, shape (n_samples_val, n_features), optional
            Validation feature set. If None, a validation set will be created from `X_train`.
        y_val : array-like, shape (n_samples_val,), optional
            Validation target values. If None, a validation set will be created from `y_train`.
        val_size : float, optional
            Proportion of data to include in the validation split if `X_val` and `y_val` are None.
        random_state : int, optional
            Random seed for reproducibility in data splitting.

        Returns
        -------
        None
        """

        if X_val is None or y_val is None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state
            )
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

        # Fit the preprocessor on training data only to avoid validation leakage
        self.preprocessor.fit(self.X_train, self.y_train)

        # Update feature info based on the actual processed data
        (
            self.cat_feature_info,
            self.num_feature_info,
        ) = self.preprocessor.get_feature_info()

    def setup(self, stage: str):
        """
        Transform the data and create DataLoaders.
        """
        if stage == "fit":
            train_preprocessed_data = self.preprocessor.transform(self.X_train)
            val_preprocessed_data = self.preprocessor.transform(self.X_val)

            # Initialize lists for tensors
            train_cat_tensors = []
            train_num_tensors = []
            val_cat_tensors = []
            val_num_tensors = []
            num_keys = []
            cat_keys = []

            # Populate tensors for categorical features, if present in processed data
            for key in self.cat_feature_info:
                cat_key = (
                    "cat_" + key
                )  # Assuming categorical keys are prefixed with 'cat_'
                if cat_key in train_preprocessed_data:
                    train_cat_tensors.append(
                        torch.tensor(train_preprocessed_data[cat_key], dtype=torch.long)
                    )
                    cat_keys.append(key)
                if cat_key in val_preprocessed_data:
                    val_cat_tensors.append(
                        torch.tensor(val_preprocessed_data[cat_key], dtype=torch.long)
                    )

                binned_key = "num_" + key  # for binned features
                if binned_key in train_preprocessed_data:
                    train_cat_tensors.append(
                        torch.tensor(
                            train_preprocessed_data[binned_key], dtype=torch.long
                        )
                    )
                    cat_keys.append(key)

                if binned_key in val_preprocessed_data:
                    val_cat_tensors.append(
                        torch.tensor(
                            val_preprocessed_data[binned_key], dtype=torch.long
                        )
                    )

            # Populate tensors for numerical features, if present in processed data
            for key in self.num_feature_info:
                num_key = (
                    "num_" + key
                )  # Assuming numerical keys are prefixed with 'num_'
                if num_key in train_preprocessed_data:
                    train_num_tensors.append(
                        torch.tensor(
                            train_preprocessed_data[num_key], dtype=torch.float32
                        )
                    )
                    num_keys.append(key)
                if num_key in val_preprocessed_data:
                    val_num_tensors.append(
                        torch.tensor(
                            val_preprocessed_data[num_key], dtype=torch.float32
                        )
                    )

            train_labels = torch.tensor(
                self.y_train, dtype=self.labels_dtype
            ).unsqueeze(dim=1)
            val_labels = torch.tensor(self.y_val, dtype=self.labels_dtype).unsqueeze(
                dim=1
            )

            # Create datasets
            self.train_dataset = NAMpyDataset(
                train_cat_tensors,
                train_num_tensors,
                train_labels,
                regression=self.regression,
                cat_keys=cat_keys,
                num_keys=num_keys,
            )
            self.val_dataset = NAMpyDataset(
                val_cat_tensors,
                val_num_tensors,
                val_labels,
                regression=self.regression,
                cat_keys=cat_keys,
                num_keys=num_keys,
            )
            self.cat_keys = cat_keys
            self.num_keys = num_keys
        elif stage == "test":
            if not self.test_preprocessor_fitted:
                raise ValueError(
                    "The preprocessor has not been fitted. Please fit the preprocessor before transforming the test data."
                )
            if self.test_labels is None:
                raise ValueError(
                    "Test labels are missing. Pass y to preprocess_test_data before calling setup('test')."
                )
            if self.cat_keys is None or self.num_keys is None:
                raise ValueError(
                    "Feature keys are missing. Call setup('fit') before setup('test')."
                )

            self.test_dataset = NAMpyDataset(
                self.test_cat_tensors,
                self.test_num_tensors,
                self.test_labels,
                regression=self.regression,
                cat_keys=self.cat_keys,
                num_keys=self.num_keys,
            )
        elif stage == "predict":
            if self.predict_cat_tensors is None or self.predict_num_tensors is None:
                raise ValueError(
                    "Predict tensors are missing. Call preprocess_predict_data before setup('predict')."
                )
            if self.cat_keys is None or self.num_keys is None:
                raise ValueError(
                    "Feature keys are missing. Call setup('fit') before setup('predict')."
                )

            self.predict_dataset = NAMpyPredictDataset(
                self.predict_cat_tensors,
                self.predict_num_tensors,
                cat_keys=self.cat_keys,
                num_keys=self.num_keys,
            )

    def preprocess_test_data(self, X, y=None):
        if self.cat_keys is None or self.num_keys is None:
            raise ValueError("Call setup('fit') before preprocess_test_data.")
        X = self._ensure_dataframe_for_predict(X)
        test_preprocessed_data = self.preprocessor.transform(X)

        # Initialize lists for categorical and numerical tensors
        test_cat_tensors = []
        test_num_tensors = []

        # Populate tensors for categorical features, including binned numeric features.
        for key in self.cat_keys:
            cat_key = "cat_" + key
            binned_key = "num_" + key
            if cat_key in test_preprocessed_data:
                test_cat_tensors.append(
                    torch.tensor(test_preprocessed_data[cat_key], dtype=torch.long)
                )
            elif binned_key in test_preprocessed_data:
                test_cat_tensors.append(
                    torch.tensor(test_preprocessed_data[binned_key], dtype=torch.long)
                )
            else:
                raise KeyError(f"Missing categorical feature '{key}' in test data.")

        # Populate tensors for numerical features, if present in processed data.
        for key in self.num_keys:
            num_key = "num_" + key
            if num_key in test_preprocessed_data:
                test_num_tensors.append(
                    torch.tensor(test_preprocessed_data[num_key], dtype=torch.float32)
                )
            else:
                raise KeyError(f"Missing numerical feature '{key}' in test data.")

        if y is not None:
            self.test_labels = torch.tensor(y, dtype=self.labels_dtype).unsqueeze(dim=1)

        self.test_cat_tensors = test_cat_tensors
        self.test_num_tensors = test_num_tensors
        self.test_preprocessor_fitted = True
        return test_cat_tensors, test_num_tensors

    def preprocess_predict_data(self, X):
        if self.cat_keys is None or self.num_keys is None:
            raise ValueError("Call setup('fit') before preprocess_predict_data.")
        X = self._ensure_dataframe_for_predict(X)
        predict_preprocessed_data = self.preprocessor.transform(X)

        predict_cat_tensors = []
        predict_num_tensors = []

        for key in self.cat_keys:
            cat_key = "cat_" + key
            binned_key = "num_" + key
            if cat_key in predict_preprocessed_data:
                predict_cat_tensors.append(
                    torch.tensor(predict_preprocessed_data[cat_key], dtype=torch.long)
                )
            elif binned_key in predict_preprocessed_data:
                predict_cat_tensors.append(
                    torch.tensor(predict_preprocessed_data[binned_key], dtype=torch.long)
                )
            else:
                raise KeyError(f"Missing categorical feature '{key}' in predict data.")

        for key in self.num_keys:
            num_key = "num_" + key
            if num_key in predict_preprocessed_data:
                predict_num_tensors.append(
                    torch.tensor(predict_preprocessed_data[num_key], dtype=torch.float32)
                )
            else:
                raise KeyError(f"Missing numerical feature '{key}' in predict data.")

        self.predict_cat_tensors = predict_cat_tensors
        self.predict_num_tensors = predict_num_tensors
        return predict_cat_tensors, predict_num_tensors

    def _expected_feature_names(self):
        if self.cat_keys is None or self.num_keys is None:
            raise ValueError("Call setup('fit') before preprocessing prediction data.")
        return list(self.cat_keys) + list(self.num_keys)

    def _ensure_dataframe_for_predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            expected = self._expected_feature_names()
            if set(X.keys()) != set(expected):
                raise ValueError(
                    f"Expected X with feature keys {expected}, got {list(X.keys())}."
                )
            return pd.DataFrame(X, columns=expected)

        if isinstance(X, pd.Series):
            expected = self._expected_feature_names()
            if len(expected) != 1:
                raise ValueError(
                    f"Expected X with {len(expected)} feature(s), got a Series."
                )
            return X.to_frame(name=expected[0])

        X_array = np.asarray(X)
        if X_array.ndim == 1:
            expected = self._expected_feature_names()
            if len(expected) != 1:
                raise ValueError(
                    f"Expected X with {len(expected)} feature(s), got 1D input."
                )
            X_array = X_array.reshape(-1, 1)

        if X_array.ndim != 2:
            raise ValueError("X must be 2D array-like for prediction.")

        expected = self._expected_feature_names()
        if X_array.shape[1] != len(expected):
            raise ValueError(
                f"Expected X with {len(expected)} feature(s), got {X_array.shape[1]}"
            )

        return pd.DataFrame(X_array, columns=expected)

    def train_dataloader(self):
        """
        Returns the training dataloader.

        Returns:
            DataLoader: DataLoader instance for the training dataset.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: DataLoader instance for the validation dataset.
        """
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )

    def test_dataloader(self):
        """
        Returns the test dataloader.

        Returns:
            DataLoader: DataLoader instance for the test dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )

    def predict_dataloader(self):
        """
        Returns the predict dataloader.

        Returns:
            DataLoader: DataLoader instance for the predict dataset.
        """
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, **self.dataloader_kwargs
        )
