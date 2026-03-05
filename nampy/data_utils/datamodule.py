import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import NAMpyDataset


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

    def _to_label_tensor(self, y):
        y_arr = np.asarray(y)
        t = torch.tensor(y_arr, dtype=self.labels_dtype)

        # For 1D targets, keep existing convention [N, 1]
        if t.ndim == 1:
            t = t.unsqueeze(1)

        # For 2D targets (e.g. Dirichlet, multivariate LSS), keep [N, K]
        return t

    def setup_data(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        val_size=0.2,
        random_state=101,
    ):
        """
        Sets up the training and validation data: splits, fits the preprocessor, and stores feature info.

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

        if (X_val is None) ^ (y_val is None):
            raise ValueError(
                "X_val and y_val must be provided together; got only one."
            )

        if X_val is None and y_val is None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state
            )
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

        combined_X = pd.concat([self.X_train, self.X_val], axis=0).reset_index(
            drop=True
        )
        combined_y = np.concatenate((self.y_train, self.y_val), axis=0)

        # Delegate to an external preprocessor (e.g. pretab) that
        # exposes get_feature_info(verbose=...) and returns
        # (num_feature_info, cat_feature_info, emb_feature_info).
        self.preprocessor.fit(combined_X, combined_y)
        num_info, cat_info, _ = self.preprocessor.get_feature_info(verbose=False)
        self.num_feature_info = num_info
        self.cat_feature_info = cat_info

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
        Backwards-compatible wrapper for the former preprocess_data API.

        This now simply delegates to setup_data, which expects a pretab-style
        preprocessor interface.
        """
        self.setup_data(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_size=val_size,
            random_state=random_state,
        )

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

            # Populate tensors for categorical features (PreTab: cat_<name>)
            for key in self.cat_feature_info:
                cat_key = "cat_" + key
                info = self.cat_feature_info[key]
                is_onehot = "onehot" in info.get("preprocessing", "").lower() or (
                    info.get("dimension", 1) > 1
                )
                cat_dtype = torch.float32 if is_onehot else torch.long
                if cat_key in train_preprocessed_data:
                    arr = train_preprocessed_data[cat_key]
                    if not is_onehot and arr.dtype.kind == "f":
                        arr = arr.astype("int64")
                    train_cat_tensors.append(torch.tensor(arr, dtype=cat_dtype))
                    cat_keys.append(key)
                if cat_key in val_preprocessed_data:
                    arr = val_preprocessed_data[cat_key]
                    if not is_onehot and arr.dtype.kind == "f":
                        arr = arr.astype("int64")
                    val_cat_tensors.append(torch.tensor(arr, dtype=cat_dtype))

            # Populate tensors for numerical features, if present in processed data
            for key in self.num_feature_info:
                num_key = "num_" + key
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

            train_labels = self._to_label_tensor(self.y_train)
            val_labels = self._to_label_tensor(self.y_val)

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
        elif stage == "test":
            if not self.test_preprocessor_fitted:
                raise ValueError(
                    "The preprocessor has not been fitted. Please fit the preprocessor before transforming the test data."
                )

            self.test_dataset = NAMpyDataset(
                self.test_cat_tensors,
                self.test_num_tensors,
                self.test_labels,
                regression=self.regression,
                cat_keys=self.cat_keys,
                num_keys=self.num_keys,
            )

    def preprocess_test_data(self, X):
        test_preprocessed_data = self.preprocessor.transform(X)

        test_cat_tensors_list = []
        test_num_tensors_list = []
        test_cat_tensors_dict = {}
        test_num_tensors_dict = {}

        cat_keys = []
        num_keys = []

        # categorical
        for key in self.cat_feature_info:
            cat_key = "cat_" + key
            info = self.cat_feature_info[key]
            is_onehot = "onehot" in info.get("preprocessing", "").lower() or (info.get("dimension", 1) > 1)
            cat_dtype = torch.float32 if is_onehot else torch.long

            if cat_key in test_preprocessed_data:
                arr = test_preprocessed_data[cat_key]
                if not is_onehot and arr.dtype.kind == "f":
                    arr = arr.astype("int64")
                t = torch.tensor(arr, dtype=cat_dtype)

                test_cat_tensors_list.append(t)
                test_cat_tensors_dict[key] = t
                cat_keys.append(key)

        # numerical
        for key in self.num_feature_info:
            num_key = "num_" + key
            if num_key in test_preprocessed_data:
                t = torch.tensor(test_preprocessed_data[num_key], dtype=torch.float32)

                test_num_tensors_list.append(t)
                test_num_tensors_dict[key] = t
                num_keys.append(key)

        n = len(next(iter(test_preprocessed_data.values())))
        self.test_labels = torch.zeros(n, dtype=self.labels_dtype).unsqueeze(1)

        # store for Lightning test_dataloader path
        self.test_cat_tensors = test_cat_tensors_list
        self.test_num_tensors = test_num_tensors_list
        self.cat_keys = cat_keys
        self.num_keys = num_keys

        self.test_preprocessor_fitted = True
        return test_cat_tensors_dict, test_num_tensors_dict

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
