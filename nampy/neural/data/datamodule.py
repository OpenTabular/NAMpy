import lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        sampling_strategy=None,
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
        if sampling_strategy not in {None, "balanced"}:
            raise ValueError("sampling_strategy must be None or 'balanced'.")
        if regression and sampling_strategy is not None:
            raise ValueError("Balanced sampling is only available for classification.")
        self.sampling_strategy = sampling_strategy
        if self.regression:
            self.labels_dtype = torch.float32
        else:
            self.labels_dtype = torch.long

        # Initialize placeholders for data
        self.X_train = None
        self.y_train = None
        self.offset_train = None
        self.offset_val = None
        self.sample_weight_train = None
        self.sample_weight_val = None
        self._train_preprocessed_data = None
        self.test_preprocessor_fitted = False
        self.dataloader_kwargs = dataloader_kwargs

    def _to_offset_tensor(self, offset):
        if offset is None:
            return None
        t = torch.tensor(np.asarray(offset, dtype=np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return t

    def _to_label_tensor(self, y):
        y_arr = np.asarray(y)
        t = torch.tensor(y_arr, dtype=self.labels_dtype)

        # For 1D targets, keep existing convention [N, 1]
        if t.ndim == 1:
            t = t.unsqueeze(1)

        # For 2D targets (e.g. Dirichlet, multivariate LSS), keep [N, K]
        return t

    @staticmethod
    def _validate_sample_weight(sample_weight, n_samples, *, name):
        if sample_weight is None:
            return None
        values = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if len(values) != n_samples:
            raise ValueError(
                f"{name} must contain {n_samples} values; got {len(values)}."
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
        if float(values.sum()) <= 0:
            raise ValueError(f"{name} must sum to a positive value.")
        return values

    @staticmethod
    def _to_weight_tensor(sample_weight):
        if sample_weight is None:
            return None
        return torch.tensor(np.asarray(sample_weight, dtype=np.float32)).reshape(-1, 1)

    @staticmethod
    def _with_cardinality(info, transformed, prefix):
        enriched = {key: dict(value) for key, value in info.items()}
        for key, value in enriched.items():
            transformed_key = f"{prefix}_{key}"
            if transformed_key not in transformed:
                continue
            array = np.asarray(transformed[transformed_key])
            if array.ndim == 1:
                array = array[:, None]
            value["n_unique"] = int(np.unique(array, axis=0).shape[0])
        return enriched

    def setup_data(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        val_size=0.2,
        random_state=101,
        stratify=None,
        offset=None,
        offset_val=None,
        sample_weight=None,
        sample_weight_val=None,
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
        stratify : array-like, optional
            Class labels used to stratify the automatic train/validation
            split. Ignored when an explicit validation set is provided.
        offset : array-like, optional
            Per-sample additive offsets on the prediction scale, aligned with
            ``X_train``. Split alongside the features for the automatic
            train/validation split.
        offset_val : array-like, optional
            Offsets for an explicitly provided validation set.

        Returns
        -------
        None
        """

        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together; got only one.")

        sample_weight = self._validate_sample_weight(
            sample_weight, len(y_train), name="sample_weight"
        )
        if X_val is None and sample_weight_val is not None:
            raise ValueError(
                "sample_weight_val can only be used with an explicit validation set."
            )

        if X_val is None and y_val is None:
            extras = []
            if offset is not None:
                extras.append(np.asarray(offset))
            if sample_weight is not None:
                extras.append(sample_weight)

            splits = train_test_split(
                X_train,
                y_train,
                *extras,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify,
            )
            self.X_train, self.X_val, self.y_train, self.y_val = splits[:4]
            rest = splits[4:]
            if offset is not None:
                self.offset_train, self.offset_val = rest[0], rest[1]
                rest = rest[2:]
            else:
                self.offset_train = None
                self.offset_val = None
            if sample_weight is not None:
                self.sample_weight_train, self.sample_weight_val = rest[0], rest[1]
            else:
                self.sample_weight_train = None
                self.sample_weight_val = None
        else:
            if offset is not None and offset_val is None:
                raise ValueError(
                    "offset_val is required when an explicit validation set "
                    "is used together with offsets."
                )
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.offset_train = offset
            self.offset_val = offset_val
            self.sample_weight_train = sample_weight
            self.sample_weight_val = self._validate_sample_weight(
                sample_weight_val, len(y_val), name="sample_weight_val"
            )

        # Fit the preprocessor on training rows only; validation rows must not
        # influence fitted statistics (supervised binning uses y).
        X_fit = (
            self.X_train.reset_index(drop=True)
            if hasattr(self.X_train, "reset_index")
            else self.X_train
        )

        # Delegate to an external preprocessor (e.g. pretab) that
        # exposes get_feature_info(verbose=...) and returns
        # (num_feature_info, cat_feature_info, emb_feature_info).
        self.preprocessor.fit(X_fit, np.asarray(self.y_train))
        num_info, cat_info, _ = self.preprocessor.get_feature_info(verbose=False)
        self._train_preprocessed_data = self.preprocessor.transform(X_fit)
        self.num_feature_info = self._with_cardinality(
            num_info, self._train_preprocessed_data, "num"
        )
        self.cat_feature_info = self._with_cardinality(
            cat_info, self._train_preprocessed_data, "cat"
        )

    def setup(self, stage: str):
        """
        Transform the data and create DataLoaders.
        """
        if stage == "fit":
            train_preprocessed_data = self._train_preprocessed_data
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
                offsets=self._to_offset_tensor(self.offset_train),
                sample_weights=self._to_weight_tensor(self.sample_weight_train),
            )
            self.val_dataset = NAMpyDataset(
                val_cat_tensors,
                val_num_tensors,
                val_labels,
                regression=self.regression,
                cat_keys=cat_keys,
                num_keys=num_keys,
                offsets=self._to_offset_tensor(self.offset_val),
                sample_weights=self._to_weight_tensor(self.sample_weight_val),
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

    def preprocess_tensors(self, X):
        """Transform arbitrary rows into architecture input dictionaries."""
        preprocessed = self.preprocessor.transform(X)
        cat_tensors = {}
        num_tensors = {}

        for key, info in self.cat_feature_info.items():
            transformed_key = "cat_" + key
            if transformed_key not in preprocessed:
                continue
            array = preprocessed[transformed_key]
            is_onehot = "onehot" in info.get("preprocessing", "").lower() or (
                info.get("dimension", 1) > 1
            )
            if not is_onehot and array.dtype.kind == "f":
                array = array.astype("int64")
            dtype = torch.float32 if is_onehot else torch.long
            cat_tensors[key] = torch.tensor(array, dtype=dtype)

        for key in self.num_feature_info:
            transformed_key = "num_" + key
            if transformed_key in preprocessed:
                num_tensors[key] = torch.tensor(
                    preprocessed[transformed_key], dtype=torch.float32
                )
        return cat_tensors, num_tensors

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
            is_onehot = "onehot" in info.get("preprocessing", "").lower() or (
                info.get("dimension", 1) > 1
            )
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

        sampler = None
        shuffle = self.shuffle
        if self.sampling_strategy == "balanced":
            if "sampler" in self.dataloader_kwargs or "batch_sampler" in self.dataloader_kwargs:
                raise ValueError(
                    "sampling_strategy cannot be combined with a custom sampler."
                )
            labels = np.asarray(self.y_train).reshape(-1)
            classes, counts = np.unique(labels, return_counts=True)
            if len(classes) < 2:
                raise ValueError("Balanced sampling requires at least two classes.")
            inverse_frequency = {
                label: 1.0 / count for label, count in zip(classes, counts, strict=True)
            }
            weights = torch.tensor(
                [inverse_frequency[label] for label in labels], dtype=torch.double
            )
            generator = torch.Generator().manual_seed(int(self.random_state))
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False

        loader_kwargs = dict(self.dataloader_kwargs)
        loader_kwargs.setdefault(
            "generator", torch.Generator().manual_seed(int(self.random_state))
        )
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **loader_kwargs,
        )

    def val_dataloader(self):
        """
        Returns the validation dataloader.

        Returns:
            DataLoader: DataLoader instance for the validation dataset.
        """
        loader_kwargs = dict(self.dataloader_kwargs)
        loader_kwargs.setdefault(
            "generator", torch.Generator().manual_seed(int(self.random_state) + 1)
        )
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, **loader_kwargs
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
