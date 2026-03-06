nampy/
├── __init__.py
├── __version__.py
│
├── arch_utils/              # Shared NN building blocks
│   ├── __init__.py
│   ├── embedding_layer.py
│   ├── mlp_utils.py
│   ├── nbm_utils.py
│   ├── neural_tree.py
│   ├── nn_utils.py
│   ├── nodegam_utils.py
│   ├── normalization_layers.py
│   └── transformer_utils.py
│
├── basemodels/              # Core model implementations (PyTorch/Lightning)
│   ├── __init__.py
│   ├── basemodel.py
│   ├── gam.py
│   ├── gpnam.py
│   ├── lightning_wrapper.py
│   ├── linreg.py
│   ├── multi_model.py
│   ├── nam.py
│   ├── namformer.py
│   ├── natt.py
│   ├── nbm.py
│   ├── nodegam.py
│   ├── qnam.py
│   ├── snam.py
│   └── treenam.py
│
├── configs/                 # Model config dataclasses
│   ├── __init__.py
│   ├── boostednam_config.py
│   ├── gpnam_config.py
│   ├── linreg_config.py
│   ├── nam_config.py
│   ├── namformer_config.py
│   ├── natt_config.py
│   ├── nbm_config.py
│   ├── nodegam_config.py
│   ├── qnam_config.py
│   ├── snam_config.py
│   └── nam_config.py (etc.)
│
├── data_utils/              # Data loading and preprocessing for training
│   ├── __init__.py
│   ├── datamodule.py
│   └── dataset.py
│
├── models/                  # Sklearn-style wrappers and model entry points
│   ├── __init__.py
│   ├── gam.py
│   ├── gpnam.py
│   ├── linreg.py
│   ├── nam.py
│   ├── namformer.py
│   ├── natt.py
│   ├── nbm.py
│   ├── nodegam.py
│   ├── qnam.py
│   ├── sklearn_classifier.py
│   ├── sklearn_lss.py
│   ├── sklearn_regressor.py
│   ├── snam.py
│   └── treenam.py
│
├── preprocessing/
│   └── __init__.py
│
├── splines/                 # Spline utilities for GAM/NAM
│   ├── __init__.py
│   ├── cubic.py
│   ├── neural_splines.py
│   ├── spline_utils.py
│   └── tensorproduct.py
│
└── utils/                   # Distributions, metrics, plotting
    ├── __init__.py
    ├── distributional_metrics.py
    ├── distributions.py
    └── plotting.py

I am working on a research software package that implements a general framework in Python for:

- **Neural Additive Models (NAMs)**
- **Neural Basis Models (NBMs)**
- **Generalized Additive Models (GAMs)**
- **Other neural/tabular architectures (NAMformer, NODE-GAM, etc.)**

The library supports **standard regression/classification** and **LSS-style distributional regression** (learning full conditional distributions, not just means). The GAM implementation is intentionally more classical/statistical and should provide **mgcv-style capabilities** (smoothing-parameter selection, EDF, diagnostics, concurvity, etc.).

Your job is to:

- **Review and improve the codebase**, catching bugs, numerical issues, and design inconsistencies.
- **Keep APIs as uniform as possible across models**, even though the core GAM differs from neural models.
- **Suggest concrete, actionable changes**, including code snippets and refactors (not just vague advice).

Below is a **very compact overview** of the structure and expectations.


Codebase structure (high-level)
-------------------------------

Top-level package layout:

- `nampy/__init__.py`, `__version__.py`
- `nampy/arch_utils/`
- `nampy/basemodels/`
- `nampy/configs/`
- `nampy/data_utils/`
- `nampy/models/`
- `nampy/preprocessing/`
- `nampy/splines/`
- `nampy/utils/`

Rough responsibilities:

- **`arch_utils/`** – low-level neural building blocks
  - `embedding_layer.py`: categorical embeddings and related helpers.
  - `mlp_utils.py`: MLP construction helpers, activations, dropout, residual/skip options.
  - `nodegam_utils.py`, `neural_tree.py`, `nbm_utils.py`, `nn_utils.py`: utilities for NODE-GAM, neural trees, NBMs, general NN helpers.

- **`basemodels/`** – core modelling engines
  - `basemodel.py`: common base/model mixins and shared training logic.
  - `gam.py`: **classical cubic-spline GAM** with Wood/mgcv-style smoothing, REML/GCV/LAML, diagnostics.
  - Other files (`nam.py`, `nbm.py`, `nodegam.py`, `treenam.py`, etc.): PyTorch/Lightning implementations of NAM, NBMs, NODE-GAM, etc.
  - `lightning_wrapper.py`: bridges raw PyTorch modules to Lightning (`TaskModel` wrapper, loss handling, logging).

- **`configs/`** – configuration objects for each model family
  - Files like `nam_config.py`, `nbm_config.py`, `nodegam_config.py`, etc. define hyperparameters and architecture choices in a structured way.

- **`data_utils/`** – dataset + Lightning datamodule
  - `dataset.py`: `NAMpyDataset` for handling numerical/categorical feature tensors and labels.
  - `datamodule.py`: `NAMpyDataModule` that uses an external preprocessor (e.g. `pretab.Preprocessor`) and exposes Lightning dataloaders.

- **`models/`** – user-facing estimators (mostly sklearn-style)
  - `gam.py`: `GAMRegressor` sklearn wrapper around `basemodels.gam.GAM`.
  - `gpnam.py`, `nam.py`, `nbm.py`, `nodegam.py`, etc.: thin, sklearn-like wrappers around corresponding `basemodels` with `.fit`/`.predict` APIs.
  - `sklearn_lss.py`: high-level LSS estimator (`SklearnBaseLSS`) that wires configs, distributions, Lightning training, and plotting.
  - `sklearn_classifier.py`, `sklearn_regressor.py`, `sklearn_lss.py`: base mixins for sklearn-style interfaces.

- **`splines/`** – spline infrastructure for GAMs and spline-based models
  - `cubic.py`: cubic regression spline basis, penalties, etc.
  - `neural_splines.py`, `spline_utils.py`, `tensorproduct.py`: spline utilities and tensor-product extensions.

- **`utils/`** – common utilities
  - `distributions.py`: distribution classes (Normal, Poisson, Gamma, Beta, Dirichlet, Student-t, Negative Binomial, Quantile, Robust Normal, etc.) with parameter transforms and NLL computation.
  - `distributional_metrics.py`: metrics for evaluating probabilistic predictions (deviances, proper scoring rules, etc.).
  - `plotting.py`: plotting helpers for partial dependence, shape functions, interaction plots, etc.

Design goals and conventions
----------------------------

- **Uniform high-level APIs**
  - Wherever possible, models should share a **consistent interface**:
    - Neural models: either **Lightning modules** + Lightning `Trainer`, or sklearn-style wrappers (`fit`, `predict`, `predict_proba`, etc.).
    - Classical GAM: sklearn-style `GAMRegressor` with methods like `fit`, `predict`, `predict_se`, `summary`, `aic_*`, `concurvity`, etc.
  - Distributions (in `utils/distributions.py`) should expose a common interface (`compute_loss`, `evaluate_nll`, consistent parameter transforms).

- **Distributional regression / LSS**
  - `SklearnBaseLSS` and friends should make it easy to switch between families like `"normal"`, `"poisson"`, `"gamma"`, `"beta"`, `"dirichlet"`, `"studentt"`, `"negativebinom"`, `"inversegamma"`, `"categorical"`, `"quantile"`, `"robustnormal"`.
  - Ensure the **parametrisations are coherent**, numerically safe (e.g. positivity via transforms), and match the metrics in `distributional_metrics.py`.

- **GAM / mgcv-style behaviour**
  - `nampy/basemodels/gam.GAM` should provide:
    - Penalised least-squares with cubic splines.
    - Smoothing-parameter selection via **GCV, ML, REML, LAML**, with stable optimisation (outer-Newton / L-BFGS-B).
    - Outputs for EDF, diagnostics (k-check, k-refit, concurvity), and variance/covariance matrices.
  - The sklearn wrapper (`nampy/models/gam.GAMRegressor`) should be thin and not duplicate core logic.

- **Clean separation of responsibilities**
  - Keep **architecture code** (`arch_utils`) separate from **model orchestration** (`basemodels`, `models`) and **data plumbing** (`data_utils`).
  - Avoid tight coupling between plotting, metrics, and core model training logic.

- **Good engineering practices**
  - Prefer **clear, explicit errors** over silent failures.
  - Aim for **numerical robustness** in linear algebra and probabilistic code (e.g. clipping, `eps`, stable transforms).
  - Keep docstrings concise but informative; avoid duplication.
  - Ensure naming and argument conventions are consistent across models and distributions.


 **Check for correctness and numerical issues.**  
   - For GAM and distributional models, pay special attention to:
     - Matrix conditioning, Cholesky/QR usage, and log-determinant computations.
     - Edge cases (small samples, nearly collinear smooths, extreme smoothing parameters).
     - Correct handling of shapes, broadcasting, and dtype conversions.

 **Enforce API consistency.**  
   - Make sure that related models expose similar method names and signatures where it makes sense.
   - Keep sklearn-style estimators compatible with sklearn expectations (e.g. `get_params`, `set_params`, `n_features_in_`).

 **Improve clarity and maintainability.**  
   - Propose refactors that reduce duplication across models and distributions.
   - Factor out reusable utilities into `arch_utils` or `utils` when appropriate.
   - Prefer small, focused functions and classes over monolithic ones.

 **Be concrete in suggestions.**  
   - When pointing out problems or design issues, always:
     - Explain **what is wrong or risky**.
     - Show **specific code changes** (or at least a clear sketch) to fix or improve it.
     - Note any relevant trade-offs or alternative designs.


Miscellaneous expectations
--------------------------

- It is fine (and expected) that the **GAM implementation looks more “statistical / mgcv-like”** than the neural models.
- Neural models should integrate cleanly with **PyTorch Lightning** and **pretab** preprocessing, and remain usable in sklearn workflows via wrappers.

Here is the structure of the codebase
├── nampy
│   │   ├── __init__.py
│   │   ├── __version__.py
│   │   ├── arch_utils
│   │   │   ├── __init__.py
│   │   │   ├── embedding_layer.py
│   │   │   ├── mlp_utils.py
│   │   │   ├── nbm_utils.py
│   │   │   ├── neural_tree.py
│   │   │   ├── nn_utils.py
│   │   │   ├── nodegam_utils.py
│   │   │   ├── normalization_layers.py
│   │   │   └── transformer_utils.py
│   │   ├── basemodels
│   │   │   ├── __init__.py
│   │   │   ├── basemodel.py
│   │   │   ├── gam.py
│   │   │   ├── gpnam.py
│   │   │   ├── lightning_wrapper.py
│   │   │   ├── linreg.py
│   │   │   ├── multi_model.py
│   │   │   ├── nam.py
│   │   │   ├── namformer.py
│   │   │   ├── natt.py
│   │   │   ├── nbm.py
│   │   │   ├── nodegam.py
│   │   │   ├── qnam.py
│   │   │   ├── snam.py
│   │   │   └── treenam.py
│   │   ├── configs
│   │   │   ├── __init__.py
│   │   │   ├── boostednam_config.py
│   │   │   ├── gpnam_config.py
│   │   │   ├── linreg_config.py
│   │   │   ├── nam_config.py
│   │   │   ├── namformer_config.py
│   │   │   ├── natt_config.py
│   │   │   ├── nbm_config.py
│   │   │   ├── nodegam_config.py
│   │   │   ├── qnam_config.py
│   │   │   └── snam_config.py
│   │   ├── data_utils
│   │   │   ├── __init__.py
│   │   │   ├── datamodule.py
│   │   │   └── dataset.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── gam.py
│   │   │   ├── gpnam.py
│   │   │   ├── linreg.py
│   │   │   ├── nam.py
│   │   │   ├── namformer.py
│   │   │   ├── natt.py
│   │   │   ├── nbm.py
│   │   │   ├── nodegam.py
│   │   │   ├── qnam.py
│   │   │   ├── sklearn_classifier.py
│   │   │   ├── sklearn_lss.py
│   │   │   ├── sklearn_regressor.py
│   │   │   ├── snam.py
│   │   │   └── treenam.py
│   │   ├── preprocessing
│   │   │   └── __init__.py
│   │   ├── splines
│   │   │   ├── __init__.py
│   │   │   ├── cubic.py
│   │   │   ├── neural_splines.py
│   │   │   ├── spline_utils.py
│   │   │   └── tensorproduct.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── distributional_metrics.py
│   │       ├── distributions.py
│   │       └── plotting.py

data_utils/datamodule.py
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

        # Initialize dictionaries for categorical and numerical tensors
        test_cat_tensors = {}
        test_num_tensors = {}

        # Populate tensors for categorical features
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
                test_cat_tensors[key] = torch.tensor(arr, dtype=cat_dtype)

        # Populate tensors for numerical features, if present in processed data
        for key in self.num_feature_info:
            num_key = "num_" + key
            if num_key in test_preprocessed_data:
                test_num_tensors[key] = torch.tensor(
                    test_preprocessed_data[num_key], dtype=torch.float32
                )

        n = len(next(iter(test_preprocessed_data.values())))
        self.test_labels = torch.zeros(n, dtype=torch.float32).unsqueeze(1)
        self.test_preprocessor_fitted = True
        return test_cat_tensors, test_num_tensors

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
data_utils/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class NAMpyDataset(Dataset):
    """
    Custom dataset for handling structured data with separate categorical and numerical features, tailored for
    both regression and classification tasks.

    Parameters:
        cat_features_list (list of Tensors): A list of tensors representing the categorical features.
        num_features_list (list of Tensors): A list of tensors representing the numerical features.
        labels (Tensor): A tensor of labels.
        regression (bool, optional): A flag indicating if the dataset is for a regression task. Defaults to True.
    """

    def __init__(
        self,
        cat_features_list,
        num_features_list,
        labels,
        regression=True,
        cat_keys=None,
        num_keys=None,
    ):
        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors

        self.regression = regression
        self.cat_keys = cat_keys
        self.num_keys = num_keys
        if not self.regression:
            self.num_classes = len(np.unique(labels))
            if self.num_classes > 2:
                self.labels = labels.view(-1)
            else:
                self.num_classes = 1
                self.labels = labels
        else:
            self.labels = labels
            self.num_classes = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index.

        Parameters:
            idx (int): The index of the data point.

        Returns:
            tuple: A tuple containing two lists of tensors (one for categorical features and one for numerical
            features) and a single label (float if regression is True).
        """
        cat_features = {
            key: feature_tensor[idx]
            for key, feature_tensor in zip(self.cat_keys, self.cat_features_list)
        }
        num_features = {
            key: torch.as_tensor(feature_tensor[idx]).clone().detach().to(torch.float32)
            for key, feature_tensor in zip(self.num_keys, self.num_features_list)
        }

        label = self.labels[idx]
        if self.regression:
            label = label.clone().detach().to(torch.float32)
        elif self.num_classes == 1:
            label = label.clone().detach().to(torch.float32)
        else:
            label = label.clone().detach().to(torch.long)

        # Keep categorical and numerical features separate
        return cat_features, num_features, label
models/gam.py
"""Classical GAM (Generalized Additive Model) with an sklearn-compatible API.

This module wraps the low-level cubic-spline GAM engine in
``nampy.basemodels.gam`` behind a familiar ``fit`` / ``predict`` / ``score``
interface.  No PyTorch, no Lightning — just penalised least-squares with
GCV / ML / REML smoothing, on top of the existing spline utilities.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..basemodels.gam import GAM


class GAMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Generalized Additive Model.

    Uses cubic regression splines with penalised least-squares estimation
    and automatic smoothing-parameter selection (GCV, ML, REML, or LAML).
    Numerical features only.

    Parameters
    ----------
    n_splines : int, default=10
        Number of basis functions (knots) per feature.  Must be >= 3.
    smoothing_params : float, array-like, or None, default=None
        Initial smoothing parameters.  ``None`` → 1.0 per feature.
        A scalar is broadcast to all features.
    method : {'GCV', 'ML', 'REML', 'LAML'}, default='GCV'
        Smoothing-parameter selection criterion.  ``'LAML'`` is a
        Laplace-approximate marginal likelihood; for Gaussian models it
        coincides with ``'REML'`` up to constants.
    optimizer : {'lbfgsb', 'outer_newton'}, default='lbfgsb'
        ``'lbfgsb'`` — L-BFGS-B on the criterion (general purpose).
        ``'outer_newton'`` — Wood-style outer Newton with analytic
        gradient / Hessian (requires ``method`` in ``{'REML', 'LAML'}``).

    Attributes
    ----------
    intercept_ : float
        Global intercept (from the core GAM, i.e. mean(y)).
    feature_names_ : list of str
        Feature names seen during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from nampy.models.gam import GAMRegressor
    >>> X = np.column_stack([np.linspace(-3, 3, 200)] * 2)
    >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
    >>> model = GAMRegressor(n_splines=12).fit(X, y)
    >>> model.score(X, y) > 0.95
    True
    """

    def __init__(
        self, n_splines=10, smoothing_params=None, method="GCV", optimizer="lbfgsb"
    ):
        self.n_splines = n_splines
        self.smoothing_params = smoothing_params
        self.method = method
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # fit / predict / score
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the GAM via penalised least-squares + automatic smoothing.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X_array, feature_names = self._validate_X(X, fitting=True)
        y_array = np.asarray(y, dtype=np.float64).ravel()

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"X has {X_array.shape[0]} samples but y has {y_array.shape[0]}"
            )

        self.feature_names_ = feature_names
        self.n_features_in_ = X_array.shape[1]

        s = self._resolve_smoothing_params(self.n_features_in_)

        self._gam = GAM(
            X_array,
            k=self.n_splines,
            s=s,
            feature_names=self.feature_names_,
        )
        self._gam.fit(y_array, optimize=True, method=self.method, optimizer=self.optimizer)

        self.intercept_ = self._gam.intercept_
        self._y_train = y_array.copy()
        return self

    def predict(self, X):
        """Predict target values for new data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, ["_gam"])
        X_array, _ = self._validate_X(X, fitting=False)
        return self._gam.predict(X_array)

    def predict_se(self, X, cov="bayes"):
        """Predict with standard errors.

        Parameters
        ----------
        X : array-like or DataFrame
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}

        Returns
        -------
        mu : ndarray, shape (n_samples,)
        se : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, ["_gam"])
        X_array, _ = self._validate_X(X, fitting=False)
        return self._gam.predict(X_array, return_se=True, cov=cov)

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        """Print a model summary (EDF per term, R-sq, GCV)."""
        check_is_fitted(self, ["_gam"])
        self._gam.summary()

    def summary_dict(self):
        """Return a structured, machine-readable model summary."""
        check_is_fitted(self, ["_gam"])
        return self._gam.summary_dict()

    def aic_conditional(self, scale="ml", cov="bayes"):
        """Conventional conditional AIC.

        Parameters
        ----------
        scale : {'ml', 'working'}
            Scale estimate for the Gaussian log-likelihood.
        cov : {'bayes', 'freq'}
            Which conditional covariance to use.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.aic_conditional(scale=scale, cov=cov)

    def aic_corrected(
        self,
        scale="ml",
        covariance_kind="wood_full",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Wood-style corrected conditional AIC.

        Parameters
        ----------
        scale : {'ml', 'working'}
            Scale estimate for the Gaussian log-likelihood.
        covariance_kind : {'kass_steffey', 'wood_full'}
            Which unconditional covariance approximation to use.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            How to invert the criterion Hessian.
        sp_uncertainty_ridge : float, default=1e-6
            Ridge constant (when ``sp_uncertainty_regularization='ridge'``).

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``,
        ``covariance_kind``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.aic_corrected(
            scale=scale,
            covariance_kind=covariance_kind,
            sp_uncertainty_regularization=sp_uncertainty_regularization,
            sp_uncertainty_ridge=sp_uncertainty_ridge,
        )

    def confidence_intervals(self, alpha=0.05, cov="bayes", include_intercept=False):
        """Wald-type CIs for spline coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}
            Covariance matrix to use for the standard errors.
            ``'kass_steffey'`` and ``'wood'`` require a prior call to
            :meth:`compute_unconditional_covariance`.
        include_intercept : bool

        Returns
        -------
        list of (float, float)
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.confidence_intervals(
            alpha=alpha, cov=cov, include_intercept=include_intercept,
        )

    def term_drop_test(self, term_index=0):
        """Drop-one-term F-test (refits reduced model)."""
        check_is_fitted(self, ["_gam"])
        if not (0 <= term_index < self.n_features_in_):
            raise IndexError(
                f"term_index must be in [0, {self.n_features_in_ - 1}], got {term_index}"
            )
        return self._gam.term_drop_test(term_index=term_index, method=self.method)

    def concurvity(self, full=True, include_intercept=False):
        """Wood/mgcv-style concurvity diagnostics.

        Parameters
        ----------
        full : bool, default=True
            If ``True``, measure each term against the whole rest of the model
            (mgcv ``full=TRUE``).  If ``False``, return pairwise matrices.
        include_intercept : bool, default=False
            Include the intercept column as a parametric term.

        Returns
        -------
        dict
            See :meth:`nampy.basemodels.gam.GAM.concurvity` for the full
            return-value specification.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.concurvity(full=full, include_intercept=include_intercept)

    def k_check(self, subsample=5000, n_rep=400, random_state=None):
        """Wood/mgcv-style basis-dimension check (k-index simulation test).

        Parameters
        ----------
        subsample : int, default=5000
            Maximum observations used (random subsample when ``n > subsample``).
        n_rep : int, default=400
            Number of residual reshuffles for the simulation p-value.
        random_state : int, np.random.Generator, or None
            Seed / generator for reproducibility.

        Returns
        -------
        dict
            Keys: ``labels``, ``table`` (shape (*m*, 4), columns k', edf,
            k-index, p-value), ``columns``, ``subsample_n``, ``n_rep``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.k_check(
            subsample=subsample, n_rep=n_rep, random_state=random_state
        )

    def k_refit_check(self, factor=2):
        """Refit-based basis-dimension sensitivity check.

        Doubles (or scales by *factor*) the basis dimension, refits with
        fresh smoothing optimisation, and compares total EDF and criterion.
        Useful as a follow-up when :meth:`k_check` flags a concern.

        Parameters
        ----------
        factor : int or float, default=2
            ``k_new = max(k + 1, factor * k)``.

        Returns
        -------
        dict
            Keys: ``k_old``, ``k_new``, ``edf_old``, ``edf_new``,
            ``criterion_old``, ``criterion_new``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.k_refit_check(factor=factor)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_X(self, X, fitting=True):
        if isinstance(X, pd.DataFrame):
            if not fitting and hasattr(self, "feature_names_"):
                missing = [c for c in self.feature_names_ if c not in X.columns]
                if missing:
                    raise ValueError(f"Missing columns for prediction: {missing}")
                X = X[self.feature_names_]
                feature_names = list(self.feature_names_)
            else:
                feature_names = list(X.columns)
            X_array = X.values.astype(np.float64)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            if fitting:
                feature_names = [f"x{i}" for i in range(X_array.shape[1])]
            else:
                feature_names = getattr(
                    self, "feature_names_",
                    [f"x{i}" for i in range(X_array.shape[1])],
                )

        if X_array.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X_array).all():
            raise ValueError("X contains NaN / Inf")

        if (
            not fitting
            and hasattr(self, "n_features_in_")
            and X_array.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_array.shape[1]}"
            )
        return X_array, feature_names

    def _resolve_smoothing_params(self, n_features):
        if self.smoothing_params is None:
            return None
        s = np.asarray(self.smoothing_params, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(n_features, s.item())
        if len(s) != n_features:
            raise ValueError(
                f"smoothing_params has length {len(s)}, expected {n_features}"
            )
        return s
models/nam.py
from ..basemodels.nam import NAM
from ..configs.nam_config import DefaultNAMConfig
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from .sklearn_regressor import SklearnBaseRegressor


class NAMRegressor(SklearnBaseRegressor):
    """
    Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the NAM model
    with the default NAM configuration.

    The accepted arguments to the NAMRegressor class include both the attributes in the DefaultNAMConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the NAM.
    activation : callable, default=nn.SELU()
        Activation function for the NAM layers.
    skip_layers : bool, default=False
        Whether to skip layers in the NAM.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the NAM.
    skip_connections : bool, default=False
        Whether to use skip connections in the NAM.
    batch_norm : bool, default=False
        Whether to use batch normalization in the NAM layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the NAM layers.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
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

    Notes
    -----
    - The accepted arguments to the NAMRegressor class are the same as the attributes in the DefaultNAMConfig dataclass.
    - NAMRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    nampy.models.SklearnBaseRegressor : The parent class for NAMRegressor.

    Examples
    --------
    >>> from nampy.models import NAMRegressor
    >>> model = NAMRegressor(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)


class NAMClassifier(SklearnBaseClassifier):
    """
    Multi-Layer Perceptron classifier. This class extends the SklearnBaseClassifier class and uses the NAM model
    with the default NAM configuration.

    The accepted arguments to the NAMClassifier class include both the attributes in the DefaultNAMConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the NAM.
    activation : callable, default=nn.SELU()
        Activation function for the NAM layers.
    skip_layers : bool, default=False
        Whether to skip layers in the NAM.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the NAM.
    skip_connections : bool, default=False
        Whether to use skip connections in the NAM.
    batch_norm : bool, default=False
        Whether to use batch normalization in the NAM layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the NAM layers.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
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

    Notes
    -----
    - The accepted arguments to the NAMClassifier class are the same as the attributes in the DefaultNAMConfig dataclass.
    - NAMClassifier uses SklearnBaseClassifieras the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    nampy.models.SklearnBaseRegressor : The parent class for NAMClassifier.

    Examples
    --------
    >>> from nampy.models import NAMClassifier
    >>> model = NAMClassifier(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)


class NAMLSS(SklearnBaseLSS):
    """
    Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the NAM model
    with the default NAM configuration.

    The accepted arguments to the NAMLSS class include both the attributes in the DefaultNAMConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the MLP.
    activation : callable, default=nn.SELU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to skip layers in the MLP.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the MLP.
    batch_norm : bool, default=False
        Whether to use batch normalization in the MLP layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the MLP layers.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    task : str, default="regression"
        Indicates the type of machine learning task ('regression' or 'classification'). This can
        influence certain preprocessing behaviors, especially when using decision tree-based binning as ple.
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

    Notes
    -----
    - The accepted arguments to the NAMLSS class are the same as the attributes in the DefaultNAMConfig dataclass.
    - NAMLSS uses SklearnBaseLSS as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    nampy.models.SklearnBaseLSS : The parent class for NAMLSS.

    Examples
    --------
    >>> from nampy.models import NAMLSS
    >>> model = NAMLSS(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=NAM, config=DefaultNAMConfig, **kwargs)
models/sklearn_classifier.py
import warnings

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from pretab.preprocessor import Preprocessor

from ..basemodels.lightning_wrapper import TaskModel
from ..data_utils.datamodule import NAMpyDataModule
from ..utils.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)


class SklearnBaseClassifier(BaseEstimator):
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
        if preprocessor_kwargs.get("task") == "regression":
            warnings.warn(
                "The task is set to 'regression'. The Classifier is designed for classification tasks.",
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
        params = dict(self.config_kwargs)  # copy to avoid mutating estimator state

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
        Trains the classification model using the provided training data. Optionally, a separate validation set can be used.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            The target values (class labels).
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
            The fitted classifier.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None:
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
            regression=False,
            **dataloader_kwargs,
        )

        self.data_module.setup_data(
            X, y, X_val=X_val, y_val=y_val, val_size=val_size, random_state=random_state
        )

        num_classes = len(np.unique(y))

        self.model = TaskModel(
            model_class=self.base_model,
            num_classes=num_classes,
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
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])

        return self

    def predict_feature_vals(self, X):
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

        # Perform inference and return raw feature/value dictionary
        with torch.no_grad():
            return self.model(num_features=num_tensor_dict, cat_features=cat_tensor_dict)

    def predict(self, X):
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
            logits = self.model(
                num_features=num_tensor_dict, cat_features=cat_tensor_dict
            )

            # Check the shape of the logits to determine binary or multi-class classification
            if logits["output"].shape[1] == 1:
                # Binary classification
                probabilities = torch.sigmoid(logits["output"])
                predictions = (probabilities > 0.5).long().squeeze()
            else:
                # Multi-class classification
                probabilities = torch.softmax(logits["output"], dim=1)
                predictions = torch.argmax(probabilities, dim=1)

        # Convert predictions to NumPy array and return
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for the given input samples.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The input samples for which to predict class probabilities.


        Notes
        -----
        The method preprocesses the input data using the same preprocessor used during training,
        sets the model to evaluation mode, and then performs inference to predict the class probabilities.
        Softmax is applied to the logits to obtain probabilities, which are then converted from a PyTorch tensor
        to a NumPy array before being returned.


        Examples
        --------
        >>> from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
        >>> # Define the metrics you want to evaluate
        >>> metrics = {
        ...     'Accuracy': (accuracy_score, False),
        ...     'Precision': (precision_score, False),
        ...     'F1 Score': (f1_score, False),
        ...     'AUC Score': (roc_auc_score, True)
        ... }
        >>> # Assuming 'X_test' and 'y_test' are your test dataset and labels
        >>> # Evaluate using the specified metrics
        >>> results = classifier.evaluate(X_test, y_test, metrics=metrics)


        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for each input sample.

        """
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
            logits = self.model(
                num_features=num_tensor_dict, cat_features=cat_tensor_dict
            )

            # Check the shape of the logits to determine binary or multi-class classification
            if logits["output"].shape[1] == 1:
                # Binary classification: sklearn-style (n_samples, 2)
                p1 = torch.sigmoid(logits["output"])
                probabilities = torch.cat([1.0 - p1, p1], dim=1)
            else:
                # Multi-class classification
                probabilities = torch.softmax(logits["output"], dim=1)

        # Convert predictions to NumPy array and return
        return probabilities.cpu().numpy()

    def evaluate(self, X, y_true, metrics=None):
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


        Returns
        -------
        scores : dict
            A dictionary with metric names as keys and their corresponding scores as values.


        Notes
        -----
        This method uses either the `predict` or `predict_proba` method depending on the metric requirements.
        """
        # Ensure input is in the correct format
        if metrics is None:
            metrics = {"Accuracy": (accuracy_score, False)}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Initialize dictionary to store results
        scores = {}

        # Generate class probabilities if any metric requires them
        if any(use_proba for _, use_proba in metrics.values()):
            probabilities = self.predict_proba(X)

        # Generate class labels if any metric requires them
        if any(not use_proba for _, use_proba in metrics.values()):
            predictions = self.predict(X)

        # Compute each metric
        for metric_name, (metric_func, use_proba) in metrics.items():
            if use_proba:
                try:
                    scores[metric_name] = metric_func(y_true, probabilities)
                except ValueError:
                    # Some binary metrics (e.g. roc_auc_score) expect p(class=1)
                    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                        scores[metric_name] = metric_func(y_true, probabilities[:, 1])
                    else:
                        raise
            else:
                scores[metric_name] = metric_func(y_true, predictions)

        return scores

    def _plot_single_feature_effects(
        self, x_plot, predictions, y_true, ax, feature_name=None, num_bins=30
    ):
        """
        Plot the effect of a single feature for classification, with separate lines for each class.

        Parameters
        ----------
        x_plot : np.ndarray
            The feature values for plotting.
        predictions : np.ndarray
            The predicted values (shape (n, k) for multi-class).
        y_true : np.ndarray
            The true target values (for scatter plot).
        ax : matplotlib.axes.Axes
            The axes on which to plot.
        feature_name : str, optional
            The name of the feature for labels.
        num_bins : int, optional
            Number of bins for density shading, by default 30.
        """
        n_classes = predictions.shape[1] if predictions.ndim > 1 else 1
        y_range = (y_true.min() - 1, y_true.max() + 1)

        plot_density_shading(ax, x_plot, y_range, num_bins)

        # Plot shape functions for each class
        for i in range(n_classes):
            contribs = predictions[:, i] if predictions.ndim > 1 else predictions
            ax.plot(x_plot, contribs, label=f"Class {i + 1}")

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
models/sklearn_lss.py
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
        params = dict(self.config_kwargs)  # copy to avoid mutating estimator state

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
        if X_val is not None:
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
models/sklearn_regressor.py
import warnings

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from pretab.preprocessor import Preprocessor

from ..basemodels.lightning_wrapper import TaskModel
from ..data_utils.datamodule import NAMpyDataModule
from ..utils.plotting import (
    create_subplot_grid,
    plot_density_shading,
    prepare_plot_data,
)


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
                "The task is set to 'classification'. The Regressor is designed for regression tasks.",
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
        params = dict(self.config_kwargs)  # copy to avoid mutating estimator state

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

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        if X_val is not None:
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
            checkpoint = torch.load(best_model_path, weights_only=False)
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
basemodels/gam.py
"""Gaussian additive model core (penalised least squares with cubic splines).

Phase A: statistically correct intercept handling, parameter-space EDF,
         proper covariance matrices, honest summary.
Phase B: GCV / exact ML / exact REML smoothing selection, prediction SEs,
         lpmatrix, term-drop tests, Kass–Steffey covariance via delta
         method, concurvity diagnostics.

ML and REML are implemented via a mixed-model reparameterization: each
smooth's basis is split into a null-space (unpenalized → fixed effects)
and a penalized space (whitened so penalty becomes λ_j I → random
effects).  The exact profiled criteria then follow from the resulting
block-structured normal equations, using Woodbury / matrix-determinant-
lemma identities to stay in the (small) coefficient space rather than
forming the n × n covariance.

ML/REML smoothing parameters are optimised in the reparameterised system;
coefficient fitting uses the equivalent original-basis penalised LS solve.
"""

import warnings

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.linalg import qr as scipy_qr, solve_triangular
from scipy.optimize import minimize
from scipy.stats import f as f_dist
from scipy.stats import norm

from ..splines.cubic import CubicSplines

_SP_LOG_BOUNDS = (-20.0, 20.0)


# ======================================================================
# Reparameterization helper (module-level, reusable)
# ======================================================================

def _reparameterize_smooth(B, P, tol=1e-10):
    """Split a smooth basis into null-space and whitened penalized-space.

    Parameters
    ----------
    B : ndarray, shape (n, d)
        Basis matrix (already centered / identifiability-constrained).
    P : ndarray, shape (d, d)
        Penalty matrix (symmetric PSD, typically rank-deficient).
    tol : float
        *Relative* eigenvalue threshold: eigenvalues <= tol * max(|evals|)
        are treated as null space.

    Returns
    -------
    B0 : ndarray, shape (n, n_null)
        Null-space basis columns (unpenalized → fixed effects).
    Zr : ndarray, shape (n, n_pen)
        Whitened penalized-space columns (penalty = λ I).
    meta : dict
        Reparameterization metadata for coefficient reconstruction:
        U0, U1, d_pos, n_null, n_pen.
    """
    P_sym = 0.5 * (P + P.T)
    evals, U = np.linalg.eigh(P_sym)

    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]

    tol_eff = tol * max(1.0, np.max(np.abs(evals)))
    null_mask = evals <= tol_eff
    pos_mask = ~null_mask

    U0 = U[:, null_mask]
    U1 = U[:, pos_mask]
    d_pos = evals[pos_mask]

    B0 = B @ U0
    B1 = B @ U1

    Zr = B1 / np.sqrt(d_pos)[np.newaxis, :] if d_pos.size else B1

    return B0, Zr, {
        "U0": U0,
        "U1": U1,
        "d_pos": d_pos,
        "n_null": int(null_mask.sum()),
        "n_pen": int(pos_mask.sum()),
    }


class GAM:
    """Gaussian additive model with penalised cubic regression splines.

    Fits:  y = alpha + Z @ beta + eps,   eps ~ N(0, sigma^2 I)

    where Z is a column-stack of sum-to-zero-constrained cubic spline
    bases (one per feature) and alpha = mean(y).  Smoothing parameters
    are selected by minimising GCV, exact ML, or exact REML.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training features (numerical only).
    k : int, default=10
        Number of basis functions (knots) per feature.  Must be >= 3.
    s : array-like, float, or None, default=None
        Initial smoothing parameters (one per feature).  ``None`` → 1.0
        per feature.  A scalar is broadcast.
    feature_names : list of str or None
        Display names; auto-generated if ``None``.
    """

    def __init__(self, X, k=10, s=None, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN / Inf")
        if k < 3:
            raise ValueError("k must be >= 3")

        self.X = X
        self.k_ = int(k)
        self.n_samples_, self.n_features_ = X.shape
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_)]
        )

        # ----- Original (constrained) basis/penalty system -----
        self.splines = [CubicSplines(X[:, i], k) for i in range(self.n_features_)]
        self.Z = np.column_stack([sp.basis for sp in self.splines])
        if self.Z.shape[0] != self.n_samples_:
            raise ValueError("Design matrix row count mismatch")

        self.penalties = [sp.penalty for sp in self.splines]

        self.slices = []
        start = 0
        for sp in self.splines:
            nb = sp.basis.shape[1]
            self.slices.append(slice(start, start + nb))
            start += nb
        self.n_coef_ = start

        self.ZTZ = self.Z.T @ self.Z

        # ----- Penalty eigendecomps (for outer-Newton / LAML) -----
        self._penalty_ranks = np.empty(self.n_features_, dtype=np.int64)
        self._penalty_logdet_plus_fixed = np.empty(
            self.n_features_, dtype=np.float64
        )
        for j in range(self.n_features_):
            Sj = self.penalties[j]
            evals_j = np.linalg.eigvalsh(0.5 * (Sj + Sj.T))
            tol_j = 1e-10 * max(1.0, np.max(np.abs(evals_j)))
            pos = evals_j[evals_j > tol_j]
            self._penalty_ranks[j] = len(pos)
            self._penalty_logdet_plus_fixed[j] = float(
                np.sum(np.log(pos)) if len(pos) > 0 else 0.0
            )

        self.smoothing_params = self._validate_smoothing_params(s)

        # ----- Reparameterized system for ML/REML -----
        self._build_reparameterized_system()

        # ----- Fitted state -----
        self.intercept_ = None
        self.coef_ = None
        self.beta = None
        self.scale_ = None
        self.edf_ = None
        self.trace_S_ = None
        self.rss_ = None
        self.Vp_ = None
        self.Vf_ = None
        self.Vp_kass_steffey_ = None
        self.Vp_wood_ = None
        self._y_train = None
        self._optim_method = None

    # ------------------------------------------------------------------
    # Reparameterized representation
    # ------------------------------------------------------------------

    def _build_reparameterized_system(self):
        """Build the mixed-model matrices (X_fix, Z_rand) once at init."""
        fix_blocks = [np.ones((self.n_samples_, 1))]
        rand_blocks = []
        self._reparam_meta = []
        self.rand_dims_per_term_ = []

        for i in range(self.n_features_):
            B = self.Z[:, self.slices[i]]
            P = self.penalties[i]
            B0, Zr, meta = _reparameterize_smooth(B, P)
            fix_blocks.append(B0)
            if Zr.shape[1] > 0:
                rand_blocks.append(Zr)
            self._reparam_meta.append(meta)
            self.rand_dims_per_term_.append(meta["n_pen"])

        X_fix_raw = np.column_stack(fix_blocks)

        _Q, R, piv = scipy_qr(X_fix_raw, pivoting=True)
        diag_R = np.abs(np.diag(R[:min(X_fix_raw.shape), :]))
        rank_tol = (
            max(X_fix_raw.shape) * np.finfo(float).eps * diag_R[0]
            if diag_R[0] > 0 else 1e-12
        )
        rank = int(np.sum(diag_R > rank_tol))
        keep_cols = np.sort(piv[:rank])
        self.X_fix_ = X_fix_raw[:, keep_cols]
        self.rank_X_fix_ = rank
        self._fix_pivot_keep = keep_cols

        if rand_blocks:
            self.Z_rand_ = np.column_stack(rand_blocks)
        else:
            self.Z_rand_ = np.empty((self.n_samples_, 0), dtype=np.float64)
        self.n_rand_ = self.Z_rand_.shape[1]

        self.ZtZ_rand_ = self.Z_rand_.T @ self.Z_rand_

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_smoothing_params(self, s):
        if s is None:
            return np.ones(self.n_features_, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(self.n_features_, s.item())
        if s.shape != (self.n_features_,):
            raise ValueError(
                f"smoothing_params shape must be ({self.n_features_},), got {s.shape}"
            )
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("smoothing_params must be finite and > 0")
        return s.copy()

    @staticmethod
    def _validate_y(y, n_expected):
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != n_expected:
            raise ValueError(
                f"y length {y.shape[0]} != n_samples {n_expected}"
            )
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN / Inf")
        return y

    # ------------------------------------------------------------------
    # Core linear algebra (original parameterization – used by all paths
    # for coefficient fitting after smoothing params are chosen)
    # ------------------------------------------------------------------

    def _assemble_penalty_block(self, smoothing_params):
        blocks = [
            smoothing_params[i] * self.penalties[i]
            for i in range(self.n_features_)
        ]
        return block_diag(*blocks)

    def _solve_given_smoothing(self, y, smoothing_params, store=False):
        """Penalised Gaussian LS for fixed smoothing parameters.

        Works in the original (constrained) parameterization.  Uses
        Cholesky factorisation of A = Z'Z + S_lambda (SPD).
        """
        y = self._validate_y(y, self.n_samples_)

        intercept = float(np.mean(y))
        y_centered = y - intercept

        P = self._assemble_penalty_block(smoothing_params)
        A = self.ZTZ + P
        ZTy = self.Z.T @ y_centered

        try:
            cA, loA = cho_factor(A, check_finite=False)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "Penalised normal equations not SPD; check penalty / data"
            )

        beta = cho_solve((cA, loA), ZTy, check_finite=False)
        fitted = intercept + self.Z @ beta
        resid = y - fitted
        rss = float(resid @ resid)

        # tr(H) = 1 (intercept) + tr(A^{-1} Z'Z)
        AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        trace_smooth = float(np.trace(AinvZTZ))
        trace_S = 1.0 + trace_smooth
        edf = trace_S

        out = {
            "intercept": intercept,
            "beta": beta,
            "fitted": fitted,
            "resid": resid,
            "rss": rss,
            "trace_S": trace_S,
            "edf": edf,
            "A": A,
            "cA": (cA, loA),
            "P": P,
            "y_centered": y_centered,
        }

        if store:
            self.smoothing_params = np.asarray(smoothing_params, dtype=np.float64).copy()
            self.intercept_ = intercept
            self.coef_ = beta
            self.beta = [beta[sl] for sl in self.slices]
            self.rss_ = rss
            self.trace_S_ = trace_S
            self.edf_ = edf

            denom = max(self.n_samples_ - edf, 1.0)
            self.scale_ = rss / denom

            A_inv = cho_solve((cA, loA), np.eye(A.shape[0]), check_finite=False)
            self.Vp_ = self.scale_ * A_inv
            self.Vf_ = self.scale_ * (A_inv @ self.ZTZ @ A_inv.T)

        return out

    # ------------------------------------------------------------------
    # Smoothing criteria
    # ------------------------------------------------------------------

    def gcv_score(self, y, log_smoothing_params):
        """GCV score using parameter-space trace (no n×n hat matrix)."""
        sp = np.exp(np.asarray(log_smoothing_params, dtype=np.float64))
        sol = self._solve_given_smoothing(y, sp, store=False)
        n = self.n_samples_
        den = 1.0 - sol["trace_S"] / n
        if den <= 1e-12 or not np.isfinite(den):
            return np.inf
        return (sol["rss"] / n) / (den ** 2)

    def _criterion_gcv(self, y, log_sp):
        """GCV criterion (original parameterization)."""
        sp = np.exp(np.asarray(log_sp, dtype=np.float64))
        sol = self._solve_given_smoothing(y, sp, store=False)
        n = self.n_samples_
        den = 1.0 - sol["trace_S"] / n
        if den <= 1e-12:
            return np.inf
        return (sol["rss"] / n) / (den ** 2)

    def _criterion_ml_reml_exact(self, y, log_sp, method):
        """Exact Gaussian ML or REML via the mixed-model reparameterization.

        ML:   J = n     * log(RSS_V / n)     + log|V_tilde|
        REML: J = (n-p) * log(RSS_V / (n-p)) + log|V_tilde| + log|X'K X|

        where K = V_tilde^{-1}, computed via Woodbury in coefficient space.
        """
        y = self._validate_y(y, self.n_samples_)
        sp = np.exp(np.asarray(log_sp, dtype=np.float64))

        Xf = self.X_fix_
        Zr = self.Z_rand_
        n = Xf.shape[0]
        p = self.rank_X_fix_
        q = self.n_rand_

        if q == 0:
            # No penalized columns → ordinary LS (degenerate case)
            XtX = Xf.T @ Xf
            try:
                cXtX, lo = cho_factor(XtX, check_finite=False)
            except np.linalg.LinAlgError:
                return np.inf
            b_hat = cho_solve((cXtX, lo), Xf.T @ y, check_finite=False)
            resid = y - Xf @ b_hat
            rss_v = max(float(resid @ resid), 1e-14)

            if method == "ML":
                return n * np.log(rss_v / n)

            # REML: need the extra log|X'X| term (K=I when q=0)
            if n <= p:
                return np.inf
            logdet_XtX = 2.0 * float(np.sum(np.log(np.diag(cXtX))))
            return (n - p) * np.log(rss_v / (n - p)) + logdet_XtX

        # Build Λ = blockdiag(λ_j I_{r_j})
        lam_vec = np.concatenate([
            np.full(rj, sp[j], dtype=np.float64)
            for j, rj in enumerate(self.rand_dims_per_term_)
            if rj > 0
        ])

        # M = Z_r' Z_r + Λ
        M = self.ZtZ_rand_ + np.diag(lam_vec)

        try:
            cM, loM = cho_factor(M, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        # V_tilde^{-1} y = y - Z_r M^{-1} Z_r' y   (Woodbury)
        ZTy = Zr.T @ y
        Minv_ZTy = cho_solve((cM, loM), ZTy, check_finite=False)
        Ky = y - Zr @ Minv_ZTy

        # V_tilde^{-1} X
        ZTX = Zr.T @ Xf
        Minv_ZTX = cho_solve((cM, loM), ZTX, check_finite=False)
        KX = Xf - Zr @ Minv_ZTX

        XtKX = Xf.T @ KX

        try:
            cXKX, loXKX = cho_factor(XtKX, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf

        XtKy = Xf.T @ Ky
        b_hat = cho_solve((cXKX, loXKX), XtKy, check_finite=False)

        rss_v = max(float(y @ Ky - XtKy @ b_hat), 1e-14)

        # log|V_tilde| = log|M| - log|Λ|   (matrix determinant lemma)
        logdet_M = 2.0 * float(np.sum(np.log(np.diag(cM))))
        logdet_Lam = float(np.sum(np.log(lam_vec)))
        logdet_Vtilde = logdet_M - logdet_Lam

        if method == "ML":
            return n * np.log(rss_v / n) + logdet_Vtilde

        # REML
        if n <= p:
            return np.inf
        logdet_XtKX = 2.0 * float(np.sum(np.log(np.abs(np.diag(cXKX)))))
        return (n - p) * np.log(rss_v / (n - p)) + logdet_Vtilde + logdet_XtKX

    def _criterion(self, y, log_smoothing_params, method="GCV"):
        m = method.upper()
        if m == "GCV":
            return self._criterion_gcv(y, log_smoothing_params)
        if m in {"ML", "REML"}:
            return self._criterion_ml_reml_exact(y, log_smoothing_params, m)
        if m == "LAML":
            return self._criterion_ml_reml_exact(y, log_smoothing_params, "REML")
        raise ValueError("method must be 'GCV', 'ML', 'REML', or 'LAML'")

    # ==================================================================
    # Outer-Newton machinery  (Wood Section 3.1)
    # ==================================================================

    # ---- A. Stable penalty log-determinant (Section 3.1.1) -----------

    def _penalty_logdet_plus_and_derivs(self, rho):
        """Stable penalty log-determinant and its first/second derivatives.

        For single-parameter-per-term blocks we have

            log|S_lambda|_+ = sum_j (rank_j * rho_j + c_j)

        where ``rank_j`` is the rank of ``S_j`` and ``c_j = log|S_j|_+`` was
        precomputed at initialisation.

        Returns ``(val, grad, hess)`` where ``hess`` is a diagonal matrix
        (cross-derivatives are zero for single-parameter blocks).
        """
        rho = np.asarray(rho, dtype=np.float64)
        m = self.n_features_
        ranks = self._penalty_ranks.astype(np.float64)
        fixed = self._penalty_logdet_plus_fixed

        val = float(np.dot(ranks, rho) + np.sum(fixed))
        grad = ranks.copy()
        hess = np.zeros((m, m), dtype=np.float64)
        return val, grad, hess

    # ---- B. Coefficient solve + implicit derivatives (Section 3.1.3) -

    def _solve_given_rho_with_derivs(self, y_centered, rho, need_second=True):
        """Penalised least-squares solve with analytic implicit derivatives.

        Solves for the spline coefficients beta given log smoothing parameters
        ``rho = log(lambda)`` and returns analytic first- and (optionally)
        second-order derivatives of beta with respect to ``rho``.

        Parameters
        ----------
        y_centered : ndarray, shape (n,)
            Response centred by ``mean(y)`` (intercept already removed).
        rho : ndarray, shape (m,)
            Log smoothing parameters.
        need_second : bool
            Whether to compute second derivatives.

        Returns
        -------
        dict with keys ``beta``, ``A``, ``cA``, ``A_inv``, ``rss``,
        ``trace_S``, ``P``, ``D`` (list of D_k), ``dbeta`` (shape p×m),
        ``d2beta`` (shape p×m×m or None), ``logdet_A``.
        """
        sp = np.exp(rho)
        m = self.n_features_
        p = self.n_coef_

        P = self._assemble_penalty_block(sp)
        A = self.ZTZ + P
        ZTy = self.Z.T @ y_centered

        cA, loA = cho_factor(A, check_finite=False)
        beta = cho_solve((cA, loA), ZTy, check_finite=False)

        fitted = self.Z @ beta
        rss = float(np.sum((y_centered - fitted) ** 2))

        AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        trace_S = 1.0 + float(np.trace(AinvZTZ))

        A_inv = cho_solve((cA, loA), np.eye(p), check_finite=False)
        logdet_A = 2.0 * float(np.sum(np.log(np.diag(cA))))

        # D_k = dA/d(rho_k) = lambda_k S_k  (since d(lambda_k S_k)/d(rho_k) = lambda_k S_k)
        D = []
        for j in range(m):
            Dj = np.zeros((p, p), dtype=np.float64)
            sl = self.slices[j]
            Dj[sl, sl] = sp[j] * self.penalties[j]
            D.append(Dj)

        # First derivatives:  d(beta)/d(rho_k) = -A^{-1} D_k beta
        dbeta = np.zeros((p, m), dtype=np.float64)
        AinvD = []
        for k in range(m):
            AiDk = A_inv @ D[k]
            AinvD.append(AiDk)
            dbeta[:, k] = -AiDk @ beta

        d2beta = None
        if need_second:
            # d2(beta)/d(rho_k)d(rho_l) =
            #   A^{-1} D_l A^{-1} D_k beta
            #   + A^{-1} D_k A^{-1} D_l beta
            #   - delta_{kl} A^{-1} D_k beta
            d2beta = np.zeros((p, m, m), dtype=np.float64)
            for k in range(m):
                Dk_beta = D[k] @ beta
                AinvDk_beta = A_inv @ Dk_beta
                for l in range(k, m):
                    Dl_beta = D[l] @ beta
                    AinvDl_beta = A_inv @ Dl_beta
                    v = (
                        A_inv @ (D[l] @ AinvDk_beta)
                        + A_inv @ (D[k] @ AinvDl_beta)
                    )
                    if k == l:
                        v -= AinvDk_beta
                    d2beta[:, k, l] = v
                    if l != k:
                        d2beta[:, l, k] = v

        return {
            "beta": beta,
            "A": A,
            "cA": (cA, loA),
            "A_inv": A_inv,
            "rss": rss,
            "trace_S": trace_S,
            "P": P,
            "D": D,
            "AinvD": AinvD,
            "dbeta": dbeta,
            "d2beta": d2beta,
            "logdet_A": logdet_A,
        }

    # ---- C. LAML/REML objective with analytic gradient & Hessian -----

    def _laml_objective_gradient_hessian(self, y_centered, rho):
        """Negative REML / LAML with analytic gradient and Hessian.

        Implements the Gaussian penalised least-squares REML / LAML objective
        in terms of ``rho = log(lambda)`` and returns its value, gradient, and
        Hessian with respect to ``rho``.  The implementation follows Wood
        (2016) where, for the Gaussian case, the Laplace-approximate marginal
        likelihood coincides with REML up to additive constants that do not
        affect optimisation.

        Returns ``(val, grad, hess)`` where ``val`` is the objective to
        minimise.
        """
        n = self.n_samples_
        m = self.n_features_
        p = self.n_coef_
        sp = np.exp(rho)

        sol = self._solve_given_rho_with_derivs(y_centered, rho, need_second=True)
        beta = sol["beta"]
        A_inv = sol["A_inv"]
        D = sol["D"]
        AinvD = sol["AinvD"]
        dbeta = sol["dbeta"]
        d2beta = sol["d2beta"]
        rss = sol["rss"]
        logdet_A = sol["logdet_A"]

        # Penalty null-space dim = sum of (k_j - rank_j)
        Mp = int(np.sum(
            np.array([sl.stop - sl.start for sl in self.slices])
            - self._penalty_ranks
        ))
        n_reml = n - Mp
        if n_reml <= 0:
            return np.inf, np.zeros(m), np.eye(m) * 1e8

        # Penalty quadratic  b^T S_lambda b
        P = sol["P"]
        bPb = float(beta @ P @ beta)

        sigma2 = max((rss + bPb) / n_reml, 1e-15)

        # log|S_lambda|+ and its derivatives
        ldet_S, dldet_S, d2ldet_S = self._penalty_logdet_plus_and_derivs(rho)

        # Objective (to minimise):
        # V = n_reml * log(sigma2) + logdet_A - ldet_S
        # (dropped constant n_reml * log(2*pi) + n_reml since irrelevant for optim)
        val = n_reml * np.log(sigma2) + logdet_A - ldet_S

        # ---------- gradient --------------------------------------------------
        grad = np.zeros(m, dtype=np.float64)
        # Pre-compute tr(A^{-1} D_k) for each k
        trAiD = np.array([float(np.trace(AinvD[k])) for k in range(m)])

        for k in range(m):
            # d(b^T P b)/d(rho_k) = 2 beta^T P dbeta_k + beta^T D_k beta
            dbPb_k = 2.0 * beta @ P @ dbeta[:, k] + beta @ D[k] @ beta
            # d(rss)/d(rho_k) = -2 y_c^T Z dbeta_k  (since rss = ||y_c - Z beta||^2)
            drss_k = -2.0 * float((y_centered - self.Z @ beta) @ (self.Z @ dbeta[:, k]))
            dsigma2_k = (drss_k + dbPb_k) / n_reml

            grad[k] = (
                n_reml * dsigma2_k / sigma2
                + trAiD[k]
                - dldet_S[k]
            )

        # ---------- Hessian ---------------------------------------------------
        hess = np.zeros((m, m), dtype=np.float64)
        resid = y_centered - self.Z @ beta

        for k in range(m):
            dbPb_k = 2.0 * beta @ P @ dbeta[:, k] + beta @ D[k] @ beta
            drss_k = -2.0 * float(resid @ (self.Z @ dbeta[:, k]))
            dsigma2_k = (drss_k + dbPb_k) / n_reml

            for l in range(k, m):
                dbPb_l = 2.0 * beta @ P @ dbeta[:, l] + beta @ D[l] @ beta
                drss_l = -2.0 * float(resid @ (self.Z @ dbeta[:, l]))
                dsigma2_l = (drss_l + dbPb_l) / n_reml

                # d2(b^T P b)/d(rho_k)d(rho_l)
                d2bPb_kl = (
                    2.0 * (dbeta[:, k] @ P @ dbeta[:, l])
                    + 2.0 * (beta @ P @ d2beta[:, k, l])
                    + 2.0 * (dbeta[:, l] @ D[k] @ beta)
                    + 2.0 * (beta @ D[l] @ dbeta[:, k])
                    + float(k == l) * (beta @ D[k] @ beta)
                )
                # d2(rss)/d(rho_k)d(rho_l)
                Zdb_k = self.Z @ dbeta[:, k]
                Zdb_l = self.Z @ dbeta[:, l]
                d2rss_kl = (
                    2.0 * float(Zdb_k @ Zdb_l)
                    - 2.0 * float(resid @ (self.Z @ d2beta[:, k, l]))
                )
                d2sigma2_kl = (d2rss_kl + d2bPb_kl) / n_reml

                # d2(logdet_A)/d(rho_k)d(rho_l)
                # = -tr(A^{-1} D_k A^{-1} D_l) + delta_{kl} tr(A^{-1} D_k)
                d2logdetA_kl = -float(np.trace(AinvD[k] @ AinvD[l]))
                if k == l:
                    d2logdetA_kl += trAiD[k]

                hess[k, l] = (
                    n_reml * (d2sigma2_kl / sigma2 - dsigma2_k * dsigma2_l / sigma2**2)
                    + d2logdetA_kl
                    - d2ldet_S[k, l]
                )
                if l != k:
                    hess[l, k] = hess[k, l]

        return float(val), grad, hess

    # ---- D. Outer Newton driver (Section 3.1 / 3.2) -----------------

    def _optimize_smoothing_outer_newton(
        self,
        y,
        method="REML",
        initial_rho=None,
        max_iter=50,
        tol=1e-6,
        max_half_steps=10,
        working_inf_pos_threshold=15.0,
        working_inf_grad_tol=1e-3,
        working_inf_hess_tol=1e-4,
    ):
        r"""Wood-style outer Newton optimiser for smoothing parameters.

        Iteratively applies Newton updates in ``rho = log(lambda)`` using
        analytic gradient and Hessian of the REML / LAML criterion.  Includes
        step-halving and faithful Wood-style "working infinity" detection.

        Parameters
        ----------
        y : ndarray, shape (n,)
            Validated response.
        method : {'REML', 'LAML'}
            Criterion.  ``'LAML'`` is a Laplace-approximate marginal
            likelihood alias; for Gaussian it coincides with ``'REML'``.
        initial_rho : ndarray or None
            Starting values for ``rho = log(lambda)``.
            ``None`` → ``log(self.smoothing_params)``.
        max_iter : int
            Maximum outer Newton iterations.
        tol : float
            Convergence tolerance on the active-set gradient norm.
        max_half_steps : int
            Maximum step halvings per iteration.
        working_inf_pos_threshold : float, default=15.0
            A coordinate ``rho_k`` is a candidate for working infinity only
            when it is large and positive (that is, ``rho_k`` above this
            threshold, meaning ``lambda_k`` tends to infinity and the
            corresponding variance component tends to zero).  Large negative
            ``rho_k`` (weak penalty) is not treated as working infinity.
        working_inf_grad_tol : float, default=1e-3
            A candidate coordinate is frozen when its gradient component
            ``|grad[k]| < working_inf_grad_tol`` (near-stationary in that
            direction).
        working_inf_hess_tol : float, default=1e-4
            A candidate coordinate is frozen when its diagonal Hessian entry
            ``hess[k, k] < working_inf_hess_tol`` (flat or indefinite curvature
            at large positive ``rho_k``).

        Returns
        -------
        dict with keys ``rho``, ``sp``, ``converged``, ``n_iter``,
        ``history``, ``frozen`` (mask of frozen coordinates).

        Notes
        -----
        "Working infinity" in the Wood sense refers specifically to
        ``lambda_k`` tending to infinity (large positive ``rho_k``), where
        the corresponding variance component or smooth is effectively zero.
        The gradient tends to zero and the Hessian diagonal becomes flat or
        indefinite at such coordinates.  The three-condition test used here
        (large positive ``rho_k`` and small gradient and small Hessian
        diagonal) is substantially more faithful to that criterion than a
        simple absolute-value threshold on ``rho_k``.
        """
        m = self.n_features_
        y = self._validate_y(y, self.n_samples_)
        intercept = float(np.mean(y))
        y_c = y - intercept

        if initial_rho is not None:
            rho = np.asarray(initial_rho, dtype=np.float64).copy()
        else:
            rho = np.log(self.smoothing_params).copy()

        active = np.ones(m, dtype=bool)
        history = []
        n_iter = 0

        for it in range(max_iter):
            n_iter = it + 1
            val, grad, hess = self._laml_objective_gradient_hessian(y_c, rho)
            active_grad_norm = float(np.linalg.norm(grad[active]))
            history.append({
                "iter": it,
                "objective": val,
                "grad_norm": active_grad_norm,
                "n_active": int(active.sum()),
            })

            if active_grad_norm < tol:
                break

            # Solve Newton system on active coordinates
            idx_a = np.where(active)[0]
            if len(idx_a) == 0:
                break

            g_a = grad[idx_a]
            H_a = hess[np.ix_(idx_a, idx_a)]

            # Stabilise: ensure H_a is PD by adding a small ridge if needed
            evals_H = np.linalg.eigvalsh(H_a)
            if evals_H.min() < 1e-8:
                H_a = H_a + (abs(evals_H.min()) + 1e-6) * np.eye(len(idx_a))

            try:
                delta_a = np.linalg.solve(H_a, g_a)
            except np.linalg.LinAlgError:
                delta_a = np.linalg.lstsq(H_a, g_a, rcond=None)[0]

            delta = np.zeros(m, dtype=np.float64)
            delta[idx_a] = delta_a

            # Step-halving line search
            step = 1.0
            rho_new = rho - step * delta
            val_new, _, _ = self._laml_objective_gradient_hessian(y_c, rho_new)

            for _ in range(max_half_steps):
                if np.isfinite(val_new) and val_new < val:
                    break
                step *= 0.5
                rho_new = rho - step * delta
                val_new, _, _ = self._laml_objective_gradient_hessian(y_c, rho_new)
            else:
                if not (np.isfinite(val_new) and val_new < val):
                    warnings.warn(
                        f"Outer Newton: step-halving failed at iteration {it}"
                    )
                    break

            rho = rho_new

            # ------------------------------------------------------------------
            # Wood-style "working infinity" detection
            #
            # A coordinate qualifies if ALL THREE conditions hold:
            #   1. rho_k is large and POSITIVE  (lambda_k -> inf; smooth -> 0)
            #      Large negative rho means near-zero penalty, which is not
            #      the same and must NOT be frozen.
            #   2. |grad[k]| is near zero  (stationary in this direction)
            #   3. hess[k,k] is near zero or non-positive
            #      (flat/indefinite curvature at the boundary)
            # ------------------------------------------------------------------
            for k in range(m):
                if (
                    rho[k] > working_inf_pos_threshold
                    and abs(grad[k]) < working_inf_grad_tol
                    and hess[k, k] < working_inf_hess_tol
                ):
                    active[k] = False

        sp_final = np.exp(rho)
        converged = float(np.linalg.norm(grad[active])) < tol if np.any(active) else True

        return {
            "rho": rho,
            "sp": sp_final,
            "converged": converged,
            "n_iter": n_iter,
            "history": history,
            "frozen": ~active,
        }

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_without_optimization(self, y):
        """Fit with current smoothing parameters (no optimisation)."""
        self._solve_given_smoothing(y, self.smoothing_params, store=True)
        return self

    def optimize_smoothing_params(
        self, y, initial_smoothing_params=None, method="GCV", optimizer="lbfgsb"
    ):
        """Optimise smoothing parameters.

        Parameters
        ----------
        y : ndarray
            Response (already validated by caller).
        initial_smoothing_params : array-like or None
            Starting values.  ``None`` → current ``self.smoothing_params``.
        method : {'GCV', 'ML', 'REML', 'LAML'}
            Smoothing-selection criterion.  ``'LAML'`` is a Laplace-
            approximate marginal likelihood; for Gaussian it coincides
            with ``'REML'`` up to constants.
        optimizer : {'lbfgsb', 'outer_newton'}
            ``'lbfgsb'`` — L-BFGS-B on the criterion (existing path).
            ``'outer_newton'`` — Wood-style outer Newton with analytic
            gradient / Hessian (requires ``method`` in ``{'REML', 'LAML'}``).

        Returns
        -------
        self
        """
        method = method.upper()
        optimizer = optimizer.lower()

        valid_methods = {"GCV", "ML", "REML", "LAML"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        if initial_smoothing_params is None:
            x0 = np.log(self.smoothing_params)
        else:
            x0 = np.log(self._validate_smoothing_params(initial_smoothing_params))

        if optimizer == "lbfgsb":
            crit_method = "REML" if method == "LAML" else method
            bounds = [_SP_LOG_BOUNDS] * self.n_features_
            result = minimize(
                lambda log_s: self._criterion(y, log_s, method=crit_method),
                x0,
                method="L-BFGS-B",
                bounds=bounds,
            )
            if not result.success:
                warnings.warn(
                    f"Smoothing optimisation did not converge: {result.message}"
                )
            self.smoothing_params = np.exp(result.x)
            self._optim_result = result

        elif optimizer == "outer_newton":
            if method not in {"REML", "LAML"}:
                raise ValueError(
                    "outer_newton optimizer requires method='REML' or 'LAML'"
                )
            result = self._optimize_smoothing_outer_newton(
                y, method=method, initial_rho=x0,
            )
            if not result["converged"]:
                warnings.warn(
                    f"Outer Newton did not converge after {result['n_iter']} iterations"
                )
            self.smoothing_params = result["sp"]
            self._optim_result = result

        else:
            raise ValueError(f"optimizer must be 'lbfgsb' or 'outer_newton'")

        self._optim_method = method
        return self

    def fit(self, y, optimize=True, method="GCV", optimizer="lbfgsb"):
        """Fit the model.

        Parameters
        ----------
        y : array-like
            Response.
        optimize : bool
            If ``True``, optimise smoothing parameters before fitting.
        method : {'GCV', 'ML', 'REML', 'LAML'}
            Smoothing-selection criterion.
        optimizer : {'lbfgsb', 'outer_newton'}
            Which optimizer to use for smoothing selection.

        Returns
        -------
        self
        """
        y = self._validate_y(y, self.n_samples_)
        if optimize:
            self.optimize_smoothing_params(y, method=method, optimizer=optimizer)
        self.fit_without_optimization(y)
        self._y_train = y.copy()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _build_new_design_matrix(self, X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_new.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_new.shape[1]}"
            )
        blocks = []
        for i, spline in enumerate(self.splines):
            raw_basis = spline.transform_new(X_new[:, i])
            blocks.append(raw_basis @ spline.center_mat)
        return np.column_stack(blocks)

    def lpmatrix(self, X_new):
        """Linear predictor matrix for coef vector [intercept, beta]."""
        Z_new = self._build_new_design_matrix(X_new)
        return np.column_stack([np.ones(Z_new.shape[0]), Z_new])

    def predict(self, X_new=None, return_se=False, cov="bayes", type="response"):
        """Predict from the fitted model.

        Parameters
        ----------
        X_new : array-like or None
            New data.  ``None`` → use training data.
        return_se : bool
            If True, return (mu, se) tuple.
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}
            Covariance matrix used for SEs.
        type : {'response', 'terms', 'lpmatrix'}
            What to return.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted")

        Z_new = self.Z if X_new is None else self._build_new_design_matrix(X_new)

        if type == "lpmatrix":
            return np.column_stack([np.ones(Z_new.shape[0]), Z_new])

        if type == "terms":
            terms = np.column_stack(
                [Z_new[:, sl] @ self.coef_[sl] for sl in self.slices]
            )
            if not return_se:
                return terms
            V = self._select_cov(cov)
            ses = []
            for sl in self.slices:
                Xi = Z_new[:, sl]
                Vi = V[sl, sl]
                v = np.einsum("ij,jk,ik->i", Xi, Vi, Xi)
                ses.append(np.sqrt(np.maximum(v, 0.0)))
            return terms, np.column_stack(ses)

        mu = self.intercept_ + Z_new @ self.coef_
        if not return_se:
            return mu

        V_full = self._full_coef_cov(cov)
        Xp = np.column_stack([np.ones(Z_new.shape[0]), Z_new])
        var = np.einsum("ij,jk,ik->i", Xp, V_full, Xp)
        se = np.sqrt(np.maximum(var, 0.0))
        return mu, se

    def _select_cov(self, cov):
        if cov == "bayes":
            V = self.Vp_
        elif cov == "freq":
            V = self.Vf_
        elif cov == "kass_steffey":
            V = self.Vp_kass_steffey_
            if V is None:
                raise RuntimeError(
                    "Kass–Steffey covariance not computed; "
                    "call compute_unconditional_covariance(kind='kass_steffey') first"
                )
        elif cov == "wood":
            V = self.Vp_wood_
            if V is None:
                raise RuntimeError(
                    "Wood covariance not computed; "
                    "call compute_unconditional_covariance(kind='wood_full') first"
                )
        else:
            raise ValueError(
                "cov must be 'bayes', 'freq', 'kass_steffey', or 'wood'"
            )
        if V is None:
            raise RuntimeError("Covariance not available; fit model first")
        return V

    def _full_coef_cov(self, cov="bayes", intercept_sigma2=None):
        """Full ``[intercept, beta]`` covariance matrix.

        The intercept variance is ``intercept_sigma2 / n`` (independent of
        spline coefficients; correct for the centered parameterisation where
        ``intercept = mean(y)``).  The smooth-coefficient block comes from
        :meth:`_select_cov`.

        Parameters
        ----------
        cov : str
            Passed to :meth:`_select_cov`.
        intercept_sigma2 : float or None
            Scale estimate used for the intercept variance.  ``None`` →
            ``self.scale_`` (the fitted residual-df scale).  Pass the same
            ``sigma2`` used to build the information matrix so the
            trace(I_hat @ V) term is internally consistent.

        Returns
        -------
        ndarray, shape (1 + n_coef, 1 + n_coef)
        """
        V_smooth = self._select_cov(cov)
        p1 = 1 + self.n_coef_
        V_full = np.zeros((p1, p1), dtype=np.float64)
        sigma2 = self.scale_ if intercept_sigma2 is None else float(intercept_sigma2)
        V_full[0, 0] = sigma2 / self.n_samples_
        V_full[1:, 1:] = V_smooth
        return V_full

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_dict(self, y=None):
        """Return a structured, machine-readable summary of the fitted model.

        Parameters
        ----------
        y : array-like or None, default=None
            Response used to form residuals and R^2-style summaries.
            ``None`` → stored training y (requires that the model was fitted
            via :meth:`fit` rather than :meth:`fit_without_optimization`).

        Returns
        -------
        dict
            A dictionary with the following top-level keys:

            - ``"model"``: overall metrics (n, EDF, R^2, deviance explained, ...).
            - ``"terms"``: per-term EDF / basis-dimension table.
            - ``"smoothing_parameters"``: per-term smoothing parameters.
            - ``"criteria"``: smoothing-selection criterion values.
            - ``"warnings"``: heuristic flags about potential issues.
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")

        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = np.asarray(y, dtype=np.float64).ravel()

        fitted = self.intercept_ + self.Z @ self.coef_
        resid = y - fitted
        rss = float(resid @ resid)
        tss = float(((y - np.mean(y)) ** 2).sum())

        n = len(y)
        edf = float(self.edf_)
        resid_df = n - edf
        scale = rss / max(resid_df, 1.0)

        r2_adj = 1.0 - (rss / max(tss, 1e-15)) * ((n - 1) / max(resid_df, 1.0))
        dev_expl = 1.0 - rss / max(tss, 1e-15)
        method = getattr(self, "_optim_method", "GCV") or "GCV"
        log_sp = np.log(self.smoothing_params)
        crit_val = self._criterion(y, log_sp, method=method)
        gcv_val = self.gcv_score(y, log_sp)

        # Per-term EDFs (same definition as in the printed summary).
        P = self._assemble_penalty_block(self.smoothing_params)
        A = self.ZTZ + P
        try:
            cA, loA = cho_factor(A, check_finite=False)
            AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        except np.linalg.LinAlgError:
            AinvZTZ = np.linalg.solve(A, self.ZTZ)

        term_summaries = []
        for i, sl in enumerate(self.slices):
            edf_i = float(np.trace(AinvZTZ[sl, sl]))
            k_i = sl.stop - sl.start
            term_summaries.append(
                {
                    "name": str(self.feature_names[i]),
                    "edf": edf_i,
                    "k": int(k_i),
                }
            )

        # Simple heuristic warnings (no heavy simulation or concurvity runs).
        rho = log_sp
        boundary_mask = rho > 15.0
        boundary_terms = [self.feature_names[i] for i in range(self.n_features_) if boundary_mask[i]]

        warnings_dict = {
            "boundary_smoothing_terms": boundary_terms,
        }

        return {
            "model": {
                "n_samples": int(n),
                "n_features": int(self.n_features_),
                "feature_names": list(self.feature_names),
                "intercept": float(self.intercept_),
                "scale": float(scale),
                "edf_total": float(edf),
                "residual_df": float(resid_df),
                "r2_adj": float(r2_adj),
                "deviance_explained": float(dev_expl),
            },
            "terms": term_summaries,
            "smoothing_parameters": {
                "per_term": {
                    str(self.feature_names[i]): float(self.smoothing_params[i])
                    for i in range(self.n_features_)
                },
                "log_smoothing_parameters": [float(v) for v in log_sp],
            },
            "criteria": {
                "smoothing_method": method,
                "criterion_name": method,
                "criterion_value": float(crit_val),
                "gcv": float(gcv_val),
            },
            "warnings": warnings_dict,
        }

    def summary(self, y=None):
        """Print an honest summary (EDF per term; significance tests not shown)."""
        summary = self.summary_dict(y=y)

        model = summary["model"]
        criteria = summary["criteria"]
        terms = summary["terms"]

        print("Gaussian Additive Model Summary")
        print("=" * 55)
        print(f"Smoothing method   : {criteria['smoothing_method']}")
        print(f"Number of samples  : {model['n_samples']}")
        print()
        print("Smooth terms (EDF only; significance tests not shown):")
        print(f"{'term':<20s} {'edf':>8s} {'k':>5s}")
        print("-" * 35)
        for term in terms:
            name = term["name"]
            edf_i = term["edf"]
            k_i = term["k"]
            print(f"s({name:<14s}) {edf_i:8.3f} {k_i:5d}")

        print("-" * 55)
        print(f"Intercept          : {model['intercept']:.6g}")
        print(f"Scale estimate     : {model['scale']:.6g}")
        print(f"EDF (total)        : {model['edf_total']:.3f}")
        print(f"Residual df        : {model['residual_df']:.3f}")
        print(f"R-sq.(adj)         : {model['r2_adj']:.6f}")
        print(f"Deviance explained : {model['deviance_explained']:.2%}")
        print(
            f"{criteria['criterion_name']} criterion     : "
            f"{criteria['criterion_value']:.6g}"
        )
        if criteria["criterion_name"] != "GCV":
            print(f"GCV (supplementary): {criteria['gcv']:.6g}")
        print(f"n                  : {model['n_samples']}")

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def confidence_intervals(self, alpha=0.05, cov="bayes", include_intercept=False):
        """Wald-type CIs for spline coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level; CIs are at ``1 - alpha`` coverage.
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}, default='bayes'
            Covariance matrix used for the standard errors.
            ``'kass_steffey'`` and ``'wood'`` require a prior call to
            :meth:`compute_unconditional_covariance`.
        include_intercept : bool, default=False
            If ``True``, prepend a CI for the intercept.

        Returns
        -------
        list of (float, float)
        """
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")

        zcrit = norm.ppf(1.0 - alpha / 2.0)
        V = self._select_cov(cov)
        ses = np.sqrt(np.maximum(np.diag(V), 0.0))

        out = []
        if include_intercept:
            se0 = np.sqrt(max(self.scale_ / self.n_samples_, 0.0))
            out.append((self.intercept_ - zcrit * se0, self.intercept_ + zcrit * se0))

        for b, se in zip(self.coef_, ses):
            out.append((b - zcrit * se, b + zcrit * se))
        return out

    # ------------------------------------------------------------------
    # AIC
    # ------------------------------------------------------------------

    def _gaussian_loglik(self, y, scale="ml"):
        """Gaussian log-likelihood at the fitted values.

        Parameters
        ----------
        scale : {'ml', 'working'}
            ``'ml'``     → sigma^2_ML = RSS / n (appropriate for AIC
            comparability).
            ``'working'`` → ``self.scale_`` (RSS / residual degrees of
            freedom), consistent with the summary.

        Returns
        -------
        loglik : float
            Value of the Gaussian log-likelihood at the fitted values.
        sigma2 : float
            The scale estimate used.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        y = self._resolve_y(y)
        fitted = self.intercept_ + self.Z @ self.coef_
        rss = float(np.sum((y - fitted) ** 2))
        n = float(self.n_samples_)

        if scale == "ml":
            sigma2 = rss / n
        elif scale == "working":
            sigma2 = self.scale_
        else:
            raise ValueError("scale must be 'ml' or 'working'")

        if sigma2 <= 0:
            sigma2 = 1e-15

        loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + rss / (n * sigma2))
        return float(loglik), float(sigma2)

    def _resolve_y(self, y):
        """Return validated y or stored training y."""
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        return self._validate_y(y, self.n_samples_)

    def _observed_information(self, sigma2):
        """Observed information matrix for ``[intercept, beta]``.

        For the Gaussian model with known (or plugged-in) variance the observed
        information is ``X_p.T @ X_p / sigma2`` where ``X_p`` is the full
        linear-predictor matrix including the intercept column.
        """
        Xp = np.column_stack([np.ones(self.n_samples_), self.Z])
        return (Xp.T @ Xp) / sigma2

    def aic_conditional(self, y=None, scale="ml", cov="bayes"):
        """Conventional conditional AIC.

        Computes

            AIC_c = -2 * loglik + 2 * trace(I_hat * V_beta),

        where ``V_beta`` is the conditional covariance (given smoothing
        parameters) and ``I_hat`` is the observed information.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        scale : {'ml', 'working'}
            Which scale estimate to use in the log-likelihood and the
            information matrix.
        cov : {'bayes', 'freq'}
            Which conditional covariance to use for the penalty term.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic`` (= ``tr(I V)``),
        ``scale``.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        loglik, sigma2 = self._gaussian_loglik(y, scale=scale)
        I_hat = self._observed_information(sigma2)
        V_full = self._full_coef_cov(cov, intercept_sigma2=sigma2)
        tau = float(np.trace(I_hat @ V_full))

        return {
            "aic": -2.0 * loglik + 2.0 * tau,
            "loglik": loglik,
            "edf_aic": tau,
            "scale": sigma2,
        }

    def aic_corrected(
        self,
        y=None,
        scale="ml",
        covariance_kind="wood_full",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Wood-style corrected conditional AIC.

        Corrected AIC is computed as:

            AIC_corr = -2 * loglik(beta_hat) + 2 * trace(I_hat @ Vbar_beta)

        where ``Vbar_beta`` is the covariance corrected for
        smoothing-parameter uncertainty (Kass–Steffey or full Wood),
        computed via :meth:`compute_unconditional_covariance` if not
        already available.

        .. note::

            The Wood (2016) corrected AIC is theoretically grounded on the
            Hessian of the **negative marginal likelihood** (REML / LAML)
            with respect to log smoothing parameters. When the model was fitted
            with ``method='GCV'`` the smoothing-parameter uncertainty
            (V_rho) is estimated from the GCV Hessian, which does not have
            the same theoretical justification.  The result is still a
            reasonable heuristic but is **not** the exact Wood corrected AIC.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        scale : {'ml', 'working'}
            Scale estimate for the log-likelihood and information.
        covariance_kind : {'kass_steffey', 'wood_full'}
            Which unconditional covariance approximation to use.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            Passed to :meth:`compute_unconditional_covariance`.
        sp_uncertainty_ridge : float
            Passed to :meth:`compute_unconditional_covariance`.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``,
        ``covariance_kind``, ``heuristic`` (bool — ``True`` when the
        corrected AIC is not theoretically exact, e.g. fitted with GCV).
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        cov_map = {"kass_steffey": "kass_steffey", "wood_full": "wood"}
        if covariance_kind not in cov_map:
            raise ValueError(
                f"covariance_kind must be 'kass_steffey' or 'wood_full', "
                f"got {covariance_kind!r}"
            )
        cov_key = cov_map[covariance_kind]

        # Warn when the fitting criterion is not REML/LAML.
        # Wood's corrected AIC uses V_rho from the marginal-likelihood Hessian;
        # using the GCV Hessian yields a heuristic rather than the exact result.
        optim_method = (self._optim_method or "GCV").upper()
        is_heuristic = optim_method not in {"REML", "LAML", "ML"}
        if is_heuristic:
            warnings.warn(
                f"aic_corrected(): model was fitted with method='{optim_method}'. "
                "Wood's corrected AIC is theoretically grounded on the REML/LAML "
                "Hessian of the negative marginal likelihood w.r.t. log(lambda). "
                "Using a GCV-based smoothing-parameter uncertainty yields a "
                "heuristic approximation, not the exact Wood corrected AIC. "
                "Refit with method='REML' (or 'LAML') for the theoretically "
                "justified result.  The returned dict includes 'heuristic': True.",
                UserWarning,
                stacklevel=2,
            )

        # Compute unconditional covariance if not already present
        attr = (
            self.Vp_kass_steffey_
            if covariance_kind == "kass_steffey"
            else self.Vp_wood_
        )
        if attr is None:
            self.compute_unconditional_covariance(
                y=y,
                kind=covariance_kind,
                sp_uncertainty_regularization=sp_uncertainty_regularization,
                sp_uncertainty_ridge=sp_uncertainty_ridge,
            )

        loglik, sigma2 = self._gaussian_loglik(y, scale=scale)
        I_hat = self._observed_information(sigma2)
        V_full = self._full_coef_cov(cov_key, intercept_sigma2=sigma2)
        tau = float(np.trace(I_hat @ V_full))

        return {
            "aic": -2.0 * loglik + 2.0 * tau,
            "loglik": loglik,
            "edf_aic": tau,
            "scale": sigma2,
            "covariance_kind": covariance_kind,
            "heuristic": is_heuristic,
        }

    # ------------------------------------------------------------------
    # Term-drop tests
    # ------------------------------------------------------------------

    def term_drop_test(self, y=None, term_index=0, method=None):
        """Refit-based approximate term significance test.

        Drops one smooth term, refits (re-optimising smoothing), and
        computes an approximate F-statistic from the change in RSS and
        EDF.  P-values are approximate because smoothing parameters are
        re-estimated in the reduced model.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)
        method = method or self._optim_method or "GCV"

        if not (0 <= term_index < self.n_features_):
            raise IndexError(
                f"term_index must be in [0, {self.n_features_ - 1}], got {term_index}"
            )

        rss_full = self.rss_
        edf_full = self.edf_
        n = self.n_samples_

        keep = [i for i in range(self.n_features_) if i != term_index]
        if not keep:
            raise ValueError("Cannot drop the only term")
        X_red = self.X[:, keep]
        names_red = [self.feature_names[i] for i in keep]

        s_red = np.array([self.smoothing_params[i] for i in keep])

        red = GAM(X_red, k=self.k_, s=s_red, feature_names=names_red)
        red.fit(y, optimize=True, method=method)

        rss_red = red.rss_
        edf_red = red.edf_
        delta_df = max(edf_full - edf_red, 1e-6)
        df_res = max(n - edf_full, 1e-6)
        ms_num = max((rss_red - rss_full) / delta_df, 0.0)
        ms_den = rss_full / df_res
        f_stat = ms_num / max(ms_den, 1e-15)
        pval = float(1.0 - f_dist.cdf(f_stat, delta_df, df_res))

        return {
            "term": self.feature_names[term_index],
            "f_stat": f_stat,
            "p_value": pval,
            "delta_df": delta_df,
            "rss_full": rss_full,
            "rss_reduced": rss_red,
            "edf_full": edf_full,
            "edf_reduced": edf_red,
        }

    # ------------------------------------------------------------------
    # Unconditional covariance (smoothing-parameter uncertainty)
    # ------------------------------------------------------------------

    def compute_unconditional_covariance(
        self,
        y=None,
        method=None,
        kind="kass_steffey",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Covariance corrected for smoothing-parameter uncertainty.

        Two approximation levels are available:

        ``kind='kass_steffey'`` — :math:`\bar V_\beta = V_\beta + J V_\rho J^\top`

            The Kass–Steffey first-order correction.  Fast, often adequate.
            Result is stored in ``Vp_kass_steffey_``; use ``cov='kass_steffey'``
            in :meth:`predict`.

        ``kind='wood_full'`` — :math:`\bar V_\beta = V_\beta + V' + V''`

            The full Wood et al. (2016) correction that also accounts for the
            derivative of the covariance factor :math:`R_\rho` where
            :math:`R_\rho^\top R_\rho = V_\beta`.  More accurate when some
            smoothing parameters are near the boundary of the penalty null
            space.  Result is stored in ``Vp_wood_``; use ``cov='wood'`` in
            :meth:`predict`.

        Parameters
        ----------
        y : array-like or None
            Response.  ``None`` → stored training y.
        method : str or None
            Criterion used for the Hessian (``'GCV'``, ``'ML'``, ``'REML'``).
            ``None`` → whatever was used during fitting.
        kind : {'kass_steffey', 'wood_full'}
            Which approximation to compute.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            How to invert the criterion Hessian :math:`H_\rho`:

            - ``'pinv'``:  Moore–Penrose pseudoinverse — eigenvalues below a
              relative tolerance (``1e-10 * max(1, max|evals|)``) are mapped
              to zero inverse, so flat/boundary directions contribute zero
              uncertainty rather than inflated uncertainty.
            - ``'ridge'``: invert ``(H_rho + kappa * I)`` where ``kappa`` is
              automatically raised if needed to keep all shifted eigenvalues
              strictly positive (equivalent to a Gaussian prior on
              :math:`\rho`).
        sp_uncertainty_ridge : float, default=1e-6
            Ridge constant used when ``sp_uncertainty_regularization='ridge'``.

        Returns
        -------
        ndarray, shape (n_coef, n_coef)
            The corrected covariance matrix.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)
        method = method or self._optim_method or "GCV"

        valid_kinds = {"kass_steffey", "wood_full"}
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}")

        sp = self.smoothing_params.copy()
        theta = np.log(sp)
        m = self.n_features_
        p = self.n_coef_

        sol = self._solve_given_smoothing(y, sp, store=False)
        cA, loA = sol["cA"]
        A_inv = cho_solve((cA, loA), np.eye(p), check_finite=False)
        beta = sol["beta"]
        sigma2 = self.scale_
        Vp = self.Vp_

        # D_k = dA/d(rho_k) = lambda_k * S_k  (block-diagonal, only one block nonzero)
        D_blocks = []
        for j in range(m):
            D = np.zeros((p, p), dtype=np.float64)
            sl = self.slices[j]
            D[sl, sl] = sp[j] * self.penalties[j]
            D_blocks.append(D)

        # J = d(beta_hat)/d(rho)  — Gaussian specialisation of implicit deriv.
        # J[:, k] = -A_inv @ D_k @ beta
        J = np.column_stack([-A_inv @ (Dk @ beta) for Dk in D_blocks])

        # V_rho = (regularised) inverse of criterion Hessian w.r.t. log(sp)
        V_rho = self._sp_uncertainty_matrix(
            y, theta, method,
            regularization=sp_uncertainty_regularization,
            ridge=sp_uncertainty_ridge,
        )

        # V' = J V_rho J^T  (Kass–Steffey term)
        V_prime = J @ V_rho @ J.T

        if kind == "kass_steffey":
            V_unc = Vp + V_prime
            self.Vp_kass_steffey_ = 0.5 * (V_unc + V_unc.T)
            return self.Vp_kass_steffey_

        # --- kind == "wood_full": also compute V'' -------------------------

        # Cholesky factor L such that V_beta = L L^T.
        # _cholesky_factor_derivative requires a genuine lower-triangular L.
        # Retry with increasing jitter up to 4 times; if Vp is still not SPD,
        # fall back to Kass–Steffey so we never pass a symmetric square-root
        # into a method that assumes a triangular factor.
        _jitter_schedule = [0.0, 1e-10, 1e-7, 1e-5, 1e-3]
        L = None
        for _jit in _jitter_schedule:
            try:
                Vp_jit = Vp if _jit == 0.0 else Vp + _jit * np.eye(p)
                L = np.linalg.cholesky(Vp_jit)
                if _jit > 0.0:
                    warnings.warn(
                        f"compute_unconditional_covariance(kind='wood_full'): "
                        f"Vp was not numerically SPD; added jitter={_jit:.0e} "
                        f"to obtain a valid Cholesky factor."
                    )
                break
            except np.linalg.LinAlgError:
                continue

        if L is None:
            warnings.warn(
                "compute_unconditional_covariance(kind='wood_full'): "
                "Vp remains non-SPD after jitter retries.  "
                "V'' cannot be computed faithfully; "
                "falling back to Kass–Steffey (V' only).  "
                "Consider using kind='kass_steffey' explicitly.",
                RuntimeWarning,
            )
            V_unc = Vp + V_prime
            self.Vp_kass_steffey_ = 0.5 * (V_unc + V_unc.T)
            # Do not set Vp_wood_ so callers requesting cov="wood" still get
            # an informative error rather than silently incorrect results.
            return self.Vp_kass_steffey_

        # dV_beta/d(rho_k) = -sigma^2 A_inv D_k A_inv
        # (ignoring d(sigma^2)/d(rho_k), consistent with standard mgcv practice)
        dVp = [-sigma2 * A_inv @ Dk @ A_inv for Dk in D_blocks]

        # Cholesky-factor derivatives:  dL_k  via  L dL_k^T + dL_k L^T = dV_k
        # Solved row-by-row (Lyapunov equation for lower-triangular L)
        dL = [self._cholesky_factor_derivative(L, dV) for dV in dVp]

        # V'' = sum_{k,l}  dL_k dL_l^T  V_rho[k,l]
        V_double_prime = np.zeros((p, p), dtype=np.float64)
        for k in range(m):
            for l in range(m):
                if V_rho[k, l] == 0.0:
                    continue
                V_double_prime += V_rho[k, l] * (dL[k] @ dL[l].T)

        V_unc = Vp + V_prime + V_double_prime
        self.Vp_wood_ = 0.5 * (V_unc + V_unc.T)
        return self.Vp_wood_

    # ---- helpers for compute_unconditional_covariance --------------------

    def _sp_uncertainty_matrix(
        self, y, theta, method, regularization="pinv", ridge=1e-6
    ):
        r"""Inverse (or pseudo-inverse) of the criterion Hessian w.r.t. log(sp).

        For ``method`` in ``{'REML', 'LAML'}`` the analytic Hessian from
        :meth:`_laml_objective_gradient_hessian` is used (stable, exact).
        For ``'GCV'`` and ``'ML'`` the adaptive numeric Hessian is the
        fallback.

        Wood notes that when a smoothing parameter is at or near the boundary
        (effectively infinite), the corresponding Hessian eigenvalue tends to
        zero.  The Moore–Penrose pseudoinverse correctly maps these directions
        to zero uncertainty rather than inflating it, while ridge
        regularisation is equivalent to placing a Gaussian prior on
        :math:`\rho`.
        """
        m_upper = method.upper()

        if m_upper in {"REML", "LAML"}:
            intercept = float(np.mean(y))
            y_c = y - intercept
            _, _, H = self._laml_objective_gradient_hessian(y_c, np.asarray(theta))
        else:
            H = self._numeric_hessian(
                lambda th: self._criterion(y, th, method=method), theta
            )

        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)

        if regularization == "pinv":
            tol = 1e-10 * max(1.0, np.max(np.abs(evals)))
            inv_evals = np.where(evals > tol, 1.0 / evals, 0.0)
        elif regularization == "ridge":
            kappa = float(ridge)
            # Ensure the shift is large enough to keep all shifted eigenvalues
            # strictly positive, so V_rho is PSD.
            min_eig = float(evals.min())
            if min_eig + kappa <= 0.0:
                kappa_eff = -min_eig + kappa + 1e-8
                warnings.warn(
                    f"_sp_uncertainty_matrix: ridge={kappa:.2e} insufficient to "
                    f"stabilise Hessian (min eigenvalue {min_eig:.2e}); "
                    f"using effective ridge={kappa_eff:.2e} to ensure PSD V_rho"
                )
            else:
                kappa_eff = kappa
            inv_evals = 1.0 / (evals + kappa_eff)
        else:
            raise ValueError(
                f"sp_uncertainty_regularization must be 'pinv' or 'ridge', "
                f"got {regularization!r}"
            )

        return (evecs * inv_evals) @ evecs.T

    @staticmethod
    def _cholesky_factor_derivative(L, dV):
        """Derivative of a Cholesky factor given :math:`dV = L dL^T + dL L^T`.

        Solves for the lower-triangular :math:`dL` using the standard
        row-by-row algorithm for the triangular Lyapunov equation.
        """
        p = L.shape[0]
        dL = np.zeros_like(L)
        for i in range(p):
            for j in range(i + 1):
                if i == j:
                    s = dV[i, i] - 2.0 * np.dot(dL[i, :j], L[i, :j])
                    dL[i, i] = s / (2.0 * L[i, i]) if abs(L[i, i]) > 1e-15 else 0.0
                else:
                    s = dV[i, j] - (
                        np.dot(dL[i, :j], L[j, :j]) + np.dot(L[i, :j], dL[j, :j])
                    )
                    dL[i, j] = s / L[j, j] if abs(L[j, j]) > 1e-15 else 0.0
        return dL

    @staticmethod
    def _numeric_hessian(func, x, rel_eps=1e-4, abs_eps=1e-4):
        r"""Central-difference Hessian with adaptive per-coordinate step sizes.

        Step for coordinate *i* is ``h_i = rel_eps * max(1, |x_i|) + abs_eps``,
        which scales with the magnitude of :math:`\rho_i` and avoids the
        degeneracy of a fixed step on flat / poorly scaled criteria.

        After computation the result is symmetry-projected and a condition
        diagnostic warning is emitted if the matrix is poorly conditioned.
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        h = rel_eps * np.maximum(1.0, np.abs(x)) + abs_eps

        fx = func(x)
        if not np.isfinite(fx):
            warnings.warn("_numeric_hessian: function value at x is non-finite")
            return np.full((n, n), np.nan)

        H = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = h[i]
            fip = func(x + ei)
            fim = func(x - ei)
            H[i, i] = (fip - 2.0 * fx + fim) / (h[i] * h[i])
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = h[j]
                fpp = func(x + ei + ej)
                fpm = func(x + ei - ej)
                fmp = func(x - ei + ej)
                fmm = func(x - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])

        H = 0.5 * (H + H.T)

        if not np.all(np.isfinite(H)):
            warnings.warn(
                "_numeric_hessian: non-finite entries detected; "
                "criterion may be ill-conditioned at this point"
            )
        else:
            evals = np.linalg.eigvalsh(H)
            emax = np.max(np.abs(evals))
            emin_pos = np.min(evals[evals > 0]) if np.any(evals > 0) else 0.0
            if emax > 0 and emin_pos > 0 and emax / emin_pos > 1e12:
                warnings.warn(
                    f"_numeric_hessian: condition number ~{emax / emin_pos:.1e}; "
                    "Hessian may be unreliable — consider using "
                    "optimizer='outer_newton' for analytic derivatives"
                )

        return H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Concurvity (Wood / mgcv-style)
    # ------------------------------------------------------------------

    @staticmethod
    def _qr_R_nopivot(A):
        """Return the R factor from a no-pivot QR decomposition."""
        # mgcv uses qr(..., LAPACK = FALSE, tol = 0) and then qr.R(...)
        # Here we use SciPy QR without pivoting.
        # mode='economic' is enough for the subsequent slicing.
        if A.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        _, R = scipy_qr(A, mode="economic", pivoting=False, check_finite=False)
        return R

    @staticmethod
    def _safe_ratio(num, den, eps=1e-15):
        """Bounded ratio helper for concurvity measures."""
        num = float(num)
        den = float(den)
        if den <= eps:
            # If the target term has effectively zero norm, concurvity is undefined;
            # return 0 for production robustness (mgcv may yield NaN in degenerate cases).
            return 0.0 if num <= eps else 1.0
        val = num / den
        # Numerical safety: these should be in [0,1] theoretically.
        return float(np.clip(val, 0.0, 1.0))

    def _concurvity_measures_for_pair(self, Xi, Xj, beta_j):
        """Compute (worst, observed, estimate) for dependence of Xj-term on Xi-space.

        This mirrors mgcv's QR-based formulas (see concurvity source in R/mgcv.r).
        """
        # Xi: basis defining the "other" space (or row term in pairwise mode)
        # Xj: basis of the term being assessed (current/full term or column term)
        # beta_j: fitted coefficients for Xj term

        r = Xi.shape[1]
        dj = Xj.shape[1]

        if dj == 0:
            return 0.0, 0.0, 0.0
        if r == 0:
            # No competing space -> no concurvity
            return 0.0, 0.0, 0.0

        # mgcv pattern:
        # R <- qr.R(qr(cbind(Xi, Xj), no pivot))[,-(1:r)]
        Rfull = self._qr_R_nopivot(np.column_stack([Xi, Xj]))
        # With economic QR and n >= p (usual case), shape is (p, p) where p=r+dj.
        # If n < p, shape is (n, p), and concurvity can be unstable/undefined.
        if Rfull.shape[0] < r:
            # Not enough rows to represent Xi block in R in the expected way
            return np.nan, np.nan, np.nan

        R = Rfull[:, r:]  # shape approx (r+dj, dj)

        # Another QR of R:
        # Rt <- qr.R(qr(R, tol=0))
        Rt = self._qr_R_nopivot(R)

        # 1) worst:
        # svd( forwardsolve(t(Rt), t(R[1:r,,drop=FALSE])) )$d[1]^2
        # In Python (0-indexed): R[:r, :]
        R_top = R[:r, :]
        try:
            # solve_triangular on lower-triangular t(Rt)
            M = solve_triangular(
                Rt.T, R_top.T, lower=True, check_finite=False, overwrite_b=False
            )
            svals = np.linalg.svd(M, compute_uv=False)
            worst = float(svals[0] ** 2) if svals.size else 0.0
        except np.linalg.LinAlgError:
            worst = np.nan

        # 2) observed:
        # sum((R[1:r,] %*% beta)^2) / sum((Rt %*% beta)^2)
        num_obs = np.sum((R_top @ beta_j) ** 2)
        den_obs = np.sum((Rt @ beta_j) ** 2)
        observed = self._safe_ratio(num_obs, den_obs)

        # 3) estimate:
        # sum(R[1:r,]^2) / sum(R^2)
        num_est = np.sum(R_top ** 2)
        den_est = np.sum(R ** 2)
        estimate = self._safe_ratio(num_est, den_est)

        return worst, observed, estimate

    def concurvity(self, full=True, include_intercept=False):
        """Wood/mgcv-style concurvity diagnostics.

        Parameters
        ----------
        full : bool, default=True
            If True, compute concurvity of each term with the whole of the rest
            of the model (mgcv full=TRUE style).
            If False, compute pairwise concurvity matrices (mgcv full=FALSE style).
        include_intercept : bool, default=False
            If True, include a 'para' component consisting only of the intercept
            column. In this model class there are no additional parametric terms.

        Returns
        -------
        If full=True:
            dict with keys 'worst', 'observed', 'estimate' each mapping term name -> value,
            and a convenience matrix-like ndarray under key 'matrix' (rows in this order).
        If full=False:
            dict with keys 'worst', 'observed', 'estimate', each an (m x m) ndarray,
            plus 'labels'.

        Notes
        -----
        This follows the mgcv concurvity source's QR-based computations closely,
        adapted to this class's term block structure.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        # Build blocks and labels
        blocks = []
        betas = []
        labels = []

        if include_intercept:
            blocks.append(np.ones((self.n_samples_, 1), dtype=np.float64))
            betas.append(np.array([self.intercept_], dtype=np.float64))
            labels.append("para")

        for i, sl in enumerate(self.slices):
            blocks.append(self.Z[:, sl])
            betas.append(self.coef_[sl])
            labels.append(self.feature_names[i])

        m = len(blocks)
        if m < 1:
            raise ValueError("No terms available for concurvity")

        measure_names = ("worst", "observed", "estimate")

        if full:
            # Each term vs the whole rest of model
            conc = np.zeros((3, m), dtype=np.float64)

            for i in range(m):
                Xj = blocks[i]
                beta_j = betas[i]
                other_blocks = [blocks[k] for k in range(m) if k != i]
                Xi = (
                    np.column_stack(other_blocks)
                    if other_blocks
                    else np.empty((self.n_samples_, 0), dtype=np.float64)
                )

                w, o, e = self._concurvity_measures_for_pair(Xi, Xj, beta_j)
                conc[:, i] = [w, o, e]

            return {
                "matrix": conc,
                "rows": list(measure_names),
                "labels": labels,
                "worst": dict(zip(labels, conc[0])),
                "observed": dict(zip(labels, conc[1])),
                "estimate": dict(zip(labels, conc[2])),
            }

        # Pairwise mode: matrices, mgcv-style
        conc = {
            "worst": np.eye(m, dtype=np.float64),
            "observed": np.eye(m, dtype=np.float64),
            "estimate": np.eye(m, dtype=np.float64),
        }

        # Row i = dependence on term i's space; column j = term j being assessed
        for i in range(m):
            Xi = blocks[i]
            for j in range(m):
                if i == j:
                    continue
                Xj = blocks[j]
                beta_j = betas[j]
                w, o, e = self._concurvity_measures_for_pair(Xi, Xj, beta_j)
                conc["worst"][i, j] = w
                conc["observed"][i, j] = o
                conc["estimate"][i, j] = e

        conc["labels"] = labels
        return conc

    # ------------------------------------------------------------------
    # k-index diagnostic (Wood/mgcv-style simulation test)
    # ------------------------------------------------------------------

    def _term_edf_vector(self):
        """Per-term EDFs from block traces of A^{-1} Z'Z.

        These are the same "display EDFs" printed by :meth:`summary`.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")

        P = self._assemble_penalty_block(self.smoothing_params)
        A = self.ZTZ + P
        try:
            cA, loA = cho_factor(A, check_finite=False)
            AinvZTZ = cho_solve((cA, loA), self.ZTZ, check_finite=False)
        except np.linalg.LinAlgError:
            AinvZTZ = np.linalg.solve(A, self.ZTZ)

        edf = np.zeros(self.n_features_, dtype=np.float64)
        for i, sl in enumerate(self.slices):
            edf[i] = float(np.trace(AinvZTZ[sl, sl]))
        return edf

    def k_check(self, y=None, subsample=5000, n_rep=400, random_state=None):
        """Wood/mgcv-style basis-dimension check (k-index) for 1-D numeric smooths.

        Matches the 1-D branch of ``mgcv::k.check()``: residuals are ordered by
        each smooth's covariate, differenced to estimate local residual variance,
        and the resulting *k-index* is compared to a simulation null obtained by
        reshuffling residuals.

        Parameters
        ----------
        y : array-like or None, default=None
            Response used to form residuals.  ``None`` → stored training y.
        subsample : int, default=5000
            When ``n > subsample`` use a random subsample without replacement
            (matches mgcv's cost-control heuristic).
        n_rep : int, default=400
            Number of residual reshuffles for the simulation p-value.
        random_state : int, np.random.Generator, or None
            Seed / generator for reproducibility.

        Returns
        -------
        dict with keys
            ``labels``       – list of term names (length *m*)
            ``table``        – ndarray, shape (*m*, 4), columns k', edf, k-index, p-value
            ``columns``      – ``["k'", "edf", "k-index", "p-value"]``
            ``subsample_n``  – actual number of observations used
            ``n_rep``        – number of simulations performed

        Notes
        -----
        A k-index **below 1** suggests remaining autocorrelation in the residuals
        at the scale of that smooth's covariate, indicating the basis dimension may
        be too small.  p-values are simulation-based and vary across runs when the
        null is true, as the mgcv documentation notes.

        This implementation covers the 1-D numeric-smooth case that is the only
        smooth type currently supported by this class.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = self._validate_y(y, self.n_samples_)

        if subsample is None or int(subsample) <= 0:
            raise ValueError("subsample must be a positive integer")
        if int(n_rep) <= 0:
            raise ValueError("n_rep must be a positive integer")

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Response residuals (Gaussian identity link)
        rsd = y - (self.intercept_ + self.Z @ self.coef_)
        n = rsd.shape[0]

        # mgcv-style cost-control subsample
        if n > int(subsample):
            idx_sub = rng.choice(n, size=int(subsample), replace=False)
            X_sub = self.X[idx_sub, :]
            rsd_sub = rsd[idx_sub]
        else:
            X_sub = self.X
            rsd_sub = rsd

        nr = rsd_sub.shape[0]
        if nr < 3:
            raise ValueError("Need at least 3 observations for k_check")

        # Global denominator: mean(rsd^2), used as sigma^2_r
        rsd_var = float(np.mean(rsd_sub ** 2))

        per_edf = self._term_edf_vector()
        m = self.n_features_
        table = np.full((m, 4), np.nan, dtype=np.float64)
        labels = list(self.feature_names)

        # Fill k' and edf unconditionally so they appear even in degenerate cases
        for j, sl in enumerate(self.slices):
            table[j, 0] = float(sl.stop - sl.start)
            table[j, 1] = float(per_edf[j])

        if not np.isfinite(rsd_var) or rsd_var <= 0.0:
            return {
                "labels": labels,
                "table": table,
                "columns": ["k'", "edf", "k-index", "p-value"],
                "subsample_n": int(nr),
                "n_rep": int(n_rep),
            }

        sim_buf = np.empty(int(n_rep), dtype=np.float64)

        for j, sl in enumerate(self.slices):
            xj = X_sub[:, j]

            if not np.issubdtype(xj.dtype, np.number):
                continue
            if not np.isfinite(xj).all():
                continue
            if np.allclose(xj.max(), xj.min()):
                # Constant covariate on the subsample → test undefined
                continue

            # mgcv 1-D branch:  e <- diff(rsd[order(x)])
            order = np.argsort(xj, kind="mergesort")
            e_obs = np.diff(rsd_sub[order])
            v_obs = float(np.mean(e_obs ** 2) / 2.0)

            # Simulation null:  e <- diff(rsd[sample(1:nr, nr)])
            for i in range(int(n_rep)):
                perm = rng.permutation(nr)
                ep = np.diff(rsd_sub[perm])
                sim_buf[i] = float(np.mean(ep ** 2) / 2.0)

            # p = proportion of simulated values *less than* observed (mgcv convention)
            p_value = float(np.mean(sim_buf < v_obs))
            k_index = float(v_obs / rsd_var)

            table[j, 2] = np.clip(k_index, 0.0, np.inf)
            table[j, 3] = np.clip(p_value, 0.0, 1.0)

        return {
            "labels": labels,
            "table": table,
            "columns": ["k'", "edf", "k-index", "p-value"],
            "subsample_n": int(nr),
            "n_rep": int(n_rep),
        }

    # ------------------------------------------------------------------
    # k-refit heuristic (kept as a complement to k_check)
    # ------------------------------------------------------------------

    def k_refit_check(self, y=None, factor=2):
        """Refit-based basis-dimension sensitivity check.

        Doubles (or scales by *factor*) the basis dimension for every smooth
        term, refits the model with freshly optimised smoothing parameters,
        and compares total EDF and the smoothing criterion.  A large increase
        in EDF suggests the current *k* may be too small; the smoothing
        criterion gives a secondary quality signal.

        This is the heuristic that mgcv documentation recommends as a
        sensible follow-up after :meth:`k_check` flags a concern.

        Parameters
        ----------
        y : array-like or None, default=None
            Response.  ``None`` → stored training y.
        factor : int or float, default=2
            ``k_new = max(k + 1, factor * k)``.

        Returns
        -------
        dict with keys ``k_old``, ``k_new``, ``edf_old``, ``edf_new``,
        ``criterion_old``, ``criterion_new``.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit first")
        if y is None:
            y = self._y_train
        if y is None:
            raise ValueError("Pass y or fit with stored training y")
        y = np.asarray(y, dtype=np.float64).ravel()
        method = self._optim_method or "GCV"

        k_current = self.k_
        k_new = int(max(k_current + 1, factor * k_current))

        refit = GAM(
            self.X, k=k_new, s=self.smoothing_params.copy(),
            feature_names=self.feature_names,
        )
        refit.fit(y, optimize=True, method=method)

        return {
            "k_old": k_current,
            "k_new": k_new,
            "edf_old": self.edf_,
            "edf_new": refit.edf_,
            "criterion_old": self._criterion(
                y, np.log(self.smoothing_params), method=method
            ),
            "criterion_new": refit._criterion(
                y, np.log(refit.smoothing_params), method=method
            ),
        }

basemodels/nam.py
from itertools import combinations

import torch
import torch.nn as nn

from ..arch_utils.normalization_layers import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    LearnableLayerScaling,
    RMSNorm,
)
from ..configs.nam_config import DefaultNAMConfig
from .basemodel import BaseModel


class NAM(BaseModel):
    """
    Neural Additive Model (NAM) class.

    This class implements a Neural Additive Model (NAM) with support for numerical and
    categorical features, interaction terms, and various normalization layers.

    Attributes
    ----------
    num_feature_networks : nn.ModuleDict
        Sub-networks for each numerical feature.
    cat_feature_networks : nn.ModuleDict
        Sub-networks for each categorical feature.
    interaction_networks : nn.ModuleDict
        Networks for modeling feature interactions (if applicable).
    interaction_degree : int, optional
        Degree of interactions to be modeled.
    intercept : torch.nn.Parameter
        Learnable intercept term, if enabled.
    feature_dropout : nn.Dropout
        Dropout layer for regularizing feature contributions.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMConfig = DefaultNAMConfig(),
        **kwargs,
    ):
        """
        Initializes the Neural Additive Model (NAM) with the given configuration.

        Parameters
        ----------
        cat_feature_info : dict
            Dictionary providing information about categorical features (e.g., input dimensions).
        num_feature_info : dict
            Dictionary providing information about numerical features (e.g., input dimensions).
        num_classes : int, optional
            Number of output classes for classification tasks, by default 1.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters for the model, by default DefaultNAMConfig.
        kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

        # Initialize sub-networks for each feature
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"], config
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"], config
            )  # Categorical features are typically encoded as single values

        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
                config=config,
            )

    def _create_subnetwork(self, input_dim, config):
        """
        Creates a subnetwork for a single feature.

        Parameters
        ----------
        input_dim : int
            Dimension of the input feature.
        config : DefaultNAMConfig
            Configuration dataclass containing model hyperparameters.

        Returns
        -------
        nn.Sequential
            A subnetwork composed of linear layers, normalization layers, and activation functions.
        """
        layers = nn.Sequential()
        layers.add_module("input", nn.Linear(input_dim, config.layer_sizes[0]))

        if config.batch_norm:
            layers.add_module("batch_norm", nn.BatchNorm1d(config.layer_sizes[0]))

        norm_layer = self.hparams.get("norm", config.norm)
        if norm_layer == "RMSNorm":
            layers.add_module("norm", RMSNorm(config.layer_sizes[0]))
        elif norm_layer == "LayerNorm":
            layers.add_module("norm", LayerNorm(config.layer_sizes[0]))
        elif norm_layer == "BatchNorm":
            layers.add_module("norm", BatchNorm(config.layer_sizes[0]))
        elif norm_layer == "InstanceNorm":
            layers.add_module("norm", InstanceNorm(config.layer_sizes[0]))
        elif norm_layer == "GroupNorm":
            layers.add_module("norm", GroupNorm(1, config.layer_sizes[0]))
        elif norm_layer == "LearnableLayerScaling":
            layers.add_module("norm", LearnableLayerScaling(config.layer_sizes[0]))

        if config.use_glu:
            layers.add_module("glu", nn.GLU())
        else:
            layers.add_module(
                "activation", self.hparams.get("activation", config.activation)
            )

        if config.dropout > 0.0:
            layers.add_module("dropout", nn.Dropout(config.dropout))

        for i in range(1, len(config.layer_sizes)):
            layers.add_module(
                f"linear_{i}",
                nn.Linear(config.layer_sizes[i - 1], config.layer_sizes[i]),
            )
            if config.batch_norm:
                layers.add_module(
                    f"batch_norm_{i}", nn.BatchNorm1d(config.layer_sizes[i])
                )
            if config.layer_norm:
                layers.add_module(
                    f"layer_norm_{i}", nn.LayerNorm(config.layer_sizes[i])
                )
            if config.use_glu:
                layers.add_module(f"glu_{i}", nn.GLU())
            else:
                layers.add_module(
                    f"activation_{i}", self.hparams.get("activation", config.activation)
                )
            if config.dropout > 0.0:
                layers.add_module(f"dropout_{i}", nn.Dropout(config.dropout))

        # Get the last layer size (handles case when layer_sizes has only 1 element)
        last_layer_size = config.layer_sizes[-1]
        last_layer_idx = len(config.layer_sizes)
        layers.add_module(
            f"linear_{last_layer_idx}",
            nn.Linear(last_layer_size, self.num_classes),
        )
        return layers

    def _create_interaction_networks(self, num_feature_info, cat_feature_info, config):
        """
        Creates networks for modeling feature interactions.

        Parameters
        ----------
        num_feature_info : dict
            Information about numerical features.
        cat_feature_info : dict
            Information about categorical features.
        config : DefaultNAMConfig
            Configuration dataclass containing model hyperparameters.
        """

        self.interaction_networks = nn.ModuleDict()
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        # Add pairwise and higher interactions up to the specified degree
        for degree in range(2, self.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                interaction_name = ":".join(interaction)  # e.g., "feature1_feature2"
                input_dim = 0

                # Calculate input dimension for the interaction
                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature][
                            "dimension"
                        ]  # Numerical features
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["dimension"]

                self.interaction_networks[interaction_name] = self._create_subnetwork(
                    input_dim, config
                )

    def _interaction_forward(self, num_features: dict, cat_features: dict):
        """
        Forward pass for the interaction networks.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Outputs from the interaction networks, keyed by interaction names.
        """
        # Handle interaction networks
        interaction_outputs = {}
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {
                **num_features,
                **cat_features,
            }  # Combine numerical and categorical features
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                )
                interaction_output = interaction_network(
                    torch.tensor(input_features, dtype=torch.float32)
                )
                interaction_outputs[interaction_name] = interaction_output

        return interaction_outputs

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Dictionary containing the output tensor and the original feature values.
        """
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name])
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = self._interaction_forward(
            num_features=num_features, cat_features=cat_features
        )

        # Sum all feature outputs (main effects) and interaction outputs
        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        # Concatenate all feature outputs: [batch_size, num_features * num_classes]
        concatenated = torch.cat(all_outputs, dim=1)
        # Apply feature dropout
        concatenated = self.feature_dropout(concatenated)

        # Reshape to [batch_size, num_features, num_classes] and sum across features
        num_features_total = len(all_outputs)
        if self.num_classes > 1:
            # Reshape to [batch_size, num_features, num_classes] and sum
            x = concatenated.view(-1, num_features_total, self.num_classes).sum(dim=1)
        else:
            # For single output, sum and keep dimension
            x = concatenated.sum(dim=1).unsqueeze(-1)

        # intercept
        if self.intercept is not None:
            x += self.intercept

        # Combine the output tensor with the original feature values
        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result
configs/nam_config.py
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn


@dataclass
class DefaultNAMConfig:
    """
    Configuration class for the default NAM with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the MLP.
    activation : callable, default=nn.SELU()
        Activation function for the MLP layers.
    skip_layers : bool, default=False
        Whether to skip layers in the MLP.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the MLP.
    skip_connections : bool, default=False
        Whether to use skip connections in the MLP.
    batch_norm : bool, default=False
        Whether to use batch normalization in the MLP layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the MLP layers.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    layer_sizes: tuple = (128, 128, 32)
    activation: Any = nn.ReLU()
    skip_layers: bool = False
    dropout: float = 0.1
    norm: Optional[str] = None
    use_glu: bool = False
    skip_connections: bool = False
    batch_norm: bool = False
    layer_norm: bool = False
    interaction_degree: Optional[int] = None
    intercept: bool = True
    feature_dropout: float = 0.0
utils/distributions.py
import numpy as np
import torch
import torch.distributions as dist


class BaseDistribution(torch.nn.Module):
    """
    The base class for various statistical distributions, providing a common interface and utilities.

    This class defines the basic structure and methods that are inherited by specific distribution
    classes, allowing for the implementation of custom distributions with specific parameter transformations
    and loss computations.

    Attributes:
        _name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
        param_count (int): The number of parameters for the distribution.
        predefined_transforms (dict): A dictionary of predefined transformation functions for parameters.

    Parameters:
        name (str): The name of the distribution.
        param_names (list of str): A list of names for the parameters of the distribution.
    """

    def __init__(self, name, param_names):
        super(BaseDistribution, self).__init__()

        self._name = name
        self.param_names = param_names
        self.param_count = len(param_names)
        # Predefined transformation functions accessible to all subclasses
        self.predefined_transforms = {
            "positive": torch.nn.functional.softplus,
            "none": lambda x: x,
            "square": lambda x: x**2,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "probabilities": lambda x: torch.softmax(x, dim=-1),
            "log": lambda x: torch.log(x + 1e-6),
            "sort": lambda x: torch.cumsum(torch.nn.functional.softplus(x), dim=-1),
        }

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def get_transform(self, transform_name):
        """
        Retrieve a transformation function by name, or return the function if it's custom.
        """
        if callable(transform_name):
            # Custom transformation function provided
            return transform_name
        return self.predefined_transforms.get(
            transform_name, lambda x: x
        )  # Default to 'none'

    def compute_loss(self, predictions, y_true):
        """
        Computes the loss (e.g., negative log likelihood) for the distribution given predictions and true values.

        This method must be implemented by subclasses.

        Parameters:
            predictions (torch.Tensor): The predicted parameters of the distribution.
            y_true (torch.Tensor): The true values.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_nll(self, y_true, y_pred):
        """
        Evaluates the negative log likelihood (NLL) for given true values and predictions.

        Parameters:
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.

        Returns:
            dict: A dictionary containing the NLL value.
        """

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Compute NLL using the provided loss function
        nll_loss_tensor = self.compute_loss(y_pred_tensor, y_true_tensor)

        # Convert the NLL loss tensor back to a numpy array and return
        return {
            "NLL": nll_loss_tensor.detach().numpy(),
        }

    def forward(self, predictions):
        """
        Apply the appropriate transformations to the predicted parameters.

        Parameters:
            predictions (torch.Tensor): The predicted parameters of the distribution.

        Returns:
            torch.Tensor: A tensor with transformed parameters.
        """
        transformed_params = []
        for idx, param_name in enumerate(self.param_names):
            transform_func = self.get_transform(
                getattr(self, f"{param_name}_transform", "none")
            )
            transformed_params.append(transform_func(predictions[:, idx]).unsqueeze(1))
        return torch.cat(transformed_params, dim=1)


class NormalDistribution(BaseDistribution):
    """
    Represents a Normal (Gaussian) distribution with parameters for mean and variance, including functionality
    for transforming these parameters and computing the loss.

    Inherits from BaseDistribution.

    Parameters:
        name (str): The name of the distribution. Defaults to "Normal".
        mean_transform (str or callable): The transformation for the mean parameter. Defaults to "none".
        var_transform (str or callable): The transformation for the variance parameter. Defaults to "positive".
    """

    def __init__(self, name="Normal", mean_transform="none", var_transform="positive"):
        param_names = [
            "mean",
            "variance",
        ]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.variance_transform = self.get_transform(var_transform)

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        variance = self.variance_transform(
            predictions[:, self.param_names.index("variance")]
        )

        normal_dist = dist.Normal(mean, variance)

        nll = -normal_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(
            y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
        )
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(
                y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
            )
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class PoissonDistribution(BaseDistribution):
    """
    Represents a Poisson distribution, typically used for modeling count data or the number of events
    occurring within a fixed interval of time or space. This class extends the BaseDistribution and
    includes parameter transformation and loss computation specific to the Poisson distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Poisson".
        rate_transform (str or callable): Transformation to apply to the rate parameter to ensure it remains positive.
    """

    def __init__(self, name="Poisson", rate_transform="positive"):
        param_names = ["rate"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Poisson distribution with the transformed parameter
        poisson_dist = dist.Poisson(rate)

        # Compute the negative log-likelihood
        nll = -poisson_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        rate = self.rate_transform(y_pred_tensor[:, self.param_names.index("rate")])

        mse_loss = torch.nn.functional.mse_loss(y_true_tensor, rate)
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = torch.nn.functional.l1_loss(y_true_tensor, rate).detach().numpy()
        poisson_deviance = 2 * torch.sum(
            y_true_tensor * torch.log(y_true_tensor / rate) - (y_true_tensor - rate)
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["poisson_deviance"] = poisson_deviance.detach().numpy()

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class InverseGammaDistribution(BaseDistribution):
    """
    Represents an Inverse Gamma distribution, often used as a prior distribution in Bayesian statistics,
    especially for scale parameters in other distributions. This class extends BaseDistribution and includes
    parameter transformation and loss computation specific to the Inverse Gamma distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "InverseGamma".
        shape_transform (str or callable): Transformation for the shape parameter to ensure it remains positive.
        scale_transform (str or callable): Transformation for the scale parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="InverseGamma",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "shape",
            "scale",
        ]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        inverse_gamma_dist = dist.InverseGamma(shape, scale)
        # Compute the negative log-likelihood
        nll = -inverse_gamma_dist.log_prob(y_true).mean()
        return nll


class BetaDistribution(BaseDistribution):
    """
    Represents a Beta distribution, a continuous distribution defined on the interval [0, 1], commonly used
    in Bayesian statistics for modeling probabilities. This class extends BaseDistribution and includes parameter
    transformation and loss computation specific to the Beta distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Beta".
        shape_transform (str or callable): Transformation for the alpha (shape) parameter to ensure it remains positive.
        scale_transform (str or callable): Transformation for the beta (scale) parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="Beta",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "alpha",
            "beta",
        ]
        super().__init__(name, param_names)

        self.alpha_transform = self.get_transform(shape_transform)
        self.beta_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        alpha = self.alpha_transform(predictions[:, self.param_names.index("alpha")])
        beta = self.beta_transform(predictions[:, self.param_names.index("beta")])

        beta_dist = dist.Beta(alpha, beta)
        # Compute the negative log-likelihood
        nll = -beta_dist.log_prob(y_true).mean()
        return nll


class DirichletDistribution(BaseDistribution):
    """
    Represents a Dirichlet distribution, a multivariate generalization of the Beta distribution. It is commonly
    used in Bayesian statistics for modeling multinomial distribution probabilities. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Dirichlet distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Dirichlet".
        concentration_transform (str or callable): Transformation to apply to concentration parameters to ensure they remain positive.
    """

    def __init__(self, name="Dirichlet", concentration_transform="positive"):
        # For Dirichlet, param_names could be dynamically set based on the dimensionality of alpha
        # For simplicity, we're not specifying individual names for each concentration parameter
        param_names = ["concentration"]  # This is a simplification
        super().__init__(name, param_names)
        # Retrieve transformation function for concentration parameters
        self.concentration_transform = self.get_transform(concentration_transform)

    def compute_loss(self, predictions, y_true):
        # Apply the transformation to ensure all concentration parameters are positive
        # Assuming predictions is a 2D tensor where each row is a set of concentration parameters for a Dirichlet distribution
        concentration = self.concentration_transform(predictions)

        dirichlet_dist = dist.Dirichlet(concentration)

        nll = -dirichlet_dist.log_prob(y_true).mean()
        return nll


class GammaDistribution(BaseDistribution):
    """
    Represents a Gamma distribution, a two-parameter family of continuous probability distributions. It's
    widely used in various fields of science for modeling a wide range of phenomena. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Gamma distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Gamma".
        shape_transform (str or callable): Transformation for the shape parameter to ensure it remains positive.
        rate_transform (str or callable): Transformation for the rate parameter to ensure it remains positive.
    """

    def __init__(
        self, name="Gamma", shape_transform="positive", rate_transform="positive"
    ):
        param_names = ["shape", "rate"]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Gamma distribution with the transformed parameters
        gamma_dist = dist.Gamma(shape, rate)

        # Compute the negative log-likelihood
        nll = -gamma_dist.log_prob(y_true).mean()
        return nll


class StudentTDistribution(BaseDistribution):
    """
    Represents a Student's t-distribution, a family of continuous probability distributions that arise when
    estimating the mean of a normally distributed population in situations where the sample size is small.
    This class extends BaseDistribution and includes parameter transformation and loss computation specific
    to the Student's t-distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "StudentT".
        df_transform (str or callable): Transformation for the degrees of freedom parameter to ensure it remains positive.
        loc_transform (str or callable): Transformation for the location parameter.
        scale_transform (str or callable): Transformation for the scale parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="StudentT",
        df_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        param_names = ["df", "loc", "scale"]
        super().__init__(name, param_names)

        self.df_transform = self.get_transform(df_transform)
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        df = self.df_transform(predictions[:, self.param_names.index("df")])
        loc = self.loc_transform(predictions[:, self.param_names.index("loc")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        student_t_dist = dist.StudentT(df, loc, scale)

        nll = -student_t_dist.log_prob(y_true).mean()
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(
            y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")]
        )
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(
                y_true_tensor, y_pred_tensor[:, self.param_names.index("loc")]
            )
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics


class NegativeBinomialDistribution(BaseDistribution):
    """
    Represents a Negative Binomial distribution, often used for count data and modeling the number of failures
    before a specified number of successes occurs in a series of Bernoulli trials. This class extends
    BaseDistribution and includes parameter transformation and loss computation specific to the Negative Binomial distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "NegativeBinomial".
        mean_transform (str or callable): Transformation for the mean parameter to ensure it remains positive.
        dispersion_transform (str or callable): Transformation for the dispersion parameter to ensure it remains positive.
    """

    def __init__(
        self,
        name="NegativeBinomial",
        mean_transform="positive",
        dispersion_transform="positive",
    ):
        param_names = ["mean", "dispersion"]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.dispersion_transform = self.get_transform(dispersion_transform)

    def compute_loss(self, predictions, y_true):
        # Apply transformations to ensure mean and dispersion parameters are positive
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        dispersion = self.dispersion_transform(
            predictions[:, self.param_names.index("dispersion")]
        )

        # Calculate the probability (p) and number of successes (r) from mean and dispersion
        # These calculations follow from the mean and variance of the negative binomial distribution
        # where variance = mean + mean^2 / dispersion
        r = 1 / dispersion
        p = r / (r + mean)

        # Define the Negative Binomial distribution with the transformed parameters
        negative_binomial_dist = dist.NegativeBinomial(total_count=r, probs=p)

        # Compute the negative log-likelihood
        nll = -negative_binomial_dist.log_prob(y_true).mean()
        return nll


class CategoricalDistribution(BaseDistribution):
    """
    Represents a Categorical distribution, a discrete distribution that describes the possible results of a
    random variable that can take on one of K possible categories, with the probability of each category
    separately specified. This class extends BaseDistribution and includes parameter transformation and loss
    computation specific to the Categorical distribution.

    Parameters:
        name (str): The name of the distribution, defaulted to "Categorical".
        prob_transform (str or callable): Transformation for the probabilities to ensure they remain valid (i.e., non-negative and sum to 1).
    """

    def __init__(self, name="Categorical", prob_transform="probabilities"):
        param_names = ["probs"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.probs_transform = self.get_transform(prob_transform)

    def compute_loss(self, predictions, y_true):
        probs = self.probs_transform(predictions)

        # Define the Poisson distribution with the transformed parameter
        cat_dist = dist.Categorical(probs=probs)

        # Compute the negative log-likelihood
        nll = -cat_dist.log_prob(y_true).mean()
        return nll


class Quantile(BaseDistribution):
    """
    Quantile Regression Loss class.

    This class computes the quantile loss (also known as pinball loss) for a set of quantiles.
    It is used to handle quantile regression tasks where we aim to predict a given quantile of the target distribution.

    Parameters
    ----------
    name : str, optional
        The name of the distribution, by default "Quantile".
    quantiles : list of float, optional
        A list of quantiles to be used for computing the loss, by default [0.25, 0.5, 0.75].

    Attributes
    ----------
    quantiles : list of float
        List of quantiles for which the pinball loss is computed.

    Methods
    -------
    compute_loss(predictions, y_true)
        Computes the quantile regression loss between the predictions and true values.
    """

    def __init__(self, name="Quantile", quantiles=None):
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        param_names = [
            f"q_{q}" for q in quantiles
        ]  # Use string representations of quantiles
        super().__init__(name, param_names)
        self.quantiles = quantiles

    def compute_loss(self, predictions, y_true):

        assert not y_true.requires_grad  # Ensure y_true does not require gradients
        assert predictions.size(0) == y_true.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - predictions[:, i]  # Calculate errors for each quantile
            # Compute the pinball loss
            quantile_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(quantile_loss)

        # Sum losses across quantiles and compute mean
        loss = torch.mean(torch.stack(losses, dim=1).sum(dim=1))
        return loss


class RobustNormalDistribution(BaseDistribution):
    """
    Represents a Normal (Gaussian) distribution with parameters for mean and variance, including functionality
    for transforming these parameters and computing the loss.

    Inherits from BaseDistribution.

    Parameters:
        name (str): The name of the distribution. Defaults to "Normal".
        mean_transform (str or callable): The transformation for the mean parameter. Defaults to "none".
        var_transform (str or callable): The transformation for the variance parameter. Defaults to "positive".
    """

    def __init__(
        self, name="Normal", mean_transform="none", var_transform="positive", rob=0.1
    ):
        param_names = [
            "mean",
            "variance",
        ]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.variance_transform = self.get_transform(var_transform)
        self.rob = rob

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        variance = self.variance_transform(
            predictions[:, self.param_names.index("variance")]
        )

        normal_dist = dist.Normal(mean, variance)
        log_likelihood = normal_dist.log_prob(y_true)

        if self.rob is not None:
            rob_tensor = torch.tensor(
                self.rob, device=log_likelihood.device, dtype=log_likelihood.dtype
            )
            log_likelihood = torch.log(
                (1 + torch.exp(log_likelihood + rob_tensor))
                / (1 + torch.exp(rob_tensor))
            )

        nll = -torch.mean(log_likelihood)
        return nll

    def evaluate_nll(self, y_true, y_pred):
        metrics = super().evaluate_nll(y_true, y_pred)

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        mse_loss = torch.nn.functional.mse_loss(
            y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
        )
        rmse = np.sqrt(mse_loss.detach().numpy())
        mae = (
            torch.nn.functional.l1_loss(
                y_true_tensor, y_pred_tensor[:, self.param_names.index("mean")]
            )
            .detach()
            .numpy()
        )

        metrics["mse"] = mse_loss.detach().numpy()
        metrics["mae"] = mae
        metrics["rmse"] = rmse

        # Convert the NLL loss tensor back to a numpy array and return
        return metrics
readme.md
 NAMpy provides interpretable additive neural models for tabular data, with support for **regression**, **classification**, and **distributional regression** tasks. Models implement scikit-learn's `BaseEstimator` interface, so they integrate with standard scikit-learn workflows for fitting, prediction, and evaluation.

## Key Features

- **Scikit-learn Compatible**: Consistent API with sklearn estimators
- **10+ Model Architectures**: NAM, GPNAM, NBM, NATT, NAMformer, and more
- **Three Task Types**: Regression, classification, and distributional regression (LSS)
- **Interpretable**: Additive structure supports feature-level interpretation
- **PyTorch Backend**: Built on modern deep learning tooling
- **Extensible**: Interfaces for custom model implementations

Most models are available for `regression`, `classification` and distributional regression, denoted by `LSS`.
Some models are specialized: `QNAM` is distributional-only, while `TreeNAM` and `SNAM` are currently regression-only.

## Integrated Models:

1. NAM
2. GPNAM
3. NBM
4. NATT
5. NAMformer
6. QNAM
7. Linear Regression (Neural)
8. NodeGAM
9. TreeNAM (Regressor)
10. SNAM (Regressor)
## Quick Start

### Fit a Model

All NAMpy models implement sklearn `BaseEstimator` methods, including `.fit`. This enables standard tooling such as scikit-learn model selection and evaluation utilities.

```python
from nampy.models import NAMClassifier

# Initialize and fit your model
model = NAMClassifier(
    numerical_preprocessing="ple",
    n_bins=50
)

# X can be a DataFrame or any array-like that can be converted to a DataFrame.
model.fit(X, y, max_epochs=150, lr=1e-04)
```

### Make Predictions

Use the standard prediction methods:

```python
# Simple predictions
preds = model.predict(X)

# Predict probabilities (for classification)
preds = model.predict_proba(X)
```

### Regression Example

```python
from nampy.models import NAMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = NAMRegressor(numerical_preprocessing="standardization")
model.fit(X_train, y_train, max_epochs=100, lr=1e-3)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")

```

## Distributional Regression with NAMLSS

NAMpy provides distributional regression through the `NAMLSS` module, which models the full response distribution rather than only the mean. This is useful when variability, skewness, or kurtosis are as important as the central tendency. Most models in NAMpy are also available as distributional models.

### Key Features of NAMLSS:

- **Full Distribution Modeling**: Unlike traditional regression models that predict a single value (e.g., the mean), `NAMLSS` models the entire distribution of the response variable. This supports predictions of quantiles, variance, and higher moments.
- **Customizable Distribution Types**: `NAMLSS` supports a variety of distribution families (e.g., Normal, Poisson, Gamma, Beta), making it suitable for response variables ranging from continuous to count data.
- **Location, Scale, Shape Parameters**: The model predicts parameters corresponding to the location, scale, and shape of the distribution, providing direct access to distributional characteristics.
- **Enhanced Predictive Uncertainty**: By modeling the full distribution, `NAMLSS` provides explicit predictive uncertainty estimates for downstream decisions.



### Available Distribution Classes:

`NAMLSS` includes a range of distribution classes for statistical modeling needs. The available distribution classes include:

- `normal`: Normal Distribution for modeling continuous data with a symmetric distribution around the mean.
- `poisson`: Poisson Distribution for modeling count data that for instance represent the number of events occurring within a fixed interval.
- `gamma`: Gamma Distribution for modeling continuous data that is skewed and bounded at zero, often used for waiting times.
- `beta`: Beta Distribution for modeling data that is bounded between 0 and 1, useful for proportions and percentages.
- `dirichlet`: Dirichlet Distribution for modeling multivariate data where individual components are correlated, and the sum is constrained to 1.
- `studentt`: Student's T-Distribution for modeling data with heavier tails than the normal distribution, useful when the sample size is small.
- `negativebinom`: Negative Binomial Distribution for modeling count data with over-dispersion relative to the Poisson distribution.
- `inversegamma`: Inverse Gamma Distribution, often used as a prior distribution in Bayesian inference for scale parameters.
- `categorical`: Categorical Distribution for modeling categorical data with more than two categories.
- `quantile`: Quantile regression for estimating conditional quantiles.
- `robustnormal`: Robust Normal Distribution for heavy-tailed targets.

These distribution classes allow `NAMLSS` to model a wide variety of data types and distributions.


### Getting Started with NAMLSS:

To integrate distributional regression into your workflow with `NAMLSS`, initialize the model with the desired configuration, similar to other NAMpy models:

```python
from nampy.models import NAMLSS

# Initialize the NAMLSS model
model = NAMLSS()

# Fit the model to your data
model.fit(
    X, 
    y, 
    max_epochs=150, 
    lr=1e-04, 
    patience=10,     
    family="normal"  # define your distribution
)

# Predict distribution parameters
dist_params = model.predict(X_test)
```


## Implement Your Own Model

NAMpy supports integration of custom models into the existing logic. Implement a PyTorch model and define its forward pass, but inherit from NAMpy's `BaseModel` rather than `nn.Module`. Each NAMpy model takes three main arguments: the number of classes (e.g., 1 for regression or 2 for binary classification), `cat_feature_info`, and `num_feature_info` for categorical and numerical feature information, respectively. These are passed as dictionaries, with variable names as the keys. Additionally, you can provide a config argument, which can either be a custom configuration or one of the provided default configs.

A key aspect of NAMpy is that the inputs to the forward passes are dictionaries of tensors. This supports models that treat different data types differently and directly maps feature/variable predictions to input features in additive models. 

Example workflow for a custom model:


1. First, define your config:
Use a dataclass to specify hyperparameters and other settings for your model.

```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
```

2. Second, define your model:
Define your custom model as you would for an `nn.Module`. The main difference is that you will inherit from `BaseModel` and use the provided feature information to construct your layers. To integrate your model into the existing API, define the architecture and the forward pass. Note that the forward pass must return a dictionary with the key "output" for the final model prediction. This can be multi-dimensional, for example for classification or distributional regression. Beyond that, the dictionary can contain anything but often includes single feature/variable predictions for further processing or plotting.

```python
from nampy.basemodels import BaseModel
import torch
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        total_input_size = sum([input_shape for input_shape in num_feature_info.values()]) + len(cat_feature_info)
        
        # Define a simple MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, 128),  # Adjust the hidden layer size as needed
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Dictionary containing the output tensor and the original feature values.
        """
        # Concatenate all numerical features into a single tensor
        num_features_tensor = torch.cat([num_features[key] for key in num_features.keys()], dim=1)

        # Concatenate all categorical features into a single tensor
        cat_features_tensor = torch.cat([cat_features[key] for key in cat_features.keys()], dim=1)

        # Concatenate all features into a single input tensor
        input_tensor = torch.cat([num_features_tensor, cat_features_tensor], dim=1)

        # Pass the concatenated tensor through the MLP
        output = self.mlp(input_tensor)

        # return a dictionary, with the key "output" for the final predictions
        # This is used, for when the model (e.g. for plotting) also returns feature predictions
        return {"output": output}


```

3. Leverage the NAMpy API:
You can build a regression, classification or distributional regression model that can leverage all of NAMpy's built-in methods, by using the following:

```python
from nampy.models import SklearnBaseRegressor

class MyRegressor(SklearnBaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(model=MyCustomModel, config=MyConfig, **kwargs)
```

4. Train and evaluate your model:
You can now fit, evaluate, and predict with your custom model using the same APIs as other NAMpy models. For classification or distributional regression, inherit from `SklearnBaseClassifier` or `SklearnBaseLSS` respectively.

```python
regressor = MyRegressor(numerical_preprocessing="ple")
regressor.fit(X_train, y_train, max_epochs=50)
predictions = regressor.predict(X_test)
```
