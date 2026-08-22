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
        offsets=None,
        sample_weights=None,
    ):
        self.cat_features_list = cat_features_list  # Categorical features tensors
        self.num_features_list = num_features_list  # Numerical features tensors

        self.regression = regression
        self.cat_keys = cat_keys
        self.num_keys = num_keys
        # Per-sample additive offsets on the prediction scale ([N, 1] float32
        # tensor, or None for a zero offset). Batches always carry an offset
        # entry so the batch arity is uniform.
        self.offsets = offsets
        self.sample_weights = sample_weights
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
            for key, feature_tensor in zip(
                self.cat_keys, self.cat_features_list, strict=True
            )
        }
        num_features = {
            key: torch.as_tensor(feature_tensor[idx]).clone().detach().to(torch.float32)
            for key, feature_tensor in zip(
                self.num_keys, self.num_features_list, strict=True
            )
        }

        label = self.labels[idx]
        if self.regression:
            label = label.clone().detach().to(torch.float32)
        elif self.num_classes == 1:
            label = label.clone().detach().to(torch.float32)
        else:
            label = label.clone().detach().to(torch.long)

        if self.offsets is None:
            offset = torch.zeros(1, dtype=torch.float32)
        else:
            offset = self.offsets[idx].clone().detach().to(torch.float32)

        if self.sample_weights is None:
            sample_weight = torch.ones(1, dtype=torch.float32)
        else:
            sample_weight = (
                self.sample_weights[idx].clone().detach().to(torch.float32)
            )

        # Keep categorical and numerical features separate
        return cat_features, num_features, label, offset, sample_weight
