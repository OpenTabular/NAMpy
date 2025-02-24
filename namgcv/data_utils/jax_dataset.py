from __future__ import annotations

import os
from typing import Literal, Tuple, Dict, Any

from mile.dataset.base import BaseLoader
from mile.config.data import DataConfig, Source, DatasetType, Task

import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import seaborn as sns


class TabularAdditiveModelDataLoader(BaseLoader):
    """
    A DataLoader that handles dictionary-formatted data of the structure:

    {
        "numerical": {
            "numerical_1": np.ndarray([...]),
            "numerical_2": np.ndarray([...]),
            ...
        },
        "categorical": {
            "categorical_1": np.ndarray([...]),
            "categorical_2": np.ndarray([...]),
            ...
        }
    }
    """

    def __init__(
        self,
        config: DataConfig,
        rng: jnp.ndarray,
        data_dict: dict,
        target_key: str = "target",
    ):
        """
        Initializer for dataset class.

        Parameters
        ----------
        config: DataConfig
            The configuration object containing dataset parameters.
        rng: jnp.ndarray
            A PRNGKey for random operations (shuffling, batching).
        data_dict: dict
            Dictionary-of-dictionaries containing numerical/categorical
            features (and optionally the target).
        target_key: str
              The key under which the target/target is stored in `data_dict`.
        """

        super().__init__(config)
        self._key = rng
        self.target_key = target_key

        assert self.config.data_type == DatasetType.TABULAR, (
            "TabularAdditiveModelDataLoader is designed for tabular-style tasks."
        )

        self.data = self._load_data(
            raw_dict=data_dict,
            shuffle=True,
            normalize=self.config.normalize
        )
        if self.config.datapoint_limit:
            self._apply_datapoint_limit()

        N = self._num_datapoints(self.data)
        train_idx = int(N * self.config.train_split)
        valid_idx = int(N * (self.config.train_split + self.config.valid_split))

        self.data_train = self._slice_data(self.data, 0, train_idx)
        self.data_valid = self._slice_data(self.data, train_idx, valid_idx)
        self.data_test  = self._slice_data(self.data, valid_idx, N)

    def __str__(self):
        """Return an informative string representation of the dataloader."""
        num_train = self._num_datapoints(self.data_train)
        num_valid = self._num_datapoints(self.data_valid)
        num_test  = self._num_datapoints(self.data_test)
        return (
            f"{super().__str__()}\n"
            f" | TabularAdditiveModelDataLoader\n"
            f" | Train: {num_train}\n"
            f" | Valid: {num_valid}\n"
            f" | Test:  {num_test}"
        )

    @property
    def key(self) -> jnp.ndarray:
        """Returns a fresh PRNG key and updates the internal key."""
        self._key, new_key = jax.random.split(self._key)
        return new_key

    @property
    def train_x(self):
        """Return the training features (dict of numerical/categorical)."""
        return self._split_features(self.data_train)

    @property
    def train_y(self):
        """Return the training targets."""
        return self._split_targets(self.data_train)

    @property
    def valid_x(self):
        return self._split_features(self.data_valid)

    @property
    def valid_y(self):
        return self._split_targets(self.data_valid)

    @property
    def test_x(self):
        return self._split_features(self.data_test)

    @property
    def test_y(self):
        return self._split_targets(self.data_test)

    def iter(
        self,
        split: Literal["train", "valid", "test"],
        batch_size: int | None = None,
        n_devices: int = 1,
    ):
        """
        Returns a generator that yields batches of the form:
            {
                "feature": {
                    "numerical": { ... },
                    "categorical": { ... }
                },
                "target": jnp.ndarray
            }
        """
        assert split in ("train", "valid", "test"), "split must be train, valid, or test."

        if split == "train":
            data_split = self.data_train
        elif split == "valid":
            data_split = self.data_valid
        else:
            data_split = self.data_test

        yield from self._iter(data_split, batch_size, n_devices)

    def shuffle(
            self,
            split: Literal["train", "valid", "test"] = "train"
    ):
        """
        Shuffle the specified split in-place. This applies the same random permutation
        across all numerical/categorical arrays (and target), preserving alignment.
        """
        if split == "train":
            self.data_train = self._shuffle_dict(self.data_train)
        elif split == "valid":
            self.data_valid = self._shuffle_dict(self.data_valid)
        else:
            self.data_test = self._shuffle_dict(self.data_test)

    def __len__(self):
        return self._num_datapoints(self.data)  # Length across all splits.

    def _load_data(
            self,
            raw_dict: dict,
            shuffle: bool,
            normalize: bool = True
    ):
        """
        Convert raw input dictionary into a unified structure:
          {
             "numerical": { "num_1": jnp.array([...]), ... },
             "categorical": { "cat_1": jnp.array([...]), ... },
             "target": jnp.array([...])     # if present
          }
        Optionally normalizes *only numerical* data. If your target is in numerical,
        you may exclude it from normalization.
        """
        data = {"numerical": {}, "categorical": {}}

        for k, v in raw_dict.get("numerical", {}).items():
            data["numerical"][k] = jnp.array(v)
        for k, v in raw_dict.get("categorical", {}).items():
            data["categorical"][k] = jnp.array(v)
        data["target"] = jnp.array(raw_dict[self.target_key])

        if normalize:
            for num_k, arr in data["numerical"].items():
                # If there's a chance 'arr' is the target, skip it. Adjust logic as needed.
                arr_mean = arr.mean()
                arr_std  = arr.std()
                # Avoid division by zero
                arr_std  = jnp.where(arr_std == 0, 1.0, arr_std)
                data["numerical"][num_k] = (arr - arr_mean) / arr_std

        if shuffle:
            data = self._shuffle_dict(data)

        return data

    def _apply_datapoint_limit(self):
        """
        Clip the entire dictionary to the first N datapoints (as defined in self.config).
        """
        limit = self.config.datapoint_limit
        for subkey in ["numerical", "categorical"]:
            for k, arr in self.data[subkey].items():
                self.data[subkey][k] = arr[:limit]
        if "target" in self.data and self.data["target"] is not None:
            self.data["target"] = self.data["target"][:limit]

    def _num_datapoints(
            self,
            data_dict: dict
    ) -> int:
        """
        Return number of samples. We assume that all arrays in 'numerical', 'categorical',
        and 'target' have the same leading dimension. Here we use 'target' if present,
        otherwise we default to the first numerical array.
        """
        if "target" in data_dict and data_dict["target"] is not None:
            return data_dict["target"].shape[0]
        # or fallback to first numerical array
        for _, arr in data_dict["numerical"].items():
            return arr.shape[0]
        # if still nothing, fallback to first categorical
        for _, arr in data_dict["categorical"].items():
            return arr.shape[0]
        return 0

    def _shuffle_dict(
            self,
            data_dict: dict
    ) -> dict:
        """Shuffle all arrays in data_dict consistently using the same permutation."""
        n = self._num_datapoints(data_dict)
        if n == 0:
            return data_dict

        indices = jax.random.permutation(self.key, jnp.arange(n))
        # Create a new dictionary with shuffled arrays
        shuffled = {"numerical": {}, "categorical": {}}
        for subkey in ["numerical", "categorical"]:
            for k, arr in data_dict[subkey].items():
                shuffled[subkey][k] = arr[indices]

        if "target" in data_dict and data_dict["target"] is not None:
            shuffled["target"] = data_dict["target"][indices]
        else:
            shuffled["target"] = data_dict.get("target", None)

        return shuffled

    def _slice_data(
            self,
            data_dict: dict,
            start: int,
            end: int
    ) -> dict:
        """Return a slice of all arrays from start to end along axis 0."""
        sliced = {"numerical": {}, "categorical": {}}
        for subkey in ["numerical", "categorical"]:
            for k, arr in data_dict[subkey].items():
                sliced[subkey][k] = arr[start:end]

        if data_dict.get("target") is not None:
            sliced["target"] = data_dict["target"][start:end]
        else:
            sliced["target"] = None

        return sliced

    def _split_features(
            self,
            data_dict: dict
    ) -> dict:
        """
        Return a dict containing only 'numerical' and 'categorical' arrays,
        which can be considered the 'features'.
        """
        return {
            "numerical": data_dict["numerical"],
            "categorical": data_dict["categorical"],
        }

    def _split_targets(
            self,
            data_dict: dict
    ) -> jnp.ndarray | None:
        """
        Return the target array as a jnp.ndarray. If classification, cast to int32.
        """
        lbl = data_dict.get("target", None)
        if lbl is None:
            return None
        if self.config.task == Task.CLASSIFICATION:
            return lbl.astype(jnp.int32)
        return lbl

    def _iter(
        self,
        data_dict: dict,
        batch_size: int | None,
        n_devices: int = 1,
    ):
        """
        Generator that yields dictionaries of the form:
            {
                "feature": {
                    "numerical": {...},
                    "categorical": {...}
                },
                "target": jnp.ndarray
            }
        in batches. If `batch_size` is None, return the entire split at once.
        If `n_devices` > 1, replicate the data across devices.
        """
        total = self._num_datapoints(data_dict)
        if total == 0:
            return  # empty generator

        # If no batching, just yield the entire dataset as one batch
        if batch_size is None:
            # Possibly replicate if multiple devices
            if n_devices > 1:
                # replicate each sub-array across device dimension
                batched_data = self._replicate_data_dict(data_dict, n_devices)
                yield self._format_batch(batched_data)
            else:
                yield self._format_batch(data_dict)
            return

        # Otherwise, compute number of batches
        n_batches = total // batch_size
        # Truncate leftover
        truncated_end = n_batches * batch_size

        # We'll create a random permutation of indices, then split them
        indices = jax.random.permutation(self.key, jnp.arange(total))
        indices = indices[:truncated_end]  # drop leftover
        # Split the indices into n_batches
        indices_split = jnp.array_split(indices, n_batches)

        for i in range(n_batches):
            # For multi-device, we replicate each batch across devices with separate permutations
            if n_devices > 1:
                # We want a separate random permutation for each device
                device_indices = [
                    jax.random.permutation(self.key, jnp.arange(total))[:batch_size]
                    for _ in range(n_devices)
                ]
                # Stack into shape (n_devices, batch_size)
                device_indices = jnp.stack(device_indices, axis=0)
                # Gather from data
                batched_data = self._gather_indices(data_dict, device_indices)
            else:
                # Single device
                batched_data = self._gather_indices(data_dict, indices_split[i])

            yield self._format_batch(batched_data)

    def _gather_indices(
            self,
            data_dict: dict,
            indices: jnp.ndarray
    ) -> dict:
        """
        Gather sub-arrays from data_dict at the given indices. If indices has shape
        (n_devices, batch_size), we gather a device dimension as well.
        """
        # Indices shape can be (batch_size,) or (n_devices, batch_size)
        gathered = {"numerical": {}, "categorical": {}}

        for subkey in ["numerical", "categorical"]:
            gathered[subkey] = {
                k: arr[indices] for k, arr in data_dict[subkey].items()
            }

        if data_dict["target"] is not None:
            gathered["target"] = data_dict["target"][indices]
        else:
            gathered["target"] = None

        return gathered

    def _format_batch(
            self,
            data_dict: dict
    ) -> dict:
        """
        Convert data_dict into the final batch format:
            {
                "feature": {
                    "numerical": {...},
                    "categorical": {...}
                },
                "target": jnp.ndarray
            }
        """
        # The target might need int32 cast if classification
        target = data_dict["target"]
        if target is not None and self.config.task == Task.CLASSIFICATION:
            target = target.astype(jnp.int32)

        return {
            "feature": {
                "numerical": data_dict["numerical"],
                "categorical": data_dict["categorical"],
            },
            "target": target,
        }

    def _replicate_data_dict(
            self,
            data_dict: dict,
            n_devices: int
    ) -> dict:
        """
        Replicate each array in data_dict along a new axis(0), such that shape becomes
        (n_devices, ...).
        """
        replicated = {"numerical": {}, "categorical": {}}
        for subkey in ["numerical", "categorical"]:
            for k, arr in data_dict[subkey].items():
                replicated[subkey][k] = jnp.repeat(arr[None, ...], n_devices, axis=0)

        if data_dict["target"] is not None:
            replicated["target"] = jnp.repeat(data_dict["target"][None, ...], n_devices, axis=0)
        else:
            replicated["target"] = None

        return replicated
