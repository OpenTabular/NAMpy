import torch
import torch.nn as nn

from ...contracts import validate_feature_names


class BaseModel(nn.Module):
    extra_reserved_feature_names = ()

    def __init__(self, **kwargs):
        """
        Initializes the BaseModel with given hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Hyperparameters to be saved and used in the model.
        """
        super(BaseModel, self).__init__()
        self.hparams = kwargs

    def save_hyperparameters(self, ignore=None):
        """
        Saves the hyperparameters while ignoring specified keys.

        Parameters
        ----------
        ignore : list, optional
            List of keys to ignore while saving hyperparameters, by default [].
        """
        if ignore is None:
            ignore = []
        self.hparams = {k: v for k, v in self.hparams.items() if k not in ignore}
        for key, value in self.hparams.items():
            setattr(self, key, value)

    def _validate_features(self, num_feature_info, cat_feature_info):
        """
        Validate feature names against the forward-output key grammar.

        Architectures call this exactly once, at the point they receive the
        feature-info dicts. The rules live in ``nampy.neural.contracts``.
        """
        validate_feature_names(
            list(cat_feature_info) + list(num_feature_info),
            owner=type(self).__name__,
            extra_reserved=self.extra_reserved_feature_names,
        )

    def get_device(self):
        """
        Get the device on which the model is located.

        Returns
        -------
        torch.device
            Device on which the model is located.
        """
        return next(self.parameters()).device


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules
        # I change the type to torch.float32 to use apex 16 precision training

    def initialize(self, *args, **kwargs):
        """initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)
