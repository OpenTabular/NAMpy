import torch.nn as nn
from torch.nn.modules.activation import ReLU
from ..configs.nbm_config import DefaultNBMConfig


class ConceptNNBasesNary(nn.Module):
    """Neural Network learning bases."""

    def __init__(
        self,
        config,
    ) -> None:
        """Initializes ConceptNNBases hyperparameters.
        Args:
            order: Order of N-ary concept interatctions.
            num_bases: Number of bases learned.
            hidden_dims: Number of units in hidden layers.
            dropout: Coefficient for dropout regularization.
            batchnorm (True): Whether to use batchnorm or not.
        """
        super(ConceptNNBasesNary, self).__init__()

        assert (
            config.order > 0
        ), "Order of N-ary interactions has to be larger than '0'."

        layers = []
        self._model_depth = len(config.hidden_dims) + 1
        self._batchnorm = config.batch_norm

        # First input_dim depends on the N-ary order
        input_dim = config.order
        for dim in config.hidden_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            if self._batchnorm is True:
                layers.append(nn.BatchNorm1d(dim))
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            layers.append(config.activation)
            input_dim = dim

        # Last MLP layer
        layers.append(nn.Linear(in_features=input_dim, out_features=config.num_bases))
        # Add batchnorm and relu for bases
        if self._batchnorm is True:
            layers.append(nn.BatchNorm1d(config.num_bases))
        layers.append(config.activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
