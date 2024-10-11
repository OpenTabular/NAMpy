import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch_utils.odst import GAM_ODST
from ..configs.nodegam_config import DefaultNodeGAMConfig
from .basemodel import BaseModel

def entmax15(input):
    """Placeholder for the entmax15 activation function."""
    # Replace this with the actual entmax15 implementation
    return F.softmax(input, dim=-1)


class NodeGAM(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNodeGAMConfig = DefaultNodeGAMConfig(),
        **kwargs,
    ):
        """
        Initializes the NodeGAM model with the given configuration.

        Parameters
        ----------
        cat_feature_info : dict
            Information about categorical features.
        num_feature_info : dict
            Information about numerical features.
        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultNodeGAMConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNodeGAMConfig().
        num_layers : int, optional
            Number of GAM_ODST layers, by default 2.
        device : str, optional
            Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
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

        # Initialize sub-networks for each feature
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, input_shape in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork2(
                input_shape, config
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, input_shape in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork2(
                1, config
            )  # Categorical features are typically encoded as single values

    def _create_subnetwork2(self, input_dim, config):
        """
        Creates a NodeGAM subnetwork for a single feature.

        Parameters
        ----------
        input_dim : int
            Dimension of the input feature.
        config : DefaultNodeGAMConfig
            Configuration dataclass containing hyperparameters.

        Returns
        -------
        nn.Sequential
            Subnetwork for the feature.
        """
        # Initialize GAM_ODST layers
        layers = nn.ModuleDict({
            f'layer_{i}': GAM_ODST(
                    in_features=input_dim
                    if i == 0
                    else config.num_trees * config.tree_dim,
                    num_trees=config.num_trees,
                    depth=config.depth,
                    tree_dim=config.tree_dim,
                    choice_function=entmax15,
                    bin_function=lambda x: (0.5 * x + 0.5).clamp_(0, 1),
                    initialize_response_=nn.init.normal_,
                    initialize_selection_logits_=nn.init.uniform_,
                    colsample_bytree=config.colsample_bytree,
                    selectors_detach=config.selectors_detach,
                    fs_normalize=config.fs_normalize,
                    ga2m=config.ga2m
                )
                for i in range(config.num_layers)
            }
        )
        return layers

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NodeGAM model.

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
            feature_output = feature_network(cat_features[feature_name])
            cat_outputs[feature_name] = feature_output

        # Sum all feature outputs
        all_outputs = list(num_outputs.values()) + list(cat_outputs.values())
        x = torch.stack(all_outputs, dim=1).sum(dim=1)

        # Combine the output tensor with the original feature values
        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)

        return result