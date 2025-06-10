import torch
import torch.nn as nn
from ..configs.nodegam_config import DefaultNodeGAMConfig
from .basemodel import BaseModel
from ..arch_utils.nodegam_utils import GAMBlock, GAMAttBlock, EM15Temp

from nampy.arch_utils.nn_utils import entmoid15

class NodeGAM(BaseModel):
    """
    Neural Additive Model (NodeGAM) class using GAMBlock/GAMAttBlock architecture.

    This class implements a Neural Additive Model (NodeGAM) with support for numerical and
    categorical features, interaction terms, and various normalization layers.

    Attributes
    ----------
    models : list of GAMBlock/GAMAttBlock
        List of models for each parameter in the distribution family.
    feature_dropout : nn.Dropout
        Dropout layer for regularizing feature contributions.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNodeGAMConfig = DefaultNodeGAMConfig(),
        **kwargs,
    ):
        """
        Initializes the Neural Additive Model (NodeGAM) with the given configuration.

        Parameters
        ----------
        cat_feature_info : dict
            Dictionary providing information about categorical features (e.g., input dimensions).
        num_feature_info : dict
            Dictionary providing information about numerical features (e.g., input dimensions).
        num_classes : int, optional
            Number of output classes for classification tasks, by default 1.
        config : DefaultNodeGAMConfig, optional
            Configuration dataclass containing hyperparameters for the model, by default DefaultNodeGAMConfig.
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

        # Calculate total input dimension
        total_input_dim = sum(info["dimension"] for info in num_feature_info.values()) + sum(info["dimension"] for info in cat_feature_info.values())

        # Initialize choice function with temperature annealing
        choice_fn = EM15Temp(max_temp=1.0, min_temp=0.01, steps=config.anneal_steps)

        # Determine which architecture to use
        the_arch = GAMBlock if config.arch == "GAM" else GAMAttBlock

        # Create a single model
        self.model = the_arch(
            in_features=total_input_dim,
            num_trees=config.num_trees,
            num_layers=config.num_layers,
            num_classes=num_classes,
            addi_tree_dim=config.addi_tree_dim,
            depth=config.depth,
            choice_function=choice_fn,
            bin_function=entmoid15,
            output_dropout=config.output_dropout,
            last_dropout=config.last_dropout,
            colsample_bytree=config.colsample_bytree,
            selectors_detach=True,
            add_last_linear=True,
            ga2m=1 if self.interaction_degree >= 2 else 0,
            l2_lambda=config.l2_lambda,
            **({} if config.arch == "GAM" else {"dim_att": config.dim_att}),
        )

        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

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
        # Combine all features into a single tensor
        all_features = []
        feature_names = []
        
        # Add numerical features
        for feature_name, feature_tensor in num_features.items():
            all_features.append(feature_tensor)
            feature_names.append(feature_name)
            
        # Add categorical features
        for feature_name, feature_tensor in cat_features.items():
            all_features.append(feature_tensor.float())
            feature_names.append(feature_name)
            
        # Concatenate all features
        x = torch.cat(all_features, dim=1)
        
        # Apply feature dropout
        x = self.feature_dropout(x)
        
        # Get prediction from the model
        output = self.model(x)

        # Create result dictionary
        result = {"output": output}
        
        # Add individual feature outputs for interpretability
        for i, feature_name in enumerate(feature_names):
            result[feature_name] = all_features[i]

        return result
