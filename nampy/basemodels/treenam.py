import numpy as np
import torch
import torch.nn as nn
from ..arch_utils.neural_tree import NeuralDecisionTree
from ..configs.boostednam_config import DefaultBoostedNAMConfig
from .basemodel import BaseModel


class BoostedNAM(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultBoostedNAMConfig = DefaultBoostedNAMConfig(),
        **kwargs,
    ):
        """
        Initializes the Boosted Neural Additive Model (NAM) with the given configuration.

        Parameters
        ----------
        cat_feature_info : Any
            Information about categorical features.
        num_feature_info : Any
            Information about numerical features.
        num_classes : int, optional
            Number of output classes, by default 1.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters, by default DefaultNAMConfig().
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

        self.num_feature_models = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_models[feature_name] = nn.ModuleList(
                [
                    NeuralDecisionTree(
                        input_dim=info["dimension"],
                        depth=config.tree_depth,
                        output_dim=num_classes,
                    )
                    for _ in range(config.n_estimators)
                ]
            )

        self.cat_feature_models = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_models[feature_name] = nn.ModuleList(
                [
                    NeuralDecisionTree(
                        input_dim=info["dimension"],
                        depth=config.tree_depth,
                        output_dim=num_classes,
                    )
                    for _ in range(config.n_estimators)
                ]
            )

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the Boosted NAM model with support for both numerical and categorical features.
        This version implements boosting, where multiple trees are summed for each feature.

        Parameters
        ----------
        num_features : dict
            Dictionary of numerical features with feature names as keys.
        cat_features : dict
            Dictionary of categorical features with feature names as keys.

        Returns
        -------
        dict
            Dictionary containing the final model output and individual feature outputs.
        """
        num_outputs = {}

        # Boosting for numerical features
        for feature_name, feature_models in self.num_feature_models.items():
            # Initialize prediction as zeros for numerical features
            num_pred = torch.zeros_like(num_features[feature_name], dtype=torch.float32)

            # Boosting: sum over all trees for each numerical feature
            for tree in feature_models:
                num_pred += self.lr * tree(
                    num_features[feature_name]
                )  # Sum predictions with learning rate

            num_outputs[feature_name] = num_pred

        cat_outputs = {}

        # Boosting for categorical features
        for feature_name, feature_models in self.cat_feature_models.items():
            # Convert categorical feature to float and initialize prediction as zeros
            cat_feature_data = cat_features[feature_name].float()
            cat_pred = torch.zeros_like(cat_feature_data, dtype=torch.float32)

            # Boosting: sum over all trees for each categorical feature
            for tree in feature_models:
                cat_pred += self.lr * tree(
                    cat_feature_data
                )  # Sum predictions with learning rate

            cat_outputs[feature_name] = cat_pred

        # Combine numerical and categorical feature outputs
        all_outputs = torch.stack(
            list(num_outputs.values()) + list(cat_outputs.values()), dim=1
        ).sum(dim=1)

        # Construct the output dictionary
        result = {"output": all_outputs}
        result.update(num_outputs)
        result.update(cat_outputs)

        return result
