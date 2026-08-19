from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs.treenam_config import DefaultTreeNAMConfig
from .basemodel import BaseModel
from .neural_tree import NeuralDecisionTree


class TreeNAM(BaseModel):
    """
    Additive model with one differentiable neural decision tree per feature,
    plus optional interaction trees.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultTreeNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultTreeNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes

        self.tree_depth = self.hparams.get("tree_depth", config.tree_depth)
        self.tree_lamda = self.hparams.get("tree_lamda", config.tree_lamda)
        self.tree_temperature = self.hparams.get(
            "tree_temperature", config.tree_temperature
        )
        self.use_hard_routing_in_eval = self.hparams.get(
            "use_hard_routing_in_eval", config.use_hard_routing_in_eval
        )
        self.feature_dropout_p = self.hparams.get(
            "feature_dropout", config.feature_dropout
        )
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )

        self.intercept: nn.Parameter | None
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(torch.zeros(num_classes))
        else:
            self.intercept = None

        self.num_feature_models = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_models[feature_name] = self._create_tree(
                input_dim=info["dimension"]
            )

        self.cat_feature_models = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_models[feature_name] = self._create_tree(
                input_dim=info["dimension"]
            )

        self.interaction_models = nn.ModuleDict()
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_trees(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

    def _create_tree(self, input_dim: int) -> NeuralDecisionTree:
        return NeuralDecisionTree(
            input_dim=input_dim,
            depth=self.tree_depth,
            output_dim=self.num_classes,
            lamda=self.tree_lamda,
            temperature=self.tree_temperature,
            use_hard_routing_in_eval=self.use_hard_routing_in_eval,
        )

    def _create_interaction_trees(self, num_feature_info, cat_feature_info):
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        for degree in range(2, self.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                interaction_name = ":".join(interaction)
                input_dim = 0

                for feature in interaction:
                    if feature in num_feature_info:
                        input_dim += num_feature_info[feature]["dimension"]
                    elif feature in cat_feature_info:
                        input_dim += cat_feature_info[feature]["dimension"]

                self.interaction_models[interaction_name] = self._create_tree(
                    input_dim=input_dim
                )

    def _interaction_forward(self, num_features: dict, cat_features: dict):
        interaction_outputs = {}
        penalty = next(self.parameters()).new_zeros(())

        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {**num_features, **cat_features}

            for interaction_name, tree in self.interaction_models.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                ).float()

                output, tree_penalty = tree(input_features, return_penalty=True)
                interaction_outputs[interaction_name] = output
                penalty = penalty + tree_penalty

        return interaction_outputs, penalty

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        num_outputs = {}
        total_penalty = next(self.parameters()).new_zeros(())

        for feature_name, tree in self.num_feature_models.items():
            output, penalty = tree(
                num_features[feature_name].float(), return_penalty=True
            )
            num_outputs[feature_name] = output
            total_penalty = total_penalty + penalty

        cat_outputs = {}
        for feature_name, tree in self.cat_feature_models.items():
            output, penalty = tree(
                cat_features[feature_name].float(), return_penalty=True
            )
            cat_outputs[feature_name] = output
            total_penalty = total_penalty + penalty

        interaction_outputs, interaction_penalty = self._interaction_forward(
            num_features=num_features,
            cat_features=cat_features,
        )
        total_penalty = total_penalty + interaction_penalty

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )

        if not all_outputs:
            raise ValueError("TreeNAM received no feature contributions to sum.")

        concatenated = torch.cat(all_outputs, dim=1)
        num_terms = len(all_outputs)
        shaped = concatenated.view(-1, num_terms, self.num_classes)

        if self.feature_dropout_p > 0.0 and self.training:
            mask = torch.ones(
                shaped.shape[0],
                num_terms,
                1,
                device=shaped.device,
                dtype=shaped.dtype,
            )
            mask = F.dropout(mask, p=self.feature_dropout_p, training=True)
            shaped = shaped * mask

        x = shaped.sum(dim=1)

        if self.intercept is not None:
            x = x + self.intercept

        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)

        if self.intercept is not None:
            result["intercept"] = self.intercept

        # Reuse the existing wrapper convention so no lightning changes are required
        if self.tree_lamda > 0.0:
            result["output_penalty"] = total_penalty

        return result
