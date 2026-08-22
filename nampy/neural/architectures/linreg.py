import torch
import torch.nn as nn

from ..configs.linreg_config import DefaultLinRegConfig
from .components.base_model import BaseModel


class LinReg(BaseModel):
    """
    Additive linear model over feature blocks.

    Each numerical/categorical feature block gets its own linear map to the
    output space, and the final prediction is the sum of all feature
    contributions plus an optional intercept.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultLinRegConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultLinRegConfig()
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

        self.intercept: nn.Parameter | None
        if self.hparams.get("intercept", getattr(config, "intercept", True)):
            self.intercept = nn.Parameter(torch.zeros(num_classes))
        else:
            self.intercept = None

        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = self._create_subnetwork(
                info["dimension"]
            )

    def _create_subnetwork(self, input_dim: int) -> nn.Module:
        return nn.Linear(input_dim, self.num_classes)

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            num_outputs[feature_name] = feature_network(
                num_features[feature_name].float()
            )

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            cat_outputs[feature_name] = feature_network(
                cat_features[feature_name].float()
            )

        all_outputs = list(num_outputs.values()) + list(cat_outputs.values())
        if not all_outputs:
            raise ValueError("LinReg received no feature contributions to sum.")

        x = torch.stack(all_outputs, dim=1).sum(dim=1)

        if self.intercept is not None:
            x = x + self.intercept

        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)

        if self.intercept is not None:
            result["intercept"] = self.intercept

        return result
