from itertools import combinations

import torch
import torch.nn as nn

from ..arch_utils.neural_splines import CubicSplineLayer
from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .basemodel import BaseModel


class SplineNAM(BaseModel):
    """
    Neural Additive Model (NAM) class with CubicSplineLayer.

    This class implements a Neural Additive Model (NAM) using cubic spline layers for feature modeling,
    with support for numerical and categorical features, interaction terms, and various normalization layers.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultSplineNAMConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultSplineNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes
        self.n_knots = int(self.hparams.get("n_knots", config.n_knots))
        self.learn_knots = bool(
            self.hparams.get("learn_knots", config.learn_knots)
        )
        self.identify = bool(self.hparams.get("identify", config.identify))
        self.smoothing = float(self.hparams.get("smoothing", config.smoothing))
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

        self.feature_dropout_p = float(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

        reserved = {"output", "intercept", "smoothness_penalty"}
        all_feature_names = set(num_feature_info) | set(cat_feature_info)
        invalid_names = reserved & all_feature_names
        invalid_names.update(name for name in all_feature_names if ":" in name)
        if invalid_names:
            raise ValueError(
                f"Feature names {sorted(invalid_names)} are reserved by SplineNAM."
            )
        if not all_feature_names:
            raise ValueError("SplineNAM requires at least one input feature.")

        for feature_name, info in {**num_feature_info, **cat_feature_info}.items():
            dimension = info.get("dimension", 1)
            if dimension is None or int(dimension) != 1:
                raise ValueError(
                    "SplineNAM requires scalar transformed features; "
                    f"feature {feature_name!r} has dimension {dimension!r}. "
                    "Use scalar preprocessing such as numerical_preprocessing='minmax' "
                    "and categorical_preprocessing='int'."
                )

        self._feature_ranges = {}
        self.num_feature_networks = nn.ModuleDict()
        for feature_name, _info in num_feature_info.items():
            self._feature_ranges[feature_name] = (0.0, 1.0)
            self.num_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=self.n_knots,
                min_val=0,
                max_val=1,
                learn_knots=self.learn_knots,
                identify=self.identify,
                n_outputs=self.num_classes,
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            categories = info.get("categories")
            max_val = float(max(int(categories) - 1, 1)) if categories else 1.0
            self._feature_ranges[feature_name] = (0.0, max_val)
            self.cat_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=self.n_knots,
                min_val=0,
                max_val=max_val,
                learn_knots=self.learn_knots,
                identify=self.identify,
                n_outputs=self.num_classes,
            )

        self.interaction_networks = nn.ModuleDict()
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

    def _create_interaction_networks(self, num_feature_info, cat_feature_info):
        all_feature_names = list(num_feature_info.keys()) + list(
            cat_feature_info.keys()
        )

        for degree in range(2, self.interaction_degree + 1):
            for interaction in combinations(all_feature_names, degree):
                interaction_name = ":".join(interaction)
                min_val = sum(self._feature_ranges[name][0] for name in interaction)
                max_val = sum(self._feature_ranges[name][1] for name in interaction)

                self.interaction_networks[interaction_name] = CubicSplineLayer(
                    n_bases=self.n_knots,
                    min_val=min_val,
                    max_val=max_val,
                    learn_knots=self.learn_knots,
                    identify=self.identify,
                    n_outputs=self.num_classes,
                )

    def _interaction_forward(self, num_features: dict, cat_features: dict):
        interaction_outputs = {}
        if self.interaction_degree is not None and self.interaction_degree >= 2:
            all_features = {
                **num_features,
                **cat_features,
            }
            for (
                interaction_name,
                interaction_network,
            ) in self.interaction_networks.items():
                feature_names = interaction_name.split(":")
                input_features = torch.cat(
                    [all_features[fn] for fn in feature_names], dim=-1
                )
                interaction_output = interaction_network(
                    input_features.float().sum(dim=-1, keepdim=True)
                )
                interaction_outputs[interaction_name] = interaction_output

        return interaction_outputs

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name].float())
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = self._interaction_forward(
            num_features=num_features, cat_features=cat_features
        )

        all_outputs = (
            list(num_outputs.values())
            + list(cat_outputs.values())
            + list(interaction_outputs.values())
        )
        term_outputs = torch.stack(all_outputs, dim=1)
        if self.feature_dropout_p > 0.0 and self.training:
            mask = term_outputs.new_ones(
                term_outputs.shape[0], term_outputs.shape[1], 1
            )
            mask = nn.functional.dropout(
                mask,
                p=self.feature_dropout_p,
                training=True,
            )
            term_outputs = term_outputs * mask
        x = term_outputs.sum(dim=1)

        if self.intercept is not None:
            x += self.intercept

        result = {"output": x}
        result.update(num_outputs)
        result.update(cat_outputs)
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept
        if self.smoothing > 0.0:
            smoothness = x.new_zeros(())
            for network in (
                list(self.num_feature_networks.values())
                + list(self.cat_feature_networks.values())
                + list(self.interaction_networks.values())
            ):
                smoothness = smoothness + network.get_smooth_penalty()
            result["smoothness_penalty"] = self.smoothing * smoothness

        return result
