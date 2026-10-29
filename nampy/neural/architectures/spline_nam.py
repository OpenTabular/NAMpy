import torch
import torch.nn as nn

from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .components.base_model import BaseModel
from .components.interactions import (
    create_interaction_networks,
    interaction_forward,
    sum_term_outputs,
)
from .components.module_dict import RawKeyModuleDict
from .components.splines import CubicSplineLayer


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
    ):
        if config is None:
            config = DefaultSplineNAMConfig()
        super().__init__()

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.n_knots = int(config.n_knots)
        self.learn_knots = bool(config.learn_knots)
        self.identify = bool(config.identify)
        self.smoothing = float(config.smoothing)
        self.interaction_degree = config.interaction_degree
        self.interactions = config.interactions
        self.intercept: nn.Parameter | None
        if config.intercept:
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        self.feature_dropout_p = float(config.feature_dropout)

        if not set(num_feature_info) | set(cat_feature_info):
            raise ValueError("SplineNAM requires at least one input feature.")

        for feature_name, info in {**num_feature_info, **cat_feature_info}.items():
            dimension = info.get("dimension", 1)
            if dimension is None or int(dimension) != 1:
                raise ValueError(
                    "SplineNAM requires scalar transformed features; "
                    f"feature {feature_name!r} has dimension {dimension!r}. "
                    "Use scalar preprocessing such as "
                    "numerical_preprocessing='minmax' and "
                    "categorical_preprocessing='int'."
                )

        self._feature_ranges = {}
        self.num_feature_networks = RawKeyModuleDict()
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

        self.cat_feature_networks = RawKeyModuleDict()
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

        self.interaction_networks = create_interaction_networks(
            list(num_feature_info.keys()) + list(cat_feature_info.keys()),
            self.interaction_degree,
            self._make_interaction_network,
            interactions=self.interactions,
        )

    def _make_interaction_network(self, interaction) -> CubicSplineLayer:
        # Interaction inputs are summed member features, so the spline range is
        # the sum of the member feature ranges.
        min_val = sum(self._feature_ranges[name][0] for name in interaction)
        max_val = sum(self._feature_ranges[name][1] for name in interaction)
        return CubicSplineLayer(
            n_bases=self.n_knots,
            min_val=min_val,
            max_val=max_val,
            learn_knots=self.learn_knots,
            identify=self.identify,
            n_outputs=self.num_classes,
        )

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name].float())
            num_outputs[feature_name] = feature_output

        cat_outputs = {}
        for feature_name, feature_network in self.cat_feature_networks.items():
            feature_output = feature_network(cat_features[feature_name].float())
            cat_outputs[feature_name] = feature_output

        interaction_outputs = interaction_forward(
            self.interaction_networks,
            self.interaction_degree,
            num_features,
            cat_features,
            lambda network, input_features: network(
                input_features.sum(dim=-1, keepdim=True)
            ),
        )

        terms = {**num_outputs, **cat_outputs, **interaction_outputs}
        result = sum_term_outputs(
            terms, self.intercept, p=self.feature_dropout_p, training=self.training
        )
        x = result["output"]

        if self.smoothing > 0.0:
            smoothness = x.new_zeros(())
            for network in (
                list(self.num_feature_networks.values())
                + list(self.cat_feature_networks.values())
                + list(self.interaction_networks.values())
            ):
                smoothness = smoothness + network.get_smooth_penalty()  # type: ignore[attr-defined,operator]
            result["smoothness_penalty"] = self.smoothing * smoothness

        return result
