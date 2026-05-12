from itertools import combinations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arch_utils.neural_splines import CubicSplineLayer, TensorProductCubicSplineLayer
from ..configs.spline_nam_config import DefaultSplineNAMConfig
from .basemodel import BaseModel
from .model_output import make_model_output, merge_terms, validate_feature_names


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
        config: Optional[DefaultSplineNAMConfig] = None,
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
        self.n_knots = int(config.spline_n_knots or config.n_knots)
        self.learn_knots = self.hparams.get("learn_knots", config.learn_knots)
        self.identify = self.hparams.get("identify", config.identify)
        self.smoothing = float(self.hparams.get("smoothing", config.smoothing))
        self.knot_distance_penalty = float(
            self.hparams.get("knot_distance_penalty", config.knot_distance_penalty)
        )
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        self.intercept: Optional[nn.Parameter]
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
        validate_feature_names(set(num_feature_info) | set(cat_feature_info))

        self.num_feature_networks = nn.ModuleDict()
        for feature_name, info in num_feature_info.items():
            self.num_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=self.n_knots,
                min_val=0,
                max_val=1,
                learn_knots=self.learn_knots,
                identify=self.identify,
                input_dim=info["dimension"],
                output_dim=self.num_classes,
            )

        self.cat_feature_networks = nn.ModuleDict()
        for feature_name, info in cat_feature_info.items():
            self.cat_feature_networks[feature_name] = CubicSplineLayer(
                n_bases=self.n_knots,
                min_val=0,
                max_val=info["dimension"],
                learn_knots=self.learn_knots,
                identify=self.identify,
                input_dim=info["dimension"],
                output_dim=self.num_classes,
            )

        if self.interaction_degree is not None and self.interaction_degree >= 2:
            self._create_interaction_networks(
                num_feature_info=num_feature_info,
                cat_feature_info=cat_feature_info,
            )

    def _create_interaction_networks(self, num_feature_info, cat_feature_info):
        self.interaction_networks = nn.ModuleDict()
        self.interaction_layer_types = {}
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

                layer_class = CubicSplineLayer
                layer_type = "additive"
                max_val = input_dim
                if input_dim == len(interaction):
                    layer_class = TensorProductCubicSplineLayer
                    layer_type = "tensor_product"
                    max_val = 1

                self.interaction_networks[interaction_name] = layer_class(
                    n_bases=self.n_knots,
                    min_val=0,
                    max_val=max_val,
                    learn_knots=self.learn_knots,
                    identify=self.identify,
                    input_dim=input_dim,
                    output_dim=self.num_classes,
                )
                self.interaction_layer_types[interaction_name] = layer_type

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
                interaction_output = interaction_network(input_features)
                interaction_outputs[interaction_name] = interaction_output

        return interaction_outputs

    def _iter_named_spline_layers(self):
        yield from self.num_feature_networks.items()
        yield from self.cat_feature_networks.items()
        if hasattr(self, "interaction_networks"):
            yield from self.interaction_networks.items()

    def _iter_spline_layers(self):
        for _, layer in self._iter_named_spline_layers():
            yield layer

    def _regularization_penalty(self):
        penalty = next(self.parameters()).new_zeros(())

        if self.smoothing > 0.0:
            smooth_penalty = next(self.parameters()).new_zeros(())
            for layer in self._iter_spline_layers():
                smooth_penalty = smooth_penalty + layer.get_smooth_penalty()
            penalty = penalty + self.smoothing * smooth_penalty

        if self.knot_distance_penalty > 0.0:
            knot_penalty = next(self.parameters()).new_zeros(())
            for layer in self._iter_spline_layers():
                knot_penalty = knot_penalty + layer.get_knot_distance_penalty()
            penalty = penalty + self.knot_distance_penalty * knot_penalty

        return penalty

    def _combine_outputs(self, all_outputs):
        if not all_outputs:
            raise ValueError("SplineNAM received no feature contributions to sum.")

        stacked = torch.stack(all_outputs, dim=1)
        if self.feature_dropout_p > 0.0 and self.training:
            mask = torch.ones(
                stacked.shape[0],
                stacked.shape[1],
                1,
                device=stacked.device,
                dtype=stacked.dtype,
            )
            mask = F.dropout(mask, p=self.feature_dropout_p, training=True)
            stacked = stacked * mask

        return stacked.sum(dim=1)

    def get_knot_locations(self):
        """
        Return current knot locations for each spline term.

        Learned knots are computed from the current learned interval distances.
        The returned tensors are detached CPU tensors so they can be inspected
        without affecting autograd.
        """
        return {
            name: layer.get_knot_locations()
            for name, layer in self._iter_named_spline_layers()
        }

    def get_spline_penalties(self):
        """
        Return unweighted and weighted regularization values by spline term.
        """
        penalties = {}
        for name, layer in self._iter_named_spline_layers():
            smooth = float(layer.get_smooth_penalty().detach().cpu())
            knot_distance = float(layer.get_knot_distance_penalty().detach().cpu())
            penalties[name] = {
                "smooth": smooth,
                "knot_distance": knot_distance,
                "weighted": self.smoothing * smooth
                + self.knot_distance_penalty * knot_distance,
            }
        return penalties

    def get_spline_diagnostics(self):
        """
        Return lightweight diagnostics for the spline terms in the model.
        """
        term_names = [name for name, _ in self._iter_named_spline_layers()]
        return {
            "n_knots": self.n_knots,
            "learn_knots": self.learn_knots,
            "identify": self.identify,
            "smoothing": self.smoothing,
            "knot_distance_penalty": self.knot_distance_penalty,
            "feature_dropout": self.feature_dropout_p,
            "terms": term_names,
            "interaction_layer_types": getattr(self, "interaction_layer_types", {}),
            "knot_locations": self.get_knot_locations(),
            "penalties": self.get_spline_penalties(),
        }

    def forward(
        self, num_features: dict, cat_features: dict, return_terms: bool = True
    ) -> dict:
        num_outputs = {}
        for feature_name, feature_network in self.num_feature_networks.items():
            feature_output = feature_network(num_features[feature_name])
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
        x = self._combine_outputs(all_outputs)

        if self.intercept is not None:
            x += self.intercept

        terms = (
            merge_terms(num_outputs, cat_outputs, interaction_outputs)
            if return_terms
            else {}
        )
        regularization = {}
        if self.smoothing > 0.0 or self.knot_distance_penalty > 0.0:
            regularization["spline"] = self._regularization_penalty()

        return make_model_output(
            prediction=x,
            terms=terms,
            intercept=self.intercept,
            regularization=regularization,
        )
