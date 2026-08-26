import torch
import torch.nn as nn

from ..configs.nam_config import DefaultNAMConfig
from .components.base_model import BaseModel
from .components.feature_metadata import ordered_feature_keys
from .components.interactions import (
    apply_feature_dropout,
    create_interaction_networks,
    interaction_forward,
    sum_feature_dims,
)
from .components.mlp import MLP
from .components.module_dict import RawKeyModuleDict
from .components.nam import NAMFeatureNN
from .components.regularization import (
    evaluating,
    mean_squared_term_outputs,
    normalized_parameter_l2,
)


class NAM(BaseModel):
    """
    Neural Additive Model (NAM) class.

    This class implements a Neural Additive Model (NAM) with support for numerical and
    categorical features, interaction terms, and various normalization layers.

    Attributes
    ----------
    num_feature_networks : RawKeyModuleDict
        Sub-networks for each numerical feature.
    cat_feature_networks : RawKeyModuleDict
        Sub-networks for each categorical feature.
    interaction_networks : RawKeyModuleDict
        Networks for modeling feature interactions (if applicable).
    interaction_degree : int, optional
        Degree of interactions to be modeled.
    intercept : torch.nn.Parameter
        Learnable intercept term, if enabled.
    feature_dropout_p : float
        Probability for feature-level dropout (drops whole feature outputs).
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNAMConfig | None = None,
        **kwargs,
    ):
        """
        Initializes the Neural Additive Model (NAM) with the given configuration.

        Parameters
        ----------
        cat_feature_info : dict
            Dictionary providing information about categorical features (e.g., input dimensions).
        num_feature_info : dict
            Dictionary providing information about numerical features (e.g., input dimensions).
        num_classes : int, optional
            Number of output classes for classification tasks, by default 1.
        config : DefaultNAMConfig, optional
            Configuration dataclass containing hyperparameters for the model, by default DefaultNAMConfig().
        kwargs : dict
            Additional keyword arguments.
        """
        if config is None:
            config = DefaultNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.feature_order = ordered_feature_keys(
            num_feature_info, cat_feature_info
        )
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )
        self.interactions = self.hparams.get("interactions", config.interactions)
        self.intercept: nn.Parameter | None
        if self.hparams.get("intercept", config.intercept):
            self.intercept = nn.Parameter(
                torch.zeros(
                    num_classes,
                )
            )
        else:
            self.intercept = None

        self.feature_dropout_p = self.hparams.get(
            "feature_dropout", config.feature_dropout
        )

        # Resolve architecture hyperparameters once, with kwargs overriding config
        self.layer_sizes = self.hparams.get("layer_sizes", config.layer_sizes)
        feature_layer = self.hparams.get("feature_layer", config.feature_layer)
        if not isinstance(feature_layer, str):
            raise TypeError("feature_layer must be a string.")
        self.feature_layer = feature_layer.lower()
        if self.feature_layer not in {"linear", "exu", "centered_relu"}:
            raise ValueError(
                "feature_layer must be 'linear', 'exu', or 'centered_relu'; "
                f"got {feature_layer!r}."
            )
        self.activation = self.hparams.get("activation", config.activation)
        if isinstance(self.activation, str):
            raise TypeError(
                "activation must be a torch.nn.Module class or instance; use "
                "feature_layer to select 'exu' or 'centered_relu'."
            )
        self.norm = self.hparams.get("norm", config.norm)
        self.use_glu = self.hparams.get("use_glu", config.use_glu)
        self.skip_connections = self.hparams.get(
            "skip_connections", config.skip_connections
        )
        self.batch_norm = self.hparams.get("batch_norm", config.batch_norm)
        self.layer_norm = self.hparams.get("layer_norm", config.layer_norm)
        self.dropout = self.hparams.get("dropout", config.dropout)
        self.adaptive_width = self.hparams.get(
            "adaptive_width", config.adaptive_width
        )
        self.num_basis_functions = int(
            self.hparams.get("num_basis_functions", config.num_basis_functions)
        )
        self.units_multiplier = int(
            self.hparams.get("units_multiplier", config.units_multiplier)
        )
        self.feature_widths = dict(
            self.hparams.get("feature_widths", config.feature_widths)
        )
        self.feature_output_bias = bool(
            self.hparams.get("feature_output_bias", config.feature_output_bias)
        )
        self.output_regularization = float(
            self.hparams.get("output_regularization", config.output_regularization)
        )
        self.l2_regularization = float(
            self.hparams.get("l2_regularization", config.l2_regularization)
        )
        self.regularize_interactions = bool(
            self.hparams.get(
                "regularize_interactions", config.regularize_interactions
            )
        )
        if self.num_basis_functions < 1 or self.units_multiplier < 1:
            raise ValueError("num_basis_functions and units_multiplier must be positive.")
        if self.output_regularization < 0 or self.l2_regularization < 0:
            raise ValueError("Regularization coefficients must be non-negative.")

        # Initialize sub-networks for each feature
        self.num_feature_networks = RawKeyModuleDict()
        self.cat_feature_networks = RawKeyModuleDict()
        info_by_kind = {"num": num_feature_info, "cat": cat_feature_info}
        networks_by_kind = {
            "num": self.num_feature_networks,
            "cat": self.cat_feature_networks,
        }
        for kind, feature_name in self.feature_order:
            info = info_by_kind[kind][feature_name]
            networks_by_kind[kind][feature_name] = self._create_subnetwork(
                info["dimension"], feature_name=feature_name, feature_info=info
            )

        self.interaction_networks = create_interaction_networks(
            [feature_name for _, feature_name in self.feature_order],
            self.interaction_degree,
            lambda interaction: self._create_subnetwork(
                sum_feature_dims(interaction, num_feature_info, cat_feature_info),
                feature_name=":".join(interaction),
            ),
            interactions=self.interactions,
        )

    def _resolved_layer_sizes(self, feature_name: str, feature_info=None) -> list[int]:
        sizes = list(self.layer_sizes)
        explicit_width = self.feature_widths.get(feature_name)
        if explicit_width is not None:
            width = int(explicit_width)
        elif self.adaptive_width and feature_info is not None:
            if "n_unique" not in feature_info:
                raise ValueError(
                    f"adaptive_width=True requires n_unique metadata for {feature_name!r}."
                )
            width = min(
                self.num_basis_functions,
                int(feature_info["n_unique"]) * self.units_multiplier,
            )
        else:
            return sizes
        if width < 1:
            raise ValueError(f"Resolved width for {feature_name!r} must be positive.")
        return [width, *sizes[1:]] if sizes else [width]

    def _create_subnetwork(
        self, input_dim: int, *, feature_name: str, feature_info=None
    ) -> nn.Module:
        """Create a subnetwork for a single feature using components.mlp.MLP."""
        layer_sizes = self._resolved_layer_sizes(feature_name, feature_info)
        if self.feature_layer in {"exu", "centered_relu"}:
            return NAMFeatureNN(
                n_input_units=input_dim,
                hidden_units_list=layer_sizes,
                n_output_units=self.num_classes,
                feature_layer=self.feature_layer,
                dropout=self.dropout,
                output_bias=self.feature_output_bias,
                use_skip_layers=self.skip_connections,
                use_batch_norm=self.batch_norm,
                use_layer_norm=self.layer_norm,
                norm=self.norm,
                use_glu=self.use_glu,
            )
        return MLP(
            n_input_units=input_dim,
            hidden_units_list=layer_sizes,
            n_output_units=self.num_classes,
            dropout=self.dropout,
            use_skip_layers=self.skip_connections,
            activation=self.activation,
            use_batch_norm=self.batch_norm,
            use_layer_norm=self.layer_norm,
            norm=self.norm,
            use_glu=self.use_glu,
            output_bias=self.feature_output_bias,
        )

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        """
        Forward pass of the NAM model.

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
            lambda network, input_features: network(input_features),
        )

        main_outputs = {
            **num_outputs,
            **cat_outputs,
        }
        all_outputs = [main_outputs[name] for _, name in self.feature_order]
        all_outputs.extend(interaction_outputs.values())
        concatenated = torch.cat(all_outputs, dim=1)
        num_features_total = len(all_outputs)

        shaped = concatenated.view(-1, num_features_total, self.num_classes)
        shaped = apply_feature_dropout(shaped, self.feature_dropout_p, self.training)

        x = shaped.sum(dim=1)  # [batch, num_classes]

        # intercept
        if self.intercept is not None:
            x += self.intercept

        # Combine the output tensor with the original feature values
        result = {"output": x}
        for _, name in self.feature_order:
            result[name] = main_outputs[name]
        result.update(interaction_outputs)
        if self.intercept is not None:
            result["intercept"] = self.intercept
        if self.output_regularization > 0:
            regularized_outputs = list(num_outputs.values()) + list(cat_outputs.values())
            if self.regularize_interactions:
                regularized_outputs.extend(interaction_outputs.values())
            if self.training:
                regularized_modules = list(self.num_feature_networks.values()) + list(
                    self.cat_feature_networks.values()
                )
                if self.regularize_interactions:
                    regularized_modules.extend(self.interaction_networks.values())
                with evaluating(regularized_modules):
                    regularized_outputs = [
                        network(num_features[name].float())
                        for name, network in self.num_feature_networks.items()
                    ] + [
                        network(cat_features[name].float())
                        for name, network in self.cat_feature_networks.items()
                    ]
                    if self.regularize_interactions:
                        deterministic_interactions = interaction_forward(
                            self.interaction_networks,
                            self.interaction_degree,
                            num_features,
                            cat_features,
                            lambda network, input_features: network(input_features),
                        )
                        regularized_outputs.extend(
                            deterministic_interactions.values()
                        )
            result["output_regularizer"] = (
                self.output_regularization
                * mean_squared_term_outputs(regularized_outputs)
            )
        if self.l2_regularization > 0:
            num_main_networks = len(num_outputs) + len(cat_outputs)
            result["parameter_regularizer"] = (
                self.l2_regularization
                * normalized_parameter_l2(
                    self, normalizer=max(num_main_networks, 1), half=True
                )
            )

        return result
