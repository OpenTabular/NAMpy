import torch
import torch.nn as nn

from ..configs.nodegam_config import DefaultNodeGAMConfig
from .components.additive_trees import GAMAttBlock, GAMBlock
from .components.base_model import BaseModel
from .components.sparse_activations import (
    EM15Temp,
    entmoid15,
    sparsemax,
    sparsemoid,
)


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

    supports_masked_pretraining = True

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: DefaultNodeGAMConfig | None = None,
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
            Configuration dataclass containing hyperparameters for the model, by default DefaultNodeGAMConfig().
        kwargs : dict
            Additional keyword arguments.
        """
        if config is None:
            config = DefaultNodeGAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.l2_lambda = self.hparams.get("l2_lambda", config.l2_lambda)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self._validate_features(num_feature_info, cat_feature_info)
        self.num_classes = num_classes
        self.interaction_degree = self.hparams.get(
            "interaction_degree", config.interaction_degree
        )

        # Calculate total input dimension
        total_input_dim = sum(
            info["dimension"] for info in num_feature_info.values()
        ) + sum(info["dimension"] for info in cat_feature_info.values())

        selector_activation = self.hparams.get(
            "selector_activation", config.selector_activation
        ).lower()
        bin_activation = self.hparams.get(
            "bin_activation", config.bin_activation
        ).lower()
        if selector_activation == "entmax15":
            choice_fn = EM15Temp(
                max_temp=1.0, min_temp=0.01, steps=config.anneal_steps
            )
        elif selector_activation == "sparsemax":
            choice_fn = sparsemax
        else:
            raise ValueError(
                "selector_activation must be 'entmax15' or 'sparsemax', "
                f"got {selector_activation!r}"
            )

        if bin_activation == "entmoid15":
            bin_fn = entmoid15
        elif bin_activation == "sparsemoid":
            bin_fn = sparsemoid
        else:
            raise ValueError(
                "bin_activation must be 'entmoid15' or 'sparsemoid', "
                f"got {bin_activation!r}"
            )

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
            bin_function=bin_fn,
            output_dropout=config.output_dropout,
            input_dropout=config.input_dropout,
            last_dropout=config.last_dropout,
            colsample_bytree=config.colsample_bytree,
            selectors_detach=True,
            add_last_linear=True,
            ga2m=1 if self.interaction_degree >= 2 else 0,
            l2_lambda=config.l2_lambda,
            l2_interactions=config.l2_interactions,
            l1_interactions=config.l1_interactions,
            **({} if config.arch == "GAM" else {"dim_att": config.dim_att}),
        )

        self.feature_dropout = nn.Dropout(
            self.hparams.get("feature_dropout", config.feature_dropout)
        )

    def forward(
        self, num_features: dict, cat_features: dict, feature_masks=None
    ) -> dict:
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
        expanded_feature_names = []

        # Add numerical features
        for feature_name, feature_tensor in num_features.items():
            all_features.append(feature_tensor)
            feature_names.append(feature_name)
            expanded_feature_names.extend([feature_name] * feature_tensor.shape[1])

        # Add categorical features
        for feature_name, feature_tensor in cat_features.items():
            all_features.append(feature_tensor.float())
            feature_names.append(feature_name)
            expanded_feature_names.extend([feature_name] * feature_tensor.shape[1])

        # Concatenate all features
        x = torch.cat(all_features, dim=1)

        # Apply feature dropout
        x = self.feature_dropout(x)

        # Get prediction (and optional regularization penalty) from the model
        penalty = None
        if (self.l2_lambda and self.l2_lambda > 0) or feature_masks is not None:
            output = self.model(
                x,
                return_outputs_penalty=True,
                feature_masks=feature_masks,
            )
            if isinstance(output, tuple):
                output, penalty = output
        else:
            output = self.model(x)

        # Create result dictionary
        result = {"output": output}
        if penalty is not None:
            result["output_penalty"] = penalty

        # Expose learned additive outputs.  Returning the raw input columns
        # here makes predict_components/plot_terms look plausible while being
        # unrelated to the fitted model (the original NODE-GAM term contract
        # aggregates tree outputs by their selected feature term).
        feature_outputs = {
            feature_name: torch.zeros_like(output)
            for feature_name in feature_names
        }
        if not self.training:
            # During selector annealing the upstream NODE-GAM representation
            # may temporarily be a dense 3+-way selector, which is not a valid
            # GAM term.  Contributions are an inference surface; keep the
            # training output contract stable without asking term extraction
            # to invent a decomposition for that unsupported state.
            term_outputs = self.model.run_with_additive_terms(x)
            terms = self.model.get_additive_terms()
            for term_index, term in enumerate(terms):
                if isinstance(term, tuple):
                    key = ":".join(expanded_feature_names[index] for index in term)
                else:
                    key = expanded_feature_names[term]
                value = term_outputs[:, term_index, :]
                result[key] = value
                if not isinstance(term, tuple):
                    feature_outputs[key] = feature_outputs[key] + value
        result.update(feature_outputs)
        result["intercept"] = self.model.bias

        return result

    def temp_step_callback(self, step: int) -> None:
        """Advance every NODE selector temperature during training."""
        for module in self.model.modules():
            choice_function = getattr(module, "choice_function", None)
            if hasattr(choice_function, "temp_step_callback"):
                choice_function.temp_step_callback(step)
