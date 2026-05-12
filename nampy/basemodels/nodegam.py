from typing import Any, Optional, cast

import torch
import torch.nn as nn

from ..arch_utils.nn_utils import entmoid15
from ..arch_utils.nodegam_utils import EM15Temp, GAMAttBlock, GAMBlock
from ..configs.nodegam_config import DefaultNodeGAMConfig
from .basemodel import BaseModel
from .model_output import make_model_output, validate_feature_names


class NodeGAMBlockHead(nn.Module):
    """One NODE-GAM head over the preprocessed feature matrix."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int,
        config: DefaultNodeGAMConfig,
        hparams: Optional[dict] = None,
    ):
        super().__init__()
        if hparams is None:
            hparams = {}

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = int(num_classes)

        self.num_feature_keys = list(num_feature_info.keys())
        self.cat_feature_keys = list(cat_feature_info.keys())
        self.input_feature_names = self._build_input_feature_names()
        self.total_input_dim = len(self.input_feature_names)
        if self.total_input_dim == 0:
            raise ValueError("NodeGAM requires at least one input feature.")

        all_feature_names = set(self.input_feature_names)
        validate_feature_names(all_feature_names)

        self.interaction_degree = hparams.get(
            "interaction_degree", config.interaction_degree
        )
        self.l2_lambda = hparams.get("l2_lambda", config.l2_lambda)
        self.feature_dropout = nn.Dropout(
            p=hparams.get("feature_dropout", config.feature_dropout)
        )

        self.choice_fn = EM15Temp(
            max_temp=1.0,
            min_temp=hparams.get("min_temp", config.min_temp),
            steps=hparams.get("anneal_steps", config.anneal_steps),
        )

        arch = hparams.get("arch", config.arch)
        if arch not in {"GAM", "GAMAtt"}:
            raise ValueError("NodeGAM arch must be 'GAM' or 'GAMAtt'.")
        the_arch = GAMBlock if arch == "GAM" else GAMAttBlock

        block_kwargs = {
            "in_features": self.total_input_dim,
            "num_trees": hparams.get("num_trees", config.num_trees),
            "num_layers": hparams.get("num_layers", config.num_layers),
            "num_classes": self.num_classes,
            "addi_tree_dim": hparams.get("addi_tree_dim", config.addi_tree_dim),
            "depth": hparams.get("depth", config.depth),
            "choice_function": self.choice_fn,
            "bin_function": entmoid15,
            "output_dropout": hparams.get("output_dropout", config.output_dropout),
            "last_dropout": hparams.get("last_dropout", config.last_dropout),
            "colsample_bytree": hparams.get(
                "colsample_bytree", config.colsample_bytree
            ),
            "selectors_detach": hparams.get(
                "selectors_detach", config.selectors_detach
            ),
            "init_bias": hparams.get("init_bias", config.init_bias),
            "add_last_linear": hparams.get("add_last_linear", config.add_last_linear),
            "ga2m": 1 if self.interaction_degree >= 2 else 0,
            "l2_lambda": self.l2_lambda,
            "l2_interactions": hparams.get("l2_interactions", config.l2_interactions),
            "l1_interactions": hparams.get("l1_interactions", config.l1_interactions),
        }
        if arch == "GAMAtt":
            block_kwargs["dim_att"] = hparams.get("dim_att", config.dim_att)

        self.block = the_arch(**block_kwargs)

    def _build_input_feature_names(self):
        names = []
        for feature_name in self.num_feature_keys:
            dim = int(self.num_feature_info[feature_name]["dimension"])
            if dim <= 0:
                raise ValueError(
                    f"Numerical feature '{feature_name}' has invalid dimension {dim}."
                )
            names.extend(self._expanded_names(feature_name, dim))

        for feature_name in self.cat_feature_keys:
            dim = int(self.cat_feature_info[feature_name]["dimension"])
            if dim <= 0:
                raise ValueError(
                    f"Categorical feature '{feature_name}' has invalid dimension {dim}."
                )
            names.extend(self._expanded_names(feature_name, dim))
        return names

    @staticmethod
    def _expanded_names(feature_name, dim):
        return [feature_name for _ in range(dim)]

    def _concat_features(self, num_features: dict, cat_features: dict):
        tensors = []

        for feature_name in self.num_feature_keys:
            x = num_features[feature_name]
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            tensors.append(x.float())

        for feature_name in self.cat_feature_keys:
            x = cat_features[feature_name]
            if x.ndim == 1:
                x = x.unsqueeze(-1)
            tensors.append(x.float())

        if not tensors:
            raise ValueError("NodeGAM received no input features.")
        return torch.cat(tensors, dim=1)

    def step_temperature_schedulers(self, step: int):
        self.choice_fn.temp_step_callback(step)

    def forward(
        self, num_features: dict, cat_features: dict, return_terms: bool = True
    ) -> dict:
        x = self._concat_features(num_features, cat_features)
        x = self.feature_dropout(x)

        if self.l2_lambda and self.l2_lambda > 0:
            output, penalty = self.block(x, return_outputs_penalty=True)
        else:
            output = self.block(x)
            penalty = None

        regularization = {}
        if penalty is not None:
            regularization["output_penalty"] = penalty

        terms = {}
        if return_terms:
            terms = self._additive_term_outputs(x)

        return make_model_output(
            prediction=output,
            terms=terms,
            intercept=self.block.bias,
            regularization=regularization,
        )

    def _additive_term_outputs(self, x: torch.Tensor):
        term_outputs = self.block.run_with_additive_terms(x)
        terms = self.block.get_additive_terms()

        result: dict[str, torch.Tensor] = {}
        for term_idx, term in enumerate(terms):
            term_name = self._term_name(term)
            value = term_outputs[:, term_idx, :]
            if term_name in result:
                result[term_name] = result[term_name] + value
            else:
                result[term_name] = value
        return result

    def _term_name(self, term):
        if isinstance(term, tuple):
            feature_names = []
            for idx in term:
                feature_name = self.input_feature_names[idx]
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
            return ":".join(feature_names)
        return self.input_feature_names[term]


class NodeGAM(BaseModel):
    """NODE-GAM base model for regression and classification."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: Optional[DefaultNodeGAMConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultNodeGAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = int(num_classes)

        self.head = NodeGAMBlockHead(
            cat_feature_info=cat_feature_info,
            num_feature_info=num_feature_info,
            num_classes=self.num_classes,
            config=config,
            hparams=self.hparams,
        )

    def step_temperature_schedulers(self, step: int):
        self.head.step_temperature_schedulers(step)

    def forward(
        self, num_features: dict, cat_features: dict, return_terms: bool = True
    ) -> dict:
        return cast(
            dict[Any, Any],
            self.head(
                num_features=num_features,
                cat_features=cat_features,
                return_terms=return_terms,
            ),
        )


class NodeGAMLSSBase(BaseModel):
    """Upstream-compatible NodeGAMLSS with one independent NodeGAM head per parameter."""

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        family=None,
        config: Optional[DefaultNodeGAMConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultNodeGAMConfig()
        if family is None:
            raise ValueError("NodeGAMLSSBase requires a distribution family.")
        if config.lss_head_mode != "independent":
            raise ValueError(
                "NodeGAMLSS currently supports only independent LSS heads."
            )

        super().__init__(**kwargs)
        self.save_hyperparameters(
            ignore=["cat_feature_info", "num_feature_info", "family"]
        )

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = int(num_classes)
        if self.num_classes != int(family.param_count):
            raise ValueError(
                "NodeGAMLSS output dimension must match family.param_count."
            )

        self.param_names = list(getattr(family, "param_names", []))
        if len(self.param_names) != self.num_classes:
            self.param_names = [f"param_{idx}" for idx in range(self.num_classes)]

        self.heads = nn.ModuleList(
            [
                NodeGAMBlockHead(
                    cat_feature_info=cat_feature_info,
                    num_feature_info=num_feature_info,
                    num_classes=1,
                    config=config,
                    hparams=self.hparams,
                )
                for _ in range(self.num_classes)
            ]
        )

    def step_temperature_schedulers(self, step: int):
        for head in self.heads:
            cast(NodeGAMBlockHead, head).step_temperature_schedulers(step)

    def forward(
        self, num_features: dict, cat_features: dict, return_terms: bool = True
    ) -> dict:
        head_results = [
            cast(NodeGAMBlockHead, head)(
                num_features=num_features,
                cat_features=cat_features,
                return_terms=return_terms,
            )
            for head in self.heads
        ]
        output = torch.cat([result["prediction"] for result in head_results], dim=1)

        penalties = [
            head_result["regularization"]["output_penalty"]
            for head_result in head_results
            if "output_penalty" in head_result["regularization"]
        ]
        regularization = {}
        if penalties:
            regularization["output_penalty"] = sum(penalties)

        terms = {}
        intercept = None
        if return_terms:
            term_names = head_results[0]["terms"].keys()
            for term_name in term_names:
                terms[term_name] = torch.cat(
                    [head_result["terms"][term_name] for head_result in head_results],
                    dim=1,
                )

            intercepts = [
                head_result["intercept"].reshape(1)
                for head_result in head_results
                if head_result["intercept"] is not None
            ]
            if intercepts:
                intercept = torch.cat(intercepts, dim=0)

        return make_model_output(
            prediction=output,
            terms=terms,
            intercept=intercept,
            regularization=regularization,
        )
