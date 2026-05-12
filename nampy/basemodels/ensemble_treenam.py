from typing import Optional

import torch
import torch.nn as nn

from ..configs.ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from .basemodel import BaseModel
from .model_output import make_model_output
from .treenam import TreeNAM


class EnsembleTreeNAM(BaseModel):
    """
    Simple ensemble of TreeNAM learners.

    Notes
    -----
    - This is a jointly trained ensemble, not bagging and not boosting.
    - Each learner is a full TreeNAM.
    - Predictions and per-feature contributions are averaged across learners.
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config: Optional[DefaultEnsembleTreeNAMConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultEnsembleTreeNAMConfig()
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.lr = self.hparams.get("lr", config.lr)
        self.lr_patience = self.hparams.get("lr_patience", config.lr_patience)
        self.weight_decay = self.hparams.get("weight_decay", config.weight_decay)
        self.lr_factor = self.hparams.get("lr_factor", config.lr_factor)

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes

        self.num_estimators = int(
            self.hparams.get("num_estimators", config.num_estimators)
        )
        if self.num_estimators < 1:
            raise ValueError("num_estimators must be >= 1")

        self.aggregation = self.hparams.get("aggregation", config.aggregation)
        if self.aggregation != "mean":
            raise ValueError(
                f"Unsupported aggregation={self.aggregation!r}. "
                "Only 'mean' is currently supported."
            )

        self.learners = nn.ModuleList(
            [
                TreeNAM(
                    cat_feature_info=cat_feature_info,
                    num_feature_info=num_feature_info,
                    num_classes=num_classes,
                    config=config,
                    **kwargs,
                )
                for _ in range(self.num_estimators)
            ]
        )

    def forward(
        self, num_features: dict, cat_features: dict, return_terms: bool = True
    ) -> dict:
        learner_results = [
            learner(
                num_features=num_features,
                cat_features=cat_features,
                return_terms=return_terms,
            )
            for learner in self.learners
        ]

        if not learner_results:
            raise RuntimeError("EnsembleTreeNAM has no learners.")

        prediction = torch.stack(
            [result["prediction"] for result in learner_results], dim=0
        ).mean(dim=0)

        terms = {}
        for key in learner_results[0]["terms"]:
            stacked = torch.stack(
                [result["terms"][key] for result in learner_results], dim=0
            )
            terms[key] = stacked.mean(dim=0)

        intercept = None
        if learner_results[0]["intercept"] is not None:
            intercept = torch.stack(
                [result["intercept"] for result in learner_results], dim=0
            ).mean(dim=0)

        regularization = {}
        for key in learner_results[0]["regularization"]:
            stacked = torch.stack(
                [result["regularization"][key] for result in learner_results], dim=0
            )
            regularization[key] = stacked.mean(dim=0)

        return make_model_output(
            prediction=prediction,
            terms=terms,
            intercept=intercept,
            regularization=regularization,
        )
