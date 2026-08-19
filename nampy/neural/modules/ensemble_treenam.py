import torch
import torch.nn as nn

from ..configs.ensemble_treenam_config import DefaultEnsembleTreeNAMConfig
from .basemodel import BaseModel
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
        config: DefaultEnsembleTreeNAMConfig | None = None,
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
        self._validate_features(num_feature_info, cat_feature_info)
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

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        learner_results = [
            learner(num_features=num_features, cat_features=cat_features)
            for learner in self.learners
        ]

        if not learner_results:
            raise RuntimeError("EnsembleTreeNAM has no learners.")

        # Aggregate all non-penalty outputs by averaging.
        template_keys = [k for k in learner_results[0].keys() if k != "output_penalty"]
        result = {}

        for key in template_keys:
            stacked = torch.stack([res[key] for res in learner_results], dim=0)
            result[key] = stacked.mean(dim=0)

        # Aggregate penalties by averaging so penalty scale stays roughly
        # comparable as num_estimators changes.
        penalties = [
            res["output_penalty"] for res in learner_results if "output_penalty" in res
        ]
        if penalties:
            result["output_penalty"] = torch.stack(penalties, dim=0).mean(dim=0)

        return result
