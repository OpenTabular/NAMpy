import torch
import torch.nn as nn


class MultiModelWrapper(nn.Module):
    """
    Wraps a base model class into N independent sub-models and concatenates outputs.

    This is useful for distributional regression where each parameter is modeled
    by its own additive model, while keeping a single interface.
    """

    def __init__(
        self,
        base_model_class,
        num_models,
        per_model_num_classes=1,
        config=None,
        cat_feature_info=None,
        num_feature_info=None,
        num_classes=None,
        **kwargs,
    ):
        super().__init__()
        if num_models is None:
            raise ValueError("num_models must be provided for MultiModelWrapper.")
        self.num_models = num_models
        self.per_model_num_classes = per_model_num_classes
        self.models = nn.ModuleList(
            [
                base_model_class(
                    config=config,
                    cat_feature_info=cat_feature_info,
                    num_feature_info=num_feature_info,
                    num_classes=per_model_num_classes,
                    **kwargs,
                )
                for _ in range(num_models)
            ]
        )

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        outputs = []
        penalties = []
        for model in self.models:
            result = model(num_features=num_features, cat_features=cat_features)
            output = result.get("output")
            if output is None:
                raise ValueError("Base model did not return an 'output' key.")
            if output.dim() == 1:
                output = output.unsqueeze(1)
            outputs.append(output)
            if "penalty" in result:
                penalties.append(result["penalty"])

        out = {"output": torch.cat(outputs, dim=1)}
        if penalties:
            out["penalty"] = sum(penalties)
        return out
