"""TaskModel-compatible container combining compiled GAM terms and a net.

The compiled design matrix reaches ``forward`` through the datamodule's
passthrough channel under :data:`GAM_DESIGN_KEY` — it never touches the
PreTab preprocessor. The combined output rides the standard forward-output
dict contract; the GAM smoothness penalty enters the loss through the
``*_penalty`` seam.
"""

from __future__ import annotations

import torch.nn as nn

from .compiled_terms import CompiledGAMTerms, CompiledGAMTermsModule

GAM_DESIGN_KEY = "__gam_design__"


class HybridAdditiveNet(nn.Module):
    """Compiled-GAM terms plus an arbitrary NAMpy architecture, additively.

    Constructed by :class:`~nampy.hybrid.joint.HybridJointRegressor` through
    the TaskModel harness; ``base_model_class`` and ``gam_payload`` arrive as
    extra TaskModel kwargs (excluded from checkpoint hyperparameters).
    """

    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 1,
        config=None,
        *,
        base_model_class=None,
        gam_payload: CompiledGAMTerms = None,
        **kwargs,
    ):
        super().__init__()
        if base_model_class is None or gam_payload is None:
            raise ValueError(
                "HybridAdditiveNet requires base_model_class and gam_payload."
            )
        if GAM_DESIGN_KEY in num_feature_info or GAM_DESIGN_KEY in cat_feature_info:
            raise ValueError(
                f"{GAM_DESIGN_KEY!r} is reserved for the compiled GAM design."
            )

        self.net = base_model_class(
            cat_feature_info=cat_feature_info,
            num_feature_info=num_feature_info,
            num_classes=num_classes,
            config=config,
            **kwargs,
        )
        self.gam = CompiledGAMTermsModule(gam_payload, num_classes=num_classes)

    def forward(self, num_features: dict, cat_features: dict) -> dict:
        num_features = dict(num_features)
        design = num_features.pop(GAM_DESIGN_KEY, None)
        if design is None:
            raise KeyError(
                f"num_features is missing the {GAM_DESIGN_KEY!r} passthrough "
                "entry with the compiled GAM design."
            )

        net_result = self.net(
            num_features=num_features, cat_features=cat_features
        )
        gam_result = self.gam(design)

        out = dict(net_result)
        out["output"] = net_result["output"] + gam_result["output"]
        # The GAM stage contributes one aggregate link-scale term ("baseline")
        # plus its quadratic smoothness penalty.
        out["gam_baseline"] = gam_result["output"]
        out["gam_penalty"] = gam_result["gam_penalty"]
        return out
