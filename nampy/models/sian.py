"""Public estimator family for Sparse Interaction Additive Networks."""

from __future__ import annotations

from ._registered import estimator_family


class _SIANSelectionMixin:
    """Resolve SIAN interactions after train-only preprocessing and splitting."""

    def interaction_selection_table(self):
        """Return the fitted detector scores and selection decisions."""
        result = getattr(self, "interaction_selection_result_", None)
        if result is None:
            raise ValueError(
                "No interaction-selection result is available; the estimator is "
                "unfitted or explicit interactions were supplied."
            )
        return result.to_frame()

    def compress_terms(self):
        """Switch a fitted SIAN to independent shape subnetworks."""
        if getattr(self, "model", None) is None:
            raise ValueError("Fit the SIAN estimator before compressing its terms.")
        self.model.model.compress()
        self.execution_mode_ = self.model.model.execution_mode_
        return self

    def block_mask_terms(self):
        """Switch a fitted SIAN back to parallel block-masked execution."""
        if getattr(self, "model", None) is None:
            raise ValueError("Fit the SIAN estimator before block-masking its terms.")
        self.model.model.block_mask()
        self.execution_mode_ = self.model.model.execution_mode_
        return self


_family = estimator_family("sian", module_name=__name__, mixin=_SIANSelectionMixin)
SIANRegressor = _family.regressor
SIANClassifier = _family.classifier
SIANLSS = _family.lss

__all__ = ["SIANClassifier", "SIANLSS", "SIANRegressor"]
