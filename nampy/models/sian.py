"""Public estimator family for Sparse Interaction Additive Networks."""

from __future__ import annotations

from dataclasses import replace

import torch

from ..neural.architectures.components.interactions import resolve_interactions
from ..neural.interaction_selection import (
    ArchipelagoDetector,
    InteractionSearchConfig,
    concatenate_feature_tensors,
    fit_reference_model,
    select_interactions,
)
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

    def _prepare_architecture_config(
        self,
        config,
        *,
        train_num_features,
        train_cat_features,
        train_targets,
        val_num_features,
        val_cat_features,
        val_targets,
        objective,
        train_offset=None,
        train_sample_weight=None,
        random_state=0,
    ):
        del val_targets
        train_inputs, train_groups = concatenate_feature_tensors(
            train_num_features, train_cat_features
        )
        selection_inputs, selection_groups = concatenate_feature_tensors(
            val_num_features, val_cat_features
        )
        if train_groups.names != selection_groups.names:
            raise RuntimeError("Training and selection feature groups do not match.")

        if config.interactions is not None or config.interaction_degree is not None:
            selected = resolve_interactions(
                train_groups.names,
                config.interaction_degree,
                config.interactions,
            )
            self.selected_interactions_ = tuple(selected)
            self.interaction_selection_result_ = None
            self.interaction_reference_model_ = None
            return config

        if str(config.interaction_detector).lower() != "archipelago":
            raise ValueError(
                "SIAN currently supports interaction_detector='archipelago'; "
                "provide explicit interactions to bypass discovery."
            )

        reference = fit_reference_model(
            train_inputs,
            train_targets,
            objective_kind=objective.kind,
            hidden_sizes=config.reference_layer_sizes,
            output_index=config.selection_output_index,
            epochs=config.reference_epochs,
            batch_size=config.reference_batch_size,
            learning_rate=config.reference_lr,
            weight_decay=config.reference_weight_decay,
            sample_weight=train_sample_weight,
            offset=train_offset,
            random_state=random_state,
            device=config.reference_device,
        )
        self.interaction_reference_model_ = reference.to("cpu")

        detector_output = (
            config.selection_output_index if objective.kind == "multiclass" else 0
        )
        detector = ArchipelagoDetector(
            baseline=config.archipelago_baseline,
            max_samples=config.selection_max_samples,
            max_pairs=config.selection_max_pairs,
            batch_size=config.selection_batch_size,
            output_index=detector_output,
            random_state=random_state,
        )
        search_config = InteractionSearchConfig(
            max_order=config.max_interaction_order,
            threshold=config.interaction_thresholds,
            threshold_mode=config.threshold_mode,
            heredity_fraction=config.heredity_fraction,
            max_candidates=config.max_candidates,
            max_terms_per_order=config.max_terms_per_order,
        )

        def predict(rows):
            with torch.no_grad():
                return self.interaction_reference_model_(rows.cpu())

        result = select_interactions(
            detector,
            predict,
            selection_inputs,
            selection_groups,
            search_config,
        )
        self.interaction_selection_result_ = result
        self.selected_interactions_ = result.selected_interactions
        return replace(
            config,
            interactions=self.selected_interactions_,
            interaction_degree=None,
        )


_family = estimator_family("sian", module_name=__name__)
SIANRegressor = _family.regressor
SIANClassifier = _family.classifier
SIANLSS = _family.lss

__all__ = ["SIANClassifier", "SIANLSS", "SIANRegressor"]
