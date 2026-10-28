"""Train-data-dependent SIAN interaction selection."""

from dataclasses import replace

import torch

from ..architectures.components.interactions import resolve_interactions
from . import (
    ArchipelagoDetector,
    InteractionSearchConfig,
    concatenate_feature_tensors,
    fit_reference_model,
    select_interactions,
)


def prepare_sian_config(
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
        return config, {
            "selected_interactions_": tuple(selected),
            "interaction_selection_result_": None,
            "interaction_reference_model_": None,
        }

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
    reference = reference.to("cpu")

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
            return reference(rows.cpu())

    result = select_interactions(
        detector,
        predict,
        selection_inputs,
        selection_groups,
        search_config,
    )
    return replace(
        config,
        interactions=result.selected_interactions,
        interaction_degree=None,
    ), {
        "interaction_selection_result_": result,
        "selected_interactions_": result.selected_interactions,
        "interaction_reference_model_": reference,
    }
