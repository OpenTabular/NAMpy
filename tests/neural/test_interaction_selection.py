from __future__ import annotations

import torch

from nampy.neural.interaction_selection import (
    ArchipelagoDetector,
    FeatureGroups,
    InteractionSearchConfig,
    concatenate_feature_tensors,
    interaction_frontier,
    select_interactions,
)


def test_feature_groups_keep_expanded_source_columns_together():
    inputs, groups = concatenate_feature_tensors(
        {"numeric": torch.ones(3, 2)},
        {"category": torch.ones(3, 4)},
    )
    assert inputs.shape == (3, 6)
    assert groups.columns == {
        "numeric": (0, 1),
        "category": (2, 3, 4, 5),
    }
    assert groups.indices(("numeric", "category")) == tuple(range(6))


def test_fractional_heredity_controls_higher_order_frontier():
    features = ("a", "b", "c")
    selected = (("a",), ("b",), ("c",), ("a", "b"), ("a", "c"))

    assert interaction_frontier(
        features, selected, order=3, heredity_fraction=0.5
    ) == [("a", "b", "c")]
    assert interaction_frontier(
        features, selected, order=3, heredity_fraction=1.0
    ) == []


def test_archipelago_detects_known_product_and_search_selects_it():
    inputs = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 2.0],
            [1.0, 1.0, 3.0],
        ]
    )
    groups = FeatureGroups(
        names=("a", "b", "c"),
        columns={"a": (0,), "b": (1,), "c": (2,)},
    )

    def predict(rows):
        return (rows[:, 0] * rows[:, 1] + 0.5 * rows[:, 2]).unsqueeze(-1)

    detector = ArchipelagoDetector(
        baseline="pairwise", max_samples=4, max_pairs=6, batch_size=64
    )
    scores = detector.score(
        predict,
        inputs,
        [("a", "b"), ("a", "c"), ("b", "c")],
        groups,
    )
    score_by_term = {item.interaction: item.score for item in scores}
    assert score_by_term[("a", "b")] > 0
    assert score_by_term[("a", "c")] == 0
    assert score_by_term[("b", "c")] == 0

    result = select_interactions(
        detector,
        predict,
        inputs,
        groups,
        InteractionSearchConfig(
            max_order=2,
            threshold=1 / 3,
            threshold_mode="fraction",
        ),
    )
    assert result.selected_interactions == (("a", "b"),)
    assert result.score_for(("a", "b")).n_contrasts == 6
