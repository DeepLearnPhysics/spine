"""Tests for interaction-level full-chain vertex reduction."""

import numpy as np
import pytest
import torch

from spine.data import IndexBatch, TensorBatch
from spine.model.full_chain import PointBatch
from spine.model.full_chain.providers.vertexing import (
    InteractionVertexingStage,
    build_interaction_vertexing_stage,
)
from spine.model.full_chain.state import ChainState
from spine.model.uresnet.ppn.vertex import vertex_raw_schema


def make_interactions() -> IndexBatch:
    """Build two reconstructed interactions in one event."""
    return IndexBatch(
        [np.array([0, 1, 2]), np.array([3, 4])],
        spans=[5],
        counts=[2],
        single_counts=[3, 2],
    )


def make_point_data() -> PointBatch:
    """Build five sparse voxels in one event."""
    rows = torch.zeros((5, 5), dtype=torch.float32)
    rows[:, 1] = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0])
    data = TensorBatch(
        rows,
        counts=[5],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    return PointBatch.from_input(data)


def test_ppn_vertexing_pools_candidates_and_falls_back() -> None:
    """PPN mode should pool passing sites and retain an argmax fallback."""
    proposals = torch.zeros((5, 5), dtype=torch.float32)
    proposals[:, 3:] = torch.tensor(
        [[0.0, 3.0], [0.0, 2.0], [3.0, 0.0], [3.0, 0.0], [2.0, 0.0]]
    )
    proposals = TensorBatch(
        proposals,
        counts=[5],
        schema=vertex_raw_schema(3),
    )
    state = ChainState(
        point_data=make_point_data(),
        vertex_proposals=proposals,
        interaction_clusts=make_interactions(),
    )

    result = InteractionVertexingStage("vertex", "ppn")(state)
    vertices = result.products["interaction_vertices"].torch_tensor()
    scores = result.products["interaction_vertex_scores"].torch_tensor()

    assert vertices.shape == (2, 3)
    assert 0.5 < vertices[0, 0] < 1.5
    assert vertices[0, 1:].tolist() == pytest.approx([0.5, 0.5])
    assert vertices[1].tolist() == pytest.approx([11.5, 0.5, 0.5])
    assert scores.tolist() == pytest.approx(
        [
            torch.softmax(torch.tensor([0.0, 3.0]), dim=0)[1].item(),
            torch.softmax(torch.tensor([2.0, 0.0]), dim=0)[1].item(),
        ]
    )


def test_grappa_vertexing_selects_most_primary_particle() -> None:
    """GrapPA mode should select one decoded primary proposal per group."""
    proposals = TensorBatch(
        torch.tensor(
            [
                [2.0, 0.0, 1.0, 2.0, 3.0],
                [0.0, 3.0, 4.0, 5.0, 6.0],
                [0.0, 2.0, 7.0, 8.0, 9.0],
            ]
        ),
        counts=[3],
    )
    group_ids = TensorBatch(torch.tensor([0, 0, 1]), counts=[3])
    state = ChainState(
        particle_vertex_proposals=proposals,
        particle_interaction_ids=group_ids,
        interaction_clusts=make_interactions(),
    )

    result = InteractionVertexingStage("vertex", "grappa")(state)

    vertices = result.products["interaction_vertices"].torch_tensor()
    np.testing.assert_allclose(vertices.numpy(), [[4, 5, 6], [7, 8, 9]])
    scores = result.products["interaction_vertex_scores"].torch_tensor()
    assert torch.all(scores > 0.8)


def test_ppn_vertexing_supports_mean_cluster_ranking() -> None:
    """Mean pooling should rank spatially distinct proposal clusters."""
    stage = InteractionVertexingStage(
        "vertex",
        "ppn",
        score_threshold=0.0,
        pool_radius=0.5,
        pool_score_fn="mean",
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]])
    scores = torch.tensor([0.7, 0.9, 0.75])

    vertex, score = stage._pool_ppn_candidates(positions, scores)

    assert 0.0 < vertex[0] < 0.1
    assert score.item() == pytest.approx(0.8)


def test_vertexing_packages_empty_interaction_batches() -> None:
    """An event with no reconstructed interactions should return empty products."""
    interactions = IndexBatch(
        [],
        spans=[0],
        counts=[0],
        single_counts=[],
        default=np.empty(0, dtype=np.int64),
    )
    proposals = TensorBatch(torch.empty((0, 5)), counts=[0])
    group_ids = TensorBatch(torch.empty(0, dtype=torch.long), counts=[0])
    state = ChainState(
        particle_vertex_proposals=proposals,
        particle_interaction_ids=group_ids,
        interaction_clusts=interactions,
    )

    result = InteractionVertexingStage("vertex", "grappa")(state)

    assert result.products["interaction_vertices"].shape == (0, 3)
    assert result.products["interaction_vertex_scores"].shape == (0,)


def test_ppn_vertexing_validates_product_alignment() -> None:
    """PPN proposals must align with points, events, and nonempty interactions."""
    stage = InteractionVertexingStage("vertex", "ppn")
    interactions = make_interactions()
    state = ChainState(
        point_data=make_point_data(),
        vertex_proposals=TensorBatch(torch.zeros((4, 5)), counts=[4]),
        interaction_clusts=interactions,
    )
    with pytest.raises(ValueError, match="row-aligned"):
        stage(state)

    state.products["vertex_proposals"] = TensorBatch(torch.zeros((5, 5)), counts=[3, 2])
    with pytest.raises(ValueError, match="same batch size"):
        stage(state)

    empty_interaction = IndexBatch(
        [np.empty(0, dtype=np.int64)],
        spans=[5],
        counts=[1],
        single_counts=[0],
    )
    state.products["vertex_proposals"] = TensorBatch(
        torch.zeros((5, 5)),
        counts=[5],
        schema=vertex_raw_schema(3),
    )
    state.products["interaction_clusts"] = empty_interaction
    with pytest.raises(ValueError, match="empty interaction"):
        stage(state)


@pytest.mark.parametrize(
    ("proposals", "group_ids", "message"),
    [
        (torch.zeros((2, 4)), torch.tensor([0, 0]), "must contain two"),
        (torch.zeros((2, 5)), torch.tensor([0]), "must be aligned"),
        (torch.zeros((2, 5)), torch.tensor([0, 0]), "do not match"),
    ],
)
def test_grappa_vertexing_validates_product_alignment(
    proposals, group_ids, message
) -> None:
    """GrapPA proposals, assignments, and interactions must align."""
    stage = InteractionVertexingStage("vertex", "grappa")
    state = ChainState(
        particle_vertex_proposals=TensorBatch(proposals, counts=[len(proposals)]),
        particle_interaction_ids=TensorBatch(group_ids, counts=[len(group_ids)]),
        interaction_clusts=make_interactions(),
    )

    with pytest.raises(ValueError, match=message):
        stage(state)


def test_vertexing_builder_constructs_non_trainable_stage() -> None:
    """The registry builder should pass provider configuration through."""
    stage = build_interaction_vertexing_stage(
        "vertex",
        {"mode": "ppn", "score_threshold": 0.7},
        object(),
    )
    assert isinstance(stage, InteractionVertexingStage)
    assert stage.score_threshold == 0.7


def test_grappa_vertexing_declares_decoding_requirements() -> None:
    """Normalized anchor decoding should require metadata and both endpoints."""
    stage = InteractionVertexingStage(
        "vertex",
        "grappa",
        normalize_positions=True,
        use_anchor_points=True,
    )
    assert {"meta", "particle_vertex_start_points", "particle_vertex_end_points"} <= (
        stage.requires
    )
    assert stage.optional == frozenset()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "bad"}, "must be `ppn` or `grappa`"),
        ({"mode": "ppn", "score_threshold": 2.0}, "between zero and one"),
        ({"mode": "ppn", "pool_radius": 0.0}, "must be positive"),
        ({"mode": "ppn", "pool_score_fn": "sum"}, "must be `max` or `mean`"),
        ({"mode": "ppn", "normalize_positions": True}, "cannot be used"),
    ],
)
def test_vertexing_rejects_invalid_configuration(kwargs, message) -> None:
    """Invalid reduction options should fail during configuration."""
    with pytest.raises(ValueError, match=message):
        InteractionVertexingStage("vertex", **kwargs)
