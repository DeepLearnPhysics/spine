"""Unit tests for the provider-driven full reconstruction chain."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from spine.constants import GHOST_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.full_chain import FullChain
from spine.model.full_chain.config import build_chain_plan
from spine.model.full_chain.ops import AggregationOperations
from spine.model.full_chain.providers.aggregation import (
    build_interaction_aggregation_loss,
)
from spine.model.full_chain.providers.fragmentation import FragmentationStage
from spine.model.full_chain.providers.image import (
    ParticleImageStage,
    build_particle_image_stage,
)
from spine.model.full_chain.providers.segmentation import (
    SegmentationLossStage,
    SegmentationStage,
)
from spine.model.full_chain.registry import ProviderSpec, register_provider
from spine.model.full_chain.stage import ChainStage
from spine.model.full_chain.state import ChainState, StageResult
from spine.utils.cluster.label import ClusterLabelAdapter


def make_data(size: int = 4) -> TensorBatch:
    """Build one sparse tensor batch with canonical coordinates and values."""
    rows = torch.zeros((size, 5), dtype=torch.float32)
    rows[:, 1] = torch.arange(size)
    rows[:, 4] = 1.0
    return TensorBatch(
        rows,
        counts=[size],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


def make_cluster_label() -> ClusterLabelBatch:
    """Build compact cluster labels for two particles in one event."""
    rows = np.array(
        [
            [0, 0, 0, 0, 1, 10, 0],
            [0, 1, 0, 0, 1, 10, 0],
            [0, 2, 0, 0, 1, 20, 1],
            [0, 3, 0, 0, 1, 20, 1],
        ],
        dtype=np.float32,
    )
    data = TensorBatch(
        rows,
        counts=[4],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    particles = {
        "shape": TensorBatch(np.array([SHOWR_SHP, TRACK_SHP]), counts=[2]),
        "group": TensorBatch(np.array([7, 9]), counts=[2]),
        "group_primary": TensorBatch(np.array([1, 0]), counts=[2]),
    }
    return ClusterLabelBatch(data, particles)


def make_clusters() -> tuple[IndexBatch, TensorBatch]:
    """Build two voxel clusters and their semantic shapes."""
    clusts = IndexBatch(
        [np.array([0, 1]), np.array([2, 3])],
        spans=[4],
        counts=[2],
        single_counts=[2, 2],
    )
    shapes = TensorBatch(np.array([SHOWR_SHP, TRACK_SHP]), counts=[2])
    return clusts, shapes


def test_chain_state_enforces_declared_replacements() -> None:
    """Canonical products cannot be silently overwritten by a provider."""
    state = ChainState(data=object())
    with pytest.raises(ValueError, match="without declaring"):
        state.publish("bad", StageResult({"data": object()}))

    replacement = object()
    state.publish(
        "good",
        StageResult({"data": replacement}, {"data_adapt": replacement}),
        frozenset({"data"}),
    )
    assert state.require("data") is replacement


def test_native_plan_resolves_used_and_loss_blocks() -> None:
    """Ordered configs resolve named model blocks without flattening them."""
    chain = {
        "stages": [
            {
                "name": "semantic",
                "provider": "segmentation",
                "uses": "backbone",
                "loss": "objective",
                "config": {"mode": "label"},
            }
        ]
    }
    plan = build_chain_plan(
        chain,
        {"backbone": {"depth": 5}, "objective": {"loss": "ce"}},
    )
    assert plan[0].config == {"backbone": {"depth": 5}, "mode": "label"}
    assert plan[0].loss_config == {"loss": "ce"}


def test_legacy_plan_translates_particle_image_tasks() -> None:
    """Historical image task modes become an explicit image provider."""
    plan = build_chain_plan(
        {
            "fragmentation": "label",
            "particle_aggregation": "label",
            "particle_identification": "image",
        },
        {"image_particle": {"encoder": {}, "heads": {"type": 5}}},
    )
    assert [stage.provider for stage in plan] == [
        "fragmentation",
        "particle_aggregation",
        "particle_image",
    ]


def test_external_provider_can_supply_multiple_capabilities() -> None:
    """A combined semantic/fragment provider needs no FullChain changes."""

    class CombinedStage(ChainStage):
        requires = frozenset({"data"})
        provides = frozenset({"seg_pred", "fragment_clusts", "fragment_shapes"})

        def forward(self, state: ChainState) -> StageResult:
            state.require("data", self.name)
            clusts = IndexBatch([np.arange(4)], [4], [1], [4])
            products = {
                "seg_pred": TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
                "fragment_clusts": clusts,
                "fragment_shapes": TensorBatch(np.array([SHOWR_SHP]), [1]),
            }
            return StageResult(products, dict(products))

    def build(name: str, _config: dict[str, Any], _owner: Any) -> ChainStage:
        return CombinedStage(name)

    provider_name = "test_combined_semantic_fragments"
    register_provider(ProviderSpec(provider_name, build))
    chain = FullChain(
        chain={
            "stages": [
                {"name": "combined", "provider": provider_name},
            ]
        }
    )
    result = chain(make_data())
    assert set(result) == {"seg_pred", "fragment_clusts", "fragment_shapes"}


def test_segmentation_deghosts_only_row_aligned_ppn_outputs() -> None:
    """Internal sparse PPN products retain their independently pruned rows."""

    class Model:
        def __call__(self, _data: TensorBatch) -> dict[str, Any]:
            return {
                "segmentation": TensorBatch(torch.zeros((3, 5)), [3]),
                "ghost": TensorBatch(
                    torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
                    [3],
                ),
                "ppn_points": TensorBatch(torch.arange(30).reshape(3, 10), [3]),
                "ppn_coords": [TensorBatch(torch.zeros((1, 4)), [1])],
            }

    stage = SegmentationStage(
        "semantic",
        "uresnet",
        Model(),  # type: ignore[arg-type]
        ClusterLabelAdapter(),
    )
    state = ChainState(data=make_data(3))
    result = stage(state)
    assert result.outputs["ppn_points"].counts.tolist() == [2]
    assert result.outputs["ppn_coords"][0].counts.tolist() == [1]


def test_segmentation_loss_aligns_cached_deghosted_rows() -> None:
    """Cached deghosting indexes align semantic labels and logits."""
    seg_label = TensorBatch(
        torch.tensor([0, GHOST_SHP, 1, 2, GHOST_SHP]).float(),
        [5],
    )
    logits = TensorBatch(torch.zeros((3, 3)), [3])
    orig_index = IndexBatch(torch.tensor([0, 2, 4]), spans=[5], counts=[3])
    aligned_label, aligned_logits = SegmentationLossStage._align(
        seg_label,
        logits,
        orig_index,
    )
    assert aligned_label.values.torch_tensor().tolist() == [0.0, 1.0]
    assert aligned_logits.shape[0] == 2


def test_label_fragmentation_uses_structured_shape_field() -> None:
    """Truth fragmentation reads shape through ClusterLabelBatch aliases."""
    stage = FragmentationStage("fragments", "label", None, None)
    state = ChainState(
        data=make_data(),
        seg_pred=TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
        clust_label=make_cluster_label(),
    )
    result = stage(state)
    assert [
        index.tolist() for index in result.products["fragment_clusts"].index_list
    ] == [
        [0, 1],
        [2, 3],
    ]
    assert result.products["fragment_shapes"].numpy_tensor().tolist() == [
        SHOWR_SHP,
        TRACK_SHP,
    ]


def test_group_builder_falls_back_when_primary_is_missing() -> None:
    """Primary-aware grouping falls back to the full group when necessary."""
    clusts, shapes = make_clusters()
    assignments = TensorBatch(np.array([5, 5]), [2])
    primary_mask = TensorBatch(np.array([False, False]), [2])
    groups, group_shapes, primaries = AggregationOperations.build_groups(
        clusts,
        shapes,
        assignments,
        primary_mask,
        aggregate_shapes=True,
        shape_use_primary=True,
        retain_primaries=True,
    )
    assert [group.tolist() for group in groups.index_list] == [[0, 1, 2, 3]]
    assert group_shapes.numpy_tensor().tolist() == [SHOWR_SHP]
    assert primaries.index_list[0].tolist() == [0, 1, 2, 3]


def test_particle_image_publishes_grappa_compatible_keys() -> None:
    """Image task predictions satisfy existing particle-builder interfaces."""

    class Model:
        def __call__(self, data: TensorBatch, objects: IndexBatch) -> dict[str, Any]:
            assert len(objects.index_list) == 2
            return {
                "type_pred": TensorBatch(torch.zeros((2, 5)), objects.counts),
                "energy_pred": TensorBatch(torch.ones((2, 1)), objects.counts),
            }

    particles, _ = make_clusters()
    stage = ParticleImageStage(
        "particle_tasks",
        Model(),  # type: ignore[arg-type]
        {"type": "type", "energy": "energy"},
    )
    result = stage(ChainState(data=make_data(), particle_clusts=particles))
    assert set(result.outputs) == {
        "particle_node_type_pred",
        "particle_node_energy_pred",
    }
    assert set(result.products) == {"particle_type_pred", "particle_energy_pred"}


def test_particle_image_builder_accepts_native_module_name() -> None:
    """Native `uses: image_particle` blocks build explicit object models."""
    owner = torch.nn.Module()
    stage = build_particle_image_stage(
        "particle_tasks",
        {
            "image_particle": {
                "objects": {"source": "explicit"},
                "encoder": {
                    "name": "cnn",
                    "num_input": 1,
                    "spatial_size": 768,
                    "filters": 4,
                    "depth": 2,
                    "reps": 1,
                },
                "heads": {"pid": 5, "primary": 2},
            }
        },
        owner,
    )
    assert stage.provides == {"particle_type_pred", "particle_primary_pred"}
    assert owner.image_particle is stage.model


def test_image_owned_task_rejects_duplicate_grappa_loss() -> None:
    """Delegated image tasks cannot retain unreachable GrapPA objectives."""
    with pytest.raises(ValueError, match="image-owned particle task.*type"):
        build_interaction_aggregation_loss(
            "interaction",
            {
                "task_modes": {"type": "image", "primary": "grappa"},
                "loss": {
                    "node_loss": {
                        "type": {"name": "class", "target": "pid"},
                        "primary": {
                            "name": "class",
                            "target": "interaction_primary",
                        },
                    }
                },
            },
            torch.nn.Module(),
        )
