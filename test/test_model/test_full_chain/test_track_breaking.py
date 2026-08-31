"""Tests for the full-chain artificial track-breaking provider."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.data import IndexBatch, TensorBatch
from spine.model.full_chain.point import PointBatch
from spine.model.full_chain.providers.transform.track_breaking import (
    LogicalTPCBoundary,
    TrackBreakingStage,
    build_track_breaking_stage,
)
from spine.model.full_chain.state import ChainState, StageResult


def make_state(shape=TRACK_SHP, sources=None) -> ChainState:
    """Build one particle spanning six voxel rows in one event."""
    points = np.asarray(
        [
            [-3.0, 0.0, 0.0, 1.0],
            [-2.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    data = TensorBatch(points, counts=[6], coord_cols=np.arange(3))
    if sources is None:
        sources = np.asarray([[2, 4]] * 3 + [[2, 5]] * 3, dtype=np.int64)
    source_batch = TensorBatch(sources, counts=[6])
    index = np.arange(6, dtype=np.int64)
    particles = IndexBatch([index], [6], [1], [6])
    shapes = TensorBatch(np.asarray([shape], dtype=np.int64), counts=[1])
    return ChainState(
        data=data,
        sources=source_batch,
        particle_clusts=particles,
        particle_shapes=shapes,
        particle_primaries=particles,
    )


def make_location(**overrides) -> LogicalTPCBoundary:
    """Build one always-on logical provenance boundary."""
    config = {
        "name": "z_zero",
        "module_id": 2,
        "tpc_pair": [4, 5],
        "frequency": 1.0,
    }
    config.update(overrides)
    return LogicalTPCBoundary.from_config(0, config)


def test_particle_track_breaking_splits_by_logical_source() -> None:
    """An eligible track becomes two particle products without a spatial gap."""
    stage = TrackBreakingStage("track_break", "particle", [make_location()])
    result = stage(make_state())

    clusts = result.products["particle_clusts"]
    assert clusts.counts.tolist() == [2]
    assert [index.tolist() for index in clusts.index_list] == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert result.products["particle_shapes"].tensor.tolist() == [TRACK_SHP] * 2
    assert [
        index.tolist() for index in result.products["particle_primaries"].index_list
    ] == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert result.outputs["track_break_parent_ids"].tensor.tolist() == [0, 0]
    assert result.outputs["track_break_location_ids"].tensor.tolist() == [0, 0]


def test_track_breaking_requires_exact_logical_contributors() -> None:
    """Physical equivalence or an extra logical source does not imply eligibility."""
    sources = np.asarray(
        [[2, 4]] * 2 + [[2, 5]] * 2 + [[2, 6]] * 2,
        dtype=np.int64,
    )
    result = TrackBreakingStage("track_break", "particle", [make_location()])(
        make_state(sources=sources)
    )

    assert result.products["particle_clusts"].counts.tolist() == [1]
    assert result.outputs["track_break_location_ids"].tensor.tolist() == [-1]


def test_grouped_boundary_can_apply_indiscriminately_across_modules() -> None:
    """Two logical-TPC partitions cover all corresponding boundaries."""
    sources = np.asarray(
        [[7, 0], [7, 1], [7, 2], [7, 2], [7, 3], [7, 3]],
        dtype=np.int64,
    )
    cathode = LogicalTPCBoundary.from_config(
        0,
        {
            "name": "all_cathodes",
            "tpc_groups": [[0, 1], [2, 3]],
            "frequency": 1.0,
        },
    )
    result = TrackBreakingStage("track_break", "particle", [cathode])(
        make_state(sources=sources)
    )

    assert [
        index.tolist() for index in result.products["particle_clusts"].index_list
    ] == [
        [0, 1],
        [2, 3, 4, 5],
    ]


def test_grouped_boundary_can_restrict_shared_response_to_one_module() -> None:
    """A module selector applies one response to every TPC in its groups."""
    boundary = LogicalTPCBoundary.from_config(
        0,
        {
            "name": "module_0_z_zero",
            "module_id": 0,
            "tpc_groups": [[0, 2], [1, 3]],
            "frequency": 1.0,
        },
    )
    sources = np.asarray([[1, 0]] * 3 + [[1, 1]] * 3, dtype=np.int64)
    result = TrackBreakingStage("track_break", "particle", [boundary])(
        make_state(sources=sources)
    )

    assert result.products["particle_clusts"].counts.tolist() == [1]


def test_grouped_boundaries_can_split_one_track_successively() -> None:
    """A four-source track can be separated at two provenance boundaries."""
    sources = np.asarray(
        [[0, 0], [0, 0], [0, 1], [0, 1], [0, 2], [0, 3]],
        dtype=np.int64,
    )
    cathode = LogicalTPCBoundary.from_config(
        0,
        {
            "name": "cathode",
            "tpc_groups": [[0, 1], [2, 3]],
            "frequency": 1.0,
        },
    )
    z_zero = LogicalTPCBoundary.from_config(
        1,
        {
            "name": "z_zero",
            "tpc_groups": [[0, 2], [1, 3]],
            "frequency": 1.0,
        },
    )

    result = TrackBreakingStage("track_break", "particle", [cathode, z_zero])(
        make_state(sources=sources)
    )

    assert result.products["particle_clusts"].counts.tolist() == [4]


def test_track_breaking_leaves_non_tracks_unchanged() -> None:
    """Only track-shaped clusters are eligible for the transformation."""
    result = TrackBreakingStage("track_break", "particle", [make_location()])(
        make_state(shape=SHOWR_SHP)
    )

    assert result.products["particle_clusts"].counts.tolist() == [1]
    assert result.products["particle_shapes"].tensor.tolist() == [SHOWR_SHP]


def test_track_breaking_can_target_fragments() -> None:
    """The same provenance transformation can run before particle aggregation."""
    state = make_state()
    state.products["fragment_clusts"] = state.products.pop("particle_clusts")
    state.products["fragment_shapes"] = state.products.pop("particle_shapes")
    state.products.pop("particle_primaries")

    result = TrackBreakingStage("track_break", "fragment", [make_location()])(state)

    assert result.products["fragment_clusts"].counts.tolist() == [2]
    assert result.products["fragment_shapes"].tensor.tolist() == [TRACK_SHP] * 2
    assert "particle_primaries" not in result.products


def test_track_breaking_applies_angular_response_and_size_threshold() -> None:
    """Location response and optional child-size threshold can veto a break."""
    angular = make_location(
        normal=[1.0, 0.0, 0.0],
        angular_response="0*x",
    )
    result = TrackBreakingStage("track_break", "particle", [angular])(make_state())
    assert result.products["particle_clusts"].counts.tolist() == [1]
    assert result.outputs["track_break_probabilities"].tensor.tolist() == [0.0]

    result = TrackBreakingStage(
        "track_break",
        "particle",
        [make_location()],
        min_voxels_per_side=4,
    )(make_state())
    assert result.products["particle_clusts"].counts.tolist() == [1]


def test_track_breaking_uses_metadata_and_run_identity_for_angular_draws() -> None:
    """Angular coordinates and random draws use stable event context."""

    class Meta:
        calls = 0

        def to_cm(self, points, center=True):
            assert center is True
            self.calls += 1
            return points

    location = make_location(
        frequency=0.5,
        normal=[1.0, 0.0, 0.0],
        angular_response="x",
    )
    stage = TrackBreakingStage("track_break", "particle", [location], seed=7)
    state = make_state()
    meta = Meta()
    state.products["meta"] = [meta]
    state.products["run_info"] = [SimpleNamespace(run=1, subrun=2, event=3)]

    first = stage(state).outputs["track_break_draws"].tensor
    second = stage(state).outputs["track_break_draws"].tensor

    assert meta.calls == 2
    np.testing.assert_array_equal(first, second)


def test_track_breaking_handles_degenerate_and_invalid_angular_responses() -> None:
    """Degenerate directions veto breaking and non-finite responses fail."""
    assert TrackBreakingStage._direction(np.zeros((1, 3))) is None
    assert TrackBreakingStage._direction(np.zeros((2, 3))) is None

    location = make_location(normal=[1.0, 0.0, 0.0], angular_response="x")
    stage = TrackBreakingStage("track_break", "particle", [location])
    masks = (
        np.asarray([True, True, False, False]),
        ~np.asarray([True, True, False, False]),
    )
    assert stage._probability(location, np.zeros((4, 3)), masks) == 0.0

    bad_response = replace(location, response=lambda _x: np.asarray([np.nan]))
    stage = TrackBreakingStage("track_break", "particle", [bad_response])
    with pytest.raises(ValueError, match="non-finite"):
        stage(make_state())


def test_track_breaking_validates_runtime_source_provenance() -> None:
    """Runtime source data must remain attached and contain two columns."""
    state = make_state()
    state.products["point_data"] = PointBatch.from_input(state.products["data"])
    with pytest.raises(ValueError, match="requires voxel-aligned"):
        TrackBreakingStage("track_break", "particle", [make_location()])(state)

    bad_sources = np.zeros((6, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="logical TPC.*pairs"):
        TrackBreakingStage("track_break", "particle", [make_location()])(
            make_state(sources=bad_sources)
        )


def test_replaced_product_updates_existing_public_alias() -> None:
    """Declared transformations synchronize canonical and public products."""
    state = ChainState(data=TensorBatch(np.ones((1, 4)), [1], coord_cols=[0, 1, 2]))
    original = object()
    replacement = object()
    state.publish(
        "producer",
        StageResult(
            products={"particle_clusts": original},
            outputs={"particle_clusts": original},
        ),
    )
    state.publish(
        "transform",
        StageResult(products={"particle_clusts": replacement}),
        frozenset({"particle_clusts"}),
    )

    assert state.products["particle_clusts"] is replacement
    assert state.outputs["particle_clusts"] is replacement


def test_track_breaking_builder_validates_logical_boundary_config() -> None:
    """The provider builder rejects ambiguous source and response settings."""
    stage = build_track_breaking_stage(
        "track_break",
        {
            "locations": [
                {
                    "module_id": 2,
                    "tpc_pair": [4, 5],
                    "frequency": 0.1,
                }
            ]
        },
        object(),
    )
    assert isinstance(stage, TrackBreakingStage)
    assert stage.target == "particle"

    with pytest.raises(ValueError, match="distinct logical TPC"):
        make_location(tpc_pair=[4, 4])
    with pytest.raises(ValueError, match="require `normal`"):
        make_location(angular_response="x")


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ([], TypeError, "must be a mapping"),
        (
            {"tpc_pair": [0, 1], "frequency": 1.0, "unknown": True},
            ValueError,
            "Unknown",
        ),
        (
            {"name": "", "tpc_pair": [0, 1], "frequency": 1.0},
            ValueError,
            "names",
        ),
        (
            {"module_id": "0", "tpc_pair": [0, 1], "frequency": 1.0},
            TypeError,
            "module_id",
        ),
        ({"frequency": 1.0}, ValueError, "exactly one"),
        (
            {
                "tpc_pair": [0, 1],
                "tpc_groups": [[0], [1]],
                "frequency": 1.0,
            },
            ValueError,
            "exactly one",
        ),
        ({"tpc_groups": [[0], []], "frequency": 1.0}, ValueError, "nonempty"),
        (
            {"tpc_groups": [[0, 0], [1]], "frequency": 1.0},
            ValueError,
            "unique",
        ),
        (
            {"tpc_groups": [[0, 1], [1, 2]], "frequency": 1.0},
            ValueError,
            "disjoint",
        ),
        ({"tpc_pair": [0, 1], "frequency": 2.0}, ValueError, r"\[0, 1\]"),
        (
            {"tpc_pair": [0, 1], "frequency": 1.0, "normal": [1, 2]},
            ValueError,
            "3-vector",
        ),
        (
            {"tpc_pair": [0, 1], "frequency": 1.0, "normal": [0, 0, 0]},
            ValueError,
            "nonzero",
        ),
        (
            {
                "tpc_pair": [0, 1],
                "frequency": 1.0,
                "normal": [1, 0, 0],
                "angular_response": 1,
            },
            TypeError,
            "must be a string",
        ),
    ],
)
def test_logical_tpc_boundary_rejects_invalid_configuration(
    config, error, message
) -> None:
    """Malformed boundary selectors and response options fail clearly."""
    with pytest.raises(error, match=message):
        LogicalTPCBoundary.from_config(0, config)


def test_track_breaking_stage_rejects_invalid_configuration() -> None:
    """Stage-level target, RNG and threshold options are validated."""
    location = make_location()
    with pytest.raises(ValueError, match="target"):
        TrackBreakingStage("track_break", "bad", [location])
    with pytest.raises(ValueError, match="at least one"):
        TrackBreakingStage("track_break", "particle", [])
    with pytest.raises(TypeError, match="seed"):
        TrackBreakingStage("track_break", "particle", [location], seed=1.5)
    with pytest.raises(ValueError, match="positive"):
        TrackBreakingStage("track_break", "particle", [location], min_voxels_per_side=0)
    with pytest.raises(ValueError, match="must be unique"):
        TrackBreakingStage("track_break", "particle", [location, location])

    with pytest.raises(TypeError, match="locations"):
        build_track_breaking_stage("track_break", {"locations": ()}, object())
