"""Tests for construct manager behavior."""

import numpy as np
import pytest

from spine.construct import BuildManager
from spine.data.larcv.particle import Particle
from spine.data.out import RecoFragment, TruthFragment

from .conftest import make_label_tensor, make_sparse_tensor


def test_manager_validates_mode_units_and_dependencies():
    """Invalid configuration should fail during manager construction."""
    with pytest.raises(ValueError, match="Run mode"):
        BuildManager(False, False, False, mode="invalid")

    with pytest.raises(ValueError, match="Units"):
        BuildManager(False, False, False, units="mm")

    with pytest.raises(ValueError, match="Interactions are built from particles"):
        BuildManager(False, False, True)

    with pytest.raises(KeyError, match="Unexpected data product"):
        BuildManager(False, False, False, sources={"bad": "data"})


def test_manager_validates_source_override_value_types():
    """Source overrides should be strings or sequences of strings."""
    with pytest.raises(TypeError, match="strings or sequences of strings"):
        BuildManager(False, False, False, sources={"data_tensor": 1})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="contain only strings"):
        BuildManager(False, False, False, sources={"data_tensor": ["data", 1]})  # type: ignore[list-item]


def test_build_sources_accepts_custom_source_names(points, depositions):
    """Configured source aliases should be used when preparing tensors."""
    BuildManager(
        fragments=False,
        particles=False,
        interactions=False,
        mode="both",
        units="px",
        sources={"data_tensor": "custom_data"},
    )
    manager = BuildManager(
        fragments=False,
        particles=False,
        interactions=False,
        mode="both",
        units="px",
        sources={"data_tensor": ["custom_data"]},
    )
    labels = make_label_tensor(points, depositions, [0, 0, 1, -1])
    data = {
        "custom_data": make_sparse_tensor(points, depositions),
        "clust_label": labels,
    }

    update = manager.build_sources(data)

    np.testing.assert_array_equal(update["points"], points)
    np.testing.assert_array_equal(update["depositions"], depositions)
    np.testing.assert_array_equal(update["points_label"], points)
    np.testing.assert_array_equal(update["depositions_label"], depositions)


def test_manager_source_overrides_do_not_mutate_defaults(points, depositions):
    """Custom source overrides should stay instance-local."""
    custom = BuildManager(
        fragments=False,
        particles=False,
        interactions=False,
        mode="both",
        units="px",
        sources={"data_tensor": "custom_data"},
    )
    default = BuildManager(
        fragments=False,
        particles=False,
        interactions=False,
        mode="both",
        units="px",
    )
    labels = make_label_tensor(points, depositions, [0, 0, 1, -1])

    custom_update = custom.build_sources(
        {"custom_data": make_sparse_tensor(points, depositions), "clust_label": labels}
    )
    default_update = default.build_sources(
        {"data": make_sparse_tensor(points, depositions), "clust_label": labels}
    )

    np.testing.assert_array_equal(custom_update["points"], points)
    np.testing.assert_array_equal(default_update["points"], points)


def test_build_sources_in_cm_requires_meta(points, depositions):
    """Centimeter source conversion should fail without metadata."""
    manager = BuildManager(False, False, False, mode="reco", units="cm")

    with pytest.raises(KeyError, match="metadata"):
        manager.build_sources(
            {"index": 0, "data": make_sparse_tensor(points, depositions)}
        )


def test_manager_builds_batched_sources(points, depositions):
    """Calling the manager on batched raw tensors should add batched source lists."""
    manager = BuildManager(False, False, False, mode="reco", units="px")
    data = {
        "index": np.array([0, 1]),
        "data": [
            make_sparse_tensor(points, depositions),
            make_sparse_tensor(points + 10, depositions + 10),
        ],
    }

    manager(data)

    assert len(data["points"]) == 2
    np.testing.assert_array_equal(data["points"][0], points)
    np.testing.assert_array_equal(data["points"][1], points + 10)


def test_manager_accepts_numpy_scalar_index(points, depositions):
    """NumPy scalar indices should be treated as single-entry inputs."""
    manager = BuildManager(False, False, False, mode="reco", units="px")
    data = {
        "index": np.int64(0),
        "data": make_sparse_tensor(points, depositions),
    }

    manager(data)

    np.testing.assert_array_equal(data["points"], points)
    np.testing.assert_array_equal(data["depositions"], depositions)


def test_manager_builds_reco_objects_end_to_end(points, depositions):
    """Manager should run fragment, particle and interaction builders together."""
    manager = BuildManager(True, True, True, mode="reco", units="px")
    data = {
        "index": 0,
        "data": make_sparse_tensor(points, depositions),
        "fragment_clusts": [np.array([0, 1]), np.array([2, 3])],
        "fragment_shapes": np.array([0, 1], dtype=np.int32),
        "fragment_group_pred": np.array([0, 1], dtype=np.int32),
        "particle_clusts": [np.array([0, 1]), np.array([2, 3])],
        "particle_shapes": np.array([0, 1], dtype=np.int32),
        "particle_start_points": np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32),
        "particle_end_points": np.array([[1, 0, 0], [3, 0, 0]], dtype=np.float32),
        "particle_group_pred": np.array([0, 0], dtype=np.int32),
        "particle_node_type_pred": np.eye(2, 6, dtype=np.float32),
        "particle_node_primary_pred": np.array([[1, 0], [0, 1]], dtype=np.float32),
    }

    manager(data)

    assert len(data["reco_fragments"]) == 2
    assert len(data["reco_particles"]) == 2
    assert len(data["reco_interactions"]) == 1


def test_manager_loads_matches_for_batched_existing_objects():
    """Manager should reconstruct stored match pairs for batched loaded objects."""
    reco = RecoFragment(
        id=0,
        is_matched=True,
        match_ids=np.array([0], dtype=np.int32),
        match_overlaps=np.array([0.5], dtype=np.float32),
    )
    truth = TruthFragment(
        id=0,
        is_matched=True,
        match_ids=np.array([0], dtype=np.int32),
        match_overlaps=np.array([0.5], dtype=np.float32),
    )
    data = {
        "index": np.array([0, 1]),
        "reco_fragments": [[reco], [RecoFragment(id=0)]],
        "truth_fragments": [[truth], [TruthFragment(id=0)]],
    }
    manager = BuildManager(True, False, False, mode="both", units="cm", lite=True)

    manager(data)

    assert data["fragment_matches_r2t"][0][0] == (reco, truth)
    assert data["fragment_matches_r2t"][1][0][1] is None


def test_build_sources_converts_points_and_truth_objects(points, depositions, meta_cm):
    """CM builds should convert point arrays and LArCV truth object positions."""
    particle = Particle(id=0, units="px", position=np.array([1, 1, 1]))
    labels = make_label_tensor(points, depositions, [0, 0, 1, 1])
    data = {
        "data": make_sparse_tensor(points, depositions),
        "clust_label": labels,
        "charge_label": make_sparse_tensor(points, depositions + 100),
        "clust_label_adapt": labels,
        "clust_label_g4": labels,
        "sources": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        "sources_label": np.array([[2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.int64),
        "orig_index": np.array([10, 11, 12, 13], dtype=np.int64),
        "orig_index_label": np.array([20, 21, 22, 23], dtype=np.int64),
        "meta": meta_cm,
        "particles": [particle],
        "fragment_start_points": np.array([[1, 1, 1]], dtype=np.float32),
    }
    manager = BuildManager(False, False, False, mode="both", units="cm")

    update = manager.build_sources(data)

    np.testing.assert_allclose(update["points"][0], [1, 1, 1])
    np.testing.assert_allclose(update["fragment_start_points"][0], [3, 3, 3])
    np.testing.assert_array_equal(update["sources"].dtype, np.dtype("int32"))
    np.testing.assert_array_equal(update["orig_index"], [10, 11, 12, 13])
    np.testing.assert_array_equal(update["depositions_q_label"], depositions + 100)
    np.testing.assert_allclose(update["points_g4"][0], [1, 1, 1])
    assert update["particles"][0].units == "cm"


def test_build_sources_selects_batched_point_attributes(points, depositions, meta_cm):
    """Batched source preparation should select entry-specific point attributes."""
    labels = make_label_tensor(points, depositions, [0, 0, 1, 1])
    data = {
        "data": [make_sparse_tensor(points, depositions)],
        "clust_label": [labels],
        "meta": [meta_cm],
        "fragment_start_points": [np.array([[1, 1, 1]], dtype=np.float32)],
    }
    manager = BuildManager(False, False, False, mode="both", units="cm")

    update = manager.build_sources(data, entry=0)

    np.testing.assert_allclose(update["fragment_start_points"][0], [3, 3, 3])


def test_manager_loads_matches_for_single_existing_objects():
    """Single-entry loaded objects should also get match-pair products."""
    reco = RecoFragment(
        id=0,
        is_matched=True,
        match_ids=np.array([0], dtype=np.int32),
        match_overlaps=np.array([0.25], dtype=np.float32),
    )
    truth = TruthFragment(
        id=0,
        is_matched=True,
        match_ids=np.array([0], dtype=np.int32),
        match_overlaps=np.array([0.25], dtype=np.float32),
    )
    data = {"index": 0, "reco_fragments": [reco], "truth_fragments": [truth]}
    manager = BuildManager(True, False, False, mode="both", units="cm", lite=True)

    manager(data)

    assert data["fragment_matches_r2t"][0] == (reco, truth)


def test_load_match_pairs_handles_matched_and_unmatched_objects():
    """Stored match metadata should reconstruct best-match object pairs."""
    reco = RecoFragment(
        id=0,
        is_matched=True,
        match_ids=np.array([1], dtype=np.int32),
        match_overlaps=np.array([0.75], dtype=np.float32),
    )
    unmatched = RecoFragment(id=1)
    truth = [TruthFragment(id=0), TruthFragment(id=1)]
    data = {"reco_fragments": [reco, unmatched], "truth_fragments": truth}

    result = BuildManager.load_match_pairs(data, "fragment")

    assert result["fragment_matches_r2t"][0] == (reco, truth[1])
    assert result["fragment_matches_r2t"][1] == (unmatched, None)
    assert result["fragment_matches_r2t_overlap"][0] == pytest.approx(0.75)
    assert result["fragment_matches_r2t_overlap"][1] == -1.0
