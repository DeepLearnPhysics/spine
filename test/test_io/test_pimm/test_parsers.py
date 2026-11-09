"""PIMM charge/label alignment, truth and malformed-input regression tests."""

import copy

import numpy as np
import pytest

from spine.io.parse.pimm import (
    PIMMClusterLabelParser,
    PIMMEventInfoParser,
    PIMMMetaParser,
    PIMMParticleCoordsParser,
    PIMMParticleGraphParser,
    PIMMParticleParser,
    PIMMParticlePointsParser,
    PIMMSegmentationParser,
    PIMMTensorParser,
)


def test_aligned_charge_and_truth(raw):
    """Charge overlaps share winners; unknown inputs keep an ignored target."""
    parser = PIMMTensorParser("float32", voxel_size_mm=1, include_unlabeled=True)
    data = parser(raw)
    seg = PIMMSegmentationParser("float32", voxel_size_mm=1, include_unlabeled=True)(
        raw
    )
    clust = PIMMClusterLabelParser("float32", voxel_size_mm=1, include_unlabeled=True)(
        raw
    )
    np.testing.assert_array_equal(data.features[:, 0], [11, 3, 4, -1])
    np.testing.assert_array_equal(seg.features[:, 0], [1, 0, 4, -1])
    np.testing.assert_array_equal(data.coords, clust.coords)
    np.testing.assert_array_equal(data.coords, seg.coords)
    np.testing.assert_array_equal(clust.particles["particle"], [0, 1, 2])
    np.testing.assert_array_equal(clust.particles["interaction"], [0, 0, 0])
    np.testing.assert_array_equal(clust.particles["pid"], [2, 0, 2])
    assert (
        PIMMMetaParser("float32", voxel_size_mm=1, include_unlabeled=True)(raw)
        is data.meta
    )
    assert len(raw["_pimm_prepared"]) == 1
    particles = PIMMParticleParser("float32", voxel_size_mm=1, include_unlabeled=True)(
        raw
    )
    assert all(np.isnan(p.energy_deposit) for p in particles)
    assert particles[0].parent_track_id == -1
    assert particles[1].pid == 0
    assert particles[0].interaction_primary == 1
    np.testing.assert_array_equal(particles[0].children_id, [1, 2])
    np.testing.assert_array_equal(
        PIMMParticleGraphParser("float32")(raw).index, [[0, 0], [1, 2]]
    )
    identity = PIMMEventInfoParser("float32")(raw)
    assert identity.features.shape == (1, 2)
    assert 12345 in identity.features


def test_endpoints_and_missing_ancestry(raw):
    """Missing endpoints remain unknown; -1 never indexes the final row."""
    table = raw["pimm"]["particle"]
    table["first_pos"][2] = np.nan
    table["ancestor"][2] = table["parent"][2] = -1
    table["fragment_instance"][2] = -1
    coords = PIMMParticleCoordsParser("float32", voxel_size_mm=1)(raw)
    assert np.isnan(coords.coordinate_data[2]).all()
    particles = PIMMParticleParser("float32", voxel_size_mm=1)(raw)
    assert particles[2].ancestor_track_id == -1
    ppn = PIMMParticlePointsParser("float32", voxel_size_mm=1)(raw)
    assert len(ppn.coords) == 3
    assert ppn.features.shape[1] == 3
    assert set(coords.coordinate_groups) == {"start", "end"}


def test_empty_event(raw):
    """Empty point clouds and tables retain stable dimensions."""
    event = raw["pimm"]
    for key, value in event["hits"].items():
        if isinstance(value, np.ndarray):
            event["hits"][key] = value[:0]
    for key, value in event["particle"].items():
        event["particle"][key] = value[:0]
    for parser in (
        PIMMTensorParser,
        PIMMSegmentationParser,
        PIMMClusterLabelParser,
        PIMMParticleCoordsParser,
        PIMMParticlePointsParser,
    ):
        assert len(parser("float32")(raw).features) == 0
    assert len(PIMMParticleParser("float32")(raw)) == 0
    assert PIMMParticleGraphParser("float32")(raw).index.shape == (2, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(voxel_size_mm=0),
        dict(voxel_size_mm=np.nan),
        dict(origin_mm=(0, 0)),
        dict(origin_mm=(0, np.inf, 0)),
    ],
)
def test_invalid_grid(kwargs):
    """Reject grids that cannot define a detector-to-voxel transform."""
    with pytest.raises(ValueError):
        PIMMTensorParser("float32", **kwargs)


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("hits", "readout_type", "wire"),
        ("metadata", "pixel_geometry", None),
        ("metadata", "hit_x_from", "true_t0"),
        ("metadata", "hit_energy", "energy"),
        ("hits", "energy", np.array([[np.nan]] * 5)),
        ("hits", "instance_fragment", np.array([[99]] * 5)),
        ("hits", "segment_motif", np.array([[8]] * 5)),
        ("hits", "energy", np.array([[1.0]])),
    ],
)
def test_invalid_event(raw, part, key, value):
    """Reject unsupported geometry, corrupt references and misaligned arrays."""
    raw["pimm"][part][key] = value
    with pytest.raises(ValueError):
        PIMMTensorParser("float32")(raw)


def test_batch_and_truth_round_trip(raw, tmp_path):
    """Particle references shift per event and physical truth survives HDF5."""
    from spine.config.factory import instantiate
    from spine.construct import BuildManager
    from spine.io.collate import CollateAll
    from spine.io.dataset import PIMMDataset
    from spine.io.read import HDF5Reader
    from spine.io.unwrap import Unwrapper
    from spine.io.write import HDF5Writer

    from .conftest import FULL_SCHEMA

    events = []
    for index in range(2):
        trees = copy.deepcopy(raw)
        event = {
            key: instantiate(
                PIMMDataset.parser_dict,
                dict(cfg, voxel_size_mm=1) if key != "event_info" else cfg,
                dtype="float32",
            )(trees)
            for key, cfg in FULL_SCHEMA.items()
        }
        # Unlabelled hits do not contribute to truth objects.
        event["index"] = index
        events.append(event)
    batch = CollateAll(data_keys=events[0])(events)
    restored = Unwrapper()(batch)
    BuildManager(True, True, True, mode="truth", units="cm")(restored)
    assert len(restored["truth_particles"][0]) > 0
    assert restored["truth_particles"][0][0].units == "cm"
    path = tmp_path / "truth.h5"
    writer = HDF5Writer(
        file_name=str(path),
        keys=list(FULL_SCHEMA)
        + ["truth_fragments", "truth_particles", "truth_interactions"],
        format_version=2,
    )
    writer(restored, {})
    writer.finalize()
    writer.close()
    reloaded = HDF5Reader(file_keys=str(path))[0]
    assert len(reloaded["truth_particles"]) == len(restored["truth_particles"][0])


def test_unavailable_truth_and_ppn_rejection(raw):
    """Unknown vertices and unsupported or missing endpoints do not become targets."""
    event = raw["pimm"]
    event["interaction"] = {}
    table = event["particle"]
    table.pop("geant4_parent_trackid")
    table["instance_sem_type"][0] = 5
    table["first_pos"][1] = np.nan
    assert len(PIMMParticlePointsParser("float32")(raw).features) == 0
    assert all(
        np.isnan(p.ancestor_position).all() for p in PIMMParticleParser("float32")(raw)
    )


def test_opposite_drift_volumes(raw):
    """Different anode faces map identical ticks to opposite physical positions."""
    geometry = raw["pimm"]["metadata"]["pixel_geometry"]
    geometry.append(
        dict(geometry[0], x_anode_mm=10.0, drift_direction=1, reference_tick=1.0)
    )
    hits = raw["pimm"]["hits"]
    hits["planes"].append("volume_1_Pixel")
    hits["plane_id"][1] = 1
    positions = PIMMTensorParser._coordinates(hits, raw["pimm"]["metadata"])
    np.testing.assert_allclose(positions[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(positions[1], [9.5, 0.5, 0.5])


def test_unlabelled_voxels_excluded_for_training(raw):
    """Default training retains charge from all contributors in labelled voxels."""
    tensor = PIMMTensorParser("float32", voxel_size_mm=1)(raw)
    labels = PIMMSegmentationParser("float32", voxel_size_mm=1)(raw)
    np.testing.assert_array_equal(tensor.features[:, 0], [11, 3, 4])
    assert np.all(labels.features >= 0)
    raw["pimm"]["hits"]["segment_motif"][:] = -1
    raw.pop("_pimm_prepared")
    assert len(PIMMTensorParser("float32")(raw).features) == 0
