"""Downstream truth contracts, provenance and physical-coordinate round trips."""

import copy

import numpy as np

from spine.construct import BuildManager
from spine.io.collate import CollateAll
from spine.io.dataset import SuperaDataset
from spine.io.parse.supera import (
    SuperaClusterLabelParser,
    SuperaParticleCoordsParser,
    SuperaParticleGraphParser,
    SuperaParticleParser,
)
from spine.io.read import HDF5Reader, SuperaReader
from spine.io.unwrap import Unwrapper
from spine.io.write import HDF5Writer

from .conftest import FULL_SCHEMA


def test_local_ids_coordinates_and_provenance(sample):
    """Sparse source IDs cannot leak into row-indexed supervision, even in batches."""
    ds = SuperaDataset(file_keys=sample, schema=FULL_SCHEMA, dtype="float32")
    events = [ds[i] for i in (0, 2, 1)]
    batch = CollateAll(data_keys=ds.data_keys)(events)
    unwrapped = Unwrapper()(batch)
    for i, event in enumerate(events):
        count = len(event["particles"])
        np.testing.assert_array_equal(
            unwrapped["clust_label"][i].particle_field("particle"), np.arange(count)
        )
        assert unwrapped["coord_label"][i].coordinate_data.shape == (count, 6)
        np.testing.assert_allclose(unwrapped["meta"][i].size, 0.3)
        np.testing.assert_array_equal(
            unwrapped["source_ids"][i].features, event["source_ids"].features
        )
        assert unwrapped["event_info"][i].features[0, 0] == [11, 13, 12][i]
        if count:
            coords = event["coord_label"].coordinate_data
            np.testing.assert_allclose(coords[0], [0.5, 0.5, 0.5, 1.5, 0.5, 0.5])
            np.testing.assert_allclose(coords[1, :3], coords[1, 3:])
            assert event["source_ids"].features.max() > count
            for row, particle in enumerate(event["particles"]):
                labels = event["clust_label"].particles
                assert particle.pid == labels["pid"][row]
                assert particle.group_primary == labels["group_primary"][row]
                assert (
                    particle.interaction_primary == labels["interaction_primary"][row]
                )


def test_missing_endpoints_and_ancestry(sample):
    """Missing links cannot wrap to the last row, and deposited endpoints are usable."""
    raw = SuperaReader(file_keys=sample)[0]
    table = raw["particles"]
    table["parent_id"][0] = -1
    table["ancestor_id"][0] = -1
    table["first_pos"][:2] = np.nan
    table["last_pos"][0] = np.inf
    table["first_pos"][2] = np.nan
    table["last_pos"][2] = np.nan
    # An unobserved extra row has neither endpoints nor a recoverable fragment.
    for key, values in table.items():
        table[key] = np.concatenate((values, values[-1:]))
    table["id"][-1] = 100000
    for key in ("parent_id", "ancestor_id", "frag_id", "inst_id", "frag_inst_id"):
        table[key][-1] = -1
    for key in ("first_pos", "last_pos"):
        table[key][-1] = np.nan
    table["pc_start"][-1] = table["pc_end"][-1] = -1
    coords = SuperaParticleCoordsParser("float32")(raw)
    assert np.isfinite(coords.coordinate_data[:5]).all()
    assert np.isnan(coords.coordinate_data[-1]).all()
    # Raw point times are microseconds; coordinate-label times are nanoseconds.
    assert coords.features[0, 0] == 2000
    np.testing.assert_allclose(coords.coordinate_data[0, :3], [1.5, 0.5, 0.5])
    labels = SuperaClusterLabelParser("float32")(raw).particles
    assert labels["interaction_primary"][0] == -1
    assert np.isnan(labels["vertex"][0]).all()
    particles = SuperaParticleParser("float32")(raw)
    assert particles[0].ancestor_track_id == -1
    assert np.isnan(particles[0].parent_position).all()
    assert particles[-1].energy_deposit == 0
    assert SuperaParticleGraphParser("float32")(raw).index.shape == (2, 0)


def test_interaction_vertex_and_genealogy(sample):
    """Stored interaction vertices override differing roots, with local graph links."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["inter"].update(x=np.array([30.0]), y=np.array([60.0]), z=np.array([90.0]))
    raw["particles"]["parent_id"][3] = 10
    raw["particles"]["ancestor_id"][3] = 10
    raw["particles"]["geant4_parent_trackid"] = np.arange(5) + 200
    raw["particles"]["geant4_parent_pdg"] = np.full(5, 211)
    labels = SuperaClusterLabelParser("float32")(raw).particles
    np.testing.assert_allclose(labels["vertex"], np.tile([10, 20, 30], (5, 1)))
    particles = SuperaParticleParser("float32")(raw)
    assert particles[3].interaction_primary == 0
    assert particles[3].ancestor_track_id == 10
    assert particles[3].parent_track_id == 203
    assert particles[3].parent_pdg_code == 211
    np.testing.assert_array_equal(particles[0].children_id, [3])
    np.testing.assert_array_equal(
        SuperaParticleGraphParser("float32")(raw).index, [[0], [3]]
    )
    # A nonfinite stored vertex falls back to valid ancestry.
    raw.pop("_supera_prepared")
    raw["inter"]["x"][:] = np.nan
    labels = SuperaClusterLabelParser("float32")(raw).particles
    np.testing.assert_allclose(labels["vertex"][0], [0.5, 0.5, 0.5])


def check_truth_round_trip(events, path):
    """Build all truth levels in cm and retain their associations through HDF5.

    Parameters
    ----------
    events : list[dict]
        Parsed events with the complete native schema and source provenance.
    path : pathlib.Path
        Temporary output HDF5 file.

    Returns
    -------
    dict
        Unwrapped truth products used for physical validation by callers.
    """
    batch = CollateAll(data_keys=events[0])(events)
    truth = Unwrapper()(batch)
    original = copy.deepcopy(truth)
    BuildManager(True, True, True, mode="truth", units="cm")(truth)
    keys = [
        "meta",
        "event_info",
        "source_ids",
        "points_label",
        "depositions_label",
        "truth_fragments",
        "truth_particles",
        "truth_interactions",
    ]
    writer = HDF5Writer(file_name=str(path), keys=keys, format_version=2)
    writer(truth, {})
    writer.finalize()
    writer.close()
    reader = HDF5Reader(file_keys=str(path))
    for index, event in enumerate(events):
        restored = reader[index]
        BuildManager(True, True, True, mode="truth", units="cm")(restored)
        # BuildManager converts parsed truth to cm; voxel centers use +0.5.
        expected = event["meta"].to_cm(
            original["clust_label"][index].coords, center=True
        )
        np.testing.assert_allclose(restored["points_label"], expected)
        assert np.isclose(
            restored["depositions_label"].sum(), event["data"].features.sum()
        )
        for key in ("truth_fragments", "truth_particles", "truth_interactions"):
            assert len(restored[key]) == len(truth[key][index])
            for before, after in zip(truth[key][index], restored[key]):
                assert after.units == "cm"
                assert before.id == after.id
                np.testing.assert_allclose(before.points, after.points)
                for field in (
                    "parent_id",
                    "children_id",
                    "orig_id",
                    "pid",
                    "is_primary",
                    "vertex",
                    "energy_deposit",
                    "fragment_ids",
                    "particle_ids",
                ):
                    if hasattr(before, field):
                        np.testing.assert_allclose(
                            getattr(before, field), getattr(after, field)
                        )
        for particle in restored["truth_particles"]:
            source = event["particles"][particle.orig_id]
            assert particle.pid == source.pid
            assert particle.is_primary == (source.interaction_primary > 0)
        for particle in truth["truth_particles"][index]:
            for fragment in particle.fragments:
                assert fragment.particle_id == particle.id
        for interaction in truth["truth_interactions"][index]:
            for particle in interaction.particles:
                assert particle.interaction_id == interaction.id
    return truth


def test_truth_round_trip(sample, tmp_path):
    """Sparse IDs, multiple events and empty truth survive the full conversion path."""
    import h5py

    with h5py.File(sample, "r+") as file:
        for row in (3, 8):
            file["particles/parent_id"][row] = 10
            file["particles/ancestor_id"][row] = 10
    ds = SuperaDataset(file_keys=sample, schema=FULL_SCHEMA, dtype="float32")
    truth = check_truth_round_trip([ds[i] for i in (0, 2, 1)], tmp_path / "truth.h5")
    # Source energy is per stored particle; grouping must count each deposit once.
    assert truth["truth_particles"][0][0].energy_deposit == 9
    np.testing.assert_array_equal(truth["truth_particles"][0][0].children_id, [1])
    assert truth["truth_particles"][0][1].parent_id == 0
    assert len(truth["truth_particles"][1]) == 0


def downstream_batch(sample):
    """Create two nodes per nonempty event, separated by an empty event.

    Parameters
    ----------
    sample : str
        Synthetic Supera file path.

    Returns
    -------
    tuple[dict, IndexBatch]
        Collated labels and fragment nodes with event-local particle references.
    """
    from spine.data import IndexBatch

    ds = SuperaDataset(file_keys=sample, schema=FULL_SCHEMA, dtype="float32")
    batch = CollateAll(data_keys=ds.data_keys)([ds[0], ds[2], ds[1]])
    clusts = IndexBatch(
        [np.array([0, 1]), np.array([2]), np.array([5, 6]), np.array([7])],
        counts=[2, 0, 2],
        spans=[5, 0, 5],
        single_counts=[2, 1, 2, 1],
    )
    return batch, clusts


def test_closest_primary_batch(sample):
    """Closest-primary selection must use local table rows, not sparse source IDs."""
    from spine.cluster.label import (
        get_cluster_closest_label_batch,
        get_cluster_closest_primary_label_batch,
    )
    from spine.data import TensorBatch

    batch, clusts = downstream_batch(sample)
    primary = TensorBatch(np.ones(4, dtype=int), counts=[2, 0, 2])
    for select in (
        get_cluster_closest_label_batch,
        get_cluster_closest_primary_label_batch,
    ):
        args = (batch["clust_label"], batch["coord_label"], clusts, primary)
        selected = (
            select(*args, default=[0, 0])
            if select == get_cluster_closest_label_batch
            else select(*args)
        )
        np.testing.assert_array_equal(selected.numpy_tensor(), [1, 0, 1, 0])
    # A fragment's PID/interaction-primary describe its host, not its own electron.
    assert (
        batch["clust_label"].particle_field("pid")[0][2]
        == batch["clust_label"].particle_field("pid")[0][0]
    )


def test_orientation_batch(sample):
    """Orientation targets follow particle-table offsets across empty batch entries."""
    import pytest

    torch = pytest.importorskip("torch")
    from spine.data import TensorBatch
    from spine.model.grappa.loss.node_orient import NodeOrientLoss

    batch, clusts = downstream_batch(sample)
    starts = TensorBatch(
        torch.tensor(
            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]
        ),
        [2, 0, 2],
    )
    ends = TensorBatch(
        torch.tensor(
            [[1.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.5, 0.5, 0.5], [2.5, 0.5, 0.5]]
        ),
        [2, 0, 2],
    )
    target = NodeOrientLoss().materialize_target(
        clust_label=batch["clust_label"].to_tensor(),
        coord_label=batch["coord_label"].to_tensor(),
        clusts=clusts,
        start_points=starts,
        end_points=ends,
    )
    np.testing.assert_array_equal(target["target"].numpy_tensor(), [1, -1, 0, -1])
    np.testing.assert_array_equal(
        target["valid"].numpy_tensor(), [True, False, True, False]
    )


def test_absorbed_fragment_uses_host_interaction(sample):
    """A deposit absorbed from another interaction follows its host's training labels."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["particles"]["interaction_id"][2] = 1
    raw["inter"].update(
        interaction_id=np.array([0, 1]),
        x=np.array([30.0, 300.0]),
        y=np.array([60.0, 600.0]),
        z=np.array([90.0, 900.0]),
    )
    labels = SuperaClusterLabelParser("float32")(raw).particles
    assert labels["interaction"][2] == 0
    assert labels["pid"][2] == labels["pid"][0]
    np.testing.assert_allclose(labels["vertex"][2], [10, 20, 30])
    particle = SuperaParticleParser("float32")(raw)[2]
    assert particle.interaction_id == 0
    np.testing.assert_allclose(particle.ancestor_position, [10, 20, 30])


def test_example_schema_is_complete(sample):
    """The documented configuration provides all products without parser overrides."""
    from pathlib import Path
    from textwrap import dedent

    import yaml

    # Exercise the published schema directly, without a standalone config file.
    doc_path = Path(__file__).resolve().parents[3] / "docs/source/api/io.rst"
    section = doc_path.read_text().split("Native pysupera input\n", 1)[1]
    example = section.split(".. code-block:: yaml\n\n", 1)[1].split("\n\n", 1)[0]
    config = yaml.safe_load(dedent(example))
    assert config["io"]["loader"]["minibatch_size"] == 1
    assert "batch_size" not in config["io"]["loader"]
    schema = config["io"]["loader"]["dataset"]["schema"]
    assert schema == FULL_SCHEMA
    event = SuperaDataset(file_keys=sample, schema=schema, dtype="float32")[0]
    BuildManager(True, True, True, mode="truth", units="cm")(event)
    assert len(event["truth_particles"]) == 3
