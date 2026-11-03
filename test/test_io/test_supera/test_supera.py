"""Native pysupera reader and aligned supervision contracts."""

import copy
import pickle

import h5py
import numpy as np
import pytest

from spine.io.collate import CollateAll
from spine.io.dataset import SuperaDataset
from spine.io.parse.supera import SuperaTensorParser
from spine.io.read import SuperaReader

SCHEMA = {
    key: {"provider": provider}
    for key, provider in (
        ("data", "tensor"),
        ("seg_label", "segmentation"),
        ("clust_label", "cluster_label"),
        ("ppn_label", "particle_points"),
        ("particles", "particle"),
    )
}


@pytest.fixture
def sample(tmp_path):
    """Two events with sparse IDs, overlap, LE in a track and empty truth."""
    path = tmp_path / "supera.h5"
    ids = np.array([10, 80, 99, 120, 150], dtype=np.int32)
    # Track, shower, LE fragment of track, delta, Michel.
    table = dict(
        id=ids,
        parent_id=ids,
        ancestor_id=ids,
        frag_id=ids,
        inst_id=np.array([10, 80, -1, 120, 150]),
        frag_inst_id=np.array([10, 80, 10, 120, 150]),
        sem_type=np.array([2, 1, 5, 3, 4]),
        frag_sem_type=np.array([2, 1, 5, 3, 4]),
        inst_sem_type=np.array([2, 1, -1, 3, 4]),
        frag_pc_start=np.array([0, 3, 2, 4, 5]),
        frag_pc_end=np.array([2, 4, 3, 5, 6]),
        inst_pc_start=np.array([0, 3, -1, 4, 5]),
        inst_pc_le_start=np.array([2, 4, -1, 5, 6]),
        inst_pc_end=np.array([3, 4, -1, 5, 6]),
        pdg=np.array([13, 11, 11, 11, 11]),
        parent_pdg=np.array([13] * 5),
        geant4_trackid=ids,
        interaction_id=np.zeros(5, dtype=int),
        mass=np.ones(5),
    )
    for key in (
        "start_pos",
        "end_pos",
        "first_pos",
        "last_pos",
        "start_mom",
        "end_mom",
    ):
        table[key] = np.tile([1.5, 1.5, 1.5, 10.0], (5, 1))
    table["last_pos"][0, :3] = [4.5, 1.5, 1.5]
    table["first_pos"][3, :3] = [10.5, 1.5, 1.5]
    table["first_pos"][4, :3] = [13.5, 1.5, 1.5]
    table["last_pos"][1] = np.nan
    points = np.array(
        [
            [1.5, 1.5, 1.5, 5, 2, 0],
            [4.5, 1.5, 1.5, 2, 3, 0],
            [7.5, 1.5, 1.5, 1, 4, 0],
            [1.5, 1.5, 1.5, 0, 9, 0],
            [10.5, 1.5, 1.5, 1, 5, 0],
            [13.5, 1.5, 1.5, 1, 6, 0],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as file:
        file["format_version"] = "4.1.0"
        file["n_events"] = 3
        file.attrs["voxel_size"] = "3.0"
        file.attrs["hit_energy"] = "charge"
        file["events/offsets"] = [0, 5, 10, 10]
        file["events/event_id"] = [11, 12, 13]
        file["points/offsets"] = [0, 6, 12, 12]
        file["points/flat"] = np.concatenate([points, points])
        file["points/columns"] = np.array(
            ["x", "y", "z", "t", "dE", "dX"], dtype=h5py.string_dtype()
        )
        for key, val in table.items():
            second = np.where(val >= 0, val + 6, val) if "pc_" in key else val
            file["particles/" + key] = np.concatenate([val, second])
        file["inter/offsets"] = [0, 1, 2, 2]
        file["inter/interaction_id"] = [0, 0]
        file["inter/part_start"] = [0, 5]
        file["inter/part_end"] = [5, 10]
        file["inter/pc_start"] = [0, 6]
        file["inter/pc_end"] = [6, 12]
        file["groups/offsets"] = [0, 9, 18, 18]
        file["groups/fragment_id"] = [10] * 18
        file["groups/is_le"] = [False] * 18
        file["groups/volume_offsets_index"] = [0, 2, 4, 6]
        file["groups/volume_offsets"] = [0, 4, 0, 4, 0, 0]
        # A huge virtual dataset: a projected read must never access it.
        hit_plane = file.create_group("hit_labels/volume0/plane0")
        hit_plane["offsets"] = [0, 2, 4, 4]
        hit_plane.create_dataset(
            "fragment_id", shape=(10**9,), dtype="i4", fillvalue=10
        )
    return str(path)


def dataset(path, **kwargs):
    """Build the complete parser schema for one synthetic input file."""
    return SuperaDataset(file_keys=path, dtype="float32", schema=SCHEMA, **kwargs)


def test_reader_projection_rebasing_multifile_and_pickle(sample, monkeypatch):
    """Preserve projection, event selection and range bases across worker copies."""
    original = h5py.Dataset.__getitem__

    def guarded(data_source, key):
        """Fail if default training reads touch group or hit-label payloads."""
        assert not data_source.name.startswith("/hit_labels/")
        assert not data_source.name.startswith("/groups/")
        return original(data_source, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    reader = SuperaReader(file_keys=[sample, sample], entry_list=[4, 0, 2])
    first, second, empty = pickle.loads(pickle.dumps(reader)).get_many([0, 1, 2])
    assert first["file_index"] == 1 and first["file_entry_index"] == 1
    np.testing.assert_array_equal(first["points"], second["points"])
    np.testing.assert_array_equal(
        first["particles"]["inst_pc_start"], second["particles"]["inst_pc_start"]
    )
    assert first["inter"]["part_start"][0] == 0
    assert empty["points"].shape == (0, 6)
    assert "groups" not in first and "hit_labels" not in first


def test_explicit_group_and_hit_slices(sample):
    """Read only the selected event from independent group and hit domains."""
    reader = SuperaReader(
        file_keys=sample, keys=["groups", "hit_labels/volume0/plane0"]
    )
    event = reader[1]
    assert len(event["groups"]["fragment_id"]) == 9
    assert len(event["hit_labels/volume0/plane0"]["fragment_id"]) == 2
    np.testing.assert_array_equal(event["groups"]["volume_offsets"], [0, 4])


def test_alignment_energy_semantics_and_endpoints(sample):
    """Keep energy, semantic, fragment and endpoint supervision aligned."""
    event = dataset(sample)[1]
    np.testing.assert_array_equal(event["data"].features[:, 0], [11, 3, 4, 5, 6])
    np.testing.assert_array_equal(event["seg_label"].features[:, 0], [1, 1, 4, 3, 2])
    np.testing.assert_array_equal(event["clust_label"].cluster_ids, [0, 0, 2, 3, 4])
    np.testing.assert_array_equal(event["clust_label"].group_ids, [0, 0, 0, 3, 4])
    np.testing.assert_array_equal(
        event["clust_label"].shapes, event["seg_label"].features[:, 0]
    )
    np.testing.assert_array_equal(
        event["data"].coordinate_data, event["seg_label"].coordinate_data
    )
    np.testing.assert_array_equal(
        event["data"].coordinate_data, event["clust_label"].coords
    )
    np.testing.assert_allclose(event["data"].meta.size, 0.3)
    ppn = event["ppn_label"]
    assert len(ppn.features) == 4  # Track start/end, delta start, Michel start.
    np.testing.assert_array_equal(ppn.features[:, 0], [1, 1, 3, 2])
    np.testing.assert_array_equal(ppn.features[:, 2], [0, 1, 0, 0])
    assert np.isfinite(ppn.coordinate_data).all()


def test_empty_event_and_collation(sample):
    """Collate empty events without changing event-local particle references."""
    data_source = dataset(sample)
    empty = data_source[2]
    assert empty["data"].features.shape == (0, 1)
    assert empty["ppn_label"].features.shape == (0, 3)
    batch = CollateAll(data_keys=SCHEMA)([data_source[0], empty, data_source[1]])
    np.testing.assert_array_equal(batch["data"].counts, [5, 0, 5])
    np.testing.assert_array_equal(
        batch["clust_label"].shapes.tensor, batch["seg_label"].features.tensor[:, 0]
    )
    assert batch["clust_label"].particle_indexes.tensor[-1] == 4
    assert batch["ppn_label"].features.tensor[-1, 1] == 4


def test_invalid_version_offsets_ranges_and_grid(sample):
    """Reject incompatible files, corrupt ranges and inconsistent parser grids."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["particles"]["frag_pc_end"][0] = 4
    with pytest.raises(ValueError, match="overlapping"):
        SuperaTensorParser("float32")(raw)
    schema = copy.deepcopy(SCHEMA)
    schema["data"]["voxel_size_mm"] = 6
    with pytest.raises(ValueError, match="same voxel"):
        SuperaDataset(file_keys=sample, dtype="float32", schema=schema)
    with h5py.File(sample, "r+") as file:
        file["points/offsets"][1] = 100
    with pytest.raises(TypeError, match="offsets"):
        SuperaReader(file_keys=sample)
    with h5py.File(sample, "r+") as file:
        file["format_version"][()] = "3.0.0"
    with pytest.raises(TypeError, match="4.1.0"):
        SuperaReader(file_keys=sample)


def test_winner_uses_time_then_source_order(sample):
    """Resolve equal-class overlaps deterministically without losing energy."""
    raw = SuperaReader(file_keys=sample)[0]
    # Make the shower a second track in the same cell. Earlier time wins.
    raw["particles"]["inst_sem_type"][1] = 2
    from spine.io.parse.supera import SuperaClusterLabelParser

    parser = SuperaClusterLabelParser("float32")
    event = parser(raw)
    assert event.cluster_ids[0] == 1
    assert event.values[0] == 11
    raw.pop("_supera_prepared")
    raw["points"][3, 3] = raw["points"][0, 3]
    assert parser(raw).cluster_ids[0] == 0


def test_unknown_and_dangling_relationships_rejected(sample):
    """Reject unsupported semantics and particle links with no target row."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["particles"]["inst_sem_type"][0] = 6
    with pytest.raises(ValueError, match="Unknown Supera semantic"):
        SuperaTensorParser("float32")(raw)
    raw = SuperaReader(file_keys=sample)[0]
    raw["particles"]["frag_inst_id"][0] = 999
    with pytest.raises(ValueError, match="Dangling"):
        SuperaTensorParser("float32")(raw)


def test_grid_transform_and_particle_energy(sample):
    """Preserve detector transforms and distinguish source from merged energy."""
    schema = copy.deepcopy(SCHEMA)
    for parser in schema.values():
        parser["voxel_size_mm"] = [3, 3, 3]
        parser["origin_mm"] = [9, -12, 3]
    event = SuperaDataset(file_keys=sample, dtype="float32", schema=schema)[0]
    np.testing.assert_allclose(event["ppn_label"].coordinate_data[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(event["data"].meta.lower, [0, 0, 0])
    # The winning track sees 11 MeV in the overlapped input cell, but its
    # source-instance truth includes only its own 2 + 3 + 4 MeV contributions.
    assert event["particles"][0].energy_deposit == 9


def test_registered_factories(sample):
    """Register native Supera readers and datasets under the same provider."""
    from spine.io.factories import dataset_factory, reader_factory

    assert isinstance(
        reader_factory({"provider": "supera", "file_keys": sample}), SuperaReader
    )
    data_source = dataset_factory(
        {"provider": "supera", "file_keys": sample, "schema": SCHEMA}, dtype="float32"
    )
    assert isinstance(data_source, SuperaDataset)
    assert len(data_source) == 3


@pytest.mark.parametrize("origin", [(0, 0), (0, np.nan, 0)])
def test_reject_invalid_origin(origin):
    """Reject malformed or nonfinite grid origins before reading events."""
    with pytest.raises(ValueError, match="origin_mm"):
        SuperaTensorParser("float32", origin_mm=origin)


@pytest.mark.parametrize("size", [0, -1, np.inf])
def test_reject_invalid_explicit_voxel_size(size):
    """Reject invalid explicit grid dimensions at parser construction."""
    with pytest.raises(ValueError, match="voxel_size_mm"):
        SuperaTensorParser("float32", voxel_size_mm=size)


@pytest.mark.parametrize("size", [None, "0", "[3, -1, 3]", np.nan])
def test_reject_invalid_file_voxel_size(sample, size):
    """Validate missing, encoded and numeric voxel-size attributes."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["metadata"]["voxel_size"] = size
    with pytest.raises(ValueError, match="voxel_size|voxel size"):
        SuperaTensorParser("float32")(raw)


@pytest.mark.parametrize(
    ("column", "row", "value", "message"),
    [
        ("id", 1, 10, "unique"),
        ("id", 1, -1, "nonnegative"),
        ("inst_pc_le_start", 0, 4, "LE range"),
        ("frag_pc_end", 0, 1, "cover every point"),
        ("frag_inst_id", 0, 80, "ownership"),
        ("frag_sem_type", 2, 2, "semantics"),
    ],
)
def test_reject_inconsistent_particle_table(sample, column, row, value, message):
    """Reject corrupt identity, range and cross-level association fields."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["particles"][column][row] = value
    with pytest.raises(ValueError, match=message):
        SuperaTensorParser("float32")(raw)


@pytest.mark.parametrize("column", [0, 3, 4])
def test_reject_nonfinite_point_values(sample, column):
    """Reject nonfinite coordinates, times and deposited energy."""
    raw = SuperaReader(file_keys=sample)[0]
    raw["points"][0, column] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        SuperaTensorParser("float32")(raw)


def test_empty_representative_retains_particle_row(sample):
    """Keep truth rows for a representative whose point ranges are absent."""
    from spine.io.parse.supera import SuperaParticleParser

    raw = SuperaReader(file_keys=sample)[0]
    raw["points"] = raw["points"][:5]
    for key in (
        "frag_pc_start",
        "frag_pc_end",
        "inst_pc_start",
        "inst_pc_end",
        "inst_pc_le_start",
    ):
        raw["particles"][key][-1] = -1
    particles = SuperaParticleParser("float32")(raw)
    assert len(particles) == 5
    assert particles[-1].num_voxels == 0
    assert particles[-1].energy_deposit == 0
    assert particles[-1].pdg_code == 11


@pytest.mark.parametrize("kind", ["le", "nucleus", "nan", "below", "above"])
def test_ppn_omits_ineligible_endpoints(sample, kind):
    """Omit LE, nuclear, nonfinite and out-of-image instance targets."""
    from spine.io.parse.supera import SuperaParticlePointsParser

    raw = SuperaReader(file_keys=sample)[0]
    table = raw["particles"]
    if kind == "le":
        table["inst_sem_type"][3] = 5
        table["frag_sem_type"][3] = 5
        table["inst_pc_le_start"][3] = table["inst_pc_start"][3]
    elif kind == "nucleus":
        table["pdg"][3] = 1000020040
    else:
        table["first_pos"][3, 0] = {"nan": np.nan, "below": -1, "above": 100}[kind]
    points = SuperaParticlePointsParser("float32")(raw)
    assert len(points.features) == 3
    assert 3 not in points.features[:, 1]


def test_reader_without_compression_plugin(sample, monkeypatch):
    """Uncompressed input remains usable without the optional filter package."""
    from spine.io.read import supera

    def missing_plugin(name):
        """Simulate an absent optional dependency without changing other imports."""
        assert name == "hdf5plugin"
        raise ImportError(name)

    monkeypatch.setattr(supera, "import_module", missing_plugin)
    assert len(SuperaReader(file_keys=sample)[0]["points"]) == 6


@pytest.mark.parametrize("key", ["unknown", "hit_labels", "hit_labels/volume0/plane9"])
def test_reject_unknown_projection(sample, key):
    """Require valid raw product names and explicit existing hit planes."""
    with pytest.raises(ValueError, match="Unsupported Supera product"):
        SuperaReader(file_keys=sample, keys=[key])


def test_reject_invalid_point_columns(sample):
    """Validate declared point columns before any parser can misinterpret them."""
    with h5py.File(sample, "r+") as file:
        file["points/columns"][0] = "wrong"
    with pytest.raises(ValueError, match="unsupported point columns"):
        SuperaReader(file_keys=sample)


def test_reject_group_in_place_of_dataset(sample):
    """Reject malformed HDF5 structure with a clear dataset diagnostic."""
    with h5py.File(sample, "r+") as file:
        del file["points/columns"]
        file.create_group("points/columns")
    with pytest.raises(TypeError, match="Expected a dataset"):
        SuperaReader(file_keys=sample)


def test_raw_dataset_projection(sample):
    """Expose raw selected products when no parser schema is supplied."""
    data = SuperaDataset(file_keys=sample, keys=["events"])[1]
    assert data["events"]["event_id"] == 12
    assert "points" not in data
    data = SuperaDataset(file_keys=sample)[0]
    assert set(SuperaReader.default_keys).issubset(data)
