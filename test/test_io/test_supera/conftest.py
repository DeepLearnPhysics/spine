"""Shared native Supera fixtures and complete schema."""

import h5py
import numpy as np
import pytest

FULL_SCHEMA = {
    key: {"provider": provider}
    for key, provider in (
        ("data", "tensor"),
        ("seg_label", "segmentation"),
        ("clust_label", "cluster_label"),
        ("ppn_label", "particle_points"),
        ("particles", "particle"),
        ("coord_label", "particle_coords"),
        ("meta", "meta"),
        ("graph_label", "particle_graph"),
        ("source_ids", "source_ids"),
        ("event_info", "event_info"),
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
        pc_start=np.array([0, 3, 2, 4, 5]),
        pc_end=np.array([2, 4, 3, 5, 6]),
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


def round_trip_cache(batch, products, path, stage):
    """Publish native products and reload them through cache collation.

    Parameters
    ----------
    batch : dict
        Collated input batch containing physical-source provenance.
    products : dict
        Batched products to cache, without source metadata.
    path : pathlib.Path
        Temporary cache repository.
    stage : str
        Cache stage name.

    Returns
    -------
    dict
        Re-collated products from the published cache stage.
    """
    from spine.io.collate import CollateAll
    from spine.io.read import CacheReader
    from spine.io.unwrap import Unwrapper
    from spine.io.write import CacheWriter

    metadata = {
        key: value
        for key, value in batch.items()
        if key == "index" or key.startswith("source_file_")
    }
    payload = {
        key: value.to_numpy() if hasattr(value, "to_numpy") else value
        for key, value in products.items()
    }
    writer = CacheWriter(path=str(path), stage=stage, keys=list(products))
    writer(Unwrapper()({**metadata, **payload}), {})
    writer.finalize()
    writer.close()
    reader = CacheReader(path=str(path), stage=stage)
    events = reader.get_many(list(range(len(batch["index"]))))
    reader.close()
    return CollateAll(data_keys=products)(events)
