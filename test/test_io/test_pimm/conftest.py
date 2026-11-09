"""Small native PIMM events used to check SPINE adaptation."""

import numpy as np
import pytest

FULL_SCHEMA = {
    key: {"provider": provider}
    for key, provider in (
        ("data", "tensor"),
        ("seg_label", "segmentation"),
        ("clust_label", "cluster_label"),
        ("particles", "particle"),
        ("coord_label", "particle_coords"),
        ("ppn_label", "particle_points"),
        ("meta", "meta"),
        ("graph_label", "particle_graph"),
        ("event_info", "event_info"),
    )
}


@pytest.fixture
def raw():
    """Pixel charge contributions with overlap, LE, an unknown hit and ancestry."""
    table = dict(
        parent=np.array([0, 0, 0]),
        ancestor=np.array([0, 0, 0]),
        fragment=np.arange(3),
        instance=np.array([0, 1, -1]),
        fragment_instance=np.array([0, 1, 0]),
        interaction=np.array([0, 0, 1]),
        pdg=np.array([13, 22, 11]),
        parent_pdg=np.array([13, 13, 13]),
        geant4_trackid=np.array([10, 80, 150]),
        geant4_parent_trackid=np.array([-1, 10, 10]),
        geant4_parent_pdg=np.array([0, 13, 13]),
        motif=np.array([1, 0, 4]),
        instance_sem_type=np.array([2, 1, -1]),
        is_primary=np.array([1, 0, 0]),
    )
    for key in ("start_pos", "end_pos", "first_pos", "last_pos"):
        table[key] = np.array(
            [[0.5, 0.5, 0.5, 10.0], [1.5, 0.5, 0.5, 20.0], [2.5, 0.5, 0.5, 30.0]]
        )
    table["last_pos"][0, 0] = 2.5
    for key in ("start_mom", "end_mom"):
        table[key] = np.tile([0.0, 0.0, 100.0, 150.0], (3, 1))
    geometry = dict(
        x_anode_mm=0.0,
        drift_direction=-1,
        reference_tick=0.0,
        mm_per_tick=1.0,
        y_min_mm=0.0,
        z_min_mm=0.0,
        pitch_mm=1.0,
    )
    return {
        "pimm": dict(
            hits=dict(
                coord=np.array(
                    [
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 1.5],
                        [0.0, 0.0, 2.5],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 3.5],
                    ]
                ),
                energy=np.array([[2.0], [3.0], [4.0], [9.0], [-1.0]]),
                plane_id=np.zeros((5, 1), int),
                planes=["volume_0_Pixel"],
                readout_type="pixel",
                segment_motif=np.array([[1], [0], [4], [0], [-1]]),
                instance_fragment=np.array([[0], [1], [2], [1], [-1]]),
                instance_particle=np.array([[0], [1], [0], [1], [-1]]),
            ),
            particle=table,
            interaction={"vertex": np.array([[10.0, 20.0, 30.0], [99.0, 99.0, 99.0]])},
            metadata=dict(
                pixel_geometry=[geometry], hit_x_from="nominal", hit_energy="charge"
            ),
            identity={"event_id": 12345, "entry": 0},
        )
    }


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
