"""Exercise the real pimm-data backend, including worker serialization."""

import json
import pickle

import h5py
import numpy as np
import pytest

from spine.io.dataset import PIMMDataset
from spine.io.read import PIMMReader

from .conftest import FULL_SCHEMA


@pytest.fixture
def corpus(tmp_path, raw):
    """Use upstream's synthetic writer instead of mocking its label joins."""
    pytest.importorskip("pimm_data")
    from pimm_data.testing import make_jaxtpc_sample

    root = str(tmp_path / "corpus")
    make_jaxtpc_sample(root, readout_type="pixel", n_events=2, n_files=2)
    for path in (tmp_path / "corpus" / "labl").glob("*.h5"):
        with h5py.File(path, "r+") as file:
            meta = raw["pimm"]["metadata"]
            file["config"].attrs["supera_pixel_geometry"] = json.dumps(
                meta["pixel_geometry"] * 2
            )
            file["config"].attrs["supera_hit_x_from"] = "nominal"
            file["config"].attrs["supera_hit_energy"] = "charge"
    return root


def test_reader_selection_pickle_and_labels(corpus):
    """Selection crosses files without changing event or particle identity."""
    reader = PIMMReader(data_root=corpus, dataset_name="sim", entry_list=[3, 0, 2])
    first = reader[0]
    assert first["file_index"] == 1
    assert first["file_entry_index"] == 1
    assert first["source_file_entry_index"] == 1
    assert (
        first["pimm"]["identity"]["event_id"]
        != reader[1]["pimm"]["identity"]["event_id"]
    )
    restored = pickle.loads(pickle.dumps(reader))
    np.testing.assert_array_equal(
        first["pimm"]["hits"]["energy"], restored[0]["pimm"]["hits"]["energy"]
    )
    reader.close()
    restored.close()


def test_dataset_factories_and_grid(corpus):
    """The public pimm provider parses the combined event through common machinery."""
    from spine.io.factories import dataset_factory, reader_factory

    cfg = dict(provider="pimm", data_root=corpus, dataset_name="sim")
    reader = reader_factory(cfg)
    assert isinstance(reader, PIMMReader)
    reader.close()
    ds = dataset_factory(dict(cfg, schema=FULL_SCHEMA), dtype="float32")
    event = ds[0]
    assert len(event["data"].features)
    np.testing.assert_array_equal(event["data"].coords, event["seg_label"].coords)
    assert np.isnan(event["particles"][0].energy_deposit)
    assert set(ds[0]) == set(ds.__getitems__([0])[0])
    ds.reader.close()
    with pytest.raises(ValueError, match="same voxel"):
        PIMMDataset(
            data_root=corpus,
            schema={
                "data": dict(provider="tensor", voxel_size_mm=1),
                "seg_label": dict(provider="segmentation", voxel_size_mm=3),
            },
            dtype="float32",
        )


def test_invalid_projection():
    """Reject legacy raw products rather than silently reading truth deposits."""
    with pytest.raises(ValueError, match="aligned"):
        PIMMReader(keys=["points"])


def test_missing_dependency(monkeypatch):
    """Base SPINE imports stay lightweight and missing extras fail clearly."""
    from spine.io.read import pimm

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(pimm, "import_module", missing)
    with pytest.raises(ImportError, match=r"spine\[pimm\]"):
        PIMMReader()


def test_missing_optional_metadata(corpus, monkeypatch, tmp_path):
    """Absent event identity or geometry is not replaced with fabricated values."""
    reader = PIMMReader(data_root=corpus, dataset_name="sim")
    monkeypatch.setattr(reader.dataset.hits_reader, "event_id_of", lambda _: None)
    assert set(reader[0]["pimm"]["identity"]) == {"entry"}
    reader.close()
    path = tmp_path / "metadata.h5"
    with h5py.File(path, "w") as file:
        file.create_group("config")
    assert PIMMReader._read_metadata(str(path))["pixel_geometry"] is None


def test_plane_local_groups(corpus):
    """The final-layout per-plane group namespace is resolved by pimm-data."""
    from pathlib import Path

    from pimm_data._shard_meta import clear_cache

    before = PIMMReader(data_root=corpus, dataset_name="sim")
    expected = before[0]["pimm"]["hits"]
    before.close()
    hits_path = next((Path(corpus) / "hits").glob("*.h5"))
    labl_path = Path(corpus) / "labl" / hits_path.name.replace("_hits_", "_labl_")
    with h5py.File(hits_path, "r+") as hits, h5py.File(labl_path, "r+") as labels:
        hits.attrs["group_id_space"] = "plane"
        for key in (k for k in hits if k.startswith("entry_")):
            for volume in (k for k in hits[key] if k.startswith("volume_")):
                plane = hits[f"{key}/{volume}/Pixel"]
                old = plane["group_ids"][:]
                # Reverse local numbering to make an accidental direct join wrong.
                plane["group_ids"][:] = np.arange(len(old))[::-1]
                target = labels[f"{key}/{volume}"].require_group("plane_Pixel")
                target["group"] = old[::-1]
    clear_cache()
    after = PIMMReader(data_root=corpus, dataset_name="sim")
    actual = after[0]["pimm"]["hits"]
    for name in (
        "coord",
        "energy",
        "instance_fragment",
        "instance_particle",
        "segment_motif",
    ):
        np.testing.assert_array_equal(actual[name], expected[name])
    after.close()


def test_runs_and_explicit_files(corpus, tmp_path):
    """Task discovery and explicit modality paths select the same events."""
    from pathlib import Path

    root = Path(corpus)
    files = {key: str(root / key / "*.h5") for key in ("hits", "labl")}
    explicit = PIMMReader(files=files, dataset_name="sim", split="")
    identity = explicit[0]["pimm"]["identity"]
    explicit.close()
    task = tmp_path / "runs" / "task"
    task.parent.mkdir()
    root.rename(task)
    runs = PIMMReader(data_root=str(task.parent), runs=["task"], dataset_name="sim")
    assert runs[0]["pimm"]["identity"] == identity
    runs.close()
