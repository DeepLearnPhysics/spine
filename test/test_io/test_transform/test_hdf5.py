"""Tests for direct V2 HDF5 transformations."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from spine.data import ObjectList, RecoParticle
from spine.io.read import HDF5Reader
from spine.io.transform import litify_hdf5
from spine.io.transform.hdf5 import _dataset_kwargs
from spine.io.write import HDF5Writer


def _write_source(path, format_version=2):
    particle = RecoParticle(
        id=3,
        index=np.asarray([1, 4, 8], dtype=np.int32),
        orig_index=np.asarray([2, 5, 9], dtype=np.int32),
        match_ids=np.asarray([7], dtype=np.int32),
        match_overlaps=np.asarray([0.75], dtype=np.float32),
        fragment_ids=np.asarray([2, 6], dtype=np.int32),
    )
    data = {
        "index": np.asarray([0, 1]),
        "particles": [
            ObjectList([particle], RecoParticle()),
            ObjectList([], RecoParticle()),
        ],
    }
    with HDF5Writer(
        str(path),
        overwrite=True,
        format_version=format_version,
    ) as writer:
        writer(data, cfg={})


def _pool_fields(group):
    return [
        tuple(yaml.safe_load(pool.attrs["fields"]))
        for pool in group["variables"].values()
    ]


def _assert_compound_equal(left, right):
    """Compare compound rows while treating corresponding NaNs as equal."""
    assert left.dtype == right.dtype
    for name in left.dtype.names:
        left_values = left[name]
        right_values = right[name]
        if np.issubdtype(left_values.dtype, np.floating):
            assert np.allclose(left_values, right_values, equal_nan=True)
        else:
            assert np.array_equal(left_values, right_values)


def test_structural_lite_matches_event_writer(tmp_path):
    """Structural lite output should match the established object policy."""
    source = tmp_path / "source.h5"
    structural = tmp_path / "structural.h5"
    event_lite = tmp_path / "event_lite.h5"
    _write_source(source)

    litify_hdf5(str(source), str(structural), keys=("particles",))

    reader = HDF5Reader(str(source))
    with HDF5Writer(
        str(event_lite),
        overwrite=True,
        format_version=2,
        lite=True,
        keys=("particles",),
    ) as writer:
        for entry in range(len(reader)):
            writer(reader.get(entry), cfg={})
    reader.close()

    with h5py.File(structural, "r") as direct, h5py.File(event_lite, "r") as rebuilt:
        direct_group = direct["products"]["particles"]
        rebuilt_group = rebuilt["products"]["particles"]
        assert direct_group["fixed"].dtype == rebuilt_group["fixed"].dtype
        _assert_compound_equal(
            direct_group["fixed"][:],
            rebuilt_group["fixed"][:],
        )
        assert np.array_equal(
            direct_group["event_offsets"][:],
            rebuilt_group["event_offsets"][:],
        )
        assert _pool_fields(direct_group) == _pool_fields(rebuilt_group)
        for direct_pool, rebuilt_pool in zip(
            direct_group["variables"].values(),
            rebuilt_group["variables"].values(),
        ):
            assert np.array_equal(
                direct_pool["values"][:],
                rebuilt_pool["values"][:],
            )

        fields = set().union(*_pool_fields(direct_group))
        assert "index" not in fields
        assert "orig_index" not in fields
        assert {"match_ids", "fragment_ids"}.issubset(fields)
        assert direct["info"].attrs["complete"]
        assert direct["info"].attrs["litified"]


def test_structural_fixed_only_removes_variable_pools(tmp_path):
    """Fixed-only mode should preserve object rows without helper columns."""
    source = tmp_path / "source.h5"
    target = tmp_path / "fixed.h5"
    _write_source(source)

    litify_hdf5(
        str(source),
        str(target),
        keys=("particles",),
        mode="fixed_only",
    )

    with h5py.File(target, "r") as out_file:
        group = out_file["products"]["particles"]
        assert len(group["variables"]) == 0
        assert not any(
            name.startswith("_var_offsets_") for name in group["fixed"].dtype.names
        )
        assert group["event_offsets"][:].tolist() == [0, 1, 1]


def test_structural_lite_validates_input_and_destination(tmp_path):
    """The transformer should reject V1, missing products and unsafe paths."""
    source = tmp_path / "source_v1.h5"
    target = tmp_path / "target.h5"
    _write_source(source, format_version=1)

    with pytest.raises(ValueError, match="version 2"):
        litify_hdf5(str(source), str(target), keys=("particles",))

    source_v2 = tmp_path / "source_v2.h5"
    _write_source(source_v2)
    with pytest.raises(KeyError, match="missing"):
        litify_hdf5(str(source_v2), str(target), keys=("absent",))
    with pytest.raises(ValueError, match="different"):
        litify_hdf5(str(source_v2), str(source_v2), keys=("particles",))

    litify_hdf5(str(source_v2), str(target), keys=("particles",))
    with pytest.raises(FileExistsError):
        litify_hdf5(str(source_v2), str(target), keys=("particles",))
    litify_hdf5(
        str(source_v2),
        str(target),
        keys=("particles",),
        overwrite=True,
    )


def test_structural_lite_validates_options_and_metadata(tmp_path):
    """Public options and required V2 metadata should fail clearly."""
    missing = tmp_path / "missing.h5"
    target = tmp_path / "target.h5"
    with pytest.raises(ValueError, match="mode"):
        litify_hdf5(str(missing), str(target), mode="unknown")
    with pytest.raises(ValueError, match="block_size"):
        litify_hdf5(str(missing), str(target), block_size=0)
    with pytest.raises(FileNotFoundError):
        litify_hdf5(str(missing), str(target))

    incomplete = tmp_path / "incomplete.h5"
    with h5py.File(incomplete, "w") as out_file:
        out_file.create_dataset("events", data=np.arange(1))
    with pytest.raises(ValueError, match="complete SPINE"):
        litify_hdf5(str(incomplete), str(target), keys=())

    no_products = tmp_path / "no_products.h5"
    with h5py.File(no_products, "w") as out_file:
        out_file.create_dataset("events", data=np.arange(1))
        info = out_file.create_group("info")
        info.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="logical-product group"):
        litify_hdf5(str(no_products), str(target), keys=())


def test_structural_lite_handles_encoded_metadata_and_source(tmp_path):
    """Byte metadata and top-level source provenance should be preserved."""
    source = tmp_path / "source.h5"
    target = tmp_path / "target.h5"
    _write_source(source)
    with h5py.File(source, "a") as out_file:
        group = out_file["products"]["particles"]
        group.attrs["class_name"] = np.bytes_("RecoParticle")
        for pool in group["variables"].values():
            fields = pool.attrs["fields"]
            del pool.attrs["fields"]
            pool.attrs["fields"] = np.bytes_(fields)
        provenance = out_file.create_group("source")
        provenance.attrs["file_name"] = "original.root"

    litify_hdf5(str(source), str(target), keys=("particles",))

    with h5py.File(target, "r") as out_file:
        assert out_file["source"].attrs["file_name"] == "original.root"


def test_structural_lite_preserves_product_owned_auxiliaries(tmp_path):
    """Unrecognized object children should be copied with their owner."""
    source = tmp_path / "source.h5"
    target = tmp_path / "target.h5"
    _write_source(source)
    with h5py.File(source, "a") as out_file:
        out_file["products"]["particles"].create_dataset(
            "diagnostic", data=np.asarray([3, 4], dtype=np.int64)
        )

    litify_hdf5(str(source), str(target), keys=("particles",))

    with h5py.File(target, "r") as out_file:
        np.testing.assert_array_equal(
            out_file["products"]["particles"]["diagnostic"][:], [3, 4]
        )


@pytest.mark.parametrize(
    ("field_value", "message"),
    [
        (4, "no string field metadata"),
        (yaml.safe_dump({"index": 1}), "string list"),
    ],
)
def test_structural_lite_rejects_invalid_pool_fields(tmp_path, field_value, message):
    """Malformed variable-pool field metadata should be rejected."""
    source = tmp_path / "source.h5"
    target = tmp_path / "target.h5"
    _write_source(source)
    with h5py.File(source, "a") as out_file:
        pool = next(iter(out_file["products"]["particles"]["variables"].values()))
        del pool.attrs["fields"]
        pool.attrs["fields"] = field_value

    with pytest.raises(TypeError, match=message):
        litify_hdf5(str(source), str(target), keys=("particles",))


def test_structural_lite_rejects_unknown_object_class(tmp_path):
    """Lite policy lookup requires a known stored SPINE object class."""
    source = tmp_path / "source.h5"
    target = tmp_path / "target.h5"
    _write_source(source)
    with h5py.File(source, "a") as out_file:
        out_file["products"]["particles"].attrs["class_name"] = "UnknownObject"

    with pytest.raises(ValueError, match="Cannot resolve"):
        litify_hdf5(str(source), str(target), keys=("particles",))


def test_dataset_creation_options_are_preserved(tmp_path):
    """Physical filters should transfer to structurally rewritten datasets."""
    path = tmp_path / "filters.h5"
    with h5py.File(path, "w") as out_file:
        filtered = out_file.create_dataset(
            "filtered",
            data=np.arange(10),
            chunks=(5,),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
            fletcher32=True,
        )
        scaled = out_file.create_dataset(
            "scaled",
            data=np.arange(10, dtype=np.float32),
            chunks=(5,),
            scaleoffset=2,
        )
        filtered_kwargs = _dataset_kwargs(filtered)
        scaled_kwargs = _dataset_kwargs(scaled)

    assert filtered_kwargs == {
        "chunks": (5,),
        "compression": "gzip",
        "compression_opts": 2,
        "shuffle": True,
        "fletcher32": True,
    }
    assert scaled_kwargs == {"chunks": (5,), "scaleoffset": 2}


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("class_name", "no class name"),
        ("fixed", "no fixed dataset"),
        ("event_offsets", "no event offsets"),
        ("variables", "no variable group"),
        ("pool", "not a group"),
        ("values", "no values"),
        ("helper", "missing helper"),
    ],
)
def test_structural_lite_rejects_corrupt_object_layout(tmp_path, corruption, message):
    """Malformed object hierarchy components should fail before publication."""
    source = tmp_path / f"source_{corruption}.h5"
    target = tmp_path / f"target_{corruption}.h5"
    _write_source(source)
    with h5py.File(source, "a") as out_file:
        group = out_file["products"]["particles"]
        if corruption == "class_name":
            del group.attrs["class_name"]
            group.attrs["class_name"] = 4
        elif corruption in ("fixed", "event_offsets"):
            del group[corruption]
            group.create_group(corruption)
        elif corruption == "variables":
            del group["variables"]
            group.create_dataset("variables", data=np.arange(1))
        elif corruption == "pool":
            del group["variables"]["pool_0"]
            group["variables"].create_dataset("pool_0", data=np.arange(1))
        elif corruption == "values":
            pool = group["variables"]["pool_0"]
            del pool["values"]
            pool.create_group("values")
        else:
            fixed = group["fixed"]
            source_rows = fixed[:]
            names = [name for name in fixed.dtype.names if name != "_var_offsets_0"]
            dtype = np.dtype([(name, fixed.dtype.fields[name][0]) for name in names])
            replacement = np.empty(len(source_rows), dtype=dtype)
            for name in names:
                replacement[name] = source_rows[name]
            del group["fixed"]
            group.create_dataset("fixed", data=replacement)

    with pytest.raises((TypeError, KeyError), match=message):
        litify_hdf5(str(source), str(target), keys=("particles",))
    assert not target.exists()
