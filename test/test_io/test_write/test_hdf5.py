"""Test that the writer classes work as intended."""

import os

import h5py
import numpy as np
import pytest
import yaml

from spine.data import (
    ClusterLabelData,
    CRTHit,
    EdgeIndexData,
    Flash,
    FlashHypothesis,
    IndexData,
    IndexListData,
    Meta,
    Neutrino,
    ObjectList,
    ObjectListData,
    Particle,
    ParticleLabel,
    RecoParticle,
    RunInfo,
    TensorData,
    TensorSchema,
    Trigger,
)
from spine.io.collate import CollateAll
from spine.io.parse import HDF5ClusterLabelParser
from spine.io.read import HDF5Reader, StageHDF5Reader
from spine.io.write import *
from spine.io.write.hdf5.common import (
    DataFormat,
    decode_string_attribute,
    require_dataset,
    require_group,
)


@pytest.fixture(name="hdf5_output")
def fixture_hdf5_output(tmp_path):
    """Create a dummy output path for an HDF5 file.

    Parameters
    ----------
    tmp_path : str
       Generic pytest fixture used to handle temporary test files
    """
    return os.path.join(tmp_path, "dummy.h5")


def stage_product_values(stage, key):
    """Return the physical values dataset for one staged V2 product."""
    return stage["products"][key]["values"][:]


@pytest.fixture(name="tensor_list")
def fixture_tensor_list(request):
    """Generates a dummy list of unwrapped tensors."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of tensors of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    tensors = []
    for i, s in enumerate(sizes):
        tensors.append(np.random.rand(s, 5))

    return tensors


@pytest.fixture(name="index_list")
def fixture_index_list(request):
    """Generates a dummy list of unwrapped index lists."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    indexes = []
    default = np.empty(0, dtype=np.int64)
    for i, s in enumerate(sizes):
        index = np.arange(s)
        if s > 1:
            index = np.split(index, [np.random.randint(1, s)])
            indexes.append(index)
        elif s == 1:
            indexes.append([index])
        else:
            indexes.append(ObjectList(index, default))

    return indexes


@pytest.fixture(name="edge_index_list")
def fixture_edge_index_list(request):
    """Generates a dummy list of unwrapped edge indexes."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    edge_indexes = []
    for i, s in enumerate(sizes):
        edge_indexes.append(np.random.rand(s, 2))

    return edge_indexes


@pytest.mark.parametrize(
    "tensor_list, index_list, edge_index_list",
    [(0, 0, 0), (10, 10, 25), ((0, 0), (0, 0), (0, 0)), ((5, 10), (5, 10), (8, 25))],
    indirect=True,
)
def test_hdf5_writer(hdf5_output, tensor_list, index_list, edge_index_list):
    """Tests the HDF5 writer."""
    # Create an output similar to that of the full chain
    batch_size = len(tensor_list)
    sizes = [len(t) for t in tensor_list]
    data = {
        "index": np.arange(batch_size),
        "dummy_data": tensor_list,
        "dummy_meta": [Meta()] * batch_size,
        "dummy_run_info": [RunInfo()] * batch_size,
        "dummy_trigger": [Trigger()] * batch_size,
        "dummy_particles": generate_object_list(Particle, sizes),
        "dummy_neutrinos": generate_object_list(Neutrino, sizes),
        "dummy_flashes": generate_object_list(Flash, sizes),
        "dummy_flash_hypotheses": generate_object_list(FlashHypothesis, sizes),
        "dummy_crthits": generate_object_list(CRTHit, sizes),
        "dummy_tensor": tensor_list,
        "dummy_clusts": index_list,
        "dummy_edge_index": edge_index_list,
    }

    # Initialize the writer
    writer = HDF5Writer(hdf5_output)

    # Write output
    writer(data)


@pytest.mark.parametrize("format_version", [1, 2])
def test_hdf5_writer_round_trips_cluster_labels(hdf5_output, format_version):
    """Both HDF5 layouts should preserve compact labels and particle tables."""
    particle = ParticleLabel(particle=4, group=4, pid=2, shape=1)
    fields = {
        name: np.asarray([value])
        for name, value in particle.as_dict(include_derived=False).items()
    }
    label = ClusterLabelData(
        np.asarray([[1, 2, 3, 5.0, 7, 0]], dtype=np.float32),
        fields,
        Meta(),
    )
    with HDF5Writer(
        hdf5_output,
        format_version=format_version,
    ) as writer:
        writer({"index": np.asarray([0]), "clust_label": [label]}, cfg={})

    reader = HDF5Reader(hdf5_output)
    entry = reader.get(0)
    reader.close()
    if format_version == 2:
        with h5py.File(hdf5_output, "r") as in_file:
            assert set(in_file) == {"events", "info", "products"}
            product = in_file["products"]["clust_label"]
            assert "particles" in product
            assert "meta" in product
            assert "clust_label_particles" not in in_file["products"]
            assert "clust_label_meta" not in in_file["products"]

        restored = entry["clust_label"]
        assert isinstance(restored, ClusterLabelData)
        assert isinstance(restored.meta, Meta)
        assert "clust_label_particles" not in entry
        assert "clust_label_meta" not in entry
    else:
        parser = HDF5ClusterLabelParser(
            dtype="float32",
            cluster_label_event="clust_label",
            particle_event="clust_label_particles",
        )
        restored = parser(
            {
                "clust_label": entry["clust_label"],
                "clust_label_particles": entry["clust_label_particles"],
            }
        )

    np.testing.assert_array_equal(restored.coords, [[1, 2, 3]])
    np.testing.assert_array_equal(restored.features, [[5, 7, 0]])
    np.testing.assert_array_equal(restored.particles["particle"], [4])
    np.testing.assert_array_equal(restored.particles["pid"], [2])


def test_hdf5_v2_round_trips_product_metadata(hdf5_output):
    """V2 should restore tensor schemas and index spans without parsers."""
    tensor = TensorData(
        coords=np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
        features=np.asarray([[7.0, 8.0]], dtype=np.float32),
        meta=Meta(),
        coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
        feature_fields={"time": (0,), "shape": (1,)},
    )
    index = IndexData(np.asarray([1, 3], dtype=np.int64), span=5)
    index_list = IndexListData([np.asarray([0, 2]), np.asarray([1])], span=4)
    edge_index = EdgeIndexData(
        np.asarray([[0, 1], [1, 2]], dtype=np.int64), span=3, directed=False
    )
    objects = ObjectListData([Particle(id=3)], Particle(), index_shifts={"id": 4})
    scalar_shift_objects = ObjectListData([Particle(id=4)], Particle(), index_shifts=6)

    with HDF5Writer(hdf5_output, format_version=2) as writer:
        writer(
            {
                "index": np.asarray([0]),
                "coordinates": [tensor],
                "selection": [index],
                "groups": [index_list],
                "edges": [edge_index],
                "particles": [objects],
                "scalar_shift_particles": [scalar_shift_objects],
            },
            cfg={},
        )

    with h5py.File(hdf5_output, "r") as in_file:
        products = in_file["products"]
        assert "product_metadata" in products["coordinates"].attrs
        assert "product_metadata" in products["selection"].attrs
        assert "product_metadata" in products["particles"].attrs
        assert "meta" in products["coordinates"]
        assert "spans" in products["selection"]
        assert "index_shifts" in products["particles"]

    reader = HDF5Reader(hdf5_output)
    entry = reader.get(0)
    reader.close()

    restored = entry["coordinates"]
    assert isinstance(restored, TensorData)
    assert isinstance(restored.meta, Meta)
    assert restored.coordinate_groups == {
        "start": (0, 1, 2),
        "end": (3, 4, 5),
    }
    np.testing.assert_array_equal(restored.coordinates("end"), [[4, 5, 6]])
    np.testing.assert_array_equal(restored.feature("shape"), [[8.0]])

    restored_index = entry["selection"]
    assert isinstance(restored_index, IndexData)
    assert restored_index.span == 5
    np.testing.assert_array_equal(restored_index.features, [1, 3])
    assert "selection_spans" not in entry
    assert "coordinates_meta" not in entry

    restored_groups = entry["groups"]
    assert isinstance(restored_groups, IndexListData)
    assert restored_groups.span == 4
    np.testing.assert_array_equal(restored_groups.features[0], [0, 2])

    restored_edges = entry["edges"]
    assert isinstance(restored_edges, EdgeIndexData)
    assert restored_edges.span == 3
    assert restored_edges.directed is False
    np.testing.assert_array_equal(restored_edges.features, [[0, 1], [1, 2]])

    restored_objects = entry["particles"]
    assert isinstance(restored_objects, ObjectListData)
    assert restored_objects.index_shifts == {"id": 4}
    assert restored_objects[0].id == 3
    assert "particles_index_shifts" not in entry

    assert entry["scalar_shift_particles"].index_shifts == 6


def test_hdf5_writer_prepares_batched_v2_products(hdf5_output):
    """V2 lowering should accept every supported batched product type."""
    events = []
    for i in range(2):
        events.append(
            {
                "tensor": TensorData(
                    coords=np.asarray([[i, i, i]]),
                    features=np.asarray([[float(i)]]),
                ),
                "index_data": IndexData(np.asarray([0]), span=1),
                "edge_data": EdgeIndexData(
                    np.asarray([[0], [0]]), span=1, directed=True
                ),
                "label": ClusterLabelData(
                    coords=np.asarray([[i, i, i]]),
                    features=np.asarray([[1.0, 0.0]]),
                ),
            }
        )
    batched = CollateAll(data_keys=tuple(events[0]))(events)
    writer = HDF5Writer(hdf5_output, format_version=2)

    prepared = writer.prepare_products(batched)

    assert writer.product_metadata["tensor"]["product_type"] == "tensor"
    assert writer.product_metadata["index_data"]["product_type"] == "index"
    assert writer.product_metadata["edge_data"]["product_type"] == "edge_index"
    assert writer.product_metadata["label"]["product_type"] == "cluster_label"
    assert prepared["edge_data"][0].shape == (1, 2)
    writer.close()


def test_hdf5_writer_expands_batched_cluster_labels_for_v1(hdf5_output):
    """Legacy lowering should support batched labels and selected sidecars."""
    particle = ParticleLabel(particle=0, group=0)
    fields = {
        name: np.asarray([value])
        for name, value in particle.as_dict(include_derived=False).items()
    }
    events = [
        {
            "label": ClusterLabelData(
                np.asarray([[0, 0, 0, 1.0, 0, 0]], dtype=np.float32), fields
            )
        }
    ]
    batch = CollateAll(data_keys=("label",))(events)["label"]
    writer = HDF5Writer(hdf5_output, format_version=1, keys=("label",))

    expanded = writer.expand_cluster_labels({"label": batch})

    assert "label_particles" in expanded
    assert "label_particles" in writer.keys

    voxel_only = CollateAll(data_keys=("label",))(
        [
            {
                "label": ClusterLabelData(
                    coords=np.asarray([[0, 0, 0]]),
                    features=np.asarray([[1.0, 0.0]]),
                )
            }
        ]
    )["label"]
    assert "label_particles" not in writer.expand_cluster_labels({"label": voxel_only})
    writer.close()


def test_hdf5_writer_rejects_inconsistent_v2_products(hdf5_output):
    """V2 schema discovery should reject inconsistent event products."""
    writer = HDF5Writer(hdf5_output, format_version=2)
    tensor = TensorData(
        features=np.asarray([[1.0]]),
        feats_only=True,
        schema=TensorSchema(feature_fields={"value": (0,)}, feats_only=True),
    )
    other_schema = TensorData(
        features=np.asarray([[1.0]]),
        feats_only=True,
        schema=TensorSchema(feature_fields={"other": (0,)}, feats_only=True),
    )
    tensor_with_meta = TensorData(
        features=np.asarray([[1.0]]), feats_only=True, meta=Meta()
    )

    with pytest.raises(TypeError, match="mixes event data classes"):
        writer.prepare_products({"mixed": [tensor, IndexData([0], span=1)]})
    with pytest.raises(TypeError, match="is not a `IndexData` collection"):
        writer._typed_entries([tensor], IndexData, "tensor")
    with pytest.raises(ValueError, match="Tensor schemas differ"):
        writer.prepare_products({"tensor": [tensor, other_schema]})
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        writer.prepare_products({"tensor": [tensor, tensor_with_meta]})

    label = ClusterLabelData(
        coords=np.asarray([[0, 0, 0]]),
        features=np.asarray([[1.0, 0.0]]),
    )
    label_with_particles = ClusterLabelData(
        coords=np.asarray([[1, 1, 1]]),
        features=np.asarray([[1.0, 0.0, 0.0]]),
        particles={"particle": np.asarray([0])},
    )
    label_with_meta = ClusterLabelData(
        coords=np.asarray([[1, 1, 1]]),
        features=np.asarray([[1.0, 0.0]]),
        meta=Meta(),
    )
    with pytest.raises(ValueError, match="particle tables are inconsistent"):
        writer.prepare_products({"label": [label, label_with_particles]})
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        writer.prepare_products({"label": [label, label_with_meta]})

    object_a = ObjectListData([Particle(id=1)], Particle(), index_shifts={"id": 1})
    object_b = ObjectListData(
        [Particle(id=2)], Particle(), index_shifts={"group_id": 1}
    )
    with pytest.raises(ValueError, match="index shifts differ"):
        writer.prepare_products({"objects": [object_a, object_b]})

    scalar_object = ObjectListData([Particle(id=2)], Particle(), index_shifts=1)
    with pytest.raises(ValueError, match="index shifts differ"):
        writer.prepare_products({"objects": [scalar_object, object_a]})
    writer.close()


def test_hdf5_writer_v2_internal_schema_guards(hdf5_output):
    """Internal V2 child registration should reject collisions and drift."""
    writer = HDF5Writer(hdf5_output, format_version=2, keys=("tensor",))
    internal_key = "__spine_v2_aux__tensor__meta"
    with pytest.raises(KeyError, match="conflicts with an internal"):
        writer._add_product_child({internal_key: object()}, "tensor", "meta", [Meta()])

    prepared = {}
    writer._add_product_child(prepared, "tensor", "meta", [Meta()])
    assert internal_key in writer.keys

    tensor = TensorData(features=np.asarray([[1.0]]), feats_only=True)
    writer.ready = False
    writer.prepare_products({"tensor": [tensor]})
    writer.ready = True
    with pytest.raises(ValueError, match="schemas changed"):
        writer.prepare_products(
            {"tensor": [TensorData(features=np.asarray([[1.0, 2.0]]), feats_only=True)]}
        )

    with pytest.raises(ValueError, match="Particle information"):
        writer._serialize_particle_tables(
            [
                ClusterLabelData(
                    coords=np.asarray([[0, 0, 0]]),
                    features=np.asarray([[1.0, 0.0]]),
                )
            ]
        )
    writer.close()


def test_hdf5_writer_v2_skips_configured_products(hdf5_output):
    """V2 preparation should honor explicit skip-key projection."""
    writer = HDF5Writer(hdf5_output, format_version=2, skip_keys=("skipped",))
    skipped = TensorData(features=np.asarray([[1.0]]), feats_only=True)

    prepared = writer.prepare_products({"skipped": [skipped]})

    assert prepared["skipped"] == [skipped]
    assert "skipped" not in writer.product_metadata
    writer.close()


def test_hdf5_writer_v2_honors_explicit_product_keys(hdf5_output):
    """V2 preparation should ignore products outside an explicit projection."""
    writer = HDF5Writer(hdf5_output, format_version=2, keys=("stored",))
    omitted = TensorData(features=np.asarray([[1.0]]), feats_only=True)

    prepared = writer.prepare_products({"omitted": [omitted]})

    assert prepared["omitted"] == [omitted]
    assert "omitted" not in writer.product_metadata
    writer.close()


def test_hdf5_writer_v2_round_trips_fixed_heterogeneous_lists(hdf5_output):
    """V2 should store each position of a fixed heterogeneous list separately."""
    jagged = [
        [
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        ]
    ]

    with HDF5Writer(hdf5_output, format_version=2) as writer:
        writer({"index": np.asarray([0]), "jagged": jagged}, cfg={})

    with h5py.File(hdf5_output, "r") as in_file:
        group = in_file["products"]["jagged"]
        assert group.attrs["kind"] == "multi_list"
        np.testing.assert_array_equal(group["element_0"]["values"], jagged[0][0])
        np.testing.assert_array_equal(group["element_1"]["values"], jagged[0][1])


def test_hdf5_writer_validates_v2_append_schema_components(hdf5_output):
    """Appending should validate the logical root, metadata, and child groups."""
    writer = HDF5Writer(hdf5_output, format_version=2)
    writer.product_metadata = {"tensor": {"product_type": "tensor"}}
    writer.product_children = {"aux": ("tensor", "meta")}

    with h5py.File(hdf5_output, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["format_version"] = 2
        with pytest.raises(ValueError, match="missing V2 products"):
            writer._validate_append_format(out_file, hdf5_output)

        products = out_file.create_group("products")
        with pytest.raises(ValueError, match="metadata is missing"):
            writer._validate_append_format(out_file, hdf5_output)

        tensor_group = products.create_group("tensor")
        with pytest.raises(ValueError, match="metadata is missing"):
            writer._validate_append_format(out_file, hdf5_output)

        tensor_group.attrs["product_metadata"] = yaml.safe_dump(
            {"product_type": "other"}
        )
        with pytest.raises(ValueError, match="schemas differ"):
            writer._validate_append_format(out_file, hdf5_output)

        tensor_group.attrs["product_metadata"] = yaml.safe_dump(
            writer.product_metadata["tensor"]
        )
        with pytest.raises(ValueError, match="stored child `meta` is missing"):
            writer._validate_append_format(out_file, hdf5_output)

        writer.product_children = {"aux": ("absent", "meta")}
        with pytest.raises(
            ValueError, match="product `absent`.*child `meta` is missing"
        ):
            writer._validate_append_format(out_file, hdf5_output)
    writer.close()


def test_hdf5_writer_common_hdf5_validation_helpers(hdf5_output):
    """Shared HDF5 helpers should validate child and attribute types."""
    with h5py.File(hdf5_output, "w") as out_file:
        out_file.create_group("group")
        out_file.create_dataset("dataset", data=np.asarray([1]))

        assert require_group(out_file, "group").name.endswith("group")
        assert require_dataset(out_file, "dataset").name.endswith("dataset")
        with pytest.raises(TypeError, match="HDF5 group"):
            require_group(out_file, "dataset")
        with pytest.raises(TypeError, match="HDF5 dataset"):
            require_dataset(out_file, "group")

    assert decode_string_attribute(b"value", "attribute") == "value"
    with pytest.raises(TypeError, match="must be a string"):
        decode_string_attribute(1, "attribute")


def test_hdf5_writer_region_backend_requires_initialized_state(hdf5_output):
    """Region-reference appends should fail clearly without initialized state."""
    writer = HDF5Writer(hdf5_output, format_version=1)
    with h5py.File(hdf5_output, "w") as out_file:
        writer.event_dtype = np.dtype(np.int64)
        writer.keys = None
        with pytest.raises(RuntimeError, match="Keys to be stored"):
            writer.append_region_entry(out_file, {}, 0)

        writer.type_dict = None
        with pytest.raises(RuntimeError, match="data formats are not initialized"):
            writer.append_region_key(out_file, np.empty(1), {}, "value", 0)

        writer.type_dict = {
            "value": DataFormat(dtype=int, width=0, merge=False, scalar=True)
        }
        writer.object_dtypes = [int]
        with pytest.raises(TypeError, match="Object dtype.*must be compound"):
            writer.append_region_key(out_file, np.empty(1), {"value": 1}, "value", 0)
    writer.close()


def test_hdf5_writer_file_name_inferred_from_prefix():
    """Test HDF5 output file name inference from input prefixes."""
    assert HDF5Writer.get_file_names(None, "input", split=False) == ["input_spine.h5"]
    assert HDF5Writer.get_file_names(None, ["a", "b"], split=True) == [
        "a_spine.h5",
        "b_spine.h5",
    ]


def test_hdf5_writer_file_name_inferred_from_prefix_with_directory(tmp_path):
    """A writer directory should relocate inferred names without changing stems."""
    output_dir = tmp_path / "outputs"
    assert HDF5Writer.get_file_names(
        None, "input.root", split=False, directory=str(output_dir), suffix="cache"
    ) == [os.path.join(output_dir, "input_cache.h5")]
    assert HDF5Writer.get_file_names(
        None,
        ["a.root", "b.root"],
        split=True,
        directory=str(output_dir),
        suffix="cache",
    ) == [
        os.path.join(output_dir, "a_cache.h5"),
        os.path.join(output_dir, "b_cache.h5"),
    ]


def test_hdf5_writer_explicit_file_name_with_directory(tmp_path):
    """A writer directory should relocate explicit output names too."""
    output_dir = tmp_path / "outputs"
    file_name = os.path.join(tmp_path, "custom.h5")
    assert HDF5Writer.get_file_names(
        file_name, split=False, directory=str(output_dir)
    ) == [os.path.join(output_dir, "custom.h5")]


def test_hdf5_writer_creates_missing_output_directory(tmp_path):
    """HDF5Writer should create a configured output directory on first write."""
    output_dir = tmp_path / "missing" / "outputs"
    writer = HDF5Writer(
        file_name="custom.h5",
        directory=str(output_dir),
        overwrite=True,
    )
    writer(
        {
            "index": np.asarray([0]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        }
    )
    writer.close()

    assert (output_dir / "custom.h5").is_file()


def test_hdf5_writer_split_explicit_single_file(hdf5_output):
    """Test split output keeps an explicit name when there is one input file."""
    assert HDF5Writer.get_file_names(hdf5_output, ["input"], split=True) == [
        hdf5_output
    ]


def test_hdf5_writer_split_explicit_multiple_files(tmp_path):
    """Test split output enumerates an explicit name for multiple input files."""
    file_name = os.path.join(tmp_path, "output.h5")

    assert HDF5Writer.get_file_names(file_name, ["a", "b", "c"], split=True) == [
        os.path.join(tmp_path, "output_0.h5"),
        os.path.join(tmp_path, "output_1.h5"),
        os.path.join(tmp_path, "output_2.h5"),
    ]


def test_hdf5_writer_split_explicit_multiple_files_with_directory(tmp_path):
    """Split explicit names should relocate into the requested directory."""
    file_name = os.path.join(tmp_path, "output.h5")
    output_dir = tmp_path / "outputs"

    assert HDF5Writer.get_file_names(
        file_name, ["a", "b", "c"], split=True, directory=str(output_dir)
    ) == [
        os.path.join(output_dir, "output_0.h5"),
        os.path.join(output_dir, "output_1.h5"),
        os.path.join(output_dir, "output_2.h5"),
    ]


def test_hdf5_writer_append_existing_file(hdf5_output):
    """Test appending a batch to an existing HDF5 output file."""
    data = {
        "index": np.arange(2),
        "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
    }

    HDF5Writer(hdf5_output)(data)
    HDF5Writer(hdf5_output, append=True)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        assert len(out_file["events"]) == 4


def test_hdf5_writer_append_missing_file(hdf5_output):
    """Test append mode creates a missing HDF5 output file."""
    data = {
        "index": np.arange(2),
        "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
    }

    HDF5Writer(hdf5_output, append=True)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        assert len(out_file["events"]) == 2


def test_hdf5_writer_auto_stores_source_entry_provenance(hdf5_output):
    """Writer should persist original source entry provenance automatically."""
    writer = HDF5Writer(
        hdf5_output,
        overwrite=True,
        keys=["dummy_data"],
    )
    writer(
        {
            "index": np.asarray([0, 1]),
            "file_index": np.asarray([3, 3]),
            "file_entry_index": np.asarray([11, 12]),
            "dummy_data": [np.random.rand(2, 3), np.random.rand(3, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "source_file_index" in out_file
        assert "source_file_entry_index" in out_file
        np.testing.assert_array_equal(out_file["source_file_index"][:], [3, 3])
        np.testing.assert_array_equal(out_file["source_file_entry_index"][:], [11, 12])


def test_hdf5_writer_skip_keys_can_disable_source_provenance(hdf5_output):
    """Explicit skip keys should still be able to drop source provenance."""
    writer = HDF5Writer(
        hdf5_output,
        overwrite=True,
        skip_keys=["source_file_index", "source_file_entry_index"],
    )
    writer(
        {
            "index": np.asarray([0]),
            "file_index": np.asarray([2]),
            "file_entry_index": np.asarray([9]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "source_file_index" not in out_file
        assert "source_file_entry_index" not in out_file


def test_hdf5_writer_finalize_marks_output_complete(hdf5_output):
    """Finalize should mark written files as complete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_close_leaves_unfinalized_output_incomplete(hdf5_output):
    """Closing alone should not mark files as complete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "dummy_data": [np.random.rand(2, 3)],
        }
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_stage_hdf5_writer_finalizes_stages_independently(hdf5_output):
    """Each cache stage should track completeness independently."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        assert not out_file["stages"]["graph_spice"]["info"].attrs["complete"]
        assert len(out_file["stages"]["deghosting"]["events"]) == 1
        assert len(out_file["stages"]["graph_spice"]["events"]) == 1


def test_stage_hdf5_v2_round_trips_typed_products(hdf5_output):
    """Stage caches should preserve V2 product metadata and sidecars."""
    tensor = TensorData(
        coords=np.asarray([[1, 2, 3]], dtype=np.int32),
        features=np.asarray([[4.0, 5.0]], dtype=np.float32),
        meta=Meta(),
        coordinate_groups={"position": (0, 1, 2)},
        feature_fields={"value": (0,), "shape": (1,)},
    )
    selection = IndexData(np.asarray([1, 3], dtype=np.int64), span=5)
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "coordinates": [tensor],
            "selection": [selection],
        },
    )
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([1]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "coordinates": [tensor],
            "selection": [selection],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(hdf5_output, "r") as in_file:
        assert in_file["info"].attrs["format"] == "stage_hdf5"
        assert in_file["info"].attrs["format_version"] == 2
        stage = in_file["stages"]["deghosting"]
        assert set(stage) == {"events", "info", "products"}
        assert stage["events"].dtype == np.dtype(np.int64)
        assert len(stage["events"]) == 2
        assert "meta" in stage["products"]["coordinates"]
        assert "spans" in stage["products"]["selection"]

    reader = StageHDF5Reader("deghosting", hdf5_output)
    entry = reader.get(0)
    reader.close()

    restored_tensor = entry["coordinates"]
    assert isinstance(restored_tensor, TensorData)
    assert isinstance(restored_tensor.meta, Meta)
    assert restored_tensor.coordinate_groups == {"position": (0, 1, 2)}
    np.testing.assert_array_equal(restored_tensor.feature("shape"), [[5.0]])
    restored_selection = entry["selection"]
    assert isinstance(restored_selection, IndexData)
    assert restored_selection.span == 5
    np.testing.assert_array_equal(restored_selection.features, [1, 3])


def test_stage_hdf5_writer_rejects_legacy_cache(tmp_path):
    """Legacy staged caches should be rebuilt rather than appended in place."""
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        info.attrs["format"] = "stage_hdf5"
        info.attrs["format_version"] = 1
        out_file.create_group("stages")

    writer = StageHDF5Writer(str(path))
    with pytest.raises(ValueError, match="format version 1.*rebuild"):
        writer.write_stage(
            "deghosting",
            {
                "index": np.asarray([0]),
                "source_file_name": np.asarray(["source.root"]),
                "source_file_size": np.asarray([10]),
                "source_file_mtime_ns": np.asarray([20]),
                "dummy_data": [np.asarray([[1.0, 2.0]])],
            },
        )
    writer.close()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "missing info group"),
        (
            {"format": "hdf5", "format_version": 2},
            "expected format 'stage_hdf5'",
        ),
    ],
)
def test_stage_hdf5_writer_rejects_invalid_container(tmp_path, metadata, message):
    """Existing outputs must identify themselves as staged V2 caches."""
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as out_file:
        if metadata:
            info = out_file.create_group("info")
            for key, value in metadata.items():
                info.attrs[key] = value
        out_file.create_group("stages")

    writer = StageHDF5Writer(str(path))
    with pytest.raises(ValueError, match=message):
        writer._ensure_stage_file(str(path))
    writer.close()


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("events", "missing its V2 event axis"),
        ("products", "different product schema"),
        ("metadata", "missing V2 metadata"),
        ("schema", "different schema"),
        ("child", "missing child 'meta'"),
    ],
)
def test_stage_hdf5_writer_validates_active_v2_schema(tmp_path, malformation, message):
    """Appending should diagnose each malformed stage-schema component."""
    path = tmp_path / "cache.h5"
    tensor = TensorData(
        coords=np.asarray([[1, 2, 3]]),
        features=np.asarray([[4.0]]),
        meta=Meta(),
    )
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "coordinates": [tensor],
        },
    )
    state = writer._stage_states["deghosting"]
    writer.close()

    with h5py.File(path, "a") as out_file:
        stage = out_file["stages"]["deghosting"]
        products = stage["products"]
        if malformation == "events":
            del stage["events"]
        elif malformation == "products":
            products.create_group("unexpected")
        elif malformation == "metadata":
            del products["coordinates"].attrs["product_metadata"]
        elif malformation == "schema":
            products["coordinates"].attrs["product_metadata"] = yaml.dump(
                {"product_type": "index"}
            )
        else:
            del products["coordinates"]["meta"]

        with pytest.raises(ValueError, match=message):
            StageHDF5Writer._validate_stage_schema(
                stage, str(path), "deghosting", state
            )


def test_stage_hdf5_writer_overwrite_stage_preserves_other_stages(hdf5_output):
    """Rewriting one stage should leave sibling stages untouched."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("graph_spice")
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[5.0, 6.0]])],
        },
        overwrite_stage=True,
    )
    writer.finalize_stage("graph_spice")
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        np.testing.assert_array_equal(
            stage_product_values(out_file["stages"]["deghosting"], "dummy_data"),
            np.asarray([[1.0, 2.0]]),
        )
        np.testing.assert_array_equal(
            stage_product_values(out_file["stages"]["graph_spice"], "dummy_data"),
            np.asarray([[5.0, 6.0]]),
        )


def test_stage_hdf5_writer_lists_written_stages(hdf5_output):
    """Stage cache writer should expose the written stage names."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.write_stage(
        "graph_spice",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[3.0, 4.0]])],
        },
    )
    assert writer.list_stages() == ("deghosting", "graph_spice")
    writer.close()


def test_stage_hdf5_writer_stores_source_provenance(tmp_path):
    """Stage cache files should persist lightweight source-file provenance."""
    source_path = tmp_path / "source.root"
    source_path.write_bytes(b"source-bytes")
    cache_path = tmp_path / "cache.h5"

    stat_result = source_path.stat()
    writer = StageHDF5Writer(str(cache_path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray([source_path.name]),
            "source_file_size": np.asarray([stat_result.st_size]),
            "source_file_mtime_ns": np.asarray([stat_result.st_mtime_ns]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(cache_path, "r") as out_file:
        source_group = out_file["source"]
        assert source_group.attrs["file_name"] == source_path.name
        assert source_group.attrs["file_size"] == stat_result.st_size
        assert source_group.attrs["file_mtime_ns"] == stat_result.st_mtime_ns


def test_stage_hdf5_writer_call_uses_configured_stage(tmp_path):
    """The standard writer interface should target the configured stage."""
    cache_path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(cache_path), stage="deghosting", overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(cache_path, "r") as out_file:
        assert "deghosting" in out_file["stages"]
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]


def test_stage_hdf5_writer_call_requires_configured_stage(tmp_path):
    """The generic writer call path should fail without a configured stage."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"), overwrite=True)
    with pytest.raises(RuntimeError, match="configured `stage`"):
        writer(
            {
                "index": np.asarray([0]),
                "source_file_name": np.asarray(["source.root"]),
                "source_file_size": np.asarray([10]),
                "source_file_mtime_ns": np.asarray([20]),
                "dummy_data": [np.asarray([[1.0, 2.0]])],
            }
        )
    with pytest.raises(RuntimeError, match="configured `stage`"):
        writer.finalize()
    writer.close()


def test_stage_hdf5_writer_respects_explicit_keys(tmp_path):
    """Stage caches should persist only the requested stage products."""
    cache_path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(
        str(cache_path),
        overwrite=True,
        keys=["dummy_data"],
    )
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "source_file_entry_index": np.asarray([5]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
            "extra_tensor": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(cache_path, "r") as out_file:
        stage_group = out_file["stages"]["deghosting"]
        products = stage_group["products"]
        assert "dummy_data" in products
        assert "extra_tensor" not in products
        assert "source_file_entry_index" in products
        assert stage_group["events"].dtype == np.dtype(np.int64)


def test_stage_hdf5_writer_preserves_source_entry_with_explicit_keys(tmp_path):
    """Stage caches should preserve source entry provenance from file_entry_index."""
    cache_path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(
        str(cache_path),
        overwrite=True,
        keys=["dummy_data"],
    )
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "file_entry_index": np.asarray([7]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
            "extra_tensor": [np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(cache_path, "r") as out_file:
        stage_group = out_file["stages"]["deghosting"]
        assert "source_file_entry_index" in stage_group["products"]
        np.testing.assert_array_equal(
            stage_product_values(stage_group, "source_file_entry_index"), [7]
        )


def test_stage_hdf5_writer_rejects_mismatched_source(tmp_path):
    """Writing a later stage with different source provenance should fail."""
    source_a = tmp_path / "source_a.root"
    source_b = tmp_path / "source_b.root"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"bb")
    cache_path = tmp_path / "cache.h5"

    stat_a = source_a.stat()
    stat_b = source_b.stat()
    writer = StageHDF5Writer(str(cache_path), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray([source_a.name]),
            "source_file_size": np.asarray([stat_a.st_size]),
            "source_file_mtime_ns": np.asarray([stat_a.st_mtime_ns]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    writer = StageHDF5Writer(str(cache_path))
    with pytest.raises(RuntimeError, match="Cache source mismatch"):
        writer.write_stage(
            "graph_spice",
            {
                "index": np.asarray([0]),
                "source_file_name": np.asarray([source_b.name]),
                "source_file_size": np.asarray([stat_b.st_size]),
                "source_file_mtime_ns": np.asarray([stat_b.st_mtime_ns]),
                "dummy_data": [np.asarray([[3.0, 4.0]])],
            },
        )
    writer.close()


def test_stage_hdf5_writer_splits_batch_by_source(tmp_path):
    """A mixed-source batch should be routed into one cache file per source."""
    output = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(output), overwrite=True)
    writer.write_stage(
        "deghosting",
        {
            "index": np.asarray([0, 1]),
            "source_file_name": np.asarray(["source_a.root", "source_b.root"]),
            "source_file_size": np.asarray([10, 20]),
            "source_file_mtime_ns": np.asarray([30, 40]),
            "source_file_entry_index": np.asarray([5, 6]),
            "dummy_data": [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])],
        },
    )
    writer.finalize_stage("deghosting")
    writer.close()

    source_a_cache = tmp_path / "source_a_stage.h5"
    source_b_cache = tmp_path / "source_b_stage.h5"
    assert source_a_cache.is_file()
    assert source_b_cache.is_file()

    with h5py.File(source_a_cache, "r") as out_file:
        assert out_file["source"].attrs["file_name"] == "source_a.root"
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        np.testing.assert_array_equal(
            stage_product_values(
                out_file["stages"]["deghosting"], "source_file_entry_index"
            ),
            [5],
        )

    with h5py.File(source_b_cache, "r") as out_file:
        assert out_file["source"].attrs["file_name"] == "source_b.root"
        assert out_file["stages"]["deghosting"]["info"].attrs["complete"]
        np.testing.assert_array_equal(
            stage_product_values(
                out_file["stages"]["deghosting"], "source_file_entry_index"
            ),
            [6],
        )


def test_stage_hdf5_writer_overwrite_removes_existing_output(hdf5_output):
    """Overwrite should remove an existing target cache file eagerly."""
    open(hdf5_output, "a", encoding="utf-8").close()
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    assert not os.path.exists(hdf5_output)
    writer.close()


def test_stage_hdf5_writer_close_swallows_handle_errors(hdf5_output):
    """Writer cleanup should clear state even if one handle raises on close."""
    writer = StageHDF5Writer(hdf5_output)

    class BadHandle:
        def close(self):
            raise OSError("boom")

    writer._handles["x"] = BadHandle()
    writer._handle_pid = 123
    writer.close()
    assert writer._handles == {}
    assert writer._handle_pid is None


def test_stage_hdf5_writer_rejects_pid_change(hdf5_output, monkeypatch):
    """Persistent staged-writer handles should stay process-local."""
    writer = StageHDF5Writer(hdf5_output)
    writer._handle_pid = 1
    monkeypatch.setattr("spine.io.write.stage_hdf5.os.getpid", lambda: 2)
    with pytest.raises(RuntimeError, match="process-local"):
        writer._check_handle_pid()


def test_stage_hdf5_writer_open_handle_keep_open_false(hdf5_output):
    """Disabling persistent handles should return a close-on-use handle."""
    writer = StageHDF5Writer(hdf5_output, keep_open=False, overwrite=True)
    handle, should_close = writer._open_handle(hdf5_output)
    assert should_close is True
    handle.close()
    writer.close()


def test_stage_hdf5_writer_open_handle_reopens_invalid_cached_handle(hdf5_output):
    """Persistent handle lookup should reopen invalid cached handles."""
    writer = StageHDF5Writer(hdf5_output, overwrite=True)
    handle, _ = writer._open_handle(hdf5_output)
    handle.close()
    reopened, should_close = writer._open_handle(hdf5_output)
    assert reopened.id.valid
    assert should_close is False
    writer.close()


def test_stage_hdf5_writer_get_batch_source_info_edges(hdf5_output):
    """Source provenance extraction should cover missing, scalar, and bad inputs."""
    writer = StageHDF5Writer(hdf5_output)
    with pytest.raises(KeyError, match="Missing keys"):
        writer.get_batch_source_info({"index": np.asarray([0])})

    info = writer.get_batch_source_info(
        {
            "source_file_name": np.asarray("source.root"),
            "source_file_size": np.asarray(10),
            "source_file_mtime_ns": np.asarray(20),
        }
    )
    assert info == {"file_name": "source.root", "file_size": 10, "file_mtime_ns": 20}

    info = writer.get_batch_source_info(
        {
            "source_file_name": "source.root",
            "source_file_size": 10,
            "source_file_mtime_ns": 20,
        }
    )
    assert info["file_size"] == 10

    with pytest.raises(ValueError, match="is empty"):
        writer.get_batch_source_info(
            {
                "source_file_name": np.asarray([], dtype=object),
                "source_file_size": np.asarray([10]),
                "source_file_mtime_ns": np.asarray([20]),
            }
        )

    with pytest.raises(ValueError, match="contains multiple values"):
        writer.get_batch_source_info(
            {
                "source_file_name": np.asarray(["a.root", "b.root"]),
                "source_file_size": np.asarray([10, 10]),
                "source_file_mtime_ns": np.asarray([20, 20]),
            }
        )


def test_stage_hdf5_writer_ensure_source_group_existing_matches_and_mismatches(
    tmp_path,
):
    """Existing source groups should validate cache-file provenance."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer._ensure_stage_file(str(path))
    handle, should_close = writer._open_handle(str(path))
    try:
        batch = {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([20]),
        }
        writer.ensure_source_group(handle, batch, str(path))
        writer.ensure_source_group(handle, batch, str(path))
        bad = dict(batch)
        bad["source_file_size"] = np.asarray([11])
        with pytest.raises(RuntimeError, match="Cache source mismatch"):
            writer.ensure_source_group(handle, bad, str(path))
    finally:
        if should_close:
            handle.close()
        writer.close()


def test_stage_hdf5_writer_prepare_batch_scalar_and_skip_keys(tmp_path):
    """Single-entry normalization and skip-key filtering should both work."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    batch, batch_size, stage_state = writer._prepare_batch(
        {
            "index": np.int64(0),
            "file_index": np.int64(1),
            "file_entry_index": np.int64(2),
            "source_file_name": "source.root",
            "source_file_size": 10,
            "source_file_mtime_ns": 20,
            "dummy_data": np.asarray([[1.0, 2.0]]),
        },
        None,
    )
    assert batch_size == 1
    assert isinstance(batch["index"], list)
    assert stage_state is not None
    state = StageHDF5Writer(str(path), overwrite=True)
    state.skip_keys = ["dummy_data"]
    _, _, stage_state = state._prepare_batch(batch, None)
    assert "dummy_data" not in stage_state.keys
    assert "source_file_name" not in stage_state.keys
    state.close()
    writer.close()


def test_stage_hdf5_writer_output_path_and_split_missing_keys(tmp_path):
    """Output-path resolution and split validation should cover edge paths."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True)
    assert writer.get_output_path({"file_name": "source.root"}) == str(path)
    assert writer.get_output_path({"file_name": "source.root"}, True).endswith(
        "source_stage.h5"
    )
    with pytest.raises(KeyError, match="Missing key"):
        writer.split_batch_by_source({"index": np.asarray([0])})
    writer.close()


def test_stage_hdf5_writer_uses_explicit_directory(tmp_path):
    """Stage cache output paths should honor an explicit output directory."""
    directory = tmp_path / "cache_dir"
    writer = StageHDF5Writer(
        str(tmp_path / "cache.h5"), overwrite=True, directory=str(directory)
    )
    assert writer.get_output_path({"file_name": "source.root"}) == os.path.join(
        directory, "cache.h5"
    )
    assert writer.get_output_path({"file_name": "source.root"}, True) == os.path.join(
        directory, "source_stage.h5"
    )
    writer.close()


def test_stage_hdf5_writer_accepts_prefix_with_directory(tmp_path):
    """Stage cache naming should support prefix-based defaults under a new directory."""
    directory = tmp_path / "cache_dir"
    writer = StageHDF5Writer(
        file_name=None,
        prefix=["input.root"],
        overwrite=True,
        directory=str(directory),
        suffix="cache",
    )
    assert writer.file_name == os.path.join(directory, "input_cache.h5")
    assert writer.get_output_path({"file_name": "source.root"}) == os.path.join(
        directory, "input_cache.h5"
    )
    assert writer.get_output_path({"file_name": "source.root"}, True) == os.path.join(
        directory, "source_cache.h5"
    )
    writer.close()


def test_stage_hdf5_writer_creates_missing_output_directory(tmp_path):
    """Stage cache writes should create a configured output directory."""
    directory = tmp_path / "missing" / "cache"
    writer = StageHDF5Writer(
        file_name=None,
        prefix=["source.root"],
        overwrite=True,
        directory=str(directory),
        stage="grappa_inter",
    )
    writer(
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["source.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([30]),
            "dummy_data": [np.asarray([[1.0, 2.0]])],
        }
    )
    writer.close()

    assert (directory / "source_stage.h5").is_file()


def test_stage_hdf5_writer_requires_split_mode(tmp_path):
    """Stage caches should reject non-split writer configuration."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"))
    assert writer.split is True
    writer.close()

    with pytest.raises(ValueError, match="split=True"):
        StageHDF5Writer(str(tmp_path / "cache.h5"), split=False)


def test_stage_hdf5_writer_split_batch_preserves_scalar_values(tmp_path):
    """Source splitting should preserve scalar-valued batch metadata."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"), overwrite=True)
    groups = writer.split_batch_by_source(
        {
            "index": np.asarray([0, 1]),
            "epoch": 3,
            "source_file_name": np.asarray(["a.root", "b.root"]),
            "source_file_size": np.asarray([10, 20]),
            "source_file_mtime_ns": np.asarray([30, 40]),
            "dummy_data": [np.asarray([[1.0]]), np.asarray([[2.0]])],
        }
    )
    assert groups[0][1]["epoch"] == 3
    assert groups[1][1]["epoch"] == 3
    writer.close()


def test_stage_hdf5_writer_uses_source_paths_for_single_source_batches(tmp_path):
    """Single-source batches from different files should still use distinct cache paths."""
    writer = StageHDF5Writer(
        file_name=str(tmp_path / "cache.h5"),
        prefix=["train_000.root", "train_001.root"],
        overwrite=True,
    )
    groups_a = writer.split_batch_by_source(
        {
            "index": np.asarray([0]),
            "source_file_name": np.asarray(["train_000.root"]),
            "source_file_size": np.asarray([10]),
            "source_file_mtime_ns": np.asarray([30]),
            "dummy_data": [np.asarray([[1.0]])],
        }
    )
    groups_b = writer.split_batch_by_source(
        {
            "index": np.asarray([1]),
            "source_file_name": np.asarray(["train_001.root"]),
            "source_file_size": np.asarray([20]),
            "source_file_mtime_ns": np.asarray([40]),
            "dummy_data": [np.asarray([[2.0]])],
        }
    )

    assert groups_a[0][0].endswith("train_000_stage.h5")
    assert groups_b[0][0].endswith("train_001_stage.h5")
    assert groups_a[0][0] != groups_b[0][0]
    writer.close()


def test_stage_hdf5_writer_stage_group_existing_paths_and_flush(tmp_path):
    """Existing-stage update branches and flush bookkeeping should be covered."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True, flush_frequency=1)
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer.write_stage("deghosting", batch, cfg={"a": 1}, attrs={"tag": "x"})
    writer.write_stage("deghosting", batch, cfg={"a": 2}, attrs={"tag": "y"})
    writer.close()

    with h5py.File(path, "r") as out_file:
        info = out_file["stages"]["deghosting"]["info"].attrs
        assert info["tag"] == "y"
        assert "cfg" in info


def test_stage_hdf5_writer_recovers_incomplete_stage_across_sessions(tmp_path):
    """A new writer should rebuild only the incomplete stage it owns."""
    path = tmp_path / "cache.h5"
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer = StageHDF5Writer(str(path), overwrite=True)
    writer.write_stage("upstream", batch)
    writer.finalize_stage("upstream")
    writer.write_stage("deghosting", batch)
    writer.close()

    writer = StageHDF5Writer(str(path))
    replacement = dict(batch)
    replacement["index"] = np.asarray([1])
    replacement["dummy_data"] = [np.asarray([[3.0, 4.0]])]
    writer.write_stage("deghosting", replacement)
    writer.finalize_stage("deghosting")
    writer.close()

    with h5py.File(path, "r") as out_file:
        stages = out_file["stages"]
        assert len(stages["upstream"]["events"]) == 1
        np.testing.assert_array_equal(
            stage_product_values(stages["upstream"], "dummy_data"),
            np.asarray([[1.0, 2.0]]),
        )
        assert len(stages["deghosting"]["events"]) == 1
        np.testing.assert_array_equal(
            stage_product_values(stages["deghosting"], "dummy_data"),
            np.asarray([[3.0, 4.0]]),
        )
        assert stages["deghosting"]["info"].attrs["complete"]


def test_stage_hdf5_writer_configured_overwrite_is_one_time(tmp_path):
    """Driver-facing overwrite should replace a complete stage only once."""
    path = tmp_path / "cache.h5"
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer = StageHDF5Writer(str(path), stage="deghosting", overwrite=True)
    writer(batch)
    writer.finalize()
    writer.close()

    writer = StageHDF5Writer(str(path), stage="deghosting")
    with pytest.raises(RuntimeError, match="already complete"):
        writer(batch)
    writer.close()

    writer = StageHDF5Writer(str(path), stage="deghosting", overwrite_stage=True)
    for index, values in ((1, [3.0, 4.0]), (2, [5.0, 6.0])):
        replacement = dict(batch)
        replacement["index"] = np.asarray([index])
        replacement["dummy_data"] = [np.asarray([values])]
        writer(replacement)
    writer.finalize()
    writer.close()

    with h5py.File(path, "r") as out_file:
        stage = out_file["stages"]["deghosting"]
        assert len(stage["events"]) == 2
        np.testing.assert_array_equal(
            stage_product_values(stage, "dummy_data"),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        )


def test_stage_hdf5_writer_keep_open_false_closes_in_write_finalize_and_list(tmp_path):
    """Close-on-use mode should exercise all staged-writer close branches."""
    path = tmp_path / "cache.h5"
    writer = StageHDF5Writer(str(path), overwrite=True, keep_open=False)
    batch = {
        "index": np.asarray([0]),
        "source_file_name": np.asarray(["source.root"]),
        "source_file_size": np.asarray([10]),
        "source_file_mtime_ns": np.asarray([20]),
        "dummy_data": [np.asarray([[1.0, 2.0]])],
    }
    writer.write_stage("deghosting", batch)
    writer.finalize_stage("deghosting")
    assert writer.list_stages() == ("deghosting",)
    writer.close()


def test_stage_hdf5_writer_finalize_and_list_ignore_missing_stage(tmp_path):
    """Finalize/list helpers should tolerate files without the requested stage."""
    writer = StageHDF5Writer(str(tmp_path / "cache.h5"), overwrite=True)
    batch = {
        "index": np.asarray([0, 1]),
        "source_file_name": np.asarray(["a.root", "b.root"]),
        "source_file_size": np.asarray([10, 20]),
        "source_file_mtime_ns": np.asarray([30, 40]),
        "dummy_data": [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])],
    }
    writer.write_stage("deghosting", batch)
    writer.finalize_stage("graph_spice")

    missing_path = sorted(writer._known_files)[0]
    del writer._handles[missing_path]["stages"]["deghosting"]
    writer.finalize_stage("deghosting")
    assert writer.list_stages() == ("deghosting",)
    writer.close()


def test_hdf5_writer_stores_stored_properties(hdf5_output):
    """Test HDF5 writer serializes stored properties on data objects."""
    particle = RecoParticle(id=3, index=np.arange(4, dtype=np.int64), pid=2)
    data = {
        "index": np.array([0]),
        "particles": [ObjectList([particle], RecoParticle())],
    }

    HDF5Writer(hdf5_output)(data)

    with h5py.File(hdf5_output, "r") as out_file:
        fields = out_file["particles"].dtype.names
        assert "size" in fields
        assert "pdg_code" in fields
        assert "mass" in fields
        assert "ke" in fields
        assert "momentum" in fields
        assert "p" in fields
        assert "reco_ke" not in fields
        assert out_file["particles"]["size"][0] == 4
        assert out_file["particles"]["pdg_code"][0] == 13


def test_hdf5_writer_rejects_existing_file(hdf5_output):
    """Writer should protect existing outputs unless overwrite/append is enabled."""
    open(hdf5_output, "a", encoding="utf-8").close()
    with pytest.raises(FileExistsError):
        HDF5Writer(hdf5_output)


def test_hdf5_writer_overwrite_removes_existing_output(hdf5_output):
    """Overwrite should remove an existing target file eagerly."""
    open(hdf5_output, "a", encoding="utf-8").close()

    writer = HDF5Writer(hdf5_output, overwrite=True)

    assert not os.path.exists(hdf5_output)
    writer.close()


def test_hdf5_writer_get_file_names_errors():
    """File name inference should reject incompatible inputs."""
    with pytest.raises(AssertionError, match="must provide the input file `prefix`"):
        HDF5Writer.get_file_names(None, None, split=False)
    with pytest.raises(
        AssertionError, match="must provide one `prefix` per input file"
    ):
        HDF5Writer.get_file_names("out.h5", "prefix", split=True)


def test_hdf5_writer_get_stored_keys_errors(hdf5_output):
    """Stored key selection should reject inconsistent requests."""
    writer = HDF5Writer(hdf5_output, keys=["a"], skip_keys=["b"])
    with pytest.raises(ValueError, match="Must not specify both"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})

    writer = HDF5Writer(hdf5_output, skip_keys=["missing"])
    with pytest.raises(KeyError, match="appears in `skip_keys`"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})

    writer = HDF5Writer(hdf5_output, keys=["missing"])
    with pytest.raises(KeyError, match="not present"):
        writer.get_stored_keys({"index": np.array([0]), "a": [1]})


def test_hdf5_writer_get_object_dtype_errors(hdf5_output):
    """Object dtype discovery should reject unsupported attribute types."""

    class BadObject:
        def as_dict(self, lite=False):
            return {"bad": {"x": 1}}

    writer = HDF5Writer(hdf5_output, overwrite=True)
    with pytest.raises(ValueError, match="unrecognized type"):
        writer.get_object_dtype(BadObject())


def test_hdf5_writer_initializes_dummy_objects(hdf5_output):
    """Configured dummy datasets should be initialized as empty SPINE objects."""
    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"dummy": "RunInfo"})
    assert isinstance(writer.dummy_ds["dummy"], RunInfo)


def test_hdf5_writer_get_stored_keys_ready_and_dummy(hdf5_output):
    """Stored-key discovery should reuse ready keys and support dummy datasets."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer.ready = True
    writer.keys = {"index", "a"}
    assert writer.get_stored_keys({"index": [0], "a": [1]}) == {"index", "a"}

    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"dummy": "RunInfo"})
    keys = writer.get_stored_keys({"index": [0], "a": [1]})
    assert "dummy" in keys

    writer = HDF5Writer(hdf5_output, overwrite=True, dummy_ds={"a": "RunInfo"})
    with pytest.raises(KeyError, match="conflicts with a real product"):
        writer.get_stored_keys({"index": [0], "a": [1]})


def test_hdf5_writer_get_stored_keys_rejects_unknown_skip_key(hdf5_output):
    """Stored-key discovery should reject skip keys that are not present in the data."""
    writer = HDF5Writer(hdf5_output, overwrite=True, skip_keys={"missing"})

    with pytest.raises(KeyError, match="skip_keys"):
        writer.get_stored_keys({"index": [0], "a": [1]})


def test_hdf5_writer_get_stored_keys_removes_requested_keys(hdf5_output):
    """Stored-key discovery should drop requested keys when they are present."""
    writer = HDF5Writer(hdf5_output, overwrite=True, skip_keys=["a"])
    keys = writer.get_stored_keys({"index": [0], "a": [1], "b": [2]})

    assert keys == {"index", "b"}


def test_hdf5_writer_get_data_type_special_cases(hdf5_output):
    """Data-type discovery should handle strings, ragged ndarrays, and unsupported containers."""
    writer = HDF5Writer(hdf5_output, overwrite=True)

    class BadContainer:
        dtype = object

        def __len__(self):
            return 1

    scalar_fmt = writer.get_data_type({"text": "hello"}, "text")
    assert scalar_fmt.scalar is True

    list_fmt = writer.get_data_type({"text": ["hello"]}, "text")
    assert list_fmt.scalar is True

    ragged_fmt = writer.get_data_type(
        {
            "jagged": [
                [np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)]
            ]
        },
        "jagged",
    )
    assert ragged_fmt.width == [2, 3]
    assert ragged_fmt.merge is False

    with pytest.raises(TypeError, match="Cannot store output"):
        writer.get_data_type({"bad": [BadContainer()]}, "bad")


def test_hdf5_writer_create_stores_cfg(hdf5_output):
    """Writer creation should persist the configuration when provided."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    data = {"index": np.array([0]), "value": [np.asarray([1.0], dtype=np.float32)]}
    writer.create(data, cfg={"io": {"loader": {}}})
    writer._ensure_file(0)
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert "cfg" in out_file["info"].attrs


def test_hdf5_writer_creates_split_outputs_lazily(tmp_path):
    """Split outputs should only be created when a file receives data."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.finalize()
    writer.close()

    assert not os.path.exists(os.path.join(tmp_path, "split_0.h5"))
    assert os.path.exists(os.path.join(tmp_path, "split_1.h5"))


def test_hdf5_writer_reuses_open_handles(monkeypatch, hdf5_output):
    """Repeated writes with one writer instance should reuse the same append handle."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.writer.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    first_open_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls == first_open_calls
    writer.close()


def test_hdf5_writer_keep_open_false_opens_per_write(monkeypatch, hdf5_output):
    """Disabling persistent handles should reopen the output file each write."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.writer.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True, keep_open=False)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    first_write_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls > first_write_calls
    assert writer._file_handles == {}


def test_hdf5_writer_reopens_invalid_persistent_handle(monkeypatch, hdf5_output):
    """Invalid persistent handles should be reopened lazily on the next write."""
    open_calls = 0
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("spine.io.write.hdf5.writer.h5py.File", counted_file)
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer._file_handles[0].close()
    first_open_calls = open_calls
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    assert open_calls > first_open_calls
    writer.close()


def test_hdf5_writer_rejects_pid_reuse(monkeypatch, hdf5_output):
    """Persistent writer handles should not be reused across process boundaries."""
    pids = iter([100, 100, 200])
    monkeypatch.setattr("spine.io.write.hdf5.writer.os.getpid", lambda: next(pids))

    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )

    with pytest.raises(RuntimeError, match="process-local"):
        writer(
            {
                "index": np.asarray([1]),
                "value": [np.asarray([2.0], dtype=np.float32)],
            }
        )


def test_hdf5_writer_context_manager_and_flush(hdf5_output):
    """Context-managed writers should flush and close persistent handles."""
    with HDF5Writer(hdf5_output, overwrite=True) as writer:
        writer(
            {
                "index": np.asarray([0]),
                "value": [np.asarray([1.0], dtype=np.float32)],
            }
        )
        assert writer._file_handles
        writer.flush()

    assert writer._file_handles == {}
    assert writer._handle_pid is None
    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_append_existing_file_keep_open_false_marks_incomplete(hdf5_output):
    """Appending to an existing file should reopen it and mark it incomplete."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.finalize()
    writer.close()

    writer = HDF5Writer(hdf5_output, append=True, keep_open=False)
    writer(
        {
            "index": np.asarray([1]),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    with h5py.File(hdf5_output, "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_flush_frequency_flushes_persistent_handle(
    monkeypatch, hdf5_output
):
    """Flush frequency should flush persistent handles after the requested interval."""
    writer = HDF5Writer(hdf5_output, overwrite=True, flush_frequency=1)
    flush_calls = 0

    original_flush = h5py.File.flush

    def counted_flush(handle):
        nonlocal flush_calls
        flush_calls += 1
        return original_flush(handle)

    monkeypatch.setattr(h5py.File, "flush", counted_flush)
    writer(
        {
            "index": np.asarray([0]),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert flush_calls >= 1


def test_hdf5_writer_close_swallows_handle_close_errors(hdf5_output):
    """Writer cleanup should clear state even if a handle raises on close."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer._handle_pid = 123

    class BadHandle:
        def close(self):
            raise OSError("boom")

    writer._file_handles[0] = BadHandle()
    writer.close()

    assert writer._file_handles == {}
    assert writer._handle_pid is None


def test_hdf5_writer_call_scalar_split_and_dummy(tmp_path):
    """Writer call should wrap scalar inputs before hitting dummy-dataset validation."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name,
        prefix=["a", "b"],
        split=True,
        overwrite=True,
        dummy_ds={"dummy": "RunInfo"},
    )
    with pytest.raises(KeyError, match="conflicts with a real product"):
        writer(
            {"index": 0, "file_index": 1, "value": np.asarray([1.0], dtype=np.float32)}
        )


def test_hdf5_writer_call_splits_entries_by_file_index(tmp_path):
    """Writer calls should fan entries out across split outputs using file_index."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "file_index": np.asarray([0, 1], dtype=np.int64),
            "value": [
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ],
        }
    )
    writer.finalize()
    writer.close()

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1


def test_hdf5_writer_call_splits_entries_by_file_index_without_persistent_handles(
    tmp_path,
):
    """Split writes should also close transient per-file handles when keep_open is disabled."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name,
        prefix=["a", "b"],
        split=True,
        overwrite=True,
        keep_open=False,
    )
    writer(
        {
            "index": np.asarray([0, 1], dtype=np.int64),
            "file_index": np.asarray([0, 1], dtype=np.int64),
            "value": [
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ],
        }
    )
    writer.finalize()

    assert writer._file_handles == {}
    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert len(out_file["events"]) == 1


def test_hdf5_writer_auto_finalizes_split_predecessors(tmp_path):
    """Sequential split writing should finalize earlier files once the writer advances."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert out_file["info"].attrs["complete"]
    with h5py.File(os.path.join(tmp_path, "split_1.h5"), "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_auto_finalizes_split_predecessors_without_persistent_handles(
    tmp_path,
):
    """Sequential split writing should finalize predecessors without persistent handles."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(
        file_name, prefix=["a", "b"], split=True, overwrite=True, keep_open=False
    )
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )

    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert out_file["info"].attrs["complete"]


def test_hdf5_writer_disables_sequential_split_finalization_on_out_of_order_writes(
    tmp_path,
):
    """Out-of-order split writes should disable automatic predecessor finalization."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([1], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert not writer._split_sequential


def test_hdf5_writer_keeps_split_predecessor_guard_on_same_file_id(tmp_path):
    """Repeated writes to the same split file should not finalize predecessors."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer(
        {
            "index": np.asarray([1], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([2.0], dtype=np.float32)],
        }
    )
    writer.close()

    assert writer._split_sequential
    with h5py.File(os.path.join(tmp_path, "split_0.h5"), "r") as out_file:
        assert not out_file["info"].attrs["complete"]


def test_hdf5_writer_rejects_writes_to_finalized_file(tmp_path):
    """Finalized files should not accept more writes."""
    file_name = os.path.join(tmp_path, "split.h5")
    writer = HDF5Writer(file_name, prefix=["a", "b"], split=True, overwrite=True)
    writer(
        {
            "index": np.asarray([0], dtype=np.int64),
            "file_index": np.asarray([0], dtype=np.int64),
            "value": [np.asarray([1.0], dtype=np.float32)],
        }
    )
    writer.finalize()

    with pytest.raises(RuntimeError, match="already finalized"):
        writer._ensure_file(0)


def test_hdf5_writer_store_jagged_and_scalar_append_region_key(hdf5_output):
    """Append-key should cover scalar fanout and jagged list storage."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    data = {
        "index": np.array([0]),
        "scalar": 5,
        "jagged": [
            [np.ones((1, 2), dtype=np.float32), np.ones((1, 3), dtype=np.float32)]
        ],
    }
    writer.create(data)
    writer._ensure_file(0)

    with h5py.File(hdf5_output, "a") as out_file:
        event = np.empty(1, writer.event_dtype)
        writer.append_region_key(out_file, event, data, "scalar", 0)
        writer.append_region_key(out_file, event, data, "jagged", 0)

        assert out_file["scalar"].shape[0] == 1
        assert isinstance(out_file["jagged"], h5py.Group)
        assert out_file["jagged"]["index"].shape[0] == 1


def generate_object_list(cls, sizes):
    """Generates a dummy list of lists of objects of the request class.

    Parameters
    ----------
    cls : object
        Class that the objects should belong to
    sizes : List[int]
        Number of objects in each list

    Returns
    -------
    List[ObjectList[obj]]
        List of typed lists of objects
    """
    return [ObjectList([cls() for _ in range(s)], cls()) for s in sizes]


def test_hdf5_writer_persists_post_processor_provenance(hdf5_output):
    """Cumulative post-processing provenance should round-trip independently."""
    writer = HDF5Writer(hdf5_output, overwrite=True)
    writer.set_post_processors(("first", "second"))
    writer({"index": np.asarray([0]), "value": np.asarray([[1.0]])}, cfg={})
    with pytest.raises(RuntimeError, match="before the first write"):
        writer.set_post_processors(("late",))
    writer.finalize()
    writer.close()

    reader = HDF5Reader(hdf5_output, build_classes=False)
    assert reader.post_processors == ("first", "second")
    reader.close()


def test_hdf5_writer_v2_uses_offsets_and_preserves_derived_fields(hdf5_output):
    """V2 should flatten variable fields while retaining advertised summaries."""
    particle = RecoParticle(
        id=7,
        index=np.asarray([2, 5, 9], dtype=np.int32),
        match_ids=np.asarray([11], dtype=np.int32),
        match_overlaps=np.asarray([0.75], dtype=np.float32),
    )
    data = {
        "index": np.asarray([0]),
        "particles": [ObjectList([particle], RecoParticle())],
        "tensor": [np.arange(6, dtype=np.float32).reshape(3, 2)],
        "label": ["event-zero"],
    }

    with HDF5Writer(hdf5_output, overwrite=True, format_version=2) as writer:
        writer(data, cfg={"test": True})

    with h5py.File(hdf5_output, "r") as out_file:
        info = out_file["info"].attrs
        assert info["format"] == "spine_hdf5"
        assert info["format_version"] == 2
        assert "spine_version" in info

        particles = out_file["products"]["particles"]
        assert particles.attrs["kind"] == "objects"
        assert particles["event_offsets"][:].tolist() == [0, 1]
        index_pool = next(
            pool
            for pool in particles["variables"].values()
            if "index" in yaml.safe_load(pool.attrs["fields"])
        )
        fields = yaml.safe_load(index_pool.attrs["fields"])
        index_column = fields.index("index")
        pool_index = int(index_pool.name.split("_")[-1])
        bounds = particles["fixed"][f"_var_offsets_{pool_index}"][
            0, index_column : index_column + 2
        ]
        assert index_pool["values"][bounds[0] : bounds[1]].tolist() == [2, 5, 9]

        # Derived properties remain directly available without SPINE classes.
        assert particles["fixed"]["size"].tolist() == [3]
        assert particles["fixed"]["best_match_id"].tolist() == [11]
        assert particles["fixed"]["best_match_overlap"].tolist() == [0.75]
        assert "num_fragments" in particles["fixed"].dtype.names

        # No dataset in the V2 product tree uses an HDF5 VLEN dtype.
        def assert_no_vlen(_, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype.names:
                    for name in obj.dtype.names:
                        assert h5py.check_dtype(vlen=obj.dtype[name]) is None
                else:
                    assert h5py.check_dtype(vlen=obj.dtype) is None

        out_file.visititems(assert_no_vlen)


def test_hdf5_writer_rejects_append_format_mismatch(hdf5_output):
    """Appending must never silently mix physical HDF5 layouts."""
    data = {"index": np.asarray([0]), "value": [np.asarray([1], dtype=np.int32)]}
    HDF5Writer(hdf5_output, overwrite=True, format_version=1)(data, cfg={})

    writer = HDF5Writer(hdf5_output, append=True, format_version=2)
    with pytest.raises(ValueError, match="format version"):
        writer(data, cfg={})
    writer.close()


def test_hdf5_writer_rejects_unknown_format_version(hdf5_output):
    """Writers should reject physical layouts they do not implement."""
    with pytest.raises(ValueError, match="Unsupported HDF5 format version"):
        HDF5Writer(hdf5_output, format_version=99)


def test_hdf5_writer_rejects_append_without_info_group(hdf5_output):
    """Append validation should reject files without format metadata."""
    writer = HDF5Writer(hdf5_output, append=True, format_version=2)
    with h5py.File(hdf5_output, "w") as out_file:
        out_file.create_dataset("events", data=np.asarray([], dtype=np.int64))
        with pytest.raises(ValueError, match="missing info group"):
            writer._validate_append_format(out_file, hdf5_output)


def test_hdf5_writer_v2_split_uses_collective_append(tmp_path):
    """Split V2 outputs should use the batch-oriented append path."""
    path = str(tmp_path / "split.h5")
    data = {
        "index": np.asarray([0, 1]),
        "file_index": np.asarray([0, 1]),
        "value": [
            np.asarray([1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
        ],
    }
    with HDF5Writer(
        path,
        prefix=["first", "second"],
        split=True,
        overwrite=True,
        format_version=2,
    ) as writer:
        writer(data, cfg={})

    for file_id in range(2):
        with h5py.File(tmp_path / f"split_{file_id}.h5", "r") as out_file:
            assert len(out_file["events"]) == 1


def test_hdf5_writer_v2_single_entry_wrapper_and_scalar(hdf5_output):
    """The single-entry compatibility wrapper should accept scalar products."""
    data = {"index": np.asarray([0]), "scalar": 5}
    writer = HDF5Writer(
        hdf5_output,
        overwrite=True,
        keep_open=False,
        format_version=2,
    )
    writer.create(data, cfg={})
    writer._ensure_file(0)

    with h5py.File(hdf5_output, "a") as out_file:
        writer.append_product_entry(out_file, data, 0)

    with h5py.File(hdf5_output, "r") as out_file:
        assert out_file["products"]["scalar"]["values"][:].tolist() == [5]
        assert len(out_file["events"]) == 1


def _make_v2_variable_object_group(out_file, fields):
    """Create the minimal V2 object group needed by pool validation tests."""
    objects = out_file.create_group("objects")
    objects.create_dataset(
        "fixed",
        (0,),
        maxshape=(None,),
        dtype=np.dtype([("_var_offsets_0", np.int64, (2,))]),
    )
    objects.create_dataset(
        "event_offsets",
        data=np.asarray([0], dtype=np.int64),
        maxshape=(None,),
    )
    pool = objects.create_group("variables").create_group("pool_0")
    pool.attrs["kind"] = "array"
    pool.attrs["fields"] = fields
    pool.create_dataset(
        "values",
        (0,),
        maxshape=(None,),
        dtype=np.float32,
    )
    return objects


def test_hdf5_writer_v2_decodes_byte_pool_fields_and_empty_lengths(
    hdf5_output,
):
    """Byte-valued field metadata and empty appends should be harmless."""
    with h5py.File(hdf5_output, "w") as out_file:
        objects = _make_v2_variable_object_group(out_file, np.bytes_(b"- field\n"))
        HDF5Writer.store_object_batches(objects, [], lite=False)
        assert objects["event_offsets"][:].tolist() == [0]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (np.void(b"- field\n"), "must be a string"),
        ("field", "list of strings"),
    ],
)
def test_hdf5_writer_v2_rejects_bad_pool_field_metadata(hdf5_output, fields, message):
    """Object pool field metadata must be a serialized string list."""
    with h5py.File(hdf5_output, "w") as out_file:
        objects = _make_v2_variable_object_group(out_file, fields)
        with pytest.raises(TypeError, match=message):
            HDF5Writer.store_object_batches(objects, [], lite=False)


def test_hdf5_writer_v2_rejects_multidimensional_variable_fields(
    hdf5_output,
):
    """Variable object fields must remain one-dimensional."""

    class BadObject:
        def as_dict(self, lite):
            return {"field": np.ones((2, 2), dtype=np.float32)}

    with h5py.File(hdf5_output, "w") as out_file:
        objects = _make_v2_variable_object_group(out_file, yaml.safe_dump(["field"]))
        with pytest.raises(ValueError, match="must be one-dimensional"):
            HDF5Writer.store_object_batches(objects, [[BadObject()]], lite=False)
