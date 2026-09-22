"""Regression tests for explicit spatial-product augmentation contracts."""

import numpy as np
import pytest

from spine.data import (
    ClusterLabelData,
    Flash,
    ObjectListData,
    Particle,
    TensorData,
    TensorSchema,
)
from spine.io.augment import AugmentManager
from spine.io.augment.spatial import (
    ClusterSpatialAdapter,
    ObjectListSpatialAdapter,
)

from .helpers import make_meta, make_tensor


def make_cluster(coords, meta, vertices=None):
    """Build compact cluster labels with optional continuous particle vertices."""
    coords = np.asarray(coords, dtype=np.int64)
    num_rows = len(coords)
    if vertices is None:
        features = np.column_stack((np.ones(num_rows), np.arange(num_rows))).astype(
            np.float32
        )
        return ClusterLabelData(coords=coords, features=features, meta=meta)

    vertices = np.asarray(vertices, dtype=np.float32)
    features = np.column_stack(
        (np.ones(num_rows), np.arange(num_rows), np.arange(num_rows))
    ).astype(np.float32)
    return ClusterLabelData(
        coords=coords,
        features=features,
        particles={"vertex": vertices},
        meta=meta,
    )


def test_manager_flips_cluster_voxels_and_particle_vertices():
    """Packed float cluster voxels remain discrete while vertices stay continuous."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    cluster = make_cluster(
        [[0, 1, 2], [3, 1, 2]],
        meta,
        vertices=[[0.5, 1.5, 2.5], [3.5, 1.5, 2.5]],
    )
    event = {
        "data": make_tensor([[0, 1, 2], [3, 1, 2]], meta),
        "clust_label": cluster,
        "meta": meta,
    }

    AugmentManager(flip={"axis": 0})(event)

    np.testing.assert_array_equal(cluster.coords, [[3, 1, 2], [0, 1, 2]])
    np.testing.assert_allclose(
        cluster.particles["vertex"], [[3.5, 1.5, 2.5], [0.5, 1.5, 2.5]]
    )
    np.testing.assert_array_equal(event["data"].coords, cluster.coords)


def test_rotation_transforms_every_tensor_coordinate_group():
    """Start and end point groups must follow the same sampled rotation."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    points = TensorData(
        coords=np.asarray([[0.5, 1.5, 2.5, 3.5, 2.5, 1.5]], dtype=np.float32),
        features=np.ones((1, 1), dtype=np.float32),
        meta=meta,
        coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
    )
    event = {"points": points, "meta": meta}

    AugmentManager(rotate={"axes": (0, 1), "k": 1})(event)

    np.testing.assert_allclose(
        points.coordinate_data,
        [[2.5, 0.5, 2.5, 1.5, 3.5, 1.5]],
    )


def test_declared_coordinate_mode_overrides_storage_dtype():
    """Sampling semantics come from the schema, not array representation."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    tensor = TensorData(
        coords=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
        features=np.ones((1, 1), dtype=np.float32),
        meta=meta,
        schema=TensorSchema(
            coordinate_groups={"voxel": (0, 1, 2)},
            coordinate_modes={"voxel": "discrete"},
        ),
    )

    AugmentManager(flip={"axis": 0})({"tensor": tensor, "meta": meta})

    np.testing.assert_array_equal(tensor.coords, [[3.0, 1.0, 2.0]])


def test_crop_selects_cluster_rows_and_preserves_associations():
    """Support-changing transforms select the complete packed cluster rows."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    cluster = make_cluster([[0, 0, 0], [3, 3, 3]], meta)
    event = {"clust_label": cluster, "meta": meta}

    AugmentManager(
        crop={
            "min_dimensions": np.full(3, 2.0, dtype=np.float32),
            "max_dimensions": np.full(3, 2.0, dtype=np.float32),
            "lower": np.zeros(3, dtype=np.float32),
            "upper": np.full(3, 2.0, dtype=np.float32),
            "keep_meta": True,
        }
    )(event)

    np.testing.assert_array_equal(cluster.coords, [[0, 0, 0]])
    np.testing.assert_array_equal(cluster.features, [[1, 0]])


def test_unsupported_spatial_object_fails_before_any_mutation():
    """An unsupported positional product rejects the whole event atomically."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2]], meta)
    original = tensor.coords.copy()
    flash = Flash(
        center=np.asarray([0.5, 1.5, 2.5], dtype=np.float32),
        width=np.ones(3, dtype=np.float32),
    )
    event = {
        "data": tensor,
        "flashes": ObjectListData([flash], Flash()),
        "meta": meta,
    }

    with pytest.raises(NotImplementedError, match="Spatial ObjectListData `flashes`"):
        AugmentManager(flip={"axis": 0})(event)

    np.testing.assert_array_equal(tensor.coords, original)


def test_object_position_fields_follow_global_transform():
    """Position-bearing objects participate without pretending to own sparse rows."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    particle = Particle(position=np.asarray([0.5, 1.5, 2.5], dtype=np.float32))
    particles = ObjectListData([particle], Particle())

    AugmentManager(flip={"axis": 0})({"particles": particles, "meta": meta})

    np.testing.assert_allclose(particle.position, [3.5, 1.5, 2.5])


def test_empty_spatial_object_list_preserves_declared_contract():
    """Empty typed lists validate through their representative object."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    particles = ObjectListData([], Particle())

    result = AugmentManager(flip={"axis": 0})({"particles": particles, "meta": meta})

    assert result["particles"] == []


def test_jitter_rejects_continuous_points_before_mutation():
    """Local stochastic correspondence must be defined rather than guessed."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    voxels = make_tensor([[0, 1, 2]], meta)
    original = voxels.coords.copy()
    points = TensorData(
        coords=np.asarray([[0.5, 1.5, 2.5]], dtype=np.float32),
        features=np.ones((1, 1), dtype=np.float32),
        meta=meta,
    )
    event = {"data": voxels, "points": points, "meta": meta}

    with pytest.raises(ValueError, match="is continuous"):
        AugmentManager(jitter={"max_offset": 1})(event)

    np.testing.assert_array_equal(voxels.coords, original)


def test_malformed_position_fails_before_any_mutation():
    """Every advertised spatial field must contain three-dimensional points."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    tensor = make_tensor([[0, 1, 2]], meta)
    original = tensor.coords.copy()
    particle = Particle()
    # Corrupt a post-construction field to exercise augmentation preflight.
    object.__setattr__(particle, "position", np.asarray([0.5, 1.5], dtype=np.float32))
    event = {
        "data": tensor,
        "particles": ObjectListData([particle], Particle()),
        "meta": meta,
    }

    with pytest.raises(ValueError, match=r"particles\.position.*\(1, 2\)"):
        AugmentManager(flip={"axis": 0})(event)

    np.testing.assert_array_equal(tensor.coords, original)


def test_tensor_coordinate_contract_rejects_absent_or_non_3d_groups():
    """Coordinate-bearing tensors must explicitly expose 3D point groups."""
    meta = make_meta()
    features = np.ones((1, 1), dtype=np.float32)

    missing = TensorData(
        coords=np.zeros((1, 3), dtype=np.float32),
        features=features,
        meta=meta,
        schema=TensorSchema(),
    )
    with pytest.raises(ValueError, match="declares no coordinate groups"):
        AugmentManager(flip={"axis": 0})({"points": missing, "meta": meta})

    malformed = TensorData(
        coords=np.zeros((1, 2), dtype=np.float32),
        features=features,
        meta=meta,
        schema=TensorSchema(
            coordinate_groups={"point": (0, 1)},
            coordinate_modes={"point": "continuous"},
        ),
    )
    with pytest.raises(ValueError, match="must have exactly three columns"):
        AugmentManager(flip={"axis": 0})({"points": malformed, "meta": meta})


def test_support_changing_augment_rejects_multiple_primary_groups():
    """One row-selection mask cannot be inferred from multiple point groups."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    points = TensorData(
        coords=np.zeros((1, 6), dtype=np.float32),
        features=np.ones((1, 1), dtype=np.float32),
        meta=meta,
        coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
    )

    with pytest.raises(ValueError, match="primary coordinate groups"):
        AugmentManager(
            crop={
                "min_dimensions": np.full(3, 2.0, dtype=np.float32),
                "max_dimensions": np.full(3, 2.0, dtype=np.float32),
            }
        )({"points": points, "meta": meta})


def test_spatial_product_without_metadata_fails_loudly():
    """A geometric transform must not assume an image frame implicitly."""
    tensor = TensorData(
        coords=np.zeros((1, 3), dtype=np.int64),
        features=np.ones((1, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="requires image metadata"):
        AugmentManager(flip={"axis": 0})({"data": tensor})


def test_crop_reframes_cluster_auxiliary_vertices():
    """Auxiliary cluster points follow a crop that changes the image frame."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    cluster = make_cluster(
        [[1, 1, 1], [2, 2, 2]],
        meta,
        vertices=[[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]],
    )
    bounds_lower = np.ones(3, dtype=np.float32)
    bounds_upper = np.full(3, 3.0, dtype=np.float32)

    AugmentManager(
        crop={
            "min_dimensions": np.full(3, 2.0, dtype=np.float32),
            "max_dimensions": np.full(3, 2.0, dtype=np.float32),
            "lower": bounds_lower,
            "upper": bounds_upper,
            "keep_meta": False,
        }
    )({"clust_label": cluster, "meta": meta})

    np.testing.assert_array_equal(cluster.coords, [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_allclose(
        cluster.particles["vertex"], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
    )
    np.testing.assert_allclose(cluster.meta.lower, bounds_lower)


def test_jitter_rejects_cluster_auxiliary_points_before_mutation():
    """Per-voxel jitter has no valid mapping for particle-level vertices."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    cluster = make_cluster([[1, 1, 1]], meta, vertices=[[1.5, 1.5, 1.5]])
    original = cluster.data.copy()

    with pytest.raises(ValueError, match="auxiliary spatial fields"):
        AugmentManager(jitter={"max_offset": 1})({"clust_label": cluster, "meta": meta})

    np.testing.assert_array_equal(cluster.data, original)


def test_object_list_contract_rejects_invalid_or_mixed_units():
    """A typed object list must identify one consistent coordinate frame."""
    invalid = Particle()
    invalid.units = "mm"
    with pytest.raises(ValueError, match="must declare `px` or `cm` units"):
        ObjectListSpatialAdapter("particles", ObjectListData([invalid], Particle()))

    first = Particle()
    second = Particle()
    second.units = "px"
    with pytest.raises(ValueError, match="one object type.*one coordinate frame"):
        ObjectListSpatialAdapter(
            "particles", ObjectListData([first, second], Particle())
        )


def test_object_list_rejects_sparse_row_selection():
    """Independent physics-object rows cannot consume a voxel support mask."""
    adapter = ObjectListSpatialAdapter(
        "particles", ObjectListData([Particle()], Particle())
    )

    with pytest.raises(ValueError, match="Row selection is undefined"):
        adapter.select_rows(np.asarray([True]))


def test_cluster_adapter_rejects_unknown_spatial_field():
    """Adapters reject misspelled product-local field names."""
    meta = make_meta()
    adapter = ClusterSpatialAdapter("clust_label", make_cluster([[0, 0, 0]], meta))

    with pytest.raises(KeyError, match="unknown"):
        adapter.set_field("unknown", np.zeros((1, 3), dtype=np.float32))


def test_translation_preserves_centimeter_object_units():
    """Voxel-frame translations round-trip centimeter-valued object points."""
    meta = make_meta(upper=(4.0, 4.0, 4.0))
    particle = Particle(position=np.asarray([0.5, 1.5, 2.5], dtype=np.float32))
    particles = ObjectListData([particle], Particle())
    manager = AugmentManager(
        translate={
            "lower": np.zeros(3, dtype=np.float32),
            "upper": np.full(3, 5.0, dtype=np.float32),
        }
    )
    manager.modules[0].generate_offset = lambda *_: np.ones(3, dtype=np.int64)

    manager({"particles": particles, "meta": meta})

    np.testing.assert_allclose(particle.position, [1.5, 2.5, 3.5])
    assert particle.units == "cm"
