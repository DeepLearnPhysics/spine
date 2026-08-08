"""Tests for whole-image and labeled-object sample construction."""

import numpy as np
import pytest

from spine.data import IndexBatch, TensorBatch
from spine.model.image.object import ImageObjectBuilder


def test_whole_image_builder_returns_one_object_per_entry(image_data):
    """Whole-image mode should preserve original event ownership."""
    objects = ImageObjectBuilder()(image_data)

    assert objects.counts.tolist() == [1, 1]
    assert objects.single_counts.tolist() == [4, 3]
    assert objects.index_list[0].tolist() == [0, 1, 2, 3]
    assert objects.index_list[1].tolist() == [4, 5, 6]


@pytest.mark.parametrize(
    ("source", "expected_counts", "expected_sizes"),
    [
        ("cluster", [2, 2], [2, 2, 2, 1]),
        ("group", [1, 2], [4, 2, 1]),
        ("ancestor", [2, 2], [2, 2, 2, 1]),
    ],
)
def test_labeled_object_builder_matches_grappa_sources(
    image_data,
    source,
    expected_counts,
    expected_sizes,
):
    """Cluster and group sources should use GrapPA-compatible semantics."""
    objects = ImageObjectBuilder(source=source)(image_data, object_data=image_data)

    assert objects.counts.tolist() == expected_counts
    assert objects.single_counts.tolist() == expected_sizes


def test_explicit_objects_override_configured_truth_source(image_data):
    """Explicit reconstructed indexes must bypass label-based construction."""
    objects = IndexBatch(
        [np.array([0, 2]), np.array([4, 6])],
        spans=np.array([4, 3]),
        counts=np.array([1, 1]),
        single_counts=np.array([2, 2]),
    )

    result = ImageObjectBuilder(source="cluster")(image_data, objects=objects)

    assert result is objects


def test_explicit_source_requires_indexes(image_data):
    """Explicit-only operation should fail before attempting truth access."""
    with pytest.raises(ValueError, match="requires `objects`"):
        ImageObjectBuilder(source="explicit")(image_data)


@pytest.mark.parametrize("source", ["image", "explicit"])
def test_direct_sources_reject_unused_shape_filters(source):
    """Direct object sources should reject shape filters they cannot apply."""
    with pytest.raises(ValueError, match="cannot be filtered by shape"):
        ImageObjectBuilder(source=source, shapes=["shower"])


def test_explicit_objects_cannot_cross_event_boundaries(image_data):
    """Object ownership metadata must agree with every referenced voxel."""
    objects = IndexBatch(
        [np.array([0, 4]), np.array([5, 6])],
        spans=np.array([4, 3]),
        counts=np.array([1, 1]),
        single_counts=np.array([2, 2]),
    )

    with pytest.raises(IndexError, match="owning batch entry"):
        ImageObjectBuilder()(image_data, objects=objects)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"min_size": -2}, ValueError, "at least -1"),
        ({"source": 3}, TypeError, "named field"),
        ({"shapes": "track"}, ValueError, "sequence"),
        ({"shapes": 1}, ValueError, "sequence"),
    ],
)
def test_object_builder_validates_configuration(kwargs, error, message):
    """Malformed source and selection policies fail during construction."""
    with pytest.raises(error, match=message):
        ImageObjectBuilder(**kwargs)


def test_labeled_sources_require_aligned_truth(image_data):
    """Truth-derived objects require voxel-aligned structured labels."""
    builder = ImageObjectBuilder(source="cluster")
    with pytest.raises(ValueError, match="requires `object_data`"):
        builder(image_data.data)

    shorter = TensorBatch(
        image_data.data.tensor[:-1],
        counts=[4, 2],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    with pytest.raises(ValueError, match="voxel-aligned"):
        builder(shorter, object_data=image_data)


def test_explicit_object_metadata_is_fully_validated(image_data):
    """Object indexes must be lists with matching batch ownership metadata."""
    flat = IndexBatch(np.array([0, 1]), spans=[2], counts=[2])
    with pytest.raises(TypeError, match="index lists"):
        ImageObjectBuilder()(image_data.data, objects=flat)

    wrong_batch = IndexBatch([np.array([0])], spans=[7], counts=[1], single_counts=[1])
    with pytest.raises(ValueError, match="batch sizes"):
        ImageObjectBuilder()(image_data.data, objects=wrong_batch)

    wrong_spans = IndexBatch(
        [np.array([0]), np.array([4])],
        spans=[3, 4],
        counts=[1, 1],
        single_counts=[1, 1],
    )
    with pytest.raises(ValueError, match="spans"):
        ImageObjectBuilder()(image_data.data, objects=wrong_spans)

    empty = IndexBatch(
        [np.array([], dtype=np.int64), np.array([4])],
        spans=[4, 3],
        counts=[1, 1],
        single_counts=[0, 1],
    )
    with pytest.raises(ValueError, match="empty indexes"):
        ImageObjectBuilder()(image_data.data, objects=empty)


def test_labeled_source_narrowing_guard(image_data):
    """A corrupted labeled-source policy fails before cluster construction."""
    builder = ImageObjectBuilder(source="cluster")
    builder.source = None
    with pytest.raises(RuntimeError, match="not initialized"):
        builder(image_data.data, object_data=image_data)
