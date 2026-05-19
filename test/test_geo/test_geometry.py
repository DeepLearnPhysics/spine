"""Tests for geometry query methods."""

import numpy as np
import pytest

from spine.geo.base import Geometry


@pytest.fixture
def simple_geo() -> Geometry:
    """Build a minimal two-TPC geometry."""
    return Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )


@pytest.fixture
def multi_module_geo() -> Geometry:
    """Build a two-module geometry with logical TPC IDs and active limits."""
    return Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [
                [-6.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [94.0, 0.0, 0.0],
                [106.0, 0.0, 0.0],
            ],
            "module_ids": [0, 0, 1, 1],
            "det_ids": [1, 0],
            "limits": {
                "intercepts": [[50.0, 0.0, 0.0]],
                "norms": [[1.0, 0.0, 0.0]],
            },
        },
    )


@pytest.fixture
def full_geo() -> Geometry:
    """Build a geometry with optical and CRT subsystems."""
    return Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
        optical={
            "volume": "module",
            "shape": "box",
            "dimensions": [2.0, 2.0, 2.0],
            "positions": [[0.0, 15.0, 0.0]],
        },
        crt={
            "dimensions": [[2.0, 2.0, 2.0]],
            "positions": [[0.0, 30.0, 0.0]],
            "normals": [1],
        },
    )


class MetaStub:
    """Minimal coordinate conversion stub for split tests."""

    def to_cm(self, points, center=False):
        """Pretend pixel coordinates are already centimeters."""
        return np.asarray(points, dtype=float)

    def to_px(self, points, floor=False):
        """Return converted pixel coordinates."""
        return np.floor(points).astype(int) if floor else points


def test_geometry_normalizes_version(simple_geo):
    """Geometry instances should store normalized version strings."""
    assert simple_geo.version == "1.0"


def test_parse_optical_rejects_unknown_volume(simple_geo):
    """Optical volume segmentation should be validated at runtime."""
    with pytest.raises(ValueError, match="TPC or module"):
        simple_geo.parse_optical("bad")


def test_geometry_initializes_optional_detectors(full_geo):
    """Optional detector systems should contribute to geometry boundaries."""
    assert full_geo.optical is not None
    assert full_geo.crt is not None

    np.testing.assert_allclose(full_geo.optical.volumes[0].centroid, [0.0, 0.0, 0.0])
    boundaries = full_geo.get_boundaries(with_optical=True, with_crt=True)
    np.testing.assert_allclose(boundaries[:, 1], [11.0, 31.0, 15.0])


def test_parse_optical_uses_tpc_offsets(simple_geo):
    """TPC-segmented optical volumes should be centered on TPC chambers."""
    optical = simple_geo.parse_optical(
        "tpc",
        shape="box",
        dimensions=[1.0, 1.0, 1.0],
        positions=[[0.0, 0.0, 0.0]],
    )

    assert optical.num_volumes == 2
    np.testing.assert_allclose(optical.volumes[0].centroid, [-6.0, 0.0, 0.0])
    np.testing.assert_allclose(optical.volumes[1].centroid, [6.0, 0.0, 0.0])


def test_get_boundaries_requires_requested_optional_detectors(simple_geo):
    """Optional detector boundaries should only be included when available."""
    with pytest.raises(ValueError, match="optical detectors"):
        simple_geo.get_boundaries(with_optical=True, with_crt=False)

    with pytest.raises(ValueError, match="CRT detectors"):
        simple_geo.get_boundaries(with_optical=False, with_crt=True)

    boundaries = simple_geo.get_boundaries(with_optical=False, with_crt=False)
    np.testing.assert_allclose(boundaries, simple_geo.tpc.boundaries)


def test_source_and_volume_queries(simple_geo):
    """Geometry should map source and volume indexes consistently."""
    sources = np.array([[0, 0], [0, 1], [0, 1]], dtype=np.int32)

    np.testing.assert_array_equal(simple_geo.get_sources(sources), sources)
    np.testing.assert_array_equal(simple_geo.get_chambers(sources), [0, 1, 1])
    contributors = simple_geo.get_contributors(sources)
    np.testing.assert_array_equal(contributors[0], [0, 0])
    np.testing.assert_array_equal(contributors[1], [0, 1])
    np.testing.assert_array_equal(simple_geo.get_volume_index(sources, 0), [0, 1, 2])
    np.testing.assert_array_equal(simple_geo.get_volume_index(sources, 0, 1), [1, 2])


def test_source_queries_apply_logical_tpc_mapping(multi_module_geo):
    """Logical TPC IDs should map to configured physical TPC IDs."""
    sources = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32)

    np.testing.assert_array_equal(
        multi_module_geo.get_sources(sources),
        [[0, 1], [0, 0], [1, 0]],
    )
    np.testing.assert_array_equal(multi_module_geo.get_chambers(sources), [1, 0, 2])
    contributors = multi_module_geo.get_contributors(sources)
    np.testing.assert_array_equal(contributors[0], [0, 0, 1])
    np.testing.assert_array_equal(contributors[1], [0, 1, 0])
    np.testing.assert_array_equal(multi_module_geo.get_volume_index(sources, 0, 0), [1])


def test_closest_volume_and_translate(simple_geo):
    """Closest-volume helpers and translation should use detector centers."""
    points = np.array([[-7.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32)

    np.testing.assert_array_equal(simple_geo.get_closest_tpc(points), [0, 1])
    np.testing.assert_array_equal(simple_geo.get_closest_module(points), [0, 0])
    indexes = simple_geo.get_closest_tpc_indexes(points)
    np.testing.assert_array_equal(indexes[0], [0])
    np.testing.assert_array_equal(indexes[1], [1])
    translated = simple_geo.translate(points, source_id=0, target_id=0)
    np.testing.assert_array_equal(translated, points)


def test_volume_offsets_and_inter_module_translation(multi_module_geo):
    """Volume offsets and translations should respect module and TPC bounds."""
    points = np.array([[120.0, 0.0, 0.0], [100.0, 30.0, 0.0]])

    offsets = multi_module_geo.get_volume_offsets(points, module_id=1)
    np.testing.assert_allclose(offsets, [[9.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    np.testing.assert_allclose(
        multi_module_geo.get_min_volume_offset(points, module_id=1),
        [9.0, 20.0, 0.0],
    )
    np.testing.assert_allclose(
        multi_module_geo.get_volume_offsets(points[:1], module_id=1, tpc_id=1),
        [[9.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        multi_module_geo.translate(points[:1], source_id=1, target_id=0, factor=0.5),
        [[70.0, 0.0, 0.0]],
    )


def test_split_points_by_module(multi_module_geo):
    """Split should migrate non-target module points to the target module."""
    points = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    sources = np.array([[0, 0], [1, 0]])

    shifted, indexes = multi_module_geo.split(
        points.copy(), target_id=0, sources=sources
    )
    np.testing.assert_allclose(shifted, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(indexes[0], [0])
    np.testing.assert_array_equal(indexes[1], [1])

    inferred, inferred_indexes = multi_module_geo.split(points.copy(), target_id=0)
    np.testing.assert_allclose(inferred, shifted)
    np.testing.assert_array_equal(inferred_indexes[0], [0])
    np.testing.assert_array_equal(inferred_indexes[1], [1])

    pixel, _ = multi_module_geo.split(points.copy(), target_id=0, meta=MetaStub())
    np.testing.assert_array_equal(pixel, shifted.astype(int))

    with pytest.raises(ValueError, match="Target ID"):
        multi_module_geo.split(points.copy(), target_id=2)


def test_containment_volume_definition_and_check(simple_geo):
    """Containment definitions should support summary and per-point checks."""
    points = np.array([[-6.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
    definition = simple_geo.define_containment_volumes(
        margin=0.0, mode="module", include_limits=False
    )

    assert simple_geo.check_containment(definition, points[:1]) is True
    assert simple_geo.check_containment(definition, points) is False
    np.testing.assert_array_equal(
        simple_geo.check_containment(definition, points, summarize=False),
        [True, False],
    )


def test_containment_rejects_unknown_mode(simple_geo):
    """Containment definitions should reject unknown modes."""
    with pytest.raises(ValueError, match="mode not recognized"):
        simple_geo.define_containment_volumes(margin=0.0, mode="bad")


def test_containment_modes_and_source_checks(multi_module_geo):
    """Containment should support detector, TPC, source and active-limit modes."""
    points = np.array([[-5.0, 0.0, 0.0], [106.0, 0.0, 0.0]])
    sources = np.array([[0, 1], [1, 0]])

    detector_definition = multi_module_geo.define_containment_volumes(
        margin=[0.0, 0.0, 0.0], mode="detector", include_limits=False
    )
    assert multi_module_geo.check_containment(detector_definition, points) is True

    tpc_definition = multi_module_geo.define_containment_volumes(
        margin=0.0, mode="tpc", include_limits=False
    )
    assert multi_module_geo.check_containment(tpc_definition, points) is False

    source_definition = multi_module_geo.define_containment_volumes(
        margin=0.0, cathode_margin=1.0, mode="source", include_limits=False
    )
    with pytest.raises(ValueError, match="provide sources"):
        multi_module_geo.check_containment(source_definition, points)
    assert (
        multi_module_geo.check_containment(
            source_definition, points, sources=sources, allow_multi_module=False
        )
        is False
    )
    assert (
        multi_module_geo.check_containment(
            source_definition, points, sources=sources, allow_multi_module=True
        )
        is True
    )

    limited_definition = multi_module_geo.define_containment_volumes(
        margin=0.0, mode="detector"
    )
    assert (
        multi_module_geo.check_containment(
            limited_definition,
            np.array([[0.0, 0.0, 0.0]]),
        )
        is True
    )
    np.testing.assert_array_equal(
        multi_module_geo.check_containment(
            limited_definition,
            np.array([[0.0, 0.0, 0.0], [75.0, 0.0, 0.0]]),
            summarize=False,
        ),
        [True, False],
    )


def test_containment_validation_and_volume_merge(multi_module_geo):
    """Containment helpers should validate margins and adapt cathode walls."""
    with pytest.raises(ValueError, match="one value per axis"):
        multi_module_geo.define_containment_volumes([1.0, 2.0], include_limits=False)

    with pytest.raises(ValueError, match="two values per axis"):
        multi_module_geo.define_containment_volumes(
            np.ones((2, 2)), include_limits=False
        )

    with pytest.raises(ValueError, match="No clear way"):
        multi_module_geo.define_containment_volumes([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Module and TPC ID"):
        multi_module_geo.adapt_volume(
            multi_module_geo.tpc[0][0].boundaries,
            np.zeros((3, 2)),
            cathode_margin=1.0,
        )

    adapted = multi_module_geo.adapt_volume(
        multi_module_geo.tpc[0][0].boundaries,
        np.ones((3, 2)),
        cathode_margin=3.0,
        module_id=0,
        tpc_id=0,
    )
    np.testing.assert_allclose(adapted[0], [-10.0, -4.0])

    merged = multi_module_geo.merge_volumes(
        np.array(
            [
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
                [[-1.0, 0.5], [-2.0, 1.0], [-3.0, 2.0]],
            ]
        )
    )
    np.testing.assert_allclose(merged, [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
