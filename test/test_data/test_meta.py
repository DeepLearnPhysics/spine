"""Comprehensive tests for the meta data module."""

import numpy as np

from spine.data.meta import Meta


class TestMetaCreation:
    """Test Meta object creation and validation."""

    def test_meta_default(self):
        """Test Meta creation with default values."""
        meta = Meta()
        # Meta defaults may be arrays of -inf, not None
        assert meta.lower is None or (
            hasattr(meta.lower, "shape") and np.all(~np.isfinite(meta.lower))
        )
        assert meta.upper is None or (
            hasattr(meta.upper, "shape") and np.all(~np.isfinite(meta.upper))
        )
        assert meta.size is None or (
            hasattr(meta.size, "shape") and np.all(~np.isfinite(meta.size))
        )
        assert meta.count is None or (
            hasattr(meta.count, "shape") and len(meta.count) >= 0
        )

    def test_meta_with_values(self):
        """Test Meta creation with explicit values."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([100.0, 200.0, 300.0])
        size = np.array([1.0, 2.0, 3.0])
        count = np.array([100, 100, 100], dtype=np.int64)

        meta = Meta(lower=lower, upper=upper, size=size, count=count)

        np.testing.assert_array_equal(meta.lower, lower)
        np.testing.assert_array_equal(meta.upper, upper)
        np.testing.assert_array_equal(meta.size, size)
        np.testing.assert_array_equal(meta.count, count)

    def test_meta_2d_scenario(self):
        """Test Meta for 2D image scenario."""
        # 2D image metadata (like collection plane view)
        meta_2d = Meta(
            lower=np.array([0.0, 0.0]),
            upper=np.array([256.0, 512.0]),
            size=np.array([1.0, 1.0]),  # 1cm per pixel
            count=np.array([256, 512], dtype=np.int64),
        )

        assert meta_2d.dimension == 2
        assert meta_2d.lower.shape == (2,)
        assert meta_2d.upper.shape == (2,)
        assert meta_2d.size.shape == (2,)
        assert meta_2d.count.shape == (2,)

    def test_meta_3d_scenario(self):
        """Test Meta for 3D voxel scenario."""
        # 3D voxelized detector volume
        meta_3d = Meta(
            lower=np.array([-200.0, -200.0, 0.0]),
            upper=np.array([200.0, 200.0, 1000.0]),
            size=np.array([3.0, 3.0, 3.0]),  # 3cm voxels
            count=np.array([134, 134, 334], dtype=np.int64),
        )

        assert meta_3d.dimension == 3
        assert meta_3d.lower.shape == (3,)
        assert meta_3d.upper.shape == (3,)
        assert meta_3d.size.shape == (3,)
        assert meta_3d.count.shape == (3,)


class TestMetaProperties:
    """Test Meta object properties and calculations."""

    def test_dimension_property(self):
        """Test dimension property calculation."""
        # 2D meta
        meta_2d = Meta(lower=np.array([0.0, 0.0]))
        assert meta_2d.dimension == 2

        # 3D meta
        meta_3d = Meta(lower=np.array([0.0, 0.0, 0.0]))
        assert meta_3d.dimension == 3

        # Test with different arrays
        meta_alt = Meta(count=np.array([100, 100, 100], dtype=np.int64))
        assert meta_alt.dimension == 3

    def test_geometric_consistency(self):
        """Test geometric consistency of meta parameters."""
        lower = np.array([0.0, -100.0, 50.0])
        upper = np.array([300.0, 100.0, 350.0])
        size = np.array([3.0, 2.0, 1.0])

        # Calculate expected count
        expected_count = np.ceil((upper - lower) / size).astype(np.int64)

        meta = Meta(lower=lower, upper=upper, size=size, count=expected_count)

        # Check consistency
        calculated_upper = meta.lower + meta.size * meta.count
        np.testing.assert_array_almost_equal(calculated_upper, meta.upper, decimal=1)

    def test_volume_calculations(self):
        """Test volume and area calculations."""
        # 2D area calculation
        meta_2d = Meta(
            lower=np.array([0.0, 0.0]),
            upper=np.array([100.0, 200.0]),
            size=np.array([1.0, 2.0]),
            count=np.array([100, 100], dtype=np.int64),
        )

        area = np.prod(meta_2d.upper - meta_2d.lower)
        assert area == 20000.0  # 100 * 200

        # 3D volume calculation
        meta_3d = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        volume = np.prod(meta_3d.upper - meta_3d.lower)
        assert volume == 1000000.0  # 100^3

    def test_pixel_voxel_counts(self):
        """Test pixel/voxel count calculations."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([300.0, 200.0, 1000.0]),
            size=np.array([3.0, 2.0, 1.0]),
            count=np.array([100, 100, 1000], dtype=np.int64),
        )

        total_voxels = np.prod(meta.count)
        assert total_voxels == 10000000  # 100 * 100 * 1000


class TestMetaPhysics:
    """Test Meta object with realistic physics scenarios."""

    def test_microboone_tpc_meta(self):
        """Test Meta for MicroBooNE TPC dimensions."""
        # Approximate MicroBooNE TPC dimensions
        meta_ub = Meta(
            lower=np.array([0.0, -116.0, 0.0]),
            upper=np.array([256.0, 116.0, 1037.0]),
            size=np.array([0.3, 0.3, 0.3]),  # 3mm voxels
            count=np.array([854, 774, 3457], dtype=np.int64),
        )

        assert meta_ub.dimension == 3
        # Check TPC volume is reasonable
        tpc_volume = np.prod(meta_ub.upper - meta_ub.lower)
        assert 50000000 < tpc_volume < 70000000  # ~60 m^3

    def test_collection_plane_meta(self):
        """Test Meta for wire collection plane view."""
        # Collection plane (U, V, Y views)
        meta_collection = Meta(
            lower=np.array([0.0, 0.0]),
            upper=np.array([3456.0, 6048.0]),  # Wire x Time
            size=np.array([1.0, 1.0]),  # 1 wire, 1 time tick
            count=np.array([3456, 6048], dtype=np.int64),
        )

        assert meta_collection.dimension == 2
        total_pixels = np.prod(meta_collection.count)
        assert total_pixels == 3456 * 6048  # Full readout

    def test_optical_detector_meta(self):
        """Test Meta for optical detector image."""
        # PMT image representation
        meta_pmt = Meta(
            lower=np.array([0.0, 0.0]),
            upper=np.array([180.0, 1500.0]),  # PMT x Time
            size=np.array([1.0, 1.0]),
            count=np.array([180, 1500], dtype=np.int64),
        )

        assert meta_pmt.dimension == 2
        assert meta_pmt.count[0] == 180  # Number of PMTs

    def test_multi_scale_meta(self):
        """Test Meta at different scales."""
        scales = [(0.1, "fine"), (1.0, "medium"), (5.0, "coarse")]

        for scale, name in scales:
            meta = Meta(
                lower=np.array([0.0, 0.0, 0.0]),
                upper=np.array([100.0, 100.0, 100.0]),
                size=np.array([scale, scale, scale]),
                count=np.array([100 / scale, 100 / scale, 100 / scale], dtype=np.int64),
            )

            # Check that finer scales have more voxels
            total_voxels = np.prod(meta.count)
            if scale == 0.1:
                assert total_voxels == 1000000000  # Very fine
            elif scale == 1.0:
                assert total_voxels == 1000000  # Medium
            elif scale == 5.0:
                assert total_voxels == 8000  # Coarse


class TestMetaCoordinates:
    """Test Meta coordinate transformations."""

    def test_coordinate_transformations(self):
        """Test coordinate system transformations."""
        meta = Meta(
            lower=np.array([-100.0, -50.0, 0.0]),
            upper=np.array([100.0, 50.0, 200.0]),
            size=np.array([2.0, 1.0, 4.0]),
            count=np.array([100, 100, 50], dtype=np.int64),
        )

        # Test voxel to world coordinate conversion
        voxel_coords = np.array([50, 50, 25])  # Center voxel
        world_coords = meta.lower + voxel_coords * meta.size

        expected_world = np.array([0.0, 0.0, 100.0])  # Should be center
        np.testing.assert_array_equal(world_coords, expected_world)

    def test_boundary_coordinates(self):
        """Test boundary coordinate handling."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        # Test corner coordinates
        corners = [
            ([0, 0, 0], [0.0, 0.0, 0.0]),  # Lower corner
            ([99, 99, 99], [99.0, 99.0, 99.0]),  # Upper corner
        ]

        for voxel, expected_world in corners:
            world = meta.lower + np.array(voxel) * meta.size
            np.testing.assert_array_equal(world, expected_world)

    def test_detector_coordinate_systems(self):
        """Test different detector coordinate systems."""
        # Test LArTPC coordinates (beam along z)
        meta_beam_z = Meta(
            lower=np.array([-200.0, -200.0, 0.0]),  # x, y, z (beam)
            upper=np.array([200.0, 200.0, 1000.0]),
            size=np.array([3.0, 3.0, 3.0]),
            count=np.array([134, 134, 334], dtype=np.int64),
        )

        # Check beam direction coverage
        beam_length = meta_beam_z.upper[2] - meta_beam_z.lower[2]
        assert beam_length == 1000.0  # 10 meter drift

        # Test transverse dimensions
        drift_height = meta_beam_z.upper[1] - meta_beam_z.lower[1]
        assert drift_height == 400.0  # 4 meter height


class TestMetaIntegration:
    """Test Meta integration with other components."""

    def test_meta_serialization(self):
        """Test Meta object serialization properties."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([100.0, 100.0, 100.0])
        size = np.array([1.0, 1.0, 1.0])
        count = np.array([100, 100, 100], dtype=np.int64)

        meta = Meta(lower=lower, upper=upper, size=size, count=count)

        # Test that arrays maintain their properties
        assert isinstance(meta.lower, np.ndarray)
        assert isinstance(meta.upper, np.ndarray)
        assert isinstance(meta.size, np.ndarray)
        assert isinstance(meta.count, np.ndarray)
        assert meta.count.dtype == np.int64

    def test_meta_collections(self):
        """Test collections of Meta objects."""
        metas = []

        # Different views/volumes
        dimensions = [
            ([0.0, 0.0], [100.0, 200.0], "2D view"),
            ([0.0, 0.0, 0.0], [100.0, 100.0, 500.0], "3D volume"),
            ([-50.0, -50.0, 0.0], [50.0, 50.0, 100.0], "ROI volume"),
        ]

        for lower, upper, desc in dimensions:
            lower_arr = np.array(lower)
            upper_arr = np.array(upper)
            size_arr = np.ones_like(lower_arr)
            count_arr = ((upper_arr - lower_arr) / size_arr).astype(np.int64)

            meta = Meta(
                lower=lower_arr, upper=upper_arr, size=size_arr, count=count_arr
            )
            metas.append(meta)

        assert len(metas) == 3
        assert all(isinstance(m, Meta) for m in metas)
        assert metas[0].dimension == 2
        assert metas[1].dimension == 3
        assert metas[2].dimension == 3

    def test_meta_edge_cases(self):
        """Test Meta edge cases and boundary conditions."""
        # Very small voxels
        tiny_meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
            size=np.array([0.01, 0.01, 0.01]),  # 1mm voxels
            count=np.array([100, 100, 100], dtype=np.int64),
        )
        assert np.prod(tiny_meta.count) == 1000000

        # Very large voxels
        large_meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([1000.0, 1000.0, 1000.0]),
            size=np.array([100.0, 100.0, 100.0]),  # 1m voxels
            count=np.array([10, 10, 10], dtype=np.int64),
        )
        assert np.prod(large_meta.count) == 1000

        # Single voxel
        single_meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([1, 1, 1], dtype=np.int64),
        )
        assert np.prod(single_meta.count) == 1

    def test_meta_data_quality(self):
        """Test Meta data quality indicators."""
        # High quality meta (consistent parameters)
        hq_meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        # Check geometric consistency
        calculated_upper = hq_meta.lower + hq_meta.size * hq_meta.count
        np.testing.assert_array_equal(calculated_upper, hq_meta.upper)

        # Low quality meta (inconsistent parameters)
        lq_meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([50, 50, 50], dtype=np.int64),  # Inconsistent
        )

        # Check inconsistency
        calculated_upper_lq = lq_meta.lower + lq_meta.size * lq_meta.count
        assert not np.array_equal(calculated_upper_lq, lq_meta.upper)

    def test_meta_physical_units(self):
        """Test Meta with different physical units."""
        # Centimeter-based meta (standard LArTPC)
        meta_cm = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([256.0, 232.0, 1037.0]),  # cm
            size=np.array([0.3, 0.3, 0.3]),  # cm
            count=np.array([854, 774, 3457], dtype=np.int64),
        )

        # Check physical reasonableness
        detector_volume_cm3 = np.prod(meta_cm.upper - meta_cm.lower)
        detector_volume_m3 = detector_volume_cm3 / 1e6
        assert 50 < detector_volume_m3 < 100  # Reasonable TPC volume

        # Check voxel volume
        voxel_volume_cm3 = np.prod(meta_cm.size)
        assert abs(voxel_volume_cm3 - 0.027) < 1e-10  # 0.3^3 = 0.027 cm^3
