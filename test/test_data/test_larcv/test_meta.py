"""Comprehensive tests for the meta data module."""

import numpy as np
import pytest

from spine.data import Meta
from spine.data.larcv import meta
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.utils.conditional import LARCV_AVAILABLE, larcv


class TestMetaCreation:
    """Test Meta object creation and validation."""

    def test_meta_default(self):
        """Test Meta creation with default values."""
        meta = Meta()

        # Meta should be initialized with NaN or -1 values, not None
        assert np.all(np.isnan(meta.lower))
        assert np.all(np.isnan(meta.upper))
        assert np.all(np.isnan(meta.size))
        assert np.all(meta.count == -1)

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
        """Test ImageMeta2D for 2D image scenario."""
        # 2D image metadata (like collection plane view)
        meta_2d = ImageMeta2D(
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
        """Test ImageMeta3D for 3D voxel scenario."""
        # 3D voxelized detector volume
        meta_3d = ImageMeta3D(
            lower=np.array([-200.0, -200.0, 0.0]),
            upper=np.array([202.0, 202.0, 1002.0]),
            size=np.array([3.0, 3.0, 3.0]),  # 3cm voxels
            count=np.array([134, 134, 334], dtype=np.int64),
        )

        assert meta_3d.dimension == 3
        assert meta_3d.lower.shape == (3,)
        assert meta_3d.upper.shape == (3,)
        assert meta_3d.size.shape == (3,)
        assert meta_3d.count.shape == (3,)

    def test_meta_data_quality(self):
        """Test Meta data quality indicators."""
        # Underconstrained meta (missing count)
        with pytest.raises(ValueError):
            Meta(
                lower=np.array([0.0, 0.0, 0.0]),
                upper=np.array([100.0, 100.0, 100.0]),
                size=np.array([1.0, 1.0, 1.0]),
            )

        # Wrong shape meta (size and count must match lower/upper)
        with pytest.raises(ValueError):
            ImageMeta2D(
                lower=np.array([0.0, 0.0]),
                upper=np.array([100.0, 100.0]),
                size=np.array([1.0, 1.0, 1.0]),  # Wrong shape
                count=np.array([100, 100], dtype=np.int64),
            )
        with pytest.raises(ValueError):
            ImageMeta3D(
                lower=np.array([0.0, 0.0, 0.0]),
                upper=np.array([100.0, 100.0, 100.0]),
                size=np.array([1.0, 1.0, 1.0]),
                count=np.array([100, 100], dtype=np.int64),  # Wrong shape
            )

        # Badly constrained meta (size and count don't match upper/lower)
        with pytest.raises(ValueError):
            Meta(
                lower=np.array([0.0, 0.0, 0.0]),
                upper=np.array([100.0, 100.0, 100.0]),
                size=np.array([2.0, 2.0, 2.0]),
                count=np.array(
                    [10, 10, 10], dtype=np.int64
                ),  # Should be 50 to match upper/lower
            )

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


class TestMetaProperties:
    """Test Meta object properties and calculations."""

    def test_meta_index_multipliers(self):
        """Test that index multipliers are computed correctly."""
        # Proper 3D meta
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([10.0, 20.0, 30.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([10, 20, 30], dtype=np.int64),
        )

        expected_multipliers = np.array([600, 30, 1], dtype=np.int64)  # [20*30, 30, 1]
        assert np.array_equal(meta.index_multipliers, expected_multipliers)

        # Uninitialized meta should raise error when accessing index multipliers
        uninit_meta = Meta()
        with pytest.raises(ValueError):
            _ = uninit_meta.index_multipliers

    def test_dimension_property(self):
        """Test dimension property calculation."""
        # 2D meta
        meta_2d = ImageMeta2D(
            lower=np.array([0.0, 0.0]),
            upper=np.array([100.0, 100.0]),
            size=np.array([1.0, 1.0]),
            count=np.array([100, 100], dtype=np.int64),
        )
        assert meta_2d.dimension == 2

        # 3D meta
        meta_3d = ImageMeta3D(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )
        assert meta_3d.dimension == 3

    def test_num_elements_property(self):
        """Test num_elements property calculation."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([10.0, 20.0, 30.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([10, 20, 30], dtype=np.int64),
        )
        assert meta.num_elements == 6000  # 10 * 20 * 30

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
        meta_2d = ImageMeta2D(
            lower=np.array([0.0, 0.0]),
            upper=np.array([100.0, 200.0]),
            size=np.array([1.0, 2.0]),
            count=np.array([100, 100], dtype=np.int64),
        )

        area = np.prod(meta_2d.upper - meta_2d.lower)
        assert area == 20000.0  # 100 * 200

        # 3D volume calculation
        meta_3d = ImageMeta3D(
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


class TestMetaCoordinates:
    """Test Meta coordinate transformations."""

    def test_index(self):
        """Test unique global index generation from coordinates."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 200.0, 300.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 200, 300], dtype=np.int64),
        )

        coords = np.array([[10, 20, 30], [99, 199, 299]], dtype=int)
        indices = meta.index(coords)

        expected_indices = (
            coords[:, 0] * (meta.count[1] * meta.count[2])
            + coords[:, 1] * meta.count[2]
            + coords.astype(int)[:, 2]
        )
        assert np.all(indices == expected_indices)

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

    def test_to_cm(self):
        """Test conversion to cm coordinates."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 200.0, 300.0]),
            size=np.array([1.0, 2.0, 3.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        pixel_coords = np.array([[10, 20, 30], [50, 50, 50]], dtype=int)
        cm_coords = meta.to_cm(pixel_coords)

        expected_cm = meta.lower + pixel_coords * meta.size
        np.testing.assert_array_equal(cm_coords, expected_cm)

        cm_coords_center = meta.to_cm(pixel_coords, center=True)

        expected_cm_center = meta.lower + pixel_coords * meta.size + meta.size / 2
        np.testing.assert_array_equal(cm_coords_center, expected_cm_center)

    def test_to_px(self):
        """Test conversion to pixel coordinates."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 200.0, 300.0]),
            size=np.array([1.0, 2.0, 3.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        cm_coords = np.array([[13.7, 22.5, 33.33], [50.0, 50.0, 50.0]])
        pixel_coords = meta.to_px(cm_coords)

        expected_pixels = (cm_coords - meta.lower) / meta.size
        np.testing.assert_array_equal(pixel_coords, expected_pixels)

        pixel_coords_floor = meta.to_px(cm_coords, floor=True)

        expected_pixels_floor = np.floor(expected_pixels)
        np.testing.assert_array_equal(pixel_coords_floor, expected_pixels_floor)

    def test_inner_mask(self):
        """Test inner_mask method for coordinate validity."""
        meta = Meta(
            lower=np.array([0.0, 0.0, 0.0]),
            upper=np.array([100.0, 100.0, 100.0]),
            size=np.array([1.0, 1.0, 1.0]),
            count=np.array([100, 100, 100], dtype=np.int64),
        )

        coords = np.array(
            [
                [10, 10, 10],  # Inside
                [99.9, 99.9, 99.9],  # Just inside
                [100, 100, 100],  # On the upper boundary (should be outside)
                [-1, -1, -1],  # Outside (negative)
                [50, 50, 150],  # Outside (z too high)
            ]
        )

        mask = meta.inner_mask(coords)

        expected_mask = np.array([True, True, False, False, False])
        np.testing.assert_array_equal(mask, expected_mask)


class TestMetaFromLArCV:
    """Tests for Meta.from_larcv() - only runs if larcv is available."""

    def test_from_larcv_mock_3d(self):
        """Test from_larcv with mock 3D Voxel3DMeta object (runs even without larcv)."""

        # Create a mock larcv Voxel3DMeta object
        class MockLArCVVoxel3DMeta:
            """Mock LArCV Voxel3DMeta for testing."""

            def pos_z(self):
                """Marker method to indicate 3D metadata."""
                return True

            def min_x(self):
                return 0.0

            def min_y(self):
                return -100.0

            def min_z(self):
                return 50.0

            def max_x(self):
                return 300.0

            def max_y(self):
                return 100.0

            def max_z(self):
                return 350.0

            def size_voxel_x(self):
                return 3.0

            def size_voxel_y(self):
                return 2.0

            def size_voxel_z(self):
                return 1.0

            def num_voxel_x(self):
                return 100

            def num_voxel_y(self):
                return 100

            def num_voxel_z(self):
                return 300

        mock_meta = MockLArCVVoxel3DMeta()
        meta = Meta.from_larcv(mock_meta)

        # Verify all attributes transferred correctly
        np.testing.assert_array_almost_equal(meta.lower, [0.0, -100.0, 50.0])
        np.testing.assert_array_almost_equal(meta.upper, [300.0, 100.0, 350.0])
        np.testing.assert_array_almost_equal(meta.size, [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(meta.count, [100, 100, 300])
        assert meta.dimension == 3

    def test_from_larcv_mock_2d(self):
        """Test from_larcv with mock 2D ImageMeta object (runs even without larcv)."""

        # Create a mock larcv ImageMeta object
        class MockLArCVImageMeta:
            """Mock LArCV ImageMeta for testing."""

            def min_x(self):
                return 0.0

            def min_y(self):
                return 0.0

            def max_x(self):
                return 3456.0

            def max_y(self):
                return 6048.0

            def pixel_height(self):
                return 1.0

            def pixel_width(self):
                return 1.0

            def cols(self):
                return 3456

            def rows(self):
                return 6048

        mock_meta = MockLArCVImageMeta()
        meta = Meta.from_larcv(mock_meta)

        # Verify all attributes transferred correctly
        np.testing.assert_array_almost_equal(meta.lower, [0.0, 0.0])
        np.testing.assert_array_almost_equal(meta.upper, [3456.0, 6048.0])
        np.testing.assert_array_almost_equal(meta.size, [1.0, 1.0])
        np.testing.assert_array_equal(meta.count, [3456, 6048])
        assert meta.dimension == 2

    @pytest.mark.skipif(not LARCV_AVAILABLE, reason="larcv not available")
    def test_from_larcv_real_2d(self):
        """Test from_larcv with real larcv ImageMeta (only if larcv installed)."""
        assert larcv is not None

        # Create a real LArCV ImageMeta
        larcv_meta = larcv.ImageMeta(
            x_min=0.0,
            y_min=0.0,
            x_max=512.0,
            y_max=256.0,
            y_row_count=256,
            x_column_count=512,
        )

        # Convert to SPINE Meta
        meta = Meta.from_larcv(larcv_meta)

        # Verify conversion
        np.testing.assert_array_almost_equal(meta.lower, [0.0, 0.0])
        np.testing.assert_array_almost_equal(meta.upper, [512.0, 256.0])
        np.testing.assert_array_equal(meta.count, [512, 256])
        assert meta.dimension == 2

        # Verify size calculation
        expected_size = (meta.upper - meta.lower) / meta.count
        np.testing.assert_array_almost_equal(meta.size, expected_size, decimal=5)

    @pytest.mark.skipif(not LARCV_AVAILABLE, reason="larcv not available")
    def test_from_larcv_real_3d(self):
        """Test from_larcv with real larcv Voxel3DMeta (only if larcv installed)."""
        assert larcv is not None

        # Create a real LArCV Voxel3DMeta
        larcv_meta = larcv.Voxel3DMeta()
        larcv_meta.set(
            xmin=0.0,
            ymin=-116.0,
            zmin=0.0,
            xmax=256.0,
            ymax=116.0,
            zmax=1037.0,
            xnum=100,
            ynum=100,
            znum=1000,
        )

        # Convert to SPINE Meta
        meta = Meta.from_larcv(larcv_meta)

        # Verify conversion
        np.testing.assert_array_almost_equal(meta.lower, [0.0, -116.0, 0.0])
        np.testing.assert_array_almost_equal(meta.upper, [256.0, 116.0, 1037.0])
        np.testing.assert_array_equal(meta.count, [100, 100, 1000])
        assert meta.dimension == 3

        # Verify size calculation is reasonable
        expected_size = (meta.upper - meta.lower) / meta.count
        np.testing.assert_array_almost_equal(meta.size, expected_size, decimal=5)
