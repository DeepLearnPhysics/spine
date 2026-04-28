"""Tests for base class functionality in spine.data.out.base module."""

import numpy as np


class TestOutBase:
    """Test OutBase functionality shared by all output classes."""

    def test_outbase_initialization(self):
        """Test OutBase initialization with default values."""
        from spine.data.out.base import OutBase

        # OutBase can be instantiated directly as it's a concrete dataclass
        obj = OutBase(id=5)

        # Test index attributes
        assert obj.id == 5

        # Test scalar attributes defaults
        assert obj.is_contained is False
        assert obj.is_time_contained is False
        assert obj.is_cathode_crosser is False
        assert obj.is_matched is False
        assert np.isnan(obj.cathode_offset)

        # Test units
        assert obj.units == "cm"

        # Test vector attributes defaults
        assert len(obj.index) == 0
        assert obj.index.dtype == np.int32
        assert obj.points.shape == (0, 3)
        assert obj.points.dtype == np.float32
        assert len(obj.depositions) == 0
        assert obj.depositions.dtype == np.float32
        assert obj.sources.shape == (0, 2)
        assert obj.sources.dtype == np.int32
        assert len(obj.match_ids) == 0
        assert obj.match_ids.dtype == np.int32
        assert len(obj.match_overlaps) == 0
        assert obj.match_overlaps.dtype == np.float32

    def test_outbase_with_voxel_data(self):
        """Test OutBase with voxel data."""
        from spine.data.out.base import OutBase

        # Create OutBase instance with voxel data
        index = np.array([10, 20, 30, 40], dtype=np.int32)
        points = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        depositions = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        sources = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)

        obj = OutBase(
            id=0,
            index=index,
            points=points,
            depositions=depositions,
            sources=sources,
        )

        # Verify data stored correctly
        np.testing.assert_array_equal(obj.index, index)
        np.testing.assert_allclose(obj.points, points)
        np.testing.assert_allclose(obj.depositions, depositions)
        np.testing.assert_array_equal(obj.sources, sources)

    def test_outbase_size_property(self):
        """Test OutBase size derived property."""
        from spine.data.out.base import OutBase

        # Empty object
        obj1 = OutBase(id=0)
        assert obj1.size == 0

        # Object with voxels
        obj2 = OutBase(
            id=1,
            index=np.array([10, 20, 30], dtype=np.int32),
        )
        assert obj2.size == 3

    def test_outbase_depositions_sum_property(self):
        """Test OutBase depositions_sum derived property."""
        from spine.data.out.base import OutBase

        # Empty object
        obj1 = OutBase(id=0)
        assert obj1.depositions_sum == 0.0

        # Object with depositions
        depositions = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        obj2 = OutBase(
            id=1,
            depositions=depositions,
        )
        expected_sum = np.sum(depositions)
        assert abs(obj2.depositions_sum - expected_sum) < 1e-6

    def test_outbase_module_ids_property(self):
        """Test OutBase module_ids derived property."""
        from spine.data.out.base import OutBase

        # Empty object
        obj1 = OutBase(id=0)
        assert len(obj1.module_ids) == 0

        # Object with sources from multiple modules
        sources = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2]], dtype=np.int32)
        obj2 = OutBase(
            id=1,
            sources=sources,
        )
        unique_modules = np.unique(sources[:, 0])
        np.testing.assert_array_equal(obj2.module_ids, unique_modules)
        assert len(obj2.module_ids) == 2  # Modules 0 and 1

    def test_outbase_reset_match(self):
        """Test OutBase reset_match method."""
        from spine.data.out.base import OutBase

        # Create object with matching information
        obj = OutBase(
            id=0,
            is_matched=True,
            match_ids=np.array([5, 7], dtype=np.int64),
            match_overlaps=np.array([0.8, 0.3], dtype=np.float32),
        )

        # Verify initial state
        assert obj.is_matched is True
        assert len(obj.match_ids) == 2

        # Reset matching
        obj.reset_match()

        # Verify reset
        assert obj.is_matched is False
        assert len(obj.match_ids) == 0

    def test_outbase_reset_cathode_crosser(self):
        """Test OutBase reset_cathode_crosser method."""
        from spine.data.out.base import OutBase

        # Create object with cathode crossing information
        obj = OutBase(
            id=0,
            is_cathode_crosser=True,
            cathode_offset=15.5,
        )

        # Verify initial state
        assert obj.is_cathode_crosser is True
        assert obj.cathode_offset == 15.5

        # Reset cathode crossing
        obj.reset_cathode_crosser()

        # Verify reset
        assert obj.is_cathode_crosser is False
        assert np.isnan(obj.cathode_offset)

    def test_outbase_matching_attributes(self):
        """Test OutBase matching attributes."""
        from spine.data.out.base import OutBase

        # Create matched object
        obj = OutBase(
            id=0,
            is_matched=True,
            match_ids=np.array([5, 7, 9], dtype=np.int64),
            match_overlaps=np.array([0.95, 0.85, 0.75], dtype=np.float32),
        )

        assert obj.is_matched is True
        assert len(obj.match_ids) == 3
        assert len(obj.match_overlaps) == 3
        np.testing.assert_array_equal(obj.match_ids, [5, 7, 9])
        np.testing.assert_array_almost_equal(obj.match_overlaps, [0.95, 0.85, 0.75])


class TestRecoBase:
    """Test RecoBase mixin functionality."""

    def test_recobase_is_truth_flag(self):
        """Test RecoBase is_truth attribute."""
        from spine.data.out import RecoFragment, RecoInteraction, RecoParticle

        # Test across all reco types
        fragment = RecoFragment(id=0)
        particle = RecoParticle(id=1)
        interaction = RecoInteraction(id=2)

        assert fragment.is_truth is False
        assert particle.is_truth is False
        assert interaction.is_truth is False


class TestTruthBase:
    """Test TruthBase mixin functionality."""

    def test_truthbase_is_truth_flag(self):
        """Test TruthBase is_truth attribute."""
        from spine.data.out import TruthFragment, TruthInteraction, TruthParticle

        # Test across all truth types
        fragment = TruthFragment(id=0)
        particle = TruthParticle(id=1)
        interaction = TruthInteraction(id=2)

        assert fragment.is_truth is True
        assert particle.is_truth is True
        assert interaction.is_truth is True

    def test_truthbase_initialization(self):
        """Test TruthBase initialization with default values."""
        from spine.data.out import TruthFragment

        obj = TruthFragment(id=5)

        # Test scalar attributes
        assert obj.orig_id == -1
        assert obj.is_truth is True

        # Test vector attributes defaults
        assert len(obj.index_adapt) == 0
        assert obj.index_adapt.dtype == np.int32
        assert len(obj.index_g4) == 0
        assert obj.index_g4.dtype == np.int32

        assert obj.points_adapt.shape == (0, 3)
        assert obj.points_adapt.dtype == np.float32
        assert obj.points_g4.shape == (0, 3)
        assert obj.points_g4.dtype == np.float32

        assert len(obj.depositions_q) == 0
        assert obj.depositions_q.dtype == np.float32
        assert len(obj.depositions_adapt) == 0
        assert obj.depositions_adapt.dtype == np.float32
        assert len(obj.depositions_adapt_q) == 0
        assert obj.depositions_adapt_q.dtype == np.float32
        assert len(obj.depositions_g4) == 0
        assert obj.depositions_g4.dtype == np.float32

        assert obj.sources_adapt.shape == (0, 2)
        assert obj.sources_adapt.dtype == np.int32

    def test_truthbase_size_adapt_property(self):
        """Test TruthBase size_adapt derived property."""
        from spine.data.out import TruthFragment

        # Empty object
        obj1 = TruthFragment(id=0)
        assert obj1.size_adapt == 0

        # Object with adapted voxels
        obj2 = TruthFragment(
            id=1,
            index_adapt=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        )
        assert obj2.size_adapt == 5

    def test_truthbase_size_g4_property(self):
        """Test TruthBase size_g4 derived property."""
        from spine.data.out import TruthFragment

        # Empty object
        obj1 = TruthFragment(id=0)
        assert obj1.size_g4 == 0

        # Object with g4 voxels
        obj2 = TruthFragment(
            id=1,
            index_g4=np.array([5, 15, 25], dtype=np.int64),
        )
        assert obj2.size_g4 == 3

    def test_truthbase_depositions_sums(self):
        """Test TruthBase depositions sum properties."""
        from spine.data.out import TruthFragment

        # Empty object
        obj1 = TruthFragment(id=0)
        assert obj1.depositions_q_sum == 0.0
        assert obj1.depositions_adapt_sum == 0.0
        assert obj1.depositions_adapt_q_sum == 0.0
        assert obj1.depositions_g4_sum == 0.0

        # Object with depositions
        dep_q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dep_adapt = np.array([1.5, 2.5], dtype=np.float32)
        dep_adapt_q = np.array([1.2, 2.2], dtype=np.float32)
        dep_g4 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        obj2 = TruthFragment(
            id=1,
            depositions_q=dep_q,
            depositions_adapt=dep_adapt,
            depositions_adapt_q=dep_adapt_q,
            depositions_g4=dep_g4,
        )

        assert abs(obj2.depositions_q_sum - np.sum(dep_q)) < 1e-6
        assert abs(obj2.depositions_adapt_sum - np.sum(dep_adapt)) < 1e-6
        assert abs(obj2.depositions_adapt_q_sum - np.sum(dep_adapt_q)) < 1e-6
        assert abs(obj2.depositions_g4_sum - np.sum(dep_g4)) < 1e-6
