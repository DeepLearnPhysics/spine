"""Test Fragment classes."""

import numpy as np


class TestFragmentBase:
    """Test FragmentBase functionality."""

    def test_fragmentbase_initialization(self):
        """Test FragmentBase initialization with default values."""
        from spine.data.out.fragment import FragmentBase

        obj = FragmentBase(id=0)

        # Test scalar attributes
        assert obj.particle_id == -1
        assert obj.interaction_id == -1
        assert obj.is_primary is False
        assert np.isnan(obj.length)
        assert obj.shape == -1

        # Test vector attributes
        assert obj.start_point.shape == (3,)
        assert obj.start_point.dtype == np.float32
        assert all(np.isnan(obj.start_point))

        assert obj.end_point.shape == (3,)
        assert obj.end_point.dtype == np.float32
        assert all(np.isnan(obj.end_point))

    def test_fragmentbase_with_data(self):
        """Test FragmentBase with complete data."""
        from spine.data.out.fragment import FragmentBase

        start = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        end = np.array([50.0, 60.0, 70.0], dtype=np.float32)

        obj = FragmentBase(
            id=5,
            particle_id=10,
            interaction_id=2,
            shape=1,  # track
            is_primary=True,
            length=69.28,
            start_point=start,
            end_point=end,
        )

        assert obj.id == 5
        assert obj.particle_id == 10
        assert obj.interaction_id == 2
        assert obj.shape == 1
        assert obj.is_primary is True
        assert abs(obj.length - 69.28) < 0.01
        np.testing.assert_allclose(obj.start_point, start)
        np.testing.assert_allclose(obj.end_point, end)

    def test_fragmentbase_str_representation(self):
        """Test FragmentBase string representation."""
        from spine.data.out.fragment import FragmentBase

        obj = FragmentBase(
            id=5,
            shape=1,  # track
            is_primary=True,
            match_ids=np.array([10], dtype=np.int64),
        )

        str_repr = str(obj)
        assert "Fragment" in str_repr
        assert "5" in str_repr  # ID
        assert "track" in str_repr.lower()
        # Primary is printed as boolean (True or 1)
        assert "True" in str_repr or "1" in str_repr


class TestFragmentCreation:
    """Test Fragment creation and basic properties."""

    def test_fragment_base_creation(self):
        """Test basic Fragment properties."""
        from spine.data.out import RecoFragment

        # Test basic fragment creation
        fragment = RecoFragment(
            id=0,
            particle_id=5,
            interaction_id=2,
            shape=1,
            is_primary=True,
            length=45.2,
            start_point=np.array([10.0, 20.0, 30.0]),
            end_point=np.array([55.2, 20.0, 30.0]),
        )

        assert fragment.id == 0
        assert fragment.particle_id == 5
        assert fragment.interaction_id == 2
        assert fragment.shape == 1  # Track
        assert fragment.is_primary is True
        assert fragment.length == 45.2
        np.testing.assert_allclose(fragment.start_point, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(fragment.end_point, [55.2, 20.0, 30.0])

    def test_fragment_shapes(self):
        """Test different fragment shapes."""
        from spine.data.out import RecoFragment

        # Different shape types
        shower_fragment = RecoFragment(id=0, shape=0, is_primary=True)  # Shower
        track_fragment = RecoFragment(id=1, shape=1, length=25.0)  # Track
        michel_fragment = RecoFragment(id=2, shape=2)  # Michel
        delta_fragment = RecoFragment(id=3, shape=3)  # Delta
        les_fragment = RecoFragment(id=4, shape=4)  # Low energy scatter

        fragments = [
            shower_fragment,
            track_fragment,
            michel_fragment,
            delta_fragment,
            les_fragment,
        ]

        # Verify shape assignments
        expected_shapes = [0, 1, 2, 3, 4]
        for i, fragment in enumerate(fragments):
            assert fragment.shape == expected_shapes[i]

        # Track should have length, others may not
        assert track_fragment.length == 25.0
        assert np.isnan(shower_fragment.length)  # Default unset


class TestTruthFragment:
    """Test TruthFragment specific functionality."""

    def test_truth_fragment_creation(self):
        """Test TruthFragment creation."""
        from spine.data.out import TruthFragment

        fragment = TruthFragment(
            id=0,
            particle_id=5,
            shape=1,
        )

        assert fragment.id == 0
        assert fragment.is_truth is True
        assert fragment.particle_id == 5

    def test_truth_fragment_str_representation(self):
        """Test TruthFragment string representation."""
        from spine.data.out import TruthFragment

        obj = TruthFragment(
            id=5,
            shape=1,  # track
            is_primary=True,
            match_ids=np.array([10], dtype=np.int64),
        )

        str_repr = str(obj)
        assert "TruthFragment" in str_repr
        assert "5" in str_repr  # ID
        assert "track" in str_repr.lower()
        # Primary is printed as boolean (True or 1)
        assert "True" in str_repr or "1" in str_repr

    def test_truth_fragment_start_dir_property(self):
        """Test TruthFragment start_dir property."""
        from spine.data.out import TruthFragment

        # Create a fragment with zero momentum should yield NaN direction
        fragment = TruthFragment(
            id=0,
        )

        assert np.isnan(fragment.start_dir).all()

        # Create fragment with specific non-zero momentum
        fragment = TruthFragment(
            id=0,
            momentum=np.array([10.0, 10.0, 10.0]),
        )

        expected_dir = np.array([10.0, 10.0, 10.0]) / np.linalg.norm([10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(fragment.start_dir, expected_dir)

    def test_truth_fragment_end_dir_property(self):
        """Test TruthFragment end_dir property."""
        from spine.data.out import TruthFragment

        # Create a fragment with zero end momentum should yield NaN direction
        fragment = TruthFragment(
            id=0,
            shape=1,  # Track
        )

        assert np.isnan(fragment.end_dir).all()

        # Create a fragment with shape not equal to track should yield NaN direction
        fragment = TruthFragment(
            id=0,
            shape=0,  # Not a track
            end_momentum=np.array([5.0, 0.0, 0.0]),
        )

        assert np.isnan(fragment.end_dir).all()

        # Create track fragment with specific non-zero end momentum
        fragment = TruthFragment(
            id=0,
            shape=1,  # Track
            end_momentum=np.array([5.0, 0.0, 0.0]),
        )

        expected_dir = np.array([1.0, 0.0, 0.0])  # Normalized direction
        np.testing.assert_array_almost_equal(fragment.end_dir, expected_dir)


class TestRecoFragment:
    """Test RecoFragment specific functionality."""

    def test_reco_fragment_creation(self):
        """Test RecoFragment creation."""
        from spine.data.out import RecoFragment

        fragment = RecoFragment(
            id=0,
            particle_id=5,
            shape=1,
        )

        assert fragment.id == 0
        assert fragment.is_truth is False
        assert fragment.particle_id == 5

    def test_reco_fragment_str_representation(self):
        """Test RecoFragment string representation."""
        from spine.data.out import RecoFragment

        obj = RecoFragment(
            id=5,
            shape=1,  # track
            is_primary=True,
            match_ids=np.array([10], dtype=np.int64),
        )

        str_repr = str(obj)
        assert "RecoFragment" in str_repr
        assert "5" in str_repr  # ID
        assert "track" in str_repr.lower()
        # Primary is printed as boolean (True or 1)
        assert "True" in str_repr or "1" in str_repr
