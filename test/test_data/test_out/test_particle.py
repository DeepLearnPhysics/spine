"""Test Particle classes."""

import numpy as np
import pytest


class TestParticleBase:
    """Test ParticleBase functionality."""

    def test_particlebase_initialization(self):
        """Test ParticleBase initialization with default values."""
        from spine.data.out.particle import ParticleBase

        obj = ParticleBase(id=0)

        # Test scalar attributes
        assert obj.interaction_id == -1
        assert obj.chi2_pid == -1
        assert obj.is_primary is False
        assert obj.is_crt_matched is False
        assert obj.is_valid is True
        assert np.isnan(obj.length)
        assert np.isnan(obj.calo_ke)
        assert np.isnan(obj.csda_ke)
        assert np.isnan(obj.mcs_ke)

        # Test enumerated attributes
        assert obj.shape == -1
        assert obj.pid == -1

        # Test vector attributes
        assert len(obj.fragment_ids) == 0
        assert obj.fragment_ids.dtype == np.int32

        assert obj.start_point.shape == (3,)
        assert all(np.isnan(obj.start_point))

        assert obj.end_point.shape == (3,)
        assert all(np.isnan(obj.end_point))

        # Test PID-related arrays (6 PID classes)
        assert len(obj.chi2_per_pid) == 6
        assert all(np.isnan(obj.chi2_per_pid))
        assert len(obj.csda_ke_per_pid) == 6
        assert all(np.isnan(obj.csda_ke_per_pid))
        assert len(obj.mcs_ke_per_pid) == 6
        assert all(np.isnan(obj.mcs_ke_per_pid))

        # Test CRT arrays
        assert len(obj.crt_ids) == 0
        assert len(obj.crt_times) == 0
        assert len(obj.crt_scores) == 0

    def test_particlebase_with_data(self):
        """Test ParticleBase with complete data."""
        from spine.data.out.particle import ParticleBase

        obj = ParticleBase(
            id=5,
            interaction_id=2,
            pid=2,  # muon
            is_primary=True,
            fragment_ids=np.array([10, 11, 12], dtype=np.int32),
        )

        assert obj.id == 5
        assert obj.interaction_id == 2
        assert obj.pid == 2
        assert obj.is_primary is True
        assert len(obj.fragment_ids) == 3
        np.testing.assert_array_equal(obj.fragment_ids, [10, 11, 12])

    def test_particlebase_num_fragments_property(self):
        """Test ParticleBase num_fragments derived property."""
        from spine.data.out.particle import ParticleBase

        # No fragments
        obj1 = ParticleBase(id=0)
        assert obj1.num_fragments == 0

        # Multiple fragments
        obj2 = ParticleBase(
            id=1,
            fragment_ids=np.array([5, 10, 15, 20], dtype=np.int32),
        )
        assert obj2.num_fragments == 4

    def test_particlebase_reset_crt_match(self):
        """Test ParticleBase reset_crt_match method."""
        from spine.data.out.particle import ParticleBase

        # Create particle with CRT matching
        obj = ParticleBase(
            id=0,
            is_crt_matched=True,
            crt_ids=np.array([5, 7], dtype=np.int32),
            crt_times=np.array([100.5, 105.2], dtype=np.float32),
            crt_scores=np.array([0.9, 0.85], dtype=np.float32),
        )

        # Verify initial state
        assert obj.is_crt_matched is True
        assert len(obj.crt_ids) == 2

        # Reset CRT matching
        obj.reset_crt_match()

        # Verify reset
        assert obj.is_crt_matched is False
        assert len(obj.crt_ids) == 0
        assert len(obj.crt_times) == 0
        assert len(obj.crt_scores) == 0

    def test_particlebase_str_representation(self):
        """Test ParticleBase string representation."""
        from spine.data.out.particle import ParticleBase

        obj = ParticleBase(
            id=7,
            pid=2,  # muon
            is_primary=True,
            match_ids=np.array([15], dtype=np.int64),
        )

        str_repr = str(obj)
        assert "Particle" in str_repr
        assert "7" in str_repr  # ID
        # Primary is printed as boolean (True or 1)
        assert "True" in str_repr or "1" in str_repr


class TestRecoParticle:
    """Test RecoParticle functionality."""

    def test_recoparticle_initialization(self):
        """Test RecoParticle initialization with default values."""
        from spine.data.out.particle import RecoParticle

        obj = RecoParticle(id=0)

        # Inherits all attributes from ParticleBase, so we can check a few key ones
        assert obj.id == 0
        assert obj.interaction_id == -1
        assert obj.is_primary is False
        assert np.isnan(obj.length)
        assert obj.shape == -1

    def test_recoparticle_with_data(self):
        """Test RecoParticle with complete data."""
        from spine.data.out.particle import RecoParticle

        obj = RecoParticle(
            id=5,
            interaction_id=2,
            pid=2,  # muon
            is_primary=True,
            fragment_ids=np.array([10, 11, 12], dtype=np.int64),
            primary_scores=np.array([0.8, 0.2], dtype=np.float32),
        )

        assert obj.id == 5
        assert obj.interaction_id == 2
        assert obj.pid == 2
        assert obj.is_primary is True
        assert len(obj.fragment_ids) == 3
        np.testing.assert_array_equal(obj.fragment_ids, [10, 11, 12])
        np.testing.assert_allclose(
            obj.primary_scores, np.array([0.8, 0.2], dtype=np.float32)
        )

    def test_recoparticle_str_representation(self):
        """Test RecoParticle string representation."""
        from spine.data.out.particle import RecoParticle

        obj = RecoParticle(
            id=7,
            pid=2,  # muon
            is_primary=True,
            match_ids=np.array([15], dtype=np.int64),
            primary_scores=np.array([0.85, 0.15], dtype=np.float32),
        )

        str_repr = str(obj)
        assert "RecoParticle" in str_repr
        assert "7" in str_repr  # ID

    def test_recoparticle_merge(self):
        """Test RecoParticle merge method."""
        from spine.data.out.particle import RecoParticle

        # Only a track can be merged into a track or shower
        obj1 = RecoParticle(id=1, shape=1)  # track
        for shape in [0, 2, 3]:
            obj2 = RecoParticle(id=2, shape=shape)
            with pytest.raises(ValueError):
                obj1.merge(obj2)

        # Cannot merge particles that have been truth-matched (i.e.
        # already assigned to a truth particle)
        obj1 = RecoParticle(id=1, shape=1, is_matched=True)
        obj2 = RecoParticle(id=2, shape=1, is_matched=True)
        with pytest.raises(ValueError):
            obj1.merge(obj2)

        # Try a successful merge of two track particles
        obj1 = RecoParticle(
            id=1,
            index=np.array([0, 1], dtype=np.int64),
            points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            depositions=np.array([1.5, 2.5], dtype=np.float32),
            start_point=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            end_point=np.array([4.0, 5.0, 6.0], dtype=np.float32),
            start_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            end_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            interaction_id=2,
            shape=1,  # track
            pid=2,  # muon
            is_primary=True,
            fragment_ids=np.array([10, 11], dtype=np.int64),
            primary_scores=np.array([0.8, 0.2], dtype=np.float32),
        )

        obj2 = RecoParticle(
            id=2,
            index=np.array([2], dtype=np.int64),
            points=np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
            depositions=np.array([3.5], dtype=np.float32),
            start_point=np.array([7.0, 8.0, 9.0], dtype=np.float32),
            end_point=np.array([10.0, 11.0, 12.0], dtype=np.float32),
            start_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            end_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            interaction_id=2,
            shape=1,  # track
            pid=2,  # muon
            is_primary=True,
            fragment_ids=np.array([12], dtype=np.int64),
            primary_scores=np.array([0.05, 0.95], dtype=np.float32),
        )

        obj1.merge(obj2)

        assert obj1.id == 1  # ID should remain the same
        assert obj1.interaction_id == 2
        assert obj1.pid == 2
        assert obj1.is_primary is True
        assert len(obj1.fragment_ids) == 3
        np.testing.assert_array_equal(obj1.index, np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(
            obj1.points,
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
            ),
        )
        np.testing.assert_array_equal(
            obj1.depositions, np.array([1.5, 2.5, 3.5], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obj1.start_point, np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obj1.end_point, np.array([10.0, 11.0, 12.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obj1.start_dir, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obj1.end_dir, np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(obj1.fragment_ids, [10, 11, 12])
        # primary_scores updated to obj2's since np.max(obj2.primary_scores)
        # > np.max(obj1.primary_scores)
        np.testing.assert_allclose(
            obj1.primary_scores, np.array([0.05, 0.95], dtype=np.float32)
        )

        # Test a successful merge of a track into a shower (shower should
        # take on track's PID and primary score if higher)
        obj1 = RecoParticle(
            id=1,
            interaction_id=2,
            start_point=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            shape=0,  # shower
            pid=1,  # electron
            is_primary=True,
            fragment_ids=np.array([20], dtype=np.int64),
            primary_scores=np.array([0.0, 1.0], dtype=np.float32),  # high shower score
            calo_ke=50.0,
        )

        obj2 = RecoParticle(
            id=2,
            interaction_id=2,
            start_point=np.array([7.0, 8.0, 9.0], dtype=np.float32),
            end_point=np.array([10.0, 11.0, 12.0], dtype=np.float32),
            start_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            end_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            shape=1,  # track
            pid=2,  # muon
            is_primary=False,
            fragment_ids=np.array([21], dtype=np.int64),
            primary_scores=np.array([0.8, 0.2], dtype=np.float32),  # high track score
            calo_ke=30.0,
        )

        obj1.merge(obj2)
        assert obj1.id == 1
        assert obj1.interaction_id == 2
        assert obj1.pid == 1  # Shower takes on track's PID
        assert obj1.is_primary is True  # Shower takes on track's primary status
        assert len(obj1.fragment_ids) == 2
        assert obj1.calo_ke == 80.0  # Calo KE should be summed
        np.testing.assert_array_equal(
            obj1.start_point, np.array([10.0, 11.0, 12.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            obj1.start_dir, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(obj1.fragment_ids, [20, 21])
        # primary_scores should remain unchanged since obj1's max score
        # is higher for shower category
        np.testing.assert_allclose(
            obj1.primary_scores, np.array([0.0, 1.0], dtype=np.float32)
        )

    def test_recoparticle_pdg_code_property(self):
        """Test RecoParticle PDG code property."""
        from spine.data.out.particle import RecoParticle

        # Test with a muon
        obj = RecoParticle(id=0, pid=2)  # muon
        assert obj.pdg_code == 13

        # Test with an electron
        obj = RecoParticle(id=1, pid=1)  # electron
        assert obj.pdg_code == 11

        # Test with a pion
        obj = RecoParticle(id=2, pid=3)  # pion
        assert obj.pdg_code == 211

        # Test with an unknown PID
        obj = RecoParticle(id=3, pid=-1)  # unknown PID
        assert obj.pdg_code == -1

    def test_recoparticle_mass_property(self):
        """Test RecoParticle mass property."""
        from spine.data.out.particle import RecoParticle

        # Test with a muon
        obj = RecoParticle(id=0, pid=2)  # muon
        assert abs(obj.mass - 105.66) < 0.01  # Muon mass in MeV/c^2

        # Test with an electron
        obj = RecoParticle(id=1, pid=1)  # electron
        assert abs(obj.mass - 0.511) < 0.001  # Electron mass in MeV/c^2

        # Test with a pion
        obj = RecoParticle(id=2, pid=3)  # pion
        assert abs(obj.mass - 139.57) < 0.01  # Pion mass in MeV/c^2

        # Test with an unknown PID
        obj = RecoParticle(id=3, pid=-1)  # unknown PID
        assert np.isnan(obj.mass)

    def test_recoparticle_ke_property(self):
        """Test RecoParticle kinetic energy property."""
        from spine.data.out.particle import RecoParticle

        # If shape is not initialized, KE always returns calo_ke
        obj = RecoParticle(id=0, calo_ke=50.0, csda_ke=30.0, mcs_ke=20.0)
        assert obj.ke == 50.0

        # Test with calo_ke only
        obj = RecoParticle(id=0, shape=0, calo_ke=50.0)
        assert obj.ke == 50.0

        # Test with csda_ke only
        obj = RecoParticle(id=1, shape=1, csda_ke=30.0, is_contained=True)
        assert obj.ke == 30.0

        # Test with mcs_ke only
        obj = RecoParticle(id=2, shape=1, mcs_ke=20.0, is_contained=False)
        assert obj.ke == 20.0

        # Test with all KE estimates (should return csda_ke if track is contained)
        obj = RecoParticle(
            id=3, shape=1, calo_ke=50.0, csda_ke=30.0, mcs_ke=20.0, is_contained=True
        )
        assert obj.ke == 30.0

        # Test with all KE estimates (should return mcs_ke if track is not contained)
        obj = RecoParticle(
            id=4, shape=1, calo_ke=25.0, csda_ke=35.0, mcs_ke=20.0, is_contained=False
        )
        assert obj.ke == 20.0

        # Test with all KE estimates (should return calo_ke if csda_ke and mcs_ke are not available)
        obj = RecoParticle(id=5, shape=1, calo_ke=40.0)
        assert obj.ke == 40.0

    def test_recoparticle_momentum_property(self):
        """Test RecoParticle momentum property."""
        from spine.data.out.particle import RecoParticle

        # Test with no momentum information (should return NaN)
        obj = RecoParticle(id=0)
        assert np.all(np.isnan(obj.momentum))

        # If start_dir is missing, momentum should return NaN
        obj = RecoParticle(
            id=1, calo_ke=50.0, shape=0
        )  # shower with KE but no direction
        assert np.all(np.isnan(obj.momentum))

        # If ke is missing, momentum should return NaN
        start_dir = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        start_dir /= np.linalg.norm(start_dir)  # Normalize direction
        obj = RecoParticle(id=2, start_dir=start_dir, shape=1)
        assert np.all(np.isnan(obj.momentum))

        # If PID is not recognized, mass is NaN and thus momentum should return NaN
        obj = RecoParticle(
            id=3,
            start_dir=start_dir,
            calo_ke=50.0,
            shape=1,
            pid=-1,
        )  # track with KE but unknown PID
        assert np.all(np.isnan(obj.momentum))

        obj = RecoParticle(
            id=4,
            start_dir=start_dir,
            calo_ke=50.0,
            shape=1,
            pid=99,
        )
        assert np.all(np.isnan(obj.momentum))

        # Test with valid momentum information
        obj = RecoParticle(
            id=1,
            start_dir=start_dir,
            calo_ke=50.0,
            shape=0,
            pid=1,
        )
        momentum = (
            50.0 + 0.511998
        ) ** 2 - 0.511998**2  # KE + mass, squared, minus mass squared
        momentum = np.sqrt(momentum)
        np.testing.assert_array_equal(obj.momentum, momentum * start_dir)

        assert np.isclose(obj.p, momentum)

    def test_recoparticle_aliases(self):
        """Test RecoParticle aliases for backward compatibility."""
        from spine.data.out.particle import RecoParticle

        start_dir = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        start_dir /= np.linalg.norm(start_dir)  # Normalize direction
        end_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        obj = RecoParticle(
            id=0,
            shape=1,
            calo_ke=50.0,
            csda_ke=30.0,
            mcs_ke=20.0,
            start_dir=start_dir,
            end_dir=end_dir,
            pid=1,
            length=100.0,
        )

        # reco_ke alias should return the same value as ke property
        assert obj.reco_ke == obj.ke

        # reco_momentum alias should return the same value as momentum property
        np.testing.assert_array_equal(obj.reco_momentum, obj.momentum)

        # reco_length alias should return the same value as length attribute
        assert obj.reco_length == obj.length

        # reco_start_dir and reco_end_dir aliases should return the same values as start_dir and end_dir
        np.testing.assert_array_equal(obj.reco_start_dir, obj.start_dir)
        np.testing.assert_array_equal(obj.reco_end_dir, obj.end_dir)


class TestTruthParticle:
    """Test TruthParticle functionality."""

    def test_truthparticle_initialization(self):
        """Test TruthParticle initialization with default values."""
        from spine.data.out.particle import TruthParticle

        obj = TruthParticle(id=0)

        # Inherits all attributes from ParticleBase, so we can check a few key ones
        assert obj.id == 0
        assert obj.interaction_id == -1
        assert obj.is_primary is False
        assert np.isnan(obj.length)
        assert obj.shape == -1

    def test_truthparticle_with_data(self):
        """Test TruthParticle with complete data."""
        from spine.data.out.particle import TruthParticle

        obj = TruthParticle(
            id=5,
            interaction_id=2,
            pid=2,  # muon
            is_primary=True,
            fragment_ids=np.array([10, 11, 12], dtype=np.int64),
        )

        assert obj.id == 5
        assert obj.interaction_id == 2
        assert obj.pid == 2
        assert obj.is_primary is True
        assert len(obj.fragment_ids) == 3
        np.testing.assert_array_equal(obj.fragment_ids, [10, 11, 12])

    def test_truthparticle_str_representation(self):
        """Test TruthParticle string representation."""
        from spine.data.out.particle import TruthParticle

        obj = TruthParticle(
            id=5,
            pid=2,  # muon
            is_primary=True,
            match_ids=np.array([15], dtype=np.int64),
        )

        str_repr = str(obj)
        assert "TruthParticle" in str_repr
        assert "5" in str_repr  # ID
        # Primary is printed as boolean (True or 1)
        assert "True" in str_repr or "1" in str_repr

    def test_truthparticle_dir_properties(self):
        """Test TruthParticle direction properties."""
        from spine.data.out.particle import TruthParticle

        # If momentum or end_momentum is not initialized, start_dir and end_dir should return NaN
        obj = TruthParticle(id=0)
        assert np.isnan(obj.start_dir).all()
        assert np.isnan(obj.end_dir).all()

        # If momemntu, or end_momentum has zero magnitude, start_dir and end_dir should return NaN
        obj = TruthParticle(
            id=1,
            momentum=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            end_momentum=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        assert np.isnan(obj.start_dir).all()
        assert np.isnan(obj.end_dir).all()

        # Initialize properly
        obj = TruthParticle(
            id=5,
            shape=1,  # track
            momentum=np.array([300.0, 400.0, 500.0], dtype=np.float32),
            end_momentum=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        )

        start_dir = obj.momentum / np.linalg.norm(obj.momentum)
        end_dir = obj.end_momentum / np.linalg.norm(obj.end_momentum)
        np.testing.assert_array_equal(obj.start_dir, start_dir)
        np.testing.assert_array_equal(obj.end_dir, end_dir)

        # If shape is not track, end_dir should return NaN even if end_momentum is set
        obj = TruthParticle(
            id=6,
            shape=0,  # shower
            end_momentum=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        )
        assert np.isnan(obj.end_dir).all()

    def test_truthparticle_ke_property(self):
        """Test TruthParticle kinetic energy property."""
        from spine.data.out.particle import TruthParticle

        # If energy_init or mass is not initialized, KE should return NaN
        obj = TruthParticle(id=0)
        assert np.isnan(obj.ke)

        obj = TruthParticle(id=1, energy_init=100.0)
        assert np.isnan(obj.ke)

        obj = TruthParticle(
            id=2, momentum=np.array([300.0, 400.0, 500.0], dtype=np.float32)
        )
        assert np.isnan(obj.ke)

        # Test with all KE estimates (should return mcs_ke if track is not contained)
        momentum = np.array([300.0, 400.0, 500.0], dtype=np.float32)
        mass = 105.66  # Muon mass in MeV/c^2
        energy_init = np.sqrt(np.sum(momentum**2) + mass**2)
        obj = TruthParticle(id=3, energy_init=energy_init, momentum=momentum)

        assert np.isclose(obj.ke, energy_init - mass)

    def test_truthparticle_reco_ke_property(self):
        """Test TruthParticle reconstructed kinetic energy property."""
        from spine.data.out.particle import TruthParticle

        # If shape is not initialized, KE always returns calo_ke
        obj = TruthParticle(id=0, calo_ke=50.0, csda_ke=30.0, mcs_ke=20.0)
        assert obj.reco_ke == 50.0

        # Test with calo_ke only
        obj = TruthParticle(id=0, shape=0, calo_ke=50.0)
        assert obj.reco_ke == 50.0

        # Test with csda_ke only
        obj = TruthParticle(id=1, shape=1, csda_ke=30.0, is_contained=True)
        assert obj.reco_ke == 30.0

        # Test with mcs_ke only
        obj = TruthParticle(id=2, shape=1, mcs_ke=20.0, is_contained=False)
        assert obj.reco_ke == 20.0

        # Test with all KE estimates (should return csda_ke if track is contained)
        obj = TruthParticle(
            id=3, shape=1, calo_ke=50.0, csda_ke=30.0, mcs_ke=20.0, is_contained=True
        )
        assert obj.reco_ke == 30.0

        # Test with all KE estimates (should return mcs_ke if track is not contained)
        obj = TruthParticle(
            id=4, shape=1, calo_ke=25.0, csda_ke=35.0, mcs_ke=20.0, is_contained=False
        )
        assert obj.reco_ke == 20.0

        # Test with all KE estimates (should return calo_ke if csda_ke and mcs_ke are not available)
        obj = TruthParticle(id=5, shape=1, calo_ke=40.0)
        assert obj.reco_ke == 40.0

    def test_truthparticle_reco_momentum_property(self):
        """Test TruthParticle reconstructed momentum property."""
        from spine.data.out.particle import TruthParticle

        # Test with no momentum information (should return NaN)
        obj = TruthParticle(id=0)
        assert np.all(np.isnan(obj.reco_momentum))

        # If start_dir is missing, momentum should return NaN
        obj = TruthParticle(
            id=1, calo_ke=50.0, shape=0
        )  # shower with KE but no direction
        assert np.all(np.isnan(obj.reco_momentum))

        # If ke is missing, momentum should return NaN
        start_dir = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        start_dir /= np.linalg.norm(start_dir)  # Normalize direction
        obj = TruthParticle(id=2, reco_start_dir=start_dir, shape=1)
        assert np.all(np.isnan(obj.reco_momentum))

        # If PID is not recognized, mass is NaN and thus momentum should return NaN
        obj = TruthParticle(
            id=3,
            reco_start_dir=start_dir,
            calo_ke=50.0,
            shape=1,
            pid=-1,
        )  # track with KE but unknown PID
        assert np.all(np.isnan(obj.reco_momentum))

        obj = TruthParticle(
            id=4,
            reco_start_dir=start_dir,
            calo_ke=50.0,
            shape=1,
            pid=99,
        )
        assert np.all(np.isnan(obj.reco_momentum))

        # Test with valid momentum information
        obj = TruthParticle(
            id=1,
            reco_start_dir=start_dir,
            calo_ke=50.0,
            shape=0,
            pid=1,
        )
        momentum = (
            50.0 + 0.511998
        ) ** 2 - 0.511998**2  # KE + mass, squared, minus mass squared
        momentum = np.sqrt(momentum)
        np.testing.assert_array_equal(obj.reco_momentum, momentum * start_dir)
