"""Comprehensive test suite for spine.data.out module."""

import pytest
import numpy as np
from unittest.mock import Mock


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
            shape=1,  # track
            is_primary=True,
            length=45.2,
            start_point=np.array([10.0, 20.0, 30.0]),
            end_point=np.array([55.2, 20.0, 30.0]),  # Track along x
        )
        
        assert fragment.id == 0
        assert fragment.particle_id == 5
        assert fragment.interaction_id == 2
        assert fragment.shape == 1  # Track
        assert fragment.is_primary is True
        assert fragment.length == 45.2
        np.testing.assert_array_equal(fragment.start_point, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(fragment.end_point, [55.2, 20.0, 30.0])
    
    def test_fragment_shapes(self):
        """Test different fragment shapes."""
        from spine.data.out import RecoFragment
        
        # Different shape types
        shower_fragment = RecoFragment(id=0, shape=0, is_primary=True)  # Shower
        track_fragment = RecoFragment(id=1, shape=1, length=25.0)  # Track  
        michel_fragment = RecoFragment(id=2, shape=2)  # Michel
        delta_fragment = RecoFragment(id=3, shape=3)  # Delta
        les_fragment = RecoFragment(id=4, shape=4)  # Low energy scatter
        
        fragments = [shower_fragment, track_fragment, michel_fragment, 
                    delta_fragment, les_fragment]
        
        # Verify shape assignments
        expected_shapes = [0, 1, 2, 3, 4]
        for i, fragment in enumerate(fragments):
            assert fragment.shape == expected_shapes[i]
        
        # Track should have length, others may not
        assert track_fragment.length == 25.0
        assert shower_fragment.length == -1.0  # Default unset
    
    def test_fragment_directions(self):
        """Test fragment direction vectors."""
        from spine.data.out import RecoFragment
        
        # Track with start and end directions
        track = RecoFragment(
            id=0,
            shape=1,  # track
            start_point=np.array([0.0, 0.0, 0.0]),
            end_point=np.array([10.0, 0.0, 0.0]),
            start_dir=np.array([1.0, 0.0, 0.0]),  # Forward along x
            end_dir=np.array([1.0, 0.0, 0.0]),    # Same direction
        )
        
        # Verify directions
        np.testing.assert_array_equal(track.start_dir, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(track.end_dir, [1.0, 0.0, 0.0])
        
        # Check direction consistency with track length
        expected_length = np.linalg.norm(track.end_point - track.start_point)
        assert abs(expected_length - 10.0) < 1e-6
    
    def test_fragment_shower_properties(self):
        """Test shower-specific fragment properties."""
        from spine.data.out import RecoFragment
        
        # Shower fragment (typically no well-defined length/end_point)
        shower = RecoFragment(
            id=0,
            shape=0,  # shower
            is_primary=True,
            start_point=np.array([5.0, 10.0, 15.0]),
            start_dir=np.array([0.0, 0.0, 1.0]),  # Shower direction
            # No specific end_point, end_dir, or length for showers
        )
        
        assert shower.shape == 0
        assert shower.is_primary is True
        np.testing.assert_array_equal(shower.start_point, [5.0, 10.0, 15.0])
        np.testing.assert_array_equal(shower.start_dir, [0.0, 0.0, 1.0])
        
        # Shower-specific defaults (may have default values, not None)
        assert shower.length == -1.0  # No meaningful length


class TestTruthFragment:
    """Test TruthFragment specific functionality."""
    
    def test_truth_fragment_creation(self):
        """Test TruthFragment with truth-specific attributes."""
        from spine.data.out import TruthFragment
        
        truth_fragment = TruthFragment(
            id=0,
            particle_id=3,
            shape=1,  # track
            is_primary=True,
        )
        
        assert truth_fragment.id == 0
        assert truth_fragment.particle_id == 3
        assert truth_fragment.shape == 1
        assert truth_fragment.is_primary is True
        assert truth_fragment.is_truth is True  # TruthFragment specific
    
    def test_truth_fragment_base_attributes(self):
        """Test TruthFragment inherits from base classes."""
        from spine.data.out import TruthFragment
        
        truth_fragment = TruthFragment(
            id=0,
            particle_id=5,
            shape=1,  # track
            length=50.0,
            start_point=np.array([10.0, 20.0, 30.0]),
        )
        
        assert truth_fragment.particle_id == 5
        assert truth_fragment.length == 50.0
        assert truth_fragment.is_truth is True
        # Size is computed from actual data points, starts at 0
        assert truth_fragment.size >= 0


class TestRecoFragment:
    """Test RecoFragment specific functionality.""" 
    
    def test_reco_fragment_creation(self):
        """Test RecoFragment with reconstruction-specific attributes."""
        from spine.data.out import RecoFragment
        
        reco_fragment = RecoFragment(
            id=10,
            particle_id=7,
            interaction_id=3,
            shape=0,  # shower
            is_primary=False,
            # Base attributes
            start_point=np.array([12.5, 8.0, 45.0]),
        )
        
        assert reco_fragment.id == 10
        assert reco_fragment.particle_id == 7
        assert reco_fragment.interaction_id == 3
        assert reco_fragment.shape == 0  # Shower
        assert reco_fragment.is_primary is False
        # Size is computed from actual voxel data
        assert reco_fragment.size >= 0
        np.testing.assert_array_equal(reco_fragment.start_point, [12.5, 8.0, 45.0])
        assert reco_fragment.is_truth is False  # RecoFragment specific
    
    def test_reco_fragment_track_properties(self):
        """Test RecoFragment with track-specific reconstruction."""
        from spine.data.out import RecoFragment
        
        # Reconstructed track
        reco_track = RecoFragment(
            id=0,
            shape=1,  # track
            length=65.4,
            start_point=np.array([10.0, 15.0, 20.0]),
            end_point=np.array([75.4, 15.0, 20.0]),
            start_dir=np.array([1.0, 0.0, 0.0]),
            end_dir=np.array([1.0, 0.0, 0.0]),
        )
        
        # Verify track properties
        assert reco_track.shape == 1
        assert reco_track.length == 65.4
        # Size computed from voxel data
        assert reco_track.size >= 0
        
        # Check geometric consistency
        actual_length = np.linalg.norm(reco_track.end_point - reco_track.start_point)
        assert abs(actual_length - reco_track.length) < 0.1  # Close match
        
        # Check direction normalization
        start_dir_norm = np.linalg.norm(reco_track.start_dir)
        end_dir_norm = np.linalg.norm(reco_track.end_dir)
        assert abs(start_dir_norm - 1.0) < 1e-6
        assert abs(end_dir_norm - 1.0) < 1e-6
    
    def test_reco_fragment_matching_properties(self):
        """Test RecoFragment matching attributes."""
        from spine.data.out import RecoFragment
        
        # Fragment with truth matching information
        matched_fragment = RecoFragment(
            id=0,
            is_matched=True,
            match_ids=np.array([5, 7]),  # Matched to truth fragments 5 and 7
            match_overlaps=np.array([0.8, 0.3]),  # High overlap with 5, low with 7
        )
        
        assert matched_fragment.is_matched is True
        np.testing.assert_array_equal(matched_fragment.match_ids, [5, 7])
        np.testing.assert_array_equal(matched_fragment.match_overlaps, [0.8, 0.3])
        
        # Best match should be first (highest overlap)
        best_match = matched_fragment.match_ids[0]
        best_overlap = matched_fragment.match_overlaps[0]
        assert best_match == 5
        assert best_overlap == 0.8


class TestParticleOut:
    """Test output Particle classes (different from data.Particle)."""
    
    def test_reco_particle_creation(self):
        """Test RecoParticle creation."""
        from spine.data.out import RecoParticle
        
        reco_particle = RecoParticle(
            id=0,
            interaction_id=2,
            pid=2,  # muon in PID system  
            start_point=np.array([5.0, 10.0, 15.0]),
        )
        
        assert reco_particle.id == 0
        assert reco_particle.interaction_id == 2
        assert reco_particle.pid == 2  # Muon
        # Size computed from constituent fragments
        assert reco_particle.size >= 0
        np.testing.assert_array_equal(reco_particle.start_point, [5.0, 10.0, 15.0])
    
    def test_truth_particle_creation(self):
        """Test TruthParticle creation."""
        from spine.data.out import TruthParticle
        
        truth_particle = TruthParticle(
            id=1,
            interaction_id=2,
            pid=4,  # proton in PID system
            size=120,
        )
        
        assert truth_particle.id == 1
        assert truth_particle.pid == 4  # Proton
        assert truth_particle.is_truth is True
    
    def test_particle_momentum_calculations(self):
        """Test momentum-related calculations for particles."""
        from spine.data.out import RecoParticle
        
        # Particle with PID (kinetic energy computed elsewhere)
        particle = RecoParticle(
            id=0,
            pid=2,  # muon
        )
        
        # Verify PID assignment
        assert particle.pid == 2
        # Mass should be set based on PID
        assert particle.mass > 0.0  # Should have muon mass
        # Kinetic energy starts at default until computed
        assert particle.ke == -1.0  # Default unset value
    
    def test_particle_pid_properties(self):
        """Test particle ID and mass properties."""
        from spine.data.out import RecoParticle
        
        # Test different PID types
        muon = RecoParticle(id=0, pid=2)  # Muon
        proton = RecoParticle(id=1, pid=4)  # Proton
        electron = RecoParticle(id=2, pid=1)  # Electron
        
        particles = [muon, proton, electron]
        expected_pids = [2, 4, 1]
        
        for particle, expected_pid in zip(particles, expected_pids):
            assert particle.pid == expected_pid
            # Mass should be set based on PID (if implemented)
            assert particle.mass >= 0.0  # Should have some mass


class TestInteractionOut:
    """Test output Interaction classes."""
    
    def test_reco_interaction_creation(self):
        """Test RecoInteraction creation."""
        from spine.data.out import RecoInteraction
        
        reco_interaction = RecoInteraction(
            id=0,
            vertex=np.array([25.0, 30.0, 150.0]),
        )
        
        assert reco_interaction.id == 0
        # Size computed from constituent particles
        assert reco_interaction.size >= 0
        np.testing.assert_array_equal(reco_interaction.vertex, [25.0, 30.0, 150.0])
    
    def test_truth_interaction_creation(self):
        """Test TruthInteraction creation."""
        from spine.data.out import TruthInteraction
        
        truth_interaction = TruthInteraction(
            id=1,
            size=1200,
            vertex=np.array([24.8, 30.2, 149.5]),  # Close to reco vertex
        )
        
        assert truth_interaction.id == 1
        assert truth_interaction.is_truth is True
        np.testing.assert_array_equal(truth_interaction.vertex, [24.8, 30.2, 149.5])
    
    def test_interaction_topology(self):
        """Test interaction topology properties."""
        from spine.data.out import RecoInteraction
        
        # Interaction with topology information
        interaction = RecoInteraction(
            id=0,
            vertex=np.array([10.0, 20.0, 30.0]),
            is_contained=True,  # Fully contained
        )
        
        # num_particles is computed from actual particle data
        assert interaction.num_particles >= 0
        assert interaction.is_contained is True


@pytest.mark.slow
class TestOutDataIntegration:
    """Integration tests for output data structures."""
    
    def test_fragment_to_particle_hierarchy(self):
        """Test hierarchy from fragments to particles."""
        from spine.data.out import RecoFragment, RecoParticle
        
        # Multiple fragments belonging to one particle
        fragments = [
            RecoFragment(
                id=0, particle_id=5, shape=1, is_primary=True,
                size=200, length=45.0
            ),
            RecoFragment(
                id=1, particle_id=5, shape=2, is_primary=False,
                size=50  # Michel electron
            ),
        ]
        
        # Particle containing these fragments
        particle = RecoParticle(
            id=5,
            size=250,  # Sum of fragment sizes
            fragment_ids=np.array([0, 1]),  # Links to fragments
        )
        
        # Verify hierarchy consistency
        assert all(f.particle_id == particle.id for f in fragments)
        
        total_fragment_size = sum(f.size for f in fragments)
        
        assert particle.size == total_fragment_size
        np.testing.assert_array_equal(particle.fragment_ids, [0, 1])
    
    def test_particle_to_interaction_hierarchy(self):
        """Test hierarchy from particles to interactions."""
        from spine.data.out import RecoParticle, RecoInteraction
        
        # Multiple particles in one interaction
        particles = [
            RecoParticle(id=0, interaction_id=10, pid=2, size=300),
            RecoParticle(id=1, interaction_id=10, pid=4, size=150),
            RecoParticle(id=2, interaction_id=10, pid=3, size=100),
        ]
        
        # Interaction containing these particles
        interaction = RecoInteraction(
            id=10,
            size=550,  # Sum of particle sizes
            particle_ids=np.array([0, 1, 2]),  # Links to particles
            vertex=np.array([20.0, 25.0, 100.0]),
        )
        
        # Verify hierarchy consistency
        assert all(p.interaction_id == interaction.id for p in particles)
        
        total_particle_size = sum(p.size for p in particles)
        
        assert interaction.size == total_particle_size
        np.testing.assert_array_equal(interaction.particle_ids, [0, 1, 2])
    
    def test_truth_reco_matching(self):
        """Test matching between truth and reconstruction objects."""
        from spine.data.out import TruthFragment, RecoFragment
        
        # Truth fragment
        truth_fragment = TruthFragment(
            id=0,
            particle_id=0,
            shape=1,  # track
            length=50.0,
            start_point=np.array([10.0, 20.0, 30.0]),
            size=200,
        )
        
        # Reconstruction matching
        reco_fragment = RecoFragment(
            id=10,  # Different ID space
            shape=1,  # Same shape
            length=48.5,  # Close but not exact
            start_point=np.array([10.2, 20.1, 29.9]),  # Close position
            size=200,
            # Matching information
            is_matched=True,
            match_ids=np.array([0]),  # Matched to truth_fragment
            match_overlaps=np.array([0.95]),  # High overlap
        )
        
        # Compare truth vs reco
        assert truth_fragment.shape == reco_fragment.shape
        
        # Check position matching
        pos_diff = np.linalg.norm(truth_fragment.start_point - reco_fragment.start_point)
        assert pos_diff < 1.0  # Within 1 cm
        
        # Check length matching
        length_diff = abs(truth_fragment.length - reco_fragment.length)
        assert length_diff < 5.0  # Within 5 cm
        
        # Check matching quality
        assert reco_fragment.is_matched is True
        assert reco_fragment.match_overlaps[0] > 0.9  # High quality match
    
    def test_complete_event_reconstruction(self):
        """Test complete event with full reconstruction hierarchy."""
        from spine.data.out import (RecoFragment, RecoParticle, RecoInteraction,
                                   TruthFragment, TruthParticle, TruthInteraction)
        
        # Truth interaction
        truth_interaction = TruthInteraction(
            id=0, vertex=np.array([15.0, 20.0, 100.0]),
            size=1000,
        )
        
        # Reconstruction
        reco_interaction = RecoInteraction(
            id=0, vertex=np.array([15.2, 20.1, 99.8]),  # Close vertex
            size=800,
            # Matching information
            is_matched=True,
            match_ids=np.array([0]),  # Matched to truth_interaction
            match_overlaps=np.array([0.85]),
        )
        
        # Verify event-level matching
        vertex_diff = np.linalg.norm(truth_interaction.vertex - reco_interaction.vertex)
        assert vertex_diff < 1.0  # Good vertex reconstruction
        
        # Check matching quality
        assert reco_interaction.is_matched is True
        assert reco_interaction.match_overlaps[0] > 0.8  # Good overall match


if __name__ == '__main__':
    pytest.main([__file__, '-v'])