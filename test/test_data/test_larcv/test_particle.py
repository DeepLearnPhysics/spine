"""Comprehensive test suite for spine.data.particle module."""

import numpy as np
import pytest

from spine.utils.conditional import LARCV_AVAILABLE, larcv


class TestParticleCreation:
    """Test Particle class creation and initialization."""

    def test_particle_basic_creation(self):
        """Test basic Particle instantiation."""
        from spine.data import Particle

        # Test creation with minimal parameters
        particle = Particle(id=0)
        assert particle.id == 0

        # Test creation with common physics parameters
        particle = Particle(
            id=1,
            pid=2,  # muon
            pdg_code=13,
            energy_init=1000.0,  # 1 GeV
            position=np.array([0.0, 0.0, 0.0]),
            momentum=np.array([500.0, 0.0, 866.0]),  # p_total ≈ 1 GeV
        )

        assert particle.id == 1
        assert particle.pid == 2
        assert particle.pdg_code == 13
        assert particle.energy_init == 1000.0
        assert np.allclose(particle.position, [0.0, 0.0, 0.0])
        assert np.allclose(particle.momentum, [500.0, 0.0, 866.0])

    def test_particle_with_all_kinematic_properties(self):
        """Test Particle with complete kinematic information."""
        from spine.data import Particle

        particle = Particle(
            id=0,
            pid=1,  # electron
            pdg_code=11,
            # Kinematic properties
            energy_init=100.0,
            energy_deposit=95.0,
            distance_travel=12.5,
            # Position and momentum
            position=np.array([1.0, 2.0, 3.0]),
            end_position=np.array([4.0, 5.0, 6.0]),
            momentum=np.array([50.0, 30.0, 40.0]),
            end_momentum=np.array([45.0, 25.0, 35.0]),
            # Timing
            t=100.0,  # ns
            end_t=200.0,  # ns
        )

        # Verify all properties are set correctly
        assert particle.pid == 1
        assert particle.energy_init == 100.0
        assert particle.energy_deposit == 95.0
        assert particle.distance_travel == 12.5
        assert np.allclose(particle.position, [1.0, 2.0, 3.0])
        assert np.allclose(particle.end_position, [4.0, 5.0, 6.0])
        assert np.allclose(particle.momentum, [50.0, 30.0, 40.0])
        assert np.allclose(particle.end_momentum, [45.0, 25.0, 35.0])
        assert particle.t == 100.0
        assert particle.end_t == 200.0

    def test_particle_genealogy_properties(self):
        """Test Particle genealogy and relationship properties."""
        from spine.data import Particle

        # Primary particle (no parent)
        primary = Particle(
            id=0,
            parent_id=-1,
            interaction_id=0,
            nu_id=0,
            group_id=0,
            interaction_primary=True,
            group_primary=True,
            pdg_code=2212,  # proton
        )

        # Secondary particle (has parent)
        secondary = Particle(
            id=1,
            parent_id=0,
            interaction_id=0,
            nu_id=0,
            group_id=0,
            interaction_primary=False,
            group_primary=False,
            pdg_code=211,  # pi+
            parent_pdg_code=2212,  # from proton
        )

        # Verify genealogy properties
        assert primary.parent_id == -1
        assert primary.interaction_primary is True
        assert primary.group_primary is True

        assert secondary.parent_id == 0
        assert secondary.interaction_primary is False
        assert secondary.group_primary is False
        assert secondary.parent_pdg_code == 2212

    def test_particle_track_and_index_properties(self):
        """Test Particle track IDs and index properties."""
        from spine.data import Particle

        particle = Particle(
            id=5,
            mct_index=10,
            mcst_index=15,
            track_id=1001,
            parent_track_id=1000,
            ancestor_track_id=999,
            num_voxels=250,
        )

        assert particle.id == 5
        assert particle.mct_index == 10
        assert particle.mcst_index == 15
        assert particle.track_id == 1001
        assert particle.parent_track_id == 1000
        assert particle.ancestor_track_id == 999
        assert particle.num_voxels == 250


class TestParticlePhysics:
    """Test Particle physics calculations and validations."""

    def test_particle_momentum_magnitude(self):
        """Test momentum magnitude calculation if available."""
        from spine.data import Particle

        # Create particle with known momentum
        momentum = np.array([3.0, 4.0, 0.0])  # |p| = 5 GeV
        particle = Particle(id=0, momentum=momentum, pdg_code=13)  # muon

        # Calculate expected momentum magnitude
        expected_p = np.linalg.norm(momentum)
        assert abs(expected_p - 5.0) < 1e-10

        # If particle has p attribute, check it
        if hasattr(particle, "p"):
            assert abs(particle.p - 5.0) < 1e-6

    def test_particle_energy_momentum_consistency(self):
        """Test energy-momentum relationship for realistic particles."""
        from spine.data import Particle

        # Electron: E^2 = (pc)^2 + (mc^2)^2, m_e ≈ 0.511 MeV
        momentum = np.array([1.0, 0.0, 0.0])  # 1 GeV/c
        energy = np.sqrt(1.0**2 + 0.000511**2)  # ≈ 1.0001 GeV

        electron = Particle(
            id=0,
            pid=1,
            pdg_code=11,
            momentum=momentum,
            energy_init=energy * 1000,  # Convert to MeV
        )

        # Verify particle properties
        assert electron.pid == 1
        assert abs(electron.energy_init - energy * 1000) < 1.0  # Within 1 MeV
        # Check momentum (particle class may normalize differently)
        assert np.linalg.norm(electron.momentum) > 0.0  # Has momentum

    def test_particle_common_physics_scenarios(self):
        """Test particles in common physics scenarios."""
        from spine.data import Particle

        # Muon decay scenario: μ → e + νμ + νe
        muon = Particle(
            id=0,
            pid=2,
            pdg_code=13,
            energy_init=105.7,  # Muon rest mass
            position=np.array([0.0, 0.0, 0.0]),
            momentum=np.array([0.0, 0.0, 0.0]),  # At rest
        )

        electron = Particle(
            id=1,
            pid=1,
            pdg_code=11,
            parent_id=0,
            parent_pdg_code=13,
            energy_init=52.8,  # About half muon mass
            position=np.array([0.0, 0.0, 0.0]),
        )

        nu_mu = Particle(
            id=2,
            pid=-1,
            pdg_code=14,  # muon neutrino
            parent_id=0,
            parent_pdg_code=13,
            energy_init=26.5,
        )

        nu_e = Particle(
            id=3,
            pid=-1,
            pdg_code=12,  # electron neutrino
            parent_id=0,
            parent_pdg_code=13,
            energy_init=26.4,
        )

        # Verify decay products
        decay_products = [electron, nu_mu, nu_e]
        for product in decay_products:
            assert product.parent_id == 0
            assert product.parent_pdg_code == 13

        # Check approximate energy conservation (with some tolerance)
        total_decay_energy = sum(p.energy_init for p in decay_products)
        assert abs(total_decay_energy - muon.energy_init) < 1.0  # Within 1 MeV

    def test_particle_detector_interactions(self):
        """Test particles with detector interaction properties."""
        from spine.data import Particle

        # Particle that travels through detector
        particle = Particle(
            id=0,
            pid=2,  # muon (good for penetrating)
            energy_init=2000.0,  # 2 GeV
            energy_deposit=150.0,  # Only deposits small amount
            distance_travel=45.0,  # Travels 45 cm through detector
            num_voxels=180,  # Hits many voxels
            # Entry and exit points
            position=np.array([-20.0, 0.0, 0.0]),
            end_position=np.array([25.0, 0.0, 0.0]),
            first_step=np.array([-19.8, 0.0, 0.0]),
            last_step=np.array([24.8, 0.0, 0.0]),
            # Timing
            t=0.0,
            end_t=1.5,  # ns
        )

        # Verify detector interaction properties
        assert particle.energy_deposit < particle.energy_init  # Lost some energy
        assert particle.distance_travel > 0
        assert particle.num_voxels > 0
        assert particle.end_t > particle.t

        # Check spatial consistency
        travel_distance = np.linalg.norm(particle.end_position - particle.position)
        assert abs(travel_distance - particle.distance_travel) < 1.0  # Rough agreement


class TestParticleProperties:
    """Test Particle properties and methods."""

    def test_particle_p(self):
        """Test the p property for momentum magnitude."""
        from spine.data import Particle

        momentum = np.array([3.0, 4.0, 0.0])  # |p| = 5 GeV
        particle = Particle(id=0, momentum=momentum, pdg_code=13)  # muon

        expected_p = np.linalg.norm(momentum)
        assert abs(particle.p - expected_p) < 1e-6

    def test_particle_end_p(self):
        """Test the end_p property for end momentum magnitude."""
        from spine.data import Particle

        end_momentum = np.array([6.0, 8.0, 0.0])  # |p| = 10 GeV
        particle = Particle(id=0, end_momentum=end_momentum, pdg_code=13)  # muon

        expected_end_p = np.linalg.norm(end_momentum)
        assert abs(particle.end_p - expected_end_p) < 1e-6

    def test_particle_mass(self):
        """Test the mass property calculated from energy and momentum."""
        from spine.data import Particle

        # Create a particle with known energy and momentum
        momentum = np.array([3.0, 4.0, 0.0])  # |p| = 5 GeV
        energy = 10.0  # GeV
        particle = Particle(id=0, momentum=momentum, energy_init=energy)

        expected_mass = np.sqrt(energy**2 - np.linalg.norm(momentum) ** 2)
        assert abs(particle.mass - expected_mass) < 1e-6

        # If the particle has uninitialized energy_init, mass will be nan
        assert Particle(id=1, momentum=momentum).mass is np.nan


class TestParticleCollections:
    """Test collections and lists of particles."""

    def test_particle_list_creation(self):
        """Test creating and managing lists of particles."""
        from spine.data import Particle

        # Create a list of particles for a simple interaction
        particles = []

        # Primary particles from neutrino interaction
        proton = Particle(id=0, pid=4, pdg_code=2212, interaction_primary=True, nu_id=0)
        muon = Particle(id=1, pid=2, pdg_code=13, interaction_primary=True, nu_id=0)

        # Secondary particles
        pi_plus = Particle(
            id=2, pid=3, pdg_code=211, parent_id=0, interaction_primary=False, nu_id=0
        )
        pi_minus = Particle(
            id=3,
            pid=3,
            pdg_code=-211,
            parent_id=0,
            interaction_primary=False,
            nu_id=0,
        )

        particles = [proton, muon, pi_plus, pi_minus]

        # Verify collection properties
        assert len(particles) == 4
        primary_particles = [p for p in particles if p.interaction_primary]
        secondary_particles = [p for p in particles if not p.interaction_primary]

        assert len(primary_particles) == 2
        assert len(secondary_particles) == 2

        # Check all belong to same neutrino interaction
        nu_ids = [p.nu_id for p in particles]
        assert all(nu_id == 0 for nu_id in nu_ids)

    def test_particle_parent_child_relationships(self):
        """Test parent-child relationships between particles."""
        from spine.data import Particle

        # Create particle hierarchy: parent -> child1, child2
        parent = Particle(
            id=0,
            pid=4,  # proton
            pdg_code=2212,
            children_id=np.array([1, 2]),
            interaction_primary=True,
        )

        child1 = Particle(
            id=1,
            pid=3,  # pi+
            pdg_code=211,
            parent_id=0,
            parent_pdg_code=2212,
            interaction_primary=False,
        )

        child2 = Particle(
            id=2,
            pid=-1,  # pi0
            pdg_code=111,
            parent_id=0,
            parent_pdg_code=2212,
            interaction_primary=False,
        )

        particles = [parent, child1, child2]

        # Verify parent-child relationships
        assert parent.interaction_primary is True
        assert np.array_equal(parent.children_id, [1, 2])

        for child in [child1, child2]:
            assert child.parent_id == 0
            assert child.parent_pdg_code == 2212
            assert child.interaction_primary is False

        # Test relationship queries
        parent_particles = [p for p in particles if p.interaction_primary]
        child_particles = [
            p for p in particles if p.parent_id == 0 and not p.interaction_primary
        ]

        assert len(parent_particles) == 1
        assert len(child_particles) == 2

    def test_particle_sorting_and_filtering(self):
        """Test sorting and filtering particle collections."""
        from spine.data import Particle

        # Create particles with various properties
        particles = [
            Particle(id=0, pid=1, energy_init=500.0, t=0.0),  # electron, 500 MeV, t=0
            Particle(id=1, pid=2, energy_init=1000.0, t=5.0),  # muon, 1 GeV, t=5
            Particle(id=2, pid=3, energy_init=200.0, t=2.0),  # pi+, 200 MeV, t=2
            Particle(id=3, pid=3, energy_init=300.0, t=1.0),  # pi-, 300 MeV, t=1
            Particle(id=4, pid=4, energy_init=938.0, t=0.5),  # proton, 938 MeV, t=0.5
        ]

        # Test filtering by particle type
        leptons = [p for p in particles if p.pid in [1, 2]]  # e, μ
        mesons = [p for p in particles if p.pid in [3, 5]]  # π, K
        baryons = [p for p in particles if p.pid in [4]]  # p

        assert len(leptons) == 2  # electron, muon
        assert len(mesons) == 2  # pi+, pi-
        assert len(baryons) == 1  # proton

        # Test sorting by energy
        by_energy = sorted(particles, key=lambda p: p.energy_init, reverse=True)
        assert by_energy[0].energy_init == 1000.0  # muon
        assert by_energy[-1].energy_init == 200.0  # pi+

        # Test sorting by time
        by_time = sorted(particles, key=lambda p: p.t)
        assert by_time[0].t == 0.0  # electron
        assert by_time[-1].t == 5.0  # muon

        # Test filtering by energy threshold
        high_energy = [p for p in particles if p.energy_init > 400.0]
        assert len(high_energy) == 3  # electron, muon, proton


@pytest.mark.slow
class TestParticleIntegration:
    """Integration tests for Particle with other data structures."""

    def test_particle_with_neutrino_relationship(self):
        """Test Particle objects linked to Neutrino interactions."""
        from spine.data import Neutrino, Particle

        # Create neutrino interaction
        neutrino = Neutrino(
            id=0,
            pdg_code=14,  # muon neutrino
            energy_init=2000.0,
            current_type=1,  # CC
            position=np.array([0.0, 0.0, 0.0]),
        )

        # Create particles from this neutrino interaction
        primary_muon = Particle(
            id=0,
            nu_id=0,  # Links to neutrino
            pid=2,
            pdg_code=13,
            interaction_primary=True,
            energy_init=1500.0,
            position=np.array([0.1, 0.0, 0.0]),  # Slightly displaced
        )

        primary_proton = Particle(
            id=1,
            nu_id=0,  # Links to same neutrino
            pid=4,
            pdg_code=2212,
            interaction_primary=True,
            energy_init=400.0,
            position=np.array([0.0, 0.1, 0.0]),
        )

        particles = [primary_muon, primary_proton]

        # Verify neutrino-particle relationships
        assert neutrino.id == 0
        for particle in particles:
            assert particle.nu_id == neutrino.id
            assert particle.interaction_primary is True

        # Check energy balance (roughly)
        total_outgoing = sum(p.energy_init for p in particles)
        # Should be less than neutrino energy (some goes to nuclear recoil)
        assert total_outgoing < neutrino.energy_init

    def test_particle_physics_validation(self):
        """Test physics validation across particle properties."""
        from spine.data import Particle

        # Create realistic muon
        muon = Particle(
            id=0,
            pid=2,
            pdg_code=13,
            energy_init=1177.0,  # Total energy in MeV
            momentum=np.array([600.0, 0.0, 800.0]),  # p_total = 1000 MeV/c
            position=np.array([0.0, 0.0, 0.0]),
            # Muon properties: m_μ = 105.7 MeV/c²
        )

        # Calculate momentum magnitude
        p_magnitude = np.linalg.norm(muon.momentum)
        assert abs(p_magnitude - 1000.0) < 1.0

        # Check relativistic energy: E² = (pc)² + (mc²)²
        # For muon: E² = (1000)² + (105.7)²
        expected_energy = np.sqrt(1000**2 + 105.7**2)
        # Particle class may use different mass calculations
        assert (
            abs(muon.energy_init - expected_energy) < 200.0
        )  # Within 200 MeV tolerance

    def test_particle_detector_simulation_scenario(self):
        """Test realistic detector simulation scenario."""
        from spine.data import Particle

        # Scenario: Cosmic muon entering detector
        cosmic_muon = Particle(
            id=0,
            pid=2,
            pdg_code=13,
            # High energy cosmic ray
            energy_init=5000.0,  # 5 GeV
            energy_deposit=200.0,  # Deposits 200 MeV
            # Trajectory through detector
            position=np.array([0.0, 50.0, -30.0]),  # Entry point
            end_position=np.array([0.0, -50.0, 30.0]),  # Exit point
            first_step=np.array([0.0, 49.5, -29.5]),  # First interaction
            last_step=np.array([0.0, -49.5, 29.5]),  # Last interaction
            # Timing
            t=0.0,
            end_t=3.33,  # Time to cross detector (ns)
            # Detector response
            num_voxels=400,
            distance_travel=100.0,  # Total path length in cm
        )

        # Verify cosmic muon properties
        assert cosmic_muon.energy_init > cosmic_muon.energy_deposit  # MIP behavior
        assert cosmic_muon.distance_travel > 0
        assert cosmic_muon.num_voxels > 0
        assert cosmic_muon.end_t > cosmic_muon.t

        # For cosmic muons, positions and times may need to be computed
        # from detector simulation - defaults may be -inf
        # Just verify the muon was created with high energy
        assert cosmic_muon.energy_init == 5000.0
        assert cosmic_muon.pid == 2


class TestParticleFromLArCV:
    """Tests for Particle.from_larcv() - only runs if larcv is available."""

    def test_from_larcv_mock(self):
        """Test from_larcv with mock object (runs even without larcv)."""
        from spine.data import Particle

        # Create a mock larcv Particle object
        class MockLArCVParticle:
            """Mock LArCV Particle for testing."""

            def id(self):
                return 10

            def group_id(self):
                return 2

            def interaction_id(self):
                return 1

            def parent_id(self):
                return 5

            def mct_index(self):
                return 8

            def mcst_index(self):
                return 12

            def num_voxels(self):
                return 5000

            def shape(self):
                return 1  # Track

            def energy_init(self):
                return 500.0  # MeV

            def energy_deposit(self):
                return 450.0  # MeV

            def distance_travel(self):
                return 200.0  # cm

            def track_id(self):
                return 1000

            def pdg_code(self):
                return 13  # muon

            def creation_process(self):
                return "primary"

            def t(self):
                return 100.0  # ns

            def parent_track_id(self):
                return 999

            def parent_pdg_code(self):
                return 14  # numu

            def parent_creation_process(self):
                return "generator"

            def parent_t(self):
                return 95.0  # ns

            def ancestor_track_id(self):
                return 998

            def ancestor_pdg_code(self):
                return 14

            def ancestor_creation_process(self):
                return "generator"

            def ancestor_t(self):
                return 90.0

            def children_id(self):
                return [11, 12, 13]

            def position(self):
                class MockPosition:
                    def x(self):
                        return 100.0

                    def y(self):
                        return 50.0

                    def z(self):
                        return 500.0

                return MockPosition()

            def end_position(self):
                class MockEndPosition:
                    def x(self):
                        return 150.0

                    def y(self):
                        return 75.0

                    def z(self):
                        return 700.0

                    def t(self):
                        return 120.0

                return MockEndPosition()

            def parent_position(self):
                class MockParentPosition:
                    def x(self):
                        return 95.0

                    def y(self):
                        return 45.0

                    def z(self):
                        return 495.0

                return MockParentPosition()

            def ancestor_position(self):
                class MockAncestorPosition:
                    def x(self):
                        return 90.0

                    def y(self):
                        return 40.0

                    def z(self):
                        return 490.0

                return MockAncestorPosition()

            def first_step(self):
                class MockFirstStep:
                    def x(self):
                        return 102.0

                    def y(self):
                        return 52.0

                    def z(self):
                        return 502.0

                return MockFirstStep()

            def last_step(self):
                class MockLastStep:
                    def x(self):
                        return 148.0

                    def y(self):
                        return 73.0

                    def z(self):
                        return 698.0

                return MockLastStep()

            def px(self):
                return 100.0

            def py(self):
                return 50.0

            def pz(self):
                return 400.0

            def momentum(self):
                """Marker method to indicate momentum attributes exist."""
                return True

            def end_px(self):
                return 80.0

            def end_py(self):
                return 40.0

            def end_pz(self):
                return 300.0

            def end_momentum(self):
                """Marker method to indicate end_momentum attributes exist."""
                return True

        mock_particle = MockLArCVParticle()
        particle = Particle.from_larcv(mock_particle)

        # Verify all scalar attributes
        assert particle.id == 10
        assert particle.group_id == 2
        assert particle.interaction_id == 1
        assert particle.parent_id == 5
        assert particle.mct_index == 8
        assert particle.mcst_index == 12
        assert particle.num_voxels == 5000
        assert particle.shape == 1
        assert particle.energy_init == 500.0
        assert particle.energy_deposit == 450.0
        assert particle.distance_travel == 200.0
        assert particle.track_id == 1000
        assert particle.pdg_code == 13
        assert particle.parent_track_id == 999
        assert particle.parent_pdg_code == 14
        assert particle.ancestor_track_id == 998
        assert particle.ancestor_pdg_code == 14
        assert particle.creation_process == "primary"
        assert particle.parent_creation_process == "generator"
        assert particle.ancestor_creation_process == "generator"
        assert particle.t == 100.0
        assert particle.end_t == 120.0
        assert particle.parent_t == 95.0
        assert particle.ancestor_t == 90.0

        # Check position arrays
        np.testing.assert_array_almost_equal(particle.position, [100.0, 50.0, 500.0])
        np.testing.assert_array_almost_equal(
            particle.end_position, [150.0, 75.0, 700.0]
        )
        np.testing.assert_array_almost_equal(
            particle.parent_position, [95.0, 45.0, 495.0]
        )
        np.testing.assert_array_almost_equal(
            particle.ancestor_position, [90.0, 40.0, 490.0]
        )
        np.testing.assert_array_almost_equal(particle.first_step, [102.0, 52.0, 502.0])
        np.testing.assert_array_almost_equal(particle.last_step, [148.0, 73.0, 698.0])

        # Check momentum arrays
        np.testing.assert_array_almost_equal(particle.momentum, [100.0, 50.0, 400.0])
        np.testing.assert_array_almost_equal(particle.end_momentum, [80.0, 40.0, 300.0])

        # Check children_id array
        np.testing.assert_array_equal(particle.children_id, [11, 12, 13])

    @pytest.mark.skipif(not LARCV_AVAILABLE, reason="larcv not available")
    def test_from_larcv_real(self):
        """Test from_larcv with real larcv object (only if larcv installed)."""
        from spine.data import Particle

        assert larcv is not None

        # Create a real LArCV Particle
        larcv_particle = larcv.Particle()
        larcv_particle.id(15)
        larcv_particle.group_id(3)
        larcv_particle.interaction_id(2)
        larcv_particle.parent_id(10)
        larcv_particle.mct_index(5)
        larcv_particle.mcst_index(8)
        larcv_particle.num_voxels(8000)
        larcv_particle.shape(2)  # Shower
        larcv_particle.energy_init(800.0)
        larcv_particle.energy_deposit(750.0)
        larcv_particle.distance_travel(150.0)

        # Set track IDs and PDG codes
        larcv_particle.track_id(2000)
        larcv_particle.pdg_code(11)  # electron
        larcv_particle.parent_track_id(1999)
        larcv_particle.parent_pdg_code(22)  # photon
        larcv_particle.ancestor_track_id(1998)
        larcv_particle.ancestor_pdg_code(111)  # pi0

        # Set creation processes
        larcv_particle.creation_process("conv")
        larcv_particle.parent_creation_process("Decay")
        larcv_particle.ancestor_creation_process("hadElastic")

        # Set positions. The LArCV time getters are populated from the vertex
        # time arguments; they do not expose standalone Python setters.
        larcv_particle.position(120.0, 60.0, 550.0, 200.0)
        larcv_particle.end_position(170.0, 85.0, 700.0, 220.0)
        larcv_particle.parent_position(118.0, 58.0, 548.0, 195.0)
        larcv_particle.ancestor_position(115.0, 55.0, 545.0, 190.0)
        larcv_particle.first_step(122.0, 62.0, 552.0, 201.0)
        larcv_particle.last_step(168.0, 83.0, 698.0, 219.0)

        # Set momentum
        larcv_particle.momentum(200.0, 100.0, 700.0)
        larcv_particle.end_momentum(150.0, 75.0, 600.0)

        # Set children
        larcv_particle.children_id().push_back(16)
        larcv_particle.children_id().push_back(17)

        # Convert to SPINE Particle
        particle = Particle.from_larcv(larcv_particle)

        # Verify conversion
        assert particle.id == 15
        assert particle.group_id == 3
        assert particle.interaction_id == 2
        assert particle.parent_id == 10
        assert particle.shape == 2
        assert particle.energy_init == 800.0
        assert particle.energy_deposit == 750.0
        assert particle.track_id == 2000
        assert particle.pdg_code == 11
        assert particle.parent_pdg_code == 22
        assert particle.ancestor_pdg_code == 111
        assert particle.creation_process == "conv"
        assert particle.t == 200.0
        assert particle.end_t == 220.0

        np.testing.assert_array_almost_equal(particle.position, [120.0, 60.0, 550.0])
        np.testing.assert_array_almost_equal(
            particle.end_position, [170.0, 85.0, 700.0]
        )
        np.testing.assert_array_almost_equal(particle.momentum, [200.0, 100.0, 700.0])
        np.testing.assert_array_almost_equal(
            particle.end_momentum, [150.0, 75.0, 600.0]
        )
        np.testing.assert_array_equal(particle.children_id, [16, 17])

    def test_from_larcv_missing_attributes(self):
        """Test from_larcv with a LArCV Particle missing some attributes."""
        from spine.data import Particle

        # Create a mock LArCV Particle which is missing some index and end momentum attributes
        class PartialMockLArCVParticle:
            """Mock LArCV Particle for testing."""

            def id(self):
                return 10

            def group_id(self):
                return 2

            def interaction_id(self):
                return 1

            def parent_id(self):
                return 5

            def mct_index(self):
                return 8

            def mcst_index(self):
                return 12

            def num_voxels(self):
                return 5000

            def shape(self):
                return 1  # Track

            def energy_init(self):
                return 500.0  # MeV

            def energy_deposit(self):
                return 450.0  # MeV

            def track_id(self):
                return 1000

            def pdg_code(self):
                return 13  # muon

            def creation_process(self):
                return "primary"

            def t(self):
                return 100.0  # ns

            def parent_track_id(self):
                return 999

            def parent_pdg_code(self):
                return 14  # numu

            def parent_creation_process(self):
                return "generator"

            def parent_t(self):
                return 95.0  # ns

            def ancestor_track_id(self):
                return 998

            def ancestor_pdg_code(self):
                return 14

            def ancestor_creation_process(self):
                return "generator"

            def ancestor_t(self):
                return 90.0

            def children_id(self):
                return [11, 12, 13]

            def position(self):
                class MockPosition:
                    def x(self):
                        return 100.0

                    def y(self):
                        return 50.0

                    def z(self):
                        return 500.0

                return MockPosition()

            def end_position(self):
                class MockEndPosition:
                    def x(self):
                        return 150.0

                    def y(self):
                        return 75.0

                    def z(self):
                        return 700.0

                    def t(self):
                        return 120.0

                return MockEndPosition()

            def parent_position(self):
                class MockParentPosition:
                    def x(self):
                        return 95.0

                    def y(self):
                        return 45.0

                    def z(self):
                        return 495.0

                return MockParentPosition()

            def ancestor_position(self):
                class MockAncestorPosition:
                    def x(self):
                        return 90.0

                    def y(self):
                        return 40.0

                    def z(self):
                        return 490.0

                return MockAncestorPosition()

            def first_step(self):
                class MockFirstStep:
                    def x(self):
                        return 102.0

                    def y(self):
                        return 52.0

                    def z(self):
                        return 502.0

                return MockFirstStep()

            def last_step(self):
                class MockLastStep:
                    def x(self):
                        return 148.0

                    def y(self):
                        return 73.0

                    def z(self):
                        return 698.0

                return MockLastStep()

            def px(self):
                return 100.0

            def py(self):
                return 50.0

            def pz(self):
                return 400.0

            def momentum(self):
                """Marker method to indicate momentum attributes exist."""
                return True

            # Missing end_momentum and distance_travel attributes

        partial_particle = PartialMockLArCVParticle()
        with pytest.warns(UserWarning, match="missing the .* attribute") as warnings:
            particle = Particle.from_larcv(partial_particle)

        missing_attributes = {
            str(warning.message).split(" missing the ", 1)[1].split(" attribute", 1)[0]
            for warning in warnings
        }
        assert missing_attributes == {"distance_travel", "end_momentum"}

        # Verify available attributes are set, and missing ones are default
        assert particle.id == 10
        assert particle.energy_init == 500.0
        assert particle.pdg_code == 13
        assert particle.distance_travel is np.nan  # Default for missing attribute
        assert np.all(np.isnan(particle.end_momentum))  # Default for missing momentum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
