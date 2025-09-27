"""Comprehensive test suite for spine.data.neutrino module."""

import numpy as np
import pytest


class TestNeutrinoCreation:
    """Test Neutrino class creation and initialization."""

    def test_neutrino_basic_creation(self):
        """Test basic Neutrino instantiation."""
        from spine.data import Neutrino

        # Test creation with minimal parameters
        neutrino = Neutrino(id=0)
        assert neutrino.id == 0

        # Test creation with common neutrino parameters
        neutrino = Neutrino(
            id=1,
            pdg_code=14,  # muon neutrino
            energy_init=2000.0,  # 2 GeV
            position=np.array([0.0, 0.0, 0.0]),
            current_type=1,  # Charged current
            interaction_mode=1,  # QE
        )

        assert neutrino.id == 1
        assert neutrino.pdg_code == 14
        assert neutrino.energy_init == 2000.0
        assert np.allclose(neutrino.position, [0.0, 0.0, 0.0])
        assert neutrino.current_type == 1
        assert neutrino.interaction_mode == 1

    def test_neutrino_physics_properties(self):
        """Test Neutrino with complete physics properties."""
        from spine.data import Neutrino

        # Charged current muon neutrino interaction
        neutrino = Neutrino(
            id=0,
            pdg_code=14,  # νμ
            lepton_pdg_code=13,  # μ⁻
            current_type=1,  # CC
            interaction_mode=1,  # QE
            interaction_type=1,  # neutrino (not antineutrino)
            # Kinematic properties
            energy_init=1500.0,  # 1.5 GeV
            energy_transfer=1200.0,  # Energy transferred to hadronic system
            momentum=np.array([0.0, 0.0, 1500.0]),  # Beam direction
            position=np.array([10.0, 5.0, 100.0]),  # Interaction vertex
            # Target properties
            target=1000060120,  # Carbon-12 nucleus
            nucleon=2212,  # Proton target
            # Timing
            t=50.0,  # ns
        )

        # Verify physics properties
        assert neutrino.pdg_code == 14  # muon neutrino
        assert neutrino.lepton_pdg_code == 13  # produces muon
        assert neutrino.current_type == 1  # charged current
        assert neutrino.interaction_mode == 1  # quasi-elastic
        assert neutrino.energy_init == 1500.0
        assert neutrino.energy_transfer == 1200.0
        assert neutrino.target == 1000060120  # Carbon-12
        assert neutrino.nucleon == 2212  # proton

    def test_neutrino_interaction_types(self):
        """Test different neutrino interaction types."""
        from spine.data import Neutrino

        # Quasi-elastic interaction
        nu_qe = Neutrino(
            id=0,
            pdg_code=14,
            current_type=1,  # CC
            interaction_mode=1,  # QE
            nucleon=2212,  # proton target
            energy_init=800.0,
        )

        # Deep inelastic scattering
        nu_dis = Neutrino(
            id=1,
            pdg_code=14,
            current_type=1,  # CC
            interaction_mode=3,  # DIS
            quark=2,  # up quark
            energy_init=5000.0,
        )

        # Neutral current interaction
        nu_nc = Neutrino(
            id=2,
            pdg_code=14,
            current_type=2,  # NC
            interaction_mode=1,  # QE
            energy_init=1000.0,
        )

        # Verify interaction classifications
        assert nu_qe.interaction_mode == 1
        assert nu_qe.nucleon == 2212

        assert nu_dis.interaction_mode == 3
        assert nu_dis.quark == 2

        assert nu_nc.current_type == 2  # Neutral current

    def test_neutrino_track_and_index_properties(self):
        """Test Neutrino track IDs and index properties."""
        from spine.data import Neutrino

        neutrino = Neutrino(
            id=0,
            interaction_id=5,  # Generator-level ID
            mct_index=12,  # MCTruth index
            track_id=2001,  # Geant4 track ID
            lepton_track_id=2002,  # Outgoing lepton track ID
        )

        assert neutrino.id == 0
        assert neutrino.interaction_id == 5
        assert neutrino.mct_index == 12
        assert neutrino.track_id == 2001
        assert neutrino.lepton_track_id == 2002


class TestNeutrinoPhysics:
    """Test Neutrino physics scenarios and calculations."""

    def test_charged_current_interactions(self):
        """Test charged current neutrino interactions."""
        from spine.data import Neutrino

        # νμ CC interaction producing μ⁻
        nu_mu_cc = Neutrino(
            id=0,
            pdg_code=14,  # νμ
            lepton_pdg_code=13,  # μ⁻
            current_type=1,  # CC
            energy_init=2000.0,
            energy_transfer=1500.0,  # Energy to hadronic system
        )

        # ν̄μ CC interaction producing μ⁺
        nubar_mu_cc = Neutrino(
            id=1,
            pdg_code=-14,  # ν̄μ
            lepton_pdg_code=-13,  # μ⁺
            current_type=1,  # CC
            energy_init=3000.0,
            energy_transfer=2200.0,
        )

        # Verify CC interaction properties
        assert nu_mu_cc.current_type == 1
        assert nu_mu_cc.pdg_code == 14
        assert nu_mu_cc.lepton_pdg_code == 13

        assert nubar_mu_cc.current_type == 1
        assert nubar_mu_cc.pdg_code == -14
        assert nubar_mu_cc.lepton_pdg_code == -13

        # Check energy conservation (roughly)
        for nu in [nu_mu_cc, nubar_mu_cc]:
            assert nu.energy_transfer < nu.energy_init  # Some goes to lepton

    def test_neutral_current_interactions(self):
        """Test neutral current neutrino interactions."""
        from spine.data import Neutrino

        # NC interaction - neutrino doesn't change flavor
        nu_nc = Neutrino(
            id=0,
            pdg_code=14,  # νμ in
            current_type=2,  # NC (no lepton production)
            energy_init=1000.0,
            energy_transfer=200.0,  # Small energy transfer
            interaction_mode=1,  # Elastic scattering
        )

        # Verify NC properties
        assert nu_nc.current_type == 2
        assert nu_nc.energy_transfer < nu_nc.energy_init
        assert nu_nc.energy_transfer / nu_nc.energy_init < 0.5  # Typical for NC

    def test_neutrino_energy_ranges(self):
        """Test neutrinos across different energy ranges."""
        from spine.data import Neutrino

        # Low energy neutrino (MeV range)
        low_e_nu = Neutrino(
            id=0,
            pdg_code=12,  # νe
            energy_init=50.0,  # 50 MeV
            interaction_mode=1,  # QE likely
        )

        # Medium energy neutrino (GeV range)
        med_e_nu = Neutrino(
            id=1,
            pdg_code=14,  # νμ
            energy_init=2000.0,  # 2 GeV
            interaction_mode=2,  # Resonant production possible
        )

        # High energy neutrino (multi-GeV)
        high_e_nu = Neutrino(
            id=2,
            pdg_code=14,  # νμ
            energy_init=10000.0,  # 10 GeV
            interaction_mode=3,  # DIS likely
        )

        # Verify energy-dependent properties
        assert low_e_nu.energy_init < 100.0
        assert 1000.0 < med_e_nu.energy_init < 5000.0
        assert high_e_nu.energy_init > 5000.0

        # Higher energy often correlates with different interaction modes
        assert low_e_nu.interaction_mode <= 1  # QE dominant at low E
        assert high_e_nu.interaction_mode >= 2  # Inelastic at high E

    def test_neutrino_flavors(self):
        """Test different neutrino flavors."""
        from spine.data import Neutrino

        # Electron neutrino
        nu_e = Neutrino(
            id=0,
            pdg_code=12,  # νe
            lepton_pdg_code=11,  # e⁻
            current_type=1,
            energy_init=500.0,
        )

        # Muon neutrino
        nu_mu = Neutrino(
            id=1,
            pdg_code=14,  # νμ
            lepton_pdg_code=13,  # μ⁻
            current_type=1,
            energy_init=1500.0,
        )

        # Tau neutrino
        nu_tau = Neutrino(
            id=2,
            pdg_code=16,  # ντ
            lepton_pdg_code=15,  # τ⁻
            current_type=1,
            energy_init=8000.0,  # Need high energy for tau production
        )

        neutrinos = [nu_e, nu_mu, nu_tau]

        # Verify flavor assignments
        pdg_codes = [12, 14, 16]  # νe, νμ, ντ
        lepton_codes = [11, 13, 15]  # e, μ, τ

        for i, nu in enumerate(neutrinos):
            assert nu.pdg_code == pdg_codes[i]
            assert nu.lepton_pdg_code == lepton_codes[i]


class TestNeutrinoInteractionScenarios:
    """Test realistic neutrino interaction scenarios."""

    def test_atmospheric_neutrino_interaction(self):
        """Test atmospheric neutrino interaction scenario."""
        from spine.data import Neutrino

        # Atmospheric νμ from cosmic ray interaction
        atm_nu = Neutrino(
            id=0,
            pdg_code=14,  # νμ
            current_type=1,  # CC
            interaction_mode=1,  # QE
            energy_init=800.0,  # Typical atmospheric energy
            energy_transfer=600.0,
            # Interaction in detector
            position=np.array([12.5, -8.3, 45.2]),
            momentum=np.array([200.0, -150.0, 750.0]),  # Downward going
            # Target nucleus
            target=1000180400,  # Argon-40 (typical LAr detector)
            nucleon=2212,  # Proton
            # Timing
            t=125.0,
        )

        # Verify atmospheric neutrino properties
        assert atm_nu.pdg_code == 14
        assert atm_nu.current_type == 1
        assert 100.0 < atm_nu.energy_init < 5000.0  # Typical atmospheric range
        assert atm_nu.target == 1000180400  # Argon target

        # Check momentum direction (should be somewhat downward)
        momentum_norm = atm_nu.momentum / np.linalg.norm(atm_nu.momentum)
        assert momentum_norm[2] > 0  # Positive z component (downward)

    def test_beam_neutrino_interaction(self):
        """Test accelerator beam neutrino interaction."""
        from spine.data import Neutrino

        # FNAL beam νμ (NuMI-style)
        beam_nu = Neutrino(
            id=0,
            pdg_code=14,  # νμ
            current_type=1,  # CC
            interaction_mode=2,  # Resonant production
            energy_init=3200.0,  # Peak of beam spectrum
            energy_transfer=2400.0,
            # Forward direction (beam along +z)
            position=np.array([2.1, -1.5, 200.0]),
            momentum=np.array([50.0, -30.0, 3190.0]),  # Nearly forward
            # Target and products
            target=1000060120,  # Carbon-12
            nucleon=2112,  # Neutron
            # Generator info
            interaction_id=12345,
            mct_index=67,
        )

        # Verify beam neutrino properties
        assert beam_nu.current_type == 1
        assert 1000.0 < beam_nu.energy_init < 10000.0  # Beam energy range
        assert beam_nu.interaction_id == 12345

        # Check beam direction (should be very forward)
        momentum_norm = beam_nu.momentum / np.linalg.norm(beam_nu.momentum)
        assert momentum_norm[2] > 0.95  # Very forward-going

    def test_solar_neutrino_interaction(self):
        """Test low-energy solar neutrino interaction."""
        from spine.data import Neutrino

        # Solar νe (8B neutrinos)
        solar_nu = Neutrino(
            id=0,
            pdg_code=12,  # νe
            current_type=1,  # CC
            interaction_mode=1,  # QE (only possibility at low E)
            energy_init=12.0,  # MeV range
            energy_transfer=8.0,  # Small energy transfer
            # Random direction
            position=np.array([0.5, 1.2, -0.8]),
            momentum=np.array([3.0, -2.0, 11.0]),
            # Light target
            target=1000010010,  # Hydrogen
            nucleon=2212,  # Proton
        )

        # Verify solar neutrino properties
        assert solar_nu.pdg_code == 12  # νe
        assert solar_nu.energy_init < 20.0  # Low energy
        assert solar_nu.interaction_mode == 1  # QE only at low E
        assert solar_nu.target == 1000010010  # Light target

    def test_supernova_neutrino_burst(self):
        """Test supernova neutrino burst scenario."""
        from spine.data import Neutrino

        # Create multiple neutrinos from SN burst
        sn_neutrinos = []

        for i in range(5):
            # Mix of flavors and energies typical of SN
            pdg_codes = [12, -12, 14, -14, 16]  # νe, ν̄e, νμ, ν̄μ, ντ
            energies = [15.0, 18.0, 25.0, 22.0, 30.0]  # MeV

            sn_nu = Neutrino(
                id=i,
                pdg_code=pdg_codes[i],
                energy_init=energies[i],
                current_type=(
                    1 if abs(pdg_codes[i]) == 12 else 2
                ),  # CC for νe, NC for others
                interaction_mode=1,  # QE at these energies
                # All from same direction (SN)
                position=np.array([0.0, 0.0, 0.0]) + np.random.normal(0, 0.1, 3),
                momentum=energies[i] * np.array([0.1, 0.05, 0.99]),  # From SN direction
                t=100.0 + i * 0.001,  # Burst within ~ms
            )
            sn_neutrinos.append(sn_nu)

        # Verify SN burst properties
        assert len(sn_neutrinos) == 5

        # Check energy range typical of SN
        for nu in sn_neutrinos:
            assert 10.0 < nu.energy_init < 50.0

        # Check time clustering (within 1 ms)
        times = [nu.t for nu in sn_neutrinos]
        time_spread = max(times) - min(times)
        assert time_spread < 0.01  # Within 0.01 ns (very tight)

        # Check directional clustering
        directions = [nu.momentum / np.linalg.norm(nu.momentum) for nu in sn_neutrinos]
        for direction in directions:
            assert direction[2] > 0.9  # All roughly from same direction


class TestNeutrinoCollections:
    """Test collections and interactions of multiple neutrinos."""

    def test_neutrino_event_collection(self):
        """Test collection of neutrinos in single event."""
        from spine.data import Neutrino

        # Single event with multiple neutrino interactions (rare but possible)
        neutrinos = [
            Neutrino(id=0, pdg_code=14, energy_init=2000.0, interaction_id=0),
            Neutrino(id=1, pdg_code=14, energy_init=1500.0, interaction_id=1),
            Neutrino(id=2, pdg_code=-14, energy_init=800.0, interaction_id=2),
        ]

        # Verify collection properties
        assert len(neutrinos) == 3

        # Check unique interaction IDs
        interaction_ids = [nu.interaction_id for nu in neutrinos]
        assert len(set(interaction_ids)) == 3  # All different

        # Check energy ordering
        by_energy = sorted(neutrinos, key=lambda nu: nu.energy_init, reverse=True)
        assert by_energy[0].energy_init == 2000.0
        assert by_energy[-1].energy_init == 800.0

    def test_neutrino_flavor_oscillations(self):
        """Test neutrino flavor mixing scenarios."""
        from spine.data import Neutrino

        # Beam νμ that oscillated to νe (simplified representation)
        original_flavor = Neutrino(
            id=0,
            pdg_code=14,  # Born as νμ
            energy_init=2000.0,
            position=np.array([0.0, 0.0, 0.0]),  # Production point
        )

        oscillated_flavor = Neutrino(
            id=1,
            pdg_code=12,  # Detected as νe
            energy_init=2000.0,  # Same energy
            position=np.array([0.0, 0.0, 735000.0]),  # 735 km baseline
        )

        # Verify oscillation scenario
        assert original_flavor.pdg_code == 14
        assert oscillated_flavor.pdg_code == 12
        assert original_flavor.energy_init == oscillated_flavor.energy_init

        # Check baseline distance
        baseline = np.linalg.norm(oscillated_flavor.position - original_flavor.position)
        assert baseline > 700000.0  # > 700 km


@pytest.mark.slow
class TestNeutrinoIntegration:
    """Integration tests for Neutrino with other data structures."""

    def test_neutrino_with_particle_products(self):
        """Test Neutrino linked to produced particles."""
        from spine.data import Neutrino, Particle

        # CC νμ interaction
        neutrino = Neutrino(
            id=0,
            pdg_code=14,  # νμ
            lepton_pdg_code=13,  # produces μ⁻
            current_type=1,  # CC
            energy_init=2500.0,
            energy_transfer=1800.0,
            lepton_track_id=3001,  # Links to muon
        )

        # Outgoing muon from this interaction
        muon = Particle(
            id=0,
            nu_id=0,  # Links to neutrino
            pid=13,
            pdg_code=13,
            track_id=3001,  # Matches neutrino.lepton_track_id
            interaction_primary=True,
            energy_init=700.0,  # Part of neutrino energy
        )

        # Recoil proton
        proton = Particle(
            id=1,
            nu_id=0,  # Same neutrino interaction
            pid=2212,
            pdg_code=2212,
            interaction_primary=True,
            energy_init=1100.0,  # Rest of hadronic energy
        )

        particles = [muon, proton]

        # Verify neutrino-particle links
        assert neutrino.lepton_track_id == muon.track_id
        for particle in particles:
            assert particle.nu_id == neutrino.id
            assert particle.interaction_primary is True

        # Check approximate energy conservation
        total_outgoing = sum(p.energy_init for p in particles)
        # Should be close to neutrino energy_transfer
        assert abs(total_outgoing - neutrino.energy_transfer) < 100.0

    def test_realistic_detector_event(self):
        """Test complete realistic neutrino detector event."""
        from spine.data import Neutrino, Particle

        # Beam νμ CC QE interaction: νμ + p → μ⁻ + p'
        beam_nu = Neutrino(
            id=0,
            pdg_code=14,
            current_type=1,
            interaction_mode=1,  # QE
            energy_init=1800.0,
            energy_transfer=1400.0,
            position=np.array([10.2, 5.7, 150.0]),
            target=1000180400,  # Argon-40
            nucleon=2212,  # Proton target
            lepton_track_id=4001,
        )

        # Primary muon
        primary_muon = Particle(
            id=0,
            nu_id=0,
            pid=13,
            pdg_code=13,
            track_id=4001,
            interaction_primary=True,
            energy_init=900.0,
            energy_deposit=180.0,  # MIP-like
            distance_travel=75.0,
            num_voxels=300,
            position=np.array([10.2, 5.7, 150.0]),  # Same vertex
            end_position=np.array([85.2, 5.7, 150.0]),  # Exits detector
        )

        # Recoil proton
        recoil_proton = Particle(
            id=1,
            nu_id=0,
            pid=2212,
            pdg_code=2212,
            interaction_primary=True,
            energy_init=500.0,
            energy_deposit=450.0,  # Stops in detector
            distance_travel=15.0,  # Short range
            num_voxels=60,
            position=np.array([10.2, 5.7, 150.0]),  # Same vertex
            end_position=np.array([25.2, 5.7, 150.0]),  # Stops
        )

        event_particles = [primary_muon, recoil_proton]

        # Verify complete event
        assert beam_nu.interaction_mode == 1  # QE
        assert len(event_particles) == 2

        # Check vertex consistency
        for particle in event_particles:
            assert np.allclose(particle.position, beam_nu.position, atol=0.1)
            assert particle.nu_id == beam_nu.id

        # Check detector response realism
        assert (
            primary_muon.distance_travel > recoil_proton.distance_travel
        )  # Muon penetrates
        assert primary_muon.energy_deposit < primary_muon.energy_init  # MIP exits
        assert (
            abs(recoil_proton.energy_deposit - recoil_proton.energy_init) <= 50.0
        )  # Proton stops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
