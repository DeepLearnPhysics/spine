"""Test Interaction classes."""

import numpy as np


class TestInteractionBase:
    """Test InteractionBase functionality."""

    def test_interactionbase_initialization(self):
        """Test InteractionBase initialization with default values."""
        from spine.data.out.interaction import InteractionBase

        obj = InteractionBase(id=0)

        # Test scalar attributes
        assert obj.is_fiducial is False
        assert obj.is_flash_matched is False
        assert np.isnan(obj.flash_total_pe)
        assert np.isnan(obj.flash_hypo_pe)

        # Test object list
        assert len(obj.particles) == 0

        # Test vector attributes
        assert len(obj.particle_ids) == 0
        assert obj.particle_ids.dtype == np.int64

        assert obj.vertex.shape == (3,)
        assert obj.vertex.dtype == np.float32
        assert all(np.isnan(obj.vertex))

        # Test flash arrays
        assert len(obj.flash_ids) == 0
        assert len(obj.flash_volume_ids) == 0
        assert len(obj.flash_times) == 0
        assert len(obj.flash_scores) == 0

    def test_interactionbase_with_data(self):
        """Test InteractionBase with complete data."""
        from spine.data.out.interaction import InteractionBase

        vertex = np.array([10.5, 20.5, 30.5], dtype=np.float32)
        obj = InteractionBase(
            id=3,
            is_fiducial=True,
            particle_ids=np.array([5, 10, 15], dtype=np.int64),
            vertex=vertex,
        )

        assert obj.id == 3
        assert obj.is_fiducial is True
        assert len(obj.particle_ids) == 3
        np.testing.assert_array_equal(obj.particle_ids, [5, 10, 15])
        np.testing.assert_allclose(obj.vertex, vertex)

    def test_interactionbase_reset_flash_match(self):
        """Test InteractionBase reset_flash_match method."""
        from spine.data.out.interaction import InteractionBase

        # Create interaction with flash matching
        obj = InteractionBase(
            id=0,
            is_flash_matched=True,
            flash_total_pe=1500.0,
            flash_hypo_pe=1400.0,
            flash_ids=np.array([5], dtype=np.int64),
            flash_volume_ids=np.array([0], dtype=np.int64),
            flash_times=np.array([100.5], dtype=np.float32),
            flash_scores=np.array([0.95], dtype=np.float32),
        )

        # Verify initial state
        assert obj.is_flash_matched is True
        assert obj.flash_total_pe == 1500.0

        # Reset flash matching
        obj.reset_flash_match()

        # Verify reset
        assert obj.is_flash_matched is False
        assert np.isnan(obj.flash_total_pe)
        assert np.isnan(obj.flash_hypo_pe)
        assert len(obj.flash_ids) == 0
        assert len(obj.flash_volume_ids) == 0
        assert len(obj.flash_times) == 0
        assert len(obj.flash_scores) == 0

    def test_interactionbase_str_representation(self):
        """Test InteractionBase string representation."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        obj = InteractionBase(
            id=2,
            match_ids=np.array([10], dtype=np.int64),
            particle_ids=np.array([5, 10], dtype=np.int64),
            particles=[ParticleBase(id=5), ParticleBase(id=10)],
        )

        str_repr = str(obj)
        print(str_repr)
        assert "Interaction" in str_repr
        assert "2" in str_repr  # ID
        assert "Particle(ID: 5" in str_repr
        assert "Particle(ID: 10" in str_repr

    def test_interactionbase_primary_particles_properties(self):
        """Test InteractionBase primary_particles property."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        # Create interaction with particles
        obj = InteractionBase(
            id=0,
            particle_ids=np.array([1, 2, 3], dtype=np.int64),
            particles=[
                ParticleBase(id=1, is_primary=True),
                ParticleBase(id=2, is_primary=False),
                ParticleBase(id=3, is_primary=True),
            ],
        )

        primary_particles = obj.primary_particles
        assert len(primary_particles) == 2
        assert all(p.is_primary for p in primary_particles)
        assert {p.id for p in primary_particles} == {1, 3}

        primary_particle_ids = obj.primary_particle_ids
        assert len(primary_particle_ids) == 2
        assert set(primary_particle_ids) == {1, 3}

    def test_interactionbase_num_particles_properties(self):
        """Test InteractionBase num_particles property."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        obj = InteractionBase(
            id=0,
            particle_ids=np.array([1, 2, 3], dtype=np.int64),
            particles=[
                ParticleBase(id=1, is_primary=True),
                ParticleBase(id=2, is_primary=False),
                ParticleBase(id=3, is_primary=False),
            ],
        )

        assert obj.num_particles == 3
        assert obj.num_primary_particles == 1

    def test_interactionbase_particle_counts_properties(self):
        """Test InteractionBase particle_counts property."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        obj = InteractionBase(
            id=0,
            particle_ids=np.array([1, 2, 3, 4], dtype=np.int64),
            particles=[
                ParticleBase(id=1, pid=0, is_primary=True),  # photon
                ParticleBase(id=2, pid=1, is_primary=False),  # electron
                ParticleBase(id=3, pid=2, is_primary=True),  # muon
                ParticleBase(id=4, pid=3, is_primary=True),  # pion
                ParticleBase(id=5, pid=3, is_primary=False),  # pion
            ],
        )

        counts = obj.particle_counts
        assert counts[0] == 1
        assert counts[1] == 1
        assert counts[2] == 1
        assert counts[3] == 2

        counts_primary = obj.primary_particle_counts
        assert counts_primary[0] == 1
        assert counts_primary[1] == 0
        assert counts_primary[2] == 1
        assert counts_primary[3] == 1

    def test_interactionbase_crt_properties(self):
        """Test InteractionBase properties related to CRT information."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        # Test on an interaction with no particles
        obj = InteractionBase(id=0)
        assert obj.is_crt_matched is False
        assert len(obj.crt_ids) == 0
        assert len(obj.crt_scores) == 0
        assert len(obj.crt_times) == 0

        # Test with valid CRT information
        obj = InteractionBase(
            id=0,
            particles=[
                ParticleBase(
                    id=1,
                    is_crt_matched=True,
                    crt_ids=np.array([5], dtype=np.int64),
                    crt_scores=np.array([0.85], dtype=np.float32),
                    crt_times=np.array([100.5], dtype=np.float32),
                ),
                ParticleBase(id=2, is_crt_matched=False),
            ],
        )

        assert obj.is_crt_matched is True
        assert len(obj.crt_ids) == 1
        assert np.array_equal(obj.crt_ids, [5])
        print(obj.crt_scores)
        assert len(obj.crt_scores) == 1
        assert np.array_equal(obj.crt_scores, np.array([0.85], dtype=np.float32))
        assert len(obj.crt_times) == 1
        assert np.array_equal(obj.crt_times, np.array([100.5], dtype=np.float32))

    def test_interactionbase_topology_property(self):
        """Test InteractionBase topology property."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        # Test on an interaction with no particles
        obj = InteractionBase(id=0)
        assert obj.topology == ""

        # Test with valid particle information
        obj = InteractionBase(
            id=0,
            particles=[
                ParticleBase(id=1, pid=0, is_primary=True),  # photon
                ParticleBase(id=2, pid=1, is_primary=False),  # electron
                ParticleBase(id=3, pid=2, is_primary=True),  # muon
                ParticleBase(id=4, pid=3, is_primary=True),  # pion
                ParticleBase(id=5, pid=3, is_primary=False),  # pion
            ],
        )

        assert obj.topology == "1g1mu1pi"

        # Test that invalid particles do not contribute to topology
        obj.particles[0].is_valid = False

        assert obj.topology == "1mu1pi"

    def test_interactionbase_from_particles(self):
        """Test InteractionBase from_particles class method."""
        from spine.data.out.interaction import InteractionBase
        from spine.data.out.particle import ParticleBase

        particles = [
            ParticleBase(
                id=1,
                is_primary=True,
                index=np.array([0], dtype=np.int64),
                points=np.array([[0, 0, 0]], dtype=np.float32),
                depositions=np.array([10.0], dtype=np.float32),
            ),
            ParticleBase(
                id=2,
                is_primary=False,
                index=np.array([1], dtype=np.int64),
                points=np.array([[1, 1, 1]], dtype=np.float32),
                depositions=np.array([20.0], dtype=np.float32),
            ),
            ParticleBase(
                id=3,
                is_primary=True,
                index=np.array([2], dtype=np.int64),
                points=np.array([[2, 2, 2]], dtype=np.float32),
                depositions=np.array([30.0], dtype=np.float32),
            ),
        ]

        obj = InteractionBase.from_particles(particles=particles)

        assert len(obj.particles) == 3
        assert all(p in obj.particles for p in particles)
        assert np.array_equal(obj.particle_ids, [1, 2, 3])
        assert np.array_equal(
            obj.points, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        )
        assert np.array_equal(
            obj.depositions, np.array([10.0, 20.0, 30.0], dtype=np.float32)
        )


class TestRecoInteraction:
    """Test RecoInteraction class."""

    def test_recointeraction_str_representation(self):
        """Test RecoInteraction string representation."""
        from spine.data.out.interaction import RecoInteraction
        from spine.data.out.particle import RecoParticle

        obj = RecoInteraction(
            id=2,
            match_ids=np.array([10], dtype=np.int64),
            particle_ids=np.array([5, 10], dtype=np.int64),
            particles=[RecoParticle(id=5), RecoParticle(id=10)],
        )

        str_repr = str(obj)
        print(str_repr)
        assert "RecoInteraction" in str_repr
        assert "2" in str_repr  # ID
        assert "RecoParticle(ID: 5" in str_repr
        assert "RecoParticle(ID: 10" in str_repr

    def test_recointeraction_leading_shower_property(self):
        """Test RecoInteraction leading_shower property."""
        from spine.data.out.interaction import RecoInteraction
        from spine.data.out.particle import RecoParticle

        # Test with no particles
        obj = RecoInteraction(id=0)
        assert obj.leading_shower is None

        # Test with particles but no showers
        obj = RecoInteraction(
            id=0,
            particles=[
                RecoParticle(id=1, shape=1),  # track
                RecoParticle(id=2, shape=1),  # track
            ],
        )
        assert obj.leading_shower is None

        # Test with valid showers
        obj = RecoInteraction(
            id=0,
            particles=[
                RecoParticle(id=1, shape=0, calo_ke=50.0, is_primary=True),  # shower
                RecoParticle(id=2, shape=0, calo_ke=150.0, is_primary=False),  # shower
                RecoParticle(id=3, shape=0, calo_ke=100.0, is_primary=True),  # shower
                RecoParticle(id=4, shape=1),  # track
            ],
        )

        leading_shower = obj.leading_shower
        assert leading_shower is not None
        assert leading_shower.id == 3


class TestTruthInteraction:
    """Test TruthInteraction class."""

    def test_truthinteraction_str_representation(self):
        """Test TruthInteraction string representation."""
        from spine.data.out.interaction import TruthInteraction
        from spine.data.out.particle import TruthParticle

        obj = TruthInteraction(
            id=2,
            match_ids=np.array([10], dtype=np.int64),
            particle_ids=np.array([5, 10], dtype=np.int64),
            particles=[TruthParticle(id=5), TruthParticle(id=10)],
        )

        str_repr = str(obj)
        print(str_repr)
        assert "TruthInteraction" in str_repr
        assert "2" in str_repr  # ID
        assert "TruthParticle(ID: 5" in str_repr
        assert "TruthParticle(ID: 10" in str_repr

    def test_truthinteraction_attach_neutrino_method(self):
        """Test TruthInteraction attach_neutrino method."""
        from spine.data.larcv.neutrino import Neutrino
        from spine.data.out.interaction import TruthInteraction

        # Regular case
        obj = TruthInteraction(id=0)
        neutrino = Neutrino(interaction_id=0, pdg_code=12, energy_init=1.0)

        obj.attach_neutrino(neutrino)

        assert obj.pdg_code == 12
        assert obj.energy_init == 1.0

        # Case where nu_id does not match neutrino id (should warn)
        obj = TruthInteraction(id=0, nu_id=999)
        neutrino = Neutrino(interaction_id=0, pdg_code=12, energy_init=1.0)

        with np.testing.assert_warns(UserWarning):
            obj.attach_neutrino(neutrino)

        assert obj.pdg_code == 12
        assert obj.energy_init == 1.0
