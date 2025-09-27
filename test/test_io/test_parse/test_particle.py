"""Test that the particle/neutrino data parsers work as intended."""

import pytest
from larcv import larcv

from spine import Meta, Neutrino, Particle
from spine.io.parse.particle import *


@pytest.mark.parametrize(
    "asis, pixel_coordinates, post_process",
    [
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
@pytest.mark.parametrize("particle_event", [0, 1, 20], indirect=True)
@pytest.mark.parametrize("neutrino_event", [0, 1, 2], indirect=True)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parse_particles(
    particle_event,
    neutrino_event,
    sparse3d_event,
    asis,
    pixel_coordinates,
    post_process,
):
    """Tests the parsing of LArCV particle information."""
    # Initialize the parser
    parser = ParticleParser(
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_event=sparse3d_event,
        asis=asis,
        pixel_coordinates=pixel_coordinates,
        post_process=post_process,
    )

    # Parse the data
    result = parser.process(
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_event=sparse3d_event,
    )

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == particle_event.size()
    if len(result):
        ref_type = larcv.Particle if asis else Particle
        assert type(result[0]) == ref_type
    if not asis:
        assert isinstance(result.default, Particle)


@pytest.mark.parametrize(
    "asis, pixel_coordinates", [(True, False), (False, False), (False, True)]
)
@pytest.mark.parametrize("neutrino_event", [0, 1, 10], indirect=True)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parse_neutrinos(neutrino_event, sparse3d_event, asis, pixel_coordinates):
    """Tests the parsing of LArCV neutrino information."""
    # Initialize the parser
    parser = NeutrinoParser(
        neutrino_event=neutrino_event,
        sparse_event=sparse3d_event,
        asis=asis,
        pixel_coordinates=pixel_coordinates,
    )

    # Parse the data
    result = parser.process(neutrino_event=neutrino_event, sparse_event=sparse3d_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == neutrino_event.as_vector().size()
    if len(result):
        ref_type = larcv.Neutrino if asis else Neutrino
        assert type(result[0]) == ref_type
    if not asis:
        assert isinstance(result.default, Neutrino)


@pytest.mark.parametrize("include_point_tagging", [True, False])
@pytest.mark.parametrize("particle_event", [0, 1, 20], indirect=True)
def test_parse_particle_points(particle_event, sparse3d_event, include_point_tagging):
    """Tests the parsing of LArCV particle points (PPN labels)."""
    # Initialize the parser
    parser = ParticlePointParser(
        particle_event=particle_event,
        sparse_event=sparse3d_event,
        include_point_tagging=include_point_tagging,
    )

    # Parse the data
    result = parser.process(particle_event=particle_event, sparse_event=sparse3d_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor with point type, particle index
    #   (and optionally a binary label of start/end status)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == 2 + int(include_point_tagging)
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize("particle_event", [0, 1, 20], indirect=True)
def test_parse_particle_coordinates(particle_event, sparse3d_event):
    """Tests the parsing of LArCV particle coordinates (GrapPA
    end cluster end points label for standalone training).
    """
    # Initialize the parser
    parser = ParticleCoordinateParser(
        particle_event=particle_event, sparse_event=sparse3d_event
    )

    # Parse the data
    result = parser.process(particle_event=particle_event, sparse_event=sparse3d_event)

    # There should be 3 components of the output
    # - The first has all 6 (!) coordinates for each particle (start/end)
    # - The second has the feature tensor with particle time and shape
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 6
    assert result[1].shape[1] == 2
    assert isinstance(result[2], Meta)

    # There should be exactly one row per particle in the input
    assert len(result[0]) == particle_event.size()


@pytest.mark.parametrize(
    "particle_event, cluster3d_event", [(0, 0), (1, 1), (20, 20)], indirect=True
)
def test_parse_particle_graph(particle_event, cluster3d_event):
    """Tests the parsing of LArCV particle information into a set of
    parentage relations.
    """
    # Initialize the parser
    parser = ParticleGraphParser(
        particle_event=particle_event, cluster_event=cluster3d_event
    )

    # Parse the data
    result = parser.process(
        particle_event=particle_event, cluster_event=cluster3d_event
    )

    # There should be 2 components of the output
    # - The first contains an (2, E) matrix with E the number of edges
    # - The second is a single number corresponding to the number of particles
    assert len(result) == 2
    assert result[0].shape[0] == 2
    assert result[1] == particle_event.size()


@pytest.mark.parametrize("particle_event", [0, 1], indirect=True)
def test_parse_particle_pid(particle_event):
    """Tests the parsing of LArCV single particle PID parser."""
    # Initialize the parser
    parser = SingleParticlePIDParser(particle_event=particle_event)

    # Parse the data
    result = parser.process(particle_event=particle_event)

    # The output should be a simple integer
    assert isinstance(result, int)
    if particle_event.size():
        assert result == 1  # Electron


@pytest.mark.parametrize("particle_event", [0, 1], indirect=True)
def test_parse_particle_energy(particle_event):
    """Tests the parsing of LArCV single particle energy parser."""
    # Initialize the parser
    parser = SingleParticleEnergyParser(particle_event=particle_event)

    # Parse the data
    result = parser.process(particle_event=particle_event)

    # The output should be a simple float
    assert isinstance(result, float)
