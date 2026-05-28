"""Test that the particle/neutrino data parsers work as intended."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import NuInteractionScheme
from spine.constants.sentinels import INVAL_ID
from spine.data.larcv import Meta, Neutrino, Particle
from spine.data.larcv.meta import ImageMeta3D
from spine.io.parse.data import ParserEdgeIndex, ParserTensor
from spine.io.parse.larcv.particle import *
from spine.utils.conditional import LARCV_AVAILABLE, larcv


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
    parser = LArCVParticleParser(
        dtype="float32",
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
    if len(result) > 0:
        ref_type = larcv.Particle if asis else Particle
        assert isinstance(result[0], ref_type)
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
    parser = LArCVNeutrinoParser(
        dtype="float32",
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
    if len(result) > 0:
        ref_type = larcv.Neutrino if asis else Neutrino
        assert isinstance(result[0], ref_type)
    if not asis:
        assert isinstance(result.default, Neutrino)
        for neutrino in result:
            assert neutrino.interaction_scheme == int(NuInteractionScheme.LARSOFT)


@pytest.mark.parametrize("neutrino_event", [1], indirect=True)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_parse_neutrinos_interaction_scheme_override(neutrino_event, sparse3d_event):
    """Tests overriding the interaction scheme used by the neutrino parser."""
    parser = LArCVNeutrinoParser(
        dtype="float32",
        neutrino_event=neutrino_event,
        sparse_event=sparse3d_event,
        interaction_scheme="genie",
    )

    result = parser.process(neutrino_event=neutrino_event, sparse_event=sparse3d_event)

    for neutrino in result:
        assert neutrino.interaction_scheme == int(NuInteractionScheme.GENIE)


@pytest.mark.parametrize("include_point_tagging", [True, False])
@pytest.mark.parametrize("particle_event", [0, 1, 20], indirect=True)
def test_parse_particle_points(particle_event, sparse3d_event, include_point_tagging):
    """Tests the parsing of LArCV particle points (PPN labels)."""
    # Initialize the parser
    parser = LArCVParticlePointParser(
        dtype="float32",
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
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == 2 + int(include_point_tagging)
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("particle_event", [0, 1, 20], indirect=True)
def test_parse_particle_coordinates(particle_event, sparse3d_event):
    """Tests the parsing of LArCV particle coordinates (GrapPA
    end cluster end points label for standalone training).
    """
    # Initialize the parser
    parser = LArCVParticleCoordinateParser(
        dtype="float32", particle_event=particle_event, sparse_event=sparse3d_event
    )

    # Parse the data
    result = parser.process(particle_event=particle_event, sparse_event=sparse3d_event)

    # There should be 3 components of the output
    # - The first has all 6 (!) coordinates for each particle (start/end)
    # - The second has the feature tensor with particle time and shape
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 6
    assert result.features.shape[1] == 2
    assert isinstance(result.meta, ImageMeta3D)

    # There should be exactly one row per particle in the input
    assert result.coords.shape[0] == particle_event.size()


@pytest.mark.parametrize(
    "particle_event, cluster3d_event", [(0, 0), (1, 1), (20, 20)], indirect=True
)
def test_parse_particle_graph(particle_event, cluster3d_event):
    """Tests the parsing of LArCV particle information into a set of
    parentage relations.
    """
    # Initialize the parser
    parser = LArCVParticleGraphParser(
        dtype="float32", particle_event=particle_event, cluster_event=cluster3d_event
    )

    # Parse the data
    result = parser.process(
        particle_event=particle_event, cluster_event=cluster3d_event
    )

    # There should be 2 components of the output
    # - The first contains an (2, E) matrix with E the number of edges
    # - The second is a single number corresponding to the number of particles
    assert isinstance(result, ParserEdgeIndex)
    assert result.features.shape[0] == 2
    assert result.global_shift == particle_event.size()


@pytest.mark.parametrize("particle_event", [0, 1], indirect=True)
def test_parse_particle_pid(particle_event):
    """Tests the parsing of LArCV single particle PID parser."""
    # Initialize the parser
    parser = LArCVSingleParticlePIDParser(
        dtype="float32", particle_event=particle_event
    )

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
    parser = LArCVSingleParticleEnergyParser(
        dtype="float32", particle_event=particle_event
    )

    # Parse the data
    result = parser.process(particle_event=particle_event)

    # The output should be a simple float
    assert isinstance(result, float)


def test_particle_parser_skip_empty_uses_placeholder():
    """Particle parsing should emit placeholders when empty particles are skipped."""

    class DummyParticle:
        def num_voxels(self):
            return 0

        def id(self):
            return 1

        def group_id(self):
            return 0

    class DummyEvent:
        def as_vector(self):
            return [DummyParticle()]

    parser = LArCVParticleParser(
        dtype="float32",
        particle_event="particle",
        skip_empty=True,
        post_process=False,
        pixel_coordinates=False,
    )
    result = parser.process(particle_event=DummyEvent())

    assert len(result) == 1
    assert isinstance(result[0], Particle)


@pytest.mark.parametrize(
    "particle_event, neutrino_event, cluster3d_event", [(20, 1, 20)], indirect=True
)
def test_particle_parser_call_paths(
    particle_event, neutrino_event, sparse3d_event, cluster3d_event
):
    """Wrapper calls should route named inputs through the particle parsers."""
    neutrino_parser = LArCVNeutrinoParser(
        dtype="float32", neutrino_event="neutrino", sparse_event="sparse"
    )
    neutrino_result = neutrino_parser(
        {"neutrino": neutrino_event, "sparse": sparse3d_event}
    )
    assert isinstance(neutrino_result.default, Neutrino)

    point_parser = LArCVParticlePointParser(
        dtype="float32", particle_event="particle", sparse_event="sparse"
    )
    assert isinstance(
        point_parser({"particle": particle_event, "sparse": sparse3d_event}),
        ParserTensor,
    )

    coord_parser = LArCVParticleCoordinateParser(
        dtype="float32", particle_event="particle", sparse_event="sparse"
    )
    assert isinstance(
        coord_parser({"particle": particle_event, "sparse": sparse3d_event}),
        ParserTensor,
    )

    pid_parser = LArCVSingleParticlePIDParser(
        dtype="float32", particle_event="particle"
    )
    assert isinstance(pid_parser({"particle": particle_event}), int)

    energy_parser = LArCVSingleParticleEnergyParser(
        dtype="float32", particle_event="particle"
    )
    assert isinstance(energy_parser({"particle": particle_event}), float)

    graph_parser = LArCVParticleGraphParser(
        dtype="float32", particle_event="particle", cluster_event="cluster"
    )
    assert isinstance(
        graph_parser({"particle": particle_event, "cluster": cluster3d_event}),
        ParserEdgeIndex,
    )


@pytest.mark.parametrize("particle_event", [20], indirect=True)
def test_vertex_point_parser_process_and_call(
    monkeypatch, particle_event, sparse3d_event
):
    """Vertex point parsing should expose tensor labels through process and wrapper paths."""
    labels = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    monkeypatch.setattr(
        "spine.io.parse.larcv.particle.get_vertex_labels",
        lambda particle_v, neutrino_v, meta, ftype: labels,
    )

    parser = LArCVVertexPointParser(
        dtype="float32", particle_event="particle", sparse_event="sparse"
    )
    result = parser({"particle": particle_event, "sparse": sparse3d_event})

    assert isinstance(result, ParserTensor)
    assert np.array_equal(result.coords, labels[:, :3])


def test_particle_graph_skips_invalid_and_fragment_edges():
    """Particle graph parsing should skip invalid parents and fragment-only edges when requested."""

    class DummyVector(list):
        def size(self):
            return len(self)

    class DummyParticle:
        def __init__(self, parent_id, group_id):
            self._parent_id = parent_id
            self._group_id = group_id

        def parent_id(self):
            return self._parent_id

        def group_id(self):
            return self._group_id

    class DummyEvent:
        def as_vector(self):
            return DummyVector([DummyParticle(INVAL_ID, 0), DummyParticle(0, 0)])

    parser = LArCVParticleGraphParser(dtype="float32", particle_event="particle")
    result = parser.process(particle_event=DummyEvent())

    assert isinstance(result, ParserEdgeIndex)
    assert result.features.size == 0


class DummyVector(list):
    """List with the ``size`` method used by LArCV containers."""

    def size(self):
        return len(self)


class DummyParticleObj:
    """Particle stand-in for validation-only parser checks."""

    def num_voxels(self):
        return 1

    def id(self):
        return 0

    def group_id(self):
        return 0


class DummyParticleEvent:
    """Particle event stand-in."""

    def __init__(self, count=1):
        self._parts = DummyVector([DummyParticleObj() for _ in range(count)])

    def as_vector(self):
        return self._parts

    def size(self):
        return len(self._parts)


class DummyNeutrinoEvent:
    """Neutrino event stand-in."""

    def __init__(self, count=1):
        self._neutrinos = DummyVector([object() for _ in range(count)])

    def as_vector(self):
        return self._neutrinos


class DummyCluster:
    """Cluster stand-in with a configurable size."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class DummyClusterEvent:
    """Cluster event stand-in."""

    def __init__(self, sizes):
        self._clusters = DummyVector([DummyCluster(size) for size in sizes])

    def as_vector(self):
        return self._clusters


def test_particle_parser_asis_validation_branches():
    """Particle parser should reject incompatible flags in as-is mode."""
    particle_event = DummyParticleEvent()

    with pytest.raises(ValueError, match="pixel_coordinates"):
        LArCVParticleParser(
            dtype="float32",
            particle_event="particle",
            asis=True,
            pixel_coordinates=True,
            post_process=False,
            skip_empty=False,
        ).process(particle_event=particle_event)

    with pytest.raises(ValueError, match="post_process"):
        LArCVParticleParser(
            dtype="float32",
            particle_event="particle",
            asis=True,
            pixel_coordinates=False,
            post_process=True,
            skip_empty=False,
        ).process(particle_event=particle_event)

    with pytest.raises(ValueError, match="skip_empty"):
        LArCVParticleParser(
            dtype="float32",
            particle_event="particle",
            asis=True,
            pixel_coordinates=False,
            post_process=False,
            skip_empty=True,
        ).process(particle_event=particle_event)


def test_particle_and_neutrino_metadata_validation():
    """Particle-family parsers should require exactly one metadata source."""
    particle_event = DummyParticleEvent()
    neutrino_event = DummyNeutrinoEvent()

    with pytest.raises(
        ValueError, match="Must provide either `sparse_event` or `cluster_event`"
    ):
        parser = LArCVParticleParser(
            dtype="float32",
            particle_event="particle",
            pixel_coordinates=True,
            post_process=False,
        )
        original = Particle.from_larcv
        Particle.from_larcv = staticmethod(
            lambda p: SimpleNamespace(id=0, nu_id=-1, to_px=lambda meta: None)
        )
        try:
            parser.process(particle_event=particle_event)
        finally:
            Particle.from_larcv = original

    with pytest.raises(
        ValueError, match="Must provide either `sparse_event` or `cluster_event`"
    ):
        parser = LArCVNeutrinoParser(
            dtype="float32",
            neutrino_event="neutrino",
            pixel_coordinates=True,
        )
        original = Neutrino.from_larcv
        Neutrino.from_larcv = staticmethod(
            lambda n, interaction_scheme=None: SimpleNamespace(
                interaction_id=-1, to_px=lambda meta: None
            )
        )
        try:
            parser.process(neutrino_event=neutrino_event)
        finally:
            Neutrino.from_larcv = original

    with pytest.raises(
        ValueError, match="Must provide either `sparse_event` or `cluster_event`"
    ):
        LArCVParticlePointParser(
            dtype="float32",
            particle_event="particle",
        ).process(particle_event=particle_event)

    with pytest.raises(
        ValueError, match="Must provide either `sparse_event` or `cluster_event`"
    ):
        LArCVParticleCoordinateParser(
            dtype="float32",
            particle_event="particle",
        ).process(particle_event=particle_event)


def test_neutrino_parser_asis_validation_branch():
    """Neutrino parser should reject pixel-coordinate conversion in as-is mode."""
    with pytest.raises(ValueError, match="pixel_coordinates"):
        LArCVNeutrinoParser(
            dtype="float32",
            neutrino_event="neutrino",
            asis=True,
            pixel_coordinates=True,
        ).process(neutrino_event=DummyNeutrinoEvent())


def test_vertex_parser_requires_exactly_one_vertex_source_and_meta():
    """Vertex parser should validate event and metadata source XORs."""
    parser = LArCVVertexPointParser(dtype="float32", particle_event="particle")

    with pytest.raises(ValueError, match="either `particle_event` or `sparse_event`"):
        parser.process()

    with pytest.raises(ValueError, match="either `particle_event` or `sparse_event`"):
        parser.process(
            particle_event=DummyParticleEvent(),
            neutrino_event=DummyNeutrinoEvent(),
        )

    with pytest.raises(
        ValueError, match="Must provide either `sparse_event` or `cluster_event`"
    ):
        parser.process(particle_event=DummyParticleEvent())


def test_particle_graph_rejects_cluster_count_mismatch():
    """Particle graph parsing should reject inconsistent cluster counts."""
    parser = LArCVParticleGraphParser(dtype="float32", particle_event="particle")

    with pytest.raises(ValueError, match="aligned with the number of clusters"):
        parser.process(
            particle_event=DummyParticleEvent(count=2),
            cluster_event=DummyClusterEvent([1, 1, 1, 1]),
        )


if not LARCV_AVAILABLE:
    _NO_LARCV_TESTS = {
        "test_particle_parser_asis_validation_branches",
        "test_particle_and_neutrino_metadata_validation",
        "test_neutrino_parser_asis_validation_branch",
        "test_vertex_parser_requires_exactly_one_vertex_source_and_meta",
        "test_particle_graph_rejects_cluster_count_mismatch",
        "test_particle_graph_skips_invalid_and_fragment_edges",
        "test_particle_parser_skip_empty_uses_placeholder",
    }
    for _name, _obj in list(globals().items()):
        if _name.startswith("test_") and _name not in _NO_LARCV_TESTS:
            globals()[_name] = pytest.mark.skip(
                reason="LArCV is required to generate parser fixtures."
            )(_obj)
