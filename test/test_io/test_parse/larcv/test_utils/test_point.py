from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import LOWES_SHP, SHOWR_SHP, TRACK_SHP
from spine.io.parse.larcv.utils.point import (
    get_ppn_labels,
    get_vertex_labels,
    image_contains,
    image_coordinates,
    image_coordinates_batch,
)


class FakePoint:
    def __init__(self, x, y, z):
        self._coords = (x, y, z)

    def x(self):
        return self._coords[0]

    def y(self):
        return self._coords[1]

    def z(self):
        return self._coords[2]


class FakeParticle:
    def __init__(
        self,
        point,
        *,
        particle_id=0,
        shape=TRACK_SHP,
        pdg=13,
        energy=10.0,
        voxels=2,
        parent_id=None,
        ancestor_pdg=13,
    ):
        self._point = point
        self._id = particle_id
        self._shape = shape
        self._pdg = pdg
        self._energy = energy
        self._voxels = voxels
        self._parent_id = particle_id if parent_id is None else parent_id
        self._ancestor_pdg = ancestor_pdg

    def vertex(self):
        return self._point

    def id(self):
        return self._id

    def energy_deposit(self):
        return self._energy

    def num_voxels(self):
        return self._voxels

    def pdg_code(self):
        return self._pdg

    def shape(self):
        return self._shape

    def first_step(self):
        return self._point

    def last_step(self):
        return FakePoint(self._point.x() + 1, self._point.y(), self._point.z())

    def parent_id(self):
        return self._parent_id

    def ancestor_pdg_code(self):
        return self._ancestor_pdg

    def ancestor_position(self):
        return self._point


def make_meta():
    """Return image metadata covering physical coordinates zero through ten."""
    return SimpleNamespace(
        min_x=lambda: 0.0,
        min_y=lambda: 0.0,
        min_z=lambda: 0.0,
        max_x=lambda: 10.0,
        max_y=lambda: 10.0,
        max_z=lambda: 10.0,
        size_voxel_x=lambda: 2.0,
        size_voxel_y=lambda: 2.0,
        size_voxel_z=lambda: 2.0,
    )


def test_image_coordinates_batch_positions_and_particle_attribute():
    meta = SimpleNamespace(
        min_x=lambda: 1.0,
        min_y=lambda: 2.0,
        min_z=lambda: 3.0,
        size_voxel_x=lambda: 2.0,
        size_voxel_y=lambda: 4.0,
        size_voxel_z=lambda: 5.0,
    )
    points = [FakePoint(3.0, 6.0, 8.0), FakePoint(5.0, 10.0, 13.0)]
    expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)

    direct = image_coordinates_batch(meta, points)
    particles = [FakeParticle(point) for point in points]
    from_particles = image_coordinates_batch(meta, particles, position_attr="vertex")

    assert np.array_equal(direct, expected)
    assert np.array_equal(from_particles, expected)


def test_ppn_label_filters_and_track_endpoints():
    """PPN labels should validate dimensions and filter unsuitable particles."""
    meta = make_meta()
    with pytest.raises(ValueError, match="dimension"):
        get_ppn_labels([], meta, np.float32, dim=4)

    particles = [
        FakeParticle(FakePoint(1, 1, 1), particle_id=9),
        FakeParticle(FakePoint(1, 1, 1), particle_id=1, energy=0.0),
        FakeParticle(FakePoint(1, 1, 1), particle_id=2, pdg=1000000001),
        FakeParticle(FakePoint(20, 1, 1), particle_id=3, pdg=11, shape=SHOWR_SHP),
        FakeParticle(FakePoint(1, 1, 1), particle_id=4, shape=LOWES_SHP),
        FakeParticle(FakePoint(2, 2, 2), particle_id=5, shape=SHOWR_SHP, pdg=11),
    ]
    with pytest.warns(UserWarning, match="does not match"):
        labels = get_ppn_labels(
            particles,
            meta,
            np.float32,
            min_energy_deposit=1.0,
            include_point_tagging=True,
        )
    # The first track contributes start/end points and the final shower a start.
    assert labels.shape == (3, 6)

    no_tags = get_ppn_labels(
        [FakeParticle(FakePoint(1, 1, 1))],
        meta,
        np.float32,
        include_point_tagging=False,
    )
    assert no_tags.shape == (2, 5)
    assert get_ppn_labels(
        [FakeParticle(FakePoint(1, 1, 1), energy=0)],
        meta,
        np.float32,
        min_energy_deposit=1,
    ).shape == (0, 6)


def test_vertex_labels_and_coordinate_variants():
    """Vertex labels should support shared ancestors, neutrinos, and empty inputs."""
    meta = make_meta()
    point = FakePoint(2, 4, 6)
    particles = [
        FakeParticle(point, particle_id=0),
        FakeParticle(point, particle_id=1, parent_id=1),
    ]
    vertices = get_vertex_labels(particles, None, meta, np.float32)
    assert vertices.shape == (1, 4)

    no_primary = [FakeParticle(point, particle_id=0, parent_id=1, ancestor_pdg=13)]
    assert get_vertex_labels(no_primary, None, meta, np.float32).shape == (0, 4)
    neutrinos = [SimpleNamespace(position=lambda: point)]
    assert get_vertex_labels(None, neutrinos, meta, np.float32).shape == (1, 4)
    outside = [SimpleNamespace(position=lambda: FakePoint(20, 20, 20))]
    assert get_vertex_labels(None, outside, meta, np.float32).shape == (0, 4)

    assert image_contains(meta, point)
    assert image_contains(meta, point, dim=2)
    assert image_coordinates(meta, point) == [1.0, 2.0, 3.0]
    assert image_coordinates(meta, point, dim=2) == [1.0, 2.0]
    generated = (p for p in [point, point])
    assert image_coordinates_batch(meta, generated, dim=2).shape == (2, 2)
