from types import SimpleNamespace

import numpy as np

from spine.utils.ppn import image_coordinates_batch


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
    def __init__(self, point):
        self._point = point

    def vertex(self):
        return self._point


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
