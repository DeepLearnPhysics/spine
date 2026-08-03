"""Tests for typed object-list data products."""

from spine.data import ObjectList, ObjectListData, Particle


class DummyObject:
    """Minimal object type for object-list product tests."""

    index_attrs = ()


def test_object_list_retains_type_for_populated_list():
    """ObjectList should behave like a list and retain its representative."""
    particles = [Particle(id=1), Particle(id=2), Particle(id=3)]
    objects = ObjectList(particles, default=Particle())

    assert list(objects) == particles
    assert isinstance(objects.default, Particle)


def test_object_list_retains_type_for_empty_list():
    """An empty ObjectList should preserve its intended element type."""
    objects = ObjectList(object_list=[], default=Particle())

    assert len(objects) == 0
    assert isinstance(objects.default, Particle)


def test_object_list_data_retains_collation_contract():
    """ObjectListData should retain scalar and named index shifts."""
    objects = ObjectListData([DummyObject(), DummyObject()], DummyObject())
    assert objects.index_shifts == 2

    shifted = ObjectListData([DummyObject()], DummyObject(), index_shifts={"a": 3})
    assert shifted.index_shifts == {"a": 3}


def test_object_list_data_cast_drops_collation_contract():
    """Casting should preserve contents and type without retaining shifts."""
    shifted = ObjectListData([DummyObject()], DummyObject(), index_shifts={"a": 3})
    objects = shifted.to_object_list

    assert type(objects) is ObjectList
    assert list(objects) == list(shifted)
    assert objects.default == shifted.default
    assert not hasattr(objects, "index_shifts")
