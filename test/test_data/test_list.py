"""Tests for ObjectList class."""


class TestObjectList:
    """Test ObjectList functionality."""

    def test_object_list_creation(self):
        """Test ObjectList creation."""
        from spine.data.larcv import Particle
        from spine.data.list import ObjectList

        # Create an ObjectList with some dummy objects and a default class
        obj_list = ObjectList(
            object_list=[Particle(id=1), Particle(id=2), Particle(id=3)],
            default=Particle(),
        )

        # Check that the list behaves like a normal list
        assert len(obj_list) == 3
        assert obj_list[0] == Particle(id=1)
        assert obj_list[1] == Particle(id=2)
        assert obj_list[2] == Particle(id=3)

        # Check that the default attribute is set correctly
        assert isinstance(obj_list.default, Particle)

    def test_object_list_empty(self):
        """Test ObjectList behavior when empty."""
        from spine.data.larcv import Particle
        from spine.data.list import ObjectList

        # Create an empty ObjectList with a default class
        obj_list = ObjectList(object_list=[], default=Particle())

        # Check that the list is empty
        assert len(obj_list) == 0

        # Check that the default attribute is set correctly
        assert isinstance(obj_list.default, Particle)
