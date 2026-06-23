"""Tests for stored_property and stored_alias decorators."""

import pytest

from spine.data.decorator import stored_alias, stored_property
from spine.data.field import FieldMetadata


class TestStoredProperty:
    """Tests for the stored_property decorator."""

    def test_stored_property_without_args(self):
        """Test stored_property decorator without arguments."""

        class MyClass:
            @property
            @stored_property
            def size(self) -> int:
                return 42

        # Check that metadata was attached
        getter = MyClass.size.fget
        assert hasattr(getter, "__stored_property_metadata__")
        metadata = getter.__stored_property_metadata__
        assert isinstance(metadata, FieldMetadata)
        assert metadata.return_type == int

    def test_stored_property_with_metadata(self):
        """Test stored_property decorator with metadata arguments."""

        class MyClass:
            @property
            @stored_property(units="MeV", index=True)
            def energy(self) -> float:
                return 100.0

        # Check that metadata was attached with correct values
        getter = MyClass.energy.fget
        assert hasattr(getter, "__stored_property_metadata__")
        metadata = getter.__stored_property_metadata__
        assert isinstance(metadata, FieldMetadata)
        assert metadata.return_type == float
        assert metadata.units == "MeV"
        assert metadata.index is True

    def test_stored_property_with_optional_type(self):
        """Test stored_property with Optional return type."""

        class MyClass:
            @property
            @stored_property
            def maybe_value(self) -> int | None:
                return None

        # Should extract int from Optional[int]
        getter = MyClass.maybe_value.fget
        metadata = getter.__stored_property_metadata__
        assert metadata.return_type == int

    def test_stored_property_no_return_type(self):
        """Test that stored_property raises error without return type annotation."""
        with pytest.raises(TypeError, match="must have a return type annotation"):

            class MyClass:
                @property
                @stored_property
                def no_type(self):
                    return 42

            # Access the property to trigger the decorator logic
            _ = MyClass()

    def test_stored_property_functionality(self):
        """Test that decorated property still works correctly."""

        class MyClass:
            def __init__(self):
                self._value = 10

            @property
            @stored_property(units="cm")
            def position(self) -> float:
                return self._value * 2.0

        obj = MyClass()
        assert obj.position == 20.0
        obj._value = 5  # pylint: disable=protected-access
        assert obj.position == 10.0

    def test_bad_type_hints(self):
        """Test that stored_property raises error if type hints cannot be resolved."""
        with pytest.raises(TypeError, match="Could not resolve type hints"):

            class MyClass:
                @property
                @stored_property
                def bad_type(self) -> "NonExistentType":  # type: ignore
                    return 42

            # Access the property to trigger the decorator logic
            _ = MyClass()

    def test_none_return_type(self):
        """Test that stored_property raises error if return type is None."""
        with pytest.raises(TypeError, match="cannot have None as return type"):

            class MyClass:
                @property
                @stored_property
                def none_type(self) -> None:
                    return None

            # Access the property to trigger the decorator logic
            _ = MyClass()


class TestStoredAlias:
    """Tests for the stored_alias decorator."""

    def test_stored_alias(self):
        """Test stored_alias decorator."""

        class MyClass:
            @property
            def energy(self) -> float:
                return 100.0

            @property
            @stored_alias("energy")
            def ke(self) -> float:
                return self.energy

        # Check that alias target was attached
        getter = MyClass.ke.fget
        assert hasattr(getter, "__stored_alias_target__")
        assert getter.__stored_alias_target__ == "energy"

    def test_stored_alias_functionality(self):
        """Test that aliased property works correctly."""

        class MyClass:
            def __init__(self, value: float):
                self._value = value

            @property
            def original(self) -> float:
                return self._value

            @property
            @stored_alias("original")
            def alias(self) -> float:
                return self.original

        obj = MyClass(42.0)
        assert obj.original == 42.0
        assert obj.alias == 42.0
        assert obj.original == obj.alias
