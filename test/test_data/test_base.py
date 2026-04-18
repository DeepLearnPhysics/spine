"""Tests for DataBase and PosDataBase classes."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from spine.data.base import DataBase, PosDataBase
from spine.data.decorator import stored_alias, stored_property
from spine.data.field import FieldMetadata


@dataclass(eq=False)
class SimpleData(DataBase):
    """Simple test data structure."""

    value: int = 0
    name: str = "test"


@dataclass(eq=False)
class ArrayData(DataBase):
    """Test data structure with arrays."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, position=True),
    )
    vector: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0], dtype=np.float64),
        metadata=FieldMetadata(length=2, dtype=np.float64, vector=True),
    )
    fixed_array: np.ndarray = field(
        default_factory=lambda: np.array([1, 2, 3], dtype=np.int32),
        metadata=FieldMetadata(length=3, dtype=np.int32),
    )
    var_array: np.ndarray = field(
        default_factory=lambda: np.array([1, 2], dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )


@dataclass(eq=False)
class IndexData(DataBase):
    """Test data structure with index attributes."""

    id: int = field(default=-1, metadata=FieldMetadata(index=True))
    parent_id: int = field(default=-1, metadata=FieldMetadata(index=True))
    cluster_ids: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 2], dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32, index=True),
    )


@dataclass(eq=False)
class SkipData(DataBase):
    """Test data structure with skip attributes."""

    visible: int = 0
    skip_field: int = field(default=1, metadata=FieldMetadata(skip=True))
    lite_skip_field: int = field(default=2, metadata=FieldMetadata(lite_skip=True))


@dataclass(eq=False)
class EnumData(DataBase):
    """Test data structure with enumerated field."""

    particle_type: int = field(
        default=0,
        metadata=FieldMetadata(enum={0: "electron", 1: "muon", 2: "pion"}),
    )


@dataclass(eq=False)
class DerivedData(DataBase):
    """Test data structure with derived properties."""

    _value: int = 0

    @property
    @stored_property(units="MeV")
    def energy(self) -> float:
        """Computed energy property."""
        return float(self._value * 2)

    @property
    @stored_alias("energy")
    def ke(self) -> float:
        """Alias for energy."""
        return self.energy


@dataclass(eq=False)
class SimplePosData(PosDataBase):
    """Simple positional data structure."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    energy: float = field(default=100.0, metadata=FieldMetadata(units="MeV"))
    length: float = field(default=10.0, metadata=FieldMetadata(units="instance"))


@dataclass(eq=False)
class SimpleNormedVectorData(PosDataBase):
    """Test data structure with normed vector attribute."""

    vector: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 2.0], dtype=np.float64),
        metadata=FieldMetadata(
            length=3, dtype=np.float64, vector=True, units="instance"
        ),
    )


@dataclass(eq=False)
class SimpleNormalizedVectorData(PosDataBase):
    """Test data structure with normalized vector attribute."""

    vector: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64),
        metadata=FieldMetadata(length=3, dtype=np.float64, vector=True),
    )


@dataclass(eq=False)
class ListData(DataBase):
    """Test data structure with object list attributes."""

    values: list[SimpleData] = field(default_factory=list)


@dataclass(eq=False)
class UnrecognizedData(DataBase):
    """Test data structure with unrecognized attribute type."""

    data: dict = field(default_factory=dict)


class TestDataBase:
    """Tests for the DataBase class."""

    def test_basic_creation(self):
        """Test basic DataBase subclass creation."""
        obj = SimpleData(value=42, name="hello")
        assert obj.value == 42
        assert obj.name == "hello"

    def test_equality_scalars(self):
        """Test equality comparison for scalar attributes."""
        obj1 = SimpleData(value=42, name="test")
        obj2 = SimpleData(value=42, name="test")
        obj3 = SimpleData(value=43, name="test")

        assert obj1 == obj2
        assert obj1 != obj3

    def test_equality_nan_floats(self):
        """Test equality comparison handles NaN floats correctly."""

        @dataclass(eq=False)
        class FloatData(DataBase):
            value: float = np.nan

        obj1 = FloatData(value=np.nan)
        obj2 = FloatData(value=np.nan)
        obj3 = FloatData(value=1.0)

        assert obj1 == obj2  # Both NaN should be equal
        assert obj1 != obj3

    def test_equality_arrays(self):
        """Test equality comparison for numpy arrays."""
        obj1 = ArrayData()
        obj2 = ArrayData()
        obj3 = ArrayData(position=np.array([4.0, 5.0, 6.0], dtype=np.float32))

        assert obj1 == obj2
        assert obj1 != obj3

    def test_equality_arrays_with_nan(self):
        """Test equality comparison for arrays containing NaN."""
        obj1 = ArrayData(position=np.array([1.0, np.nan, 3.0], dtype=np.float32))
        obj2 = ArrayData(position=np.array([1.0, np.nan, 3.0], dtype=np.float32))

        assert obj1 == obj2

    def test_equality_different_classes(self):
        """Test that different classes are not equal."""
        obj1 = SimpleData()
        obj2 = ArrayData()

        assert obj1 != obj2

    def test_equality_lists(self):
        """Test equality comparison for list attributes."""
        obj1 = ListData(
            values=[SimpleData(value=1), SimpleData(value=2), SimpleData(value=3)]
        )
        obj2 = ListData(
            values=[SimpleData(value=1), SimpleData(value=2), SimpleData(value=3)]
        )
        obj3 = ListData(
            values=[SimpleData(value=4), SimpleData(value=5), SimpleData(value=6)]
        )

        assert obj1 == obj2
        assert obj1 != obj3

    def test_equality_unrecognized_type(self):
        """Test that equality comparison raises error for unrecognized types."""
        obj1 = UnrecognizedData(data={"key": "value"})
        obj2 = UnrecognizedData(data={"key": "value"})

        with pytest.raises(TypeError, match="Cannot compare the `data` attribute"):
            _ = obj1 == obj2

    def test_array_dtype_casting(self):
        """Test that arrays are cast to correct dtype in __post_init__."""
        obj = ArrayData(position=np.array([1, 2, 3]))  # int array
        assert obj.position.dtype == np.float32  # Should be cast to float32

    def test_array_length_validation(self):
        """Test that array length is validated in __post_init__."""
        with pytest.raises(ValueError, match="must have length 3"):
            ArrayData(position=np.array([1.0, 2.0], dtype=np.float32))  # Wrong length

    def test_set_precision(self):
        """Test set_precision method."""
        obj = ArrayData()
        assert obj.position.dtype == np.float32
        assert obj.vector.dtype == np.float64

        obj.set_precision(8)
        assert obj.position.dtype == np.float64
        assert obj.vector.dtype == np.float64

        obj.set_precision(4)
        assert obj.position.dtype == np.float32
        assert obj.vector.dtype == np.float32

    def test_set_precision_invalid(self):
        """Test that set_precision raises error for invalid precision."""
        obj = ArrayData()

        with pytest.raises(ValueError, match="Supported precisions"):
            obj.set_precision(16)

    def test_shift_indexes_scalar(self):
        """Test shift_indexes with scalar shift."""
        obj = IndexData(id=5, parent_id=10)

        obj.shift_indexes(100)

        assert obj.id == 105
        assert obj.parent_id == 110

    def test_shift_indexes_invalid_not_shifted(self):
        """Test that invalid indexes (-1) are not shifted."""
        obj = IndexData(id=-1, parent_id=10)

        obj.shift_indexes(100)

        assert obj.id == -1  # Should remain -1
        assert obj.parent_id == 110

    def test_shift_indexes_array(self):
        """Test shift_indexes with array attribute."""
        obj = IndexData()

        obj.shift_indexes(10)

        assert np.array_equal(obj.cluster_ids, np.array([10, 11, 12]))

    def test_shift_indexes_dict(self):
        """Test shift_indexes with dictionary of shifts."""
        obj = IndexData(id=5, parent_id=10)

        obj.shift_indexes({"id": 100, "parent_id": 200, "cluster_ids": 1000})

        assert obj.id == 105
        assert obj.parent_id == 210
        assert np.array_equal(obj.cluster_ids, np.array([1000, 1001, 1002]))

    def test_as_dict(self):
        """Test as_dict method."""
        obj = SkipData(visible=42, skip_field=1, lite_skip_field=2)

        result = obj.as_dict()

        assert "visible" in result
        assert "skip_field" not in result  # Should be skipped
        assert "lite_skip_field" in result  # Should not be skipped in normal mode

    def test_as_dict_lite(self):
        """Test as_dict method with lite=True."""
        obj = SkipData(visible=42, skip_field=1, lite_skip_field=2)

        result = obj.as_dict(lite=True)

        assert "visible" in result
        assert "skip_field" not in result
        assert "lite_skip_field" not in result  # Should be skipped in lite mode

    def test_scalar_dict_scalars(self):
        """Test scalar_dict with scalar attributes."""
        obj = SimpleData(value=42, name="test")

        result = obj.scalar_dict()

        assert result["value"] == 42
        assert result["name"] == "test"

    def test_scalar_dict_position(self):
        """Test scalar_dict expands position vectors."""
        obj = ArrayData()

        result = obj.scalar_dict()

        assert result["position_x"] == 1.0
        assert result["position_y"] == 2.0
        assert result["position_z"] == 3.0

    def test_scalar_dict_fixed_array(self):
        """Test scalar_dict expands fixed-length arrays."""
        obj = ArrayData()

        result = obj.scalar_dict()

        assert result["fixed_array_0"] == 1
        assert result["fixed_array_1"] == 2
        assert result["fixed_array_2"] == 3

    def test_scalar_dict_var_array_with_length(self):
        """Test scalar_dict with variable-length array and specified length."""
        obj = ArrayData()

        result = obj.scalar_dict(lengths={"var_array": 4})

        assert result["var_array_0"] == 1
        assert result["var_array_1"] == 2
        assert result["var_array_2"] is None  # Padded with None
        assert result["var_array_3"] is None

    def test_scalar_dict_specific_attrs(self):
        """Test scalar_dict with specific attribute list."""
        obj = ArrayData()

        result = obj.scalar_dict(attrs=["position", "fixed_array"])

        assert "position_x" in result
        assert "fixed_array_0" in result
        assert "vector_x" not in result  # Not requested

    def test_scalar_dict_missing_attr(self):
        """Test scalar_dict raises error for missing attribute."""
        obj = SimpleData()

        with pytest.raises(AttributeError, match="do\\(es\\) not appear"):
            obj.scalar_dict(attrs=["nonexistent"])

    def test_scalar_dict_var_array_no_length(self):
        """Test scalar_dict raises error for var array without length."""
        obj = ArrayData()

        with pytest.raises(ValueError, match="must provide a fixed length"):
            obj.scalar_dict(attrs=["var_array"])

    def test_scalar_dict_unrecognized_type(self):
        """Test scalar_dict raises error for unrecognized attribute type."""
        obj = UnrecognizedData()

        with pytest.raises(ValueError, match="Cannot expand the `data` attribute"):
            obj.scalar_dict(attrs=["data"])

    def test_enum_dicts(self):
        """Test enum_dicts property."""
        obj = EnumData()

        enum_dicts = obj.enum_dicts

        assert "particle_type" in enum_dicts
        assert enum_dicts["particle_type"]["electron"] == 0
        assert enum_dicts["particle_type"]["muon"] == 1

    def test_field_units(self):
        """Test field_units property."""
        obj = DerivedData()

        units = obj.field_units

        assert units["energy"] == "MeV"

    def test_from_hdf5_basic(self):
        """Test from_hdf5 class method."""
        data_dict = {"value": 42, "name": "test"}

        obj = SimpleData.from_hdf5(data_dict)

        assert obj.value == 42
        assert obj.name == "test"

    def test_from_hdf5_binary_string(self):
        """Test from_hdf5 converts binary strings."""
        data_dict = {"value": 42, "name": b"test"}

        obj = SimpleData.from_hdf5(data_dict)

        assert obj.name == "test"  # Should be decoded

    def test_from_hdf5_bool_array(self):
        """Test from_hdf5 converts uint8 arrays to booleans."""

        @dataclass(eq=False)
        class BoolData(DataBase):
            flag: bool = False

        data_dict = {"flag": np.array([1], dtype=np.uint8)}

        obj = BoolData.from_hdf5(data_dict)

        assert obj.flag is True
        assert isinstance(obj.flag, bool)

    def test_from_hdf5_excludes_derived(self):
        """Test from_hdf5 excludes derived properties."""
        data_dict = {"_value": 10, "energy": 999}  # energy should be ignored

        obj = DerivedData.from_hdf5(data_dict)

        assert obj._value == 10  # pylint: disable=protected-access
        assert obj.energy == 20  # Computed, not loaded from dict

    def test_cached_attrs_mechanism(self):
        """Test that attribute caching works correctly."""

        # Create first instance
        obj1 = SimpleData()  # pylint: disable=unused-variable
        assert SimpleData._attrs_cached is True  # pylint: disable=protected-access

        # Create second instance (should use cached values)
        obj2 = SimpleData()
        assert obj2._attrs_cached is True  # pylint: disable=protected-access

    def test_get_stored_properties(self):
        """Test _get_stored_properties class method."""
        props = DerivedData._get_stored_properties()  # pylint: disable=protected-access

        assert "energy" in props
        assert props["energy"].units == "MeV"
        assert "ke" in props  # Alias should be discovered
        assert props["ke"].skip is True  # Aliases are marked as skip

    def test_get_stored_properties_override(self):
        """Test that a subclass stored property can override a parent class stored property."""

        @dataclass(eq=False)
        class Parent(DataBase):
            @property
            @stored_property(units="MeV")
            def energy(self) -> float:
                return 100.0

        @dataclass(eq=False)
        class Child(Parent):
            @property
            @stored_property(units="GeV")
            def energy(self) -> float:
                return 0.1

        parent_props = (
            Parent._get_stored_properties()  # pylint: disable=protected-access
        )
        child_props = Child._get_stored_properties()  # pylint: disable=protected-access

        assert parent_props["energy"].units == "MeV"
        assert child_props["energy"].units == "GeV"

    def test_get_stored_properties_alias_field(self):
        """Test that _get_stored_properties correctly identifies aliases of fields."""

        # With a well formed metadata of the field
        @dataclass(eq=False)
        class MyData(DataBase):
            value: int = field(default=42, metadata=FieldMetadata(units="MeV"))

            @property
            @stored_alias("value")
            def energy(self) -> int:
                return self.value

        props = MyData._get_stored_properties()  # pylint: disable=protected-access

        assert "energy" in props
        assert props["energy"].units == "MeV"  # Should inherit metadata from 'value'

        # With no metadata on the field (should not fail, just no metadata for alias)
        @dataclass(eq=False)
        class MyDataNoMeta(DataBase):
            value: int = 42  # No metadata

            @property
            @stored_alias("value")
            def energy(self) -> int:
                return self.value

        props_no_meta = (
            MyDataNoMeta._get_stored_properties()  # pylint: disable=protected-access
        )

        assert "energy" in props_no_meta
        assert props_no_meta["energy"].units is None  # No metadata to inherit

    def test_get_stored_properties_bad_alias_target(self):
        """Test that aliases to missing targets fail loudly."""

        @dataclass(eq=False)
        class MyData(DataBase):
            value: int = 42

            @property
            @stored_alias("missing")
            def energy(self) -> int:
                return self.value

        with pytest.raises(AttributeError, match="targets unknown attribute"):
            MyData._get_stored_properties()  # pylint: disable=protected-access

    def test_inheritance_independent_caches(self):
        """Test that subclasses have independent cached attributes."""

        @dataclass(eq=False)
        class Parent(DataBase):
            x: int = 0
            data: np.ndarray = field(
                default_factory=lambda: np.array([1, 2], dtype=np.float32),
                metadata=FieldMetadata(length=2, dtype=np.float32),
            )

        @dataclass(eq=False)
        class Child(Parent):
            y: int = 0
            other_data: np.ndarray = field(
                default_factory=lambda: np.array([1, 2, 3], dtype=np.float32),
                metadata=FieldMetadata(length=3, dtype=np.float32),
            )

        # Create instances to trigger caching
        _ = Parent()
        _ = Child()

        # Each should have its own cache
        assert Parent._attrs_cached is True  # pylint: disable=protected-access
        assert Child._attrs_cached is True  # pylint: disable=protected-access

        # Child should have more fixed_length_attrs than Parent
        assert "data" in Parent._fixed_length_attrs  # pylint: disable=protected-access
        assert "data" in Child._fixed_length_attrs  # pylint: disable=protected-access
        assert (
            "other_data"
            in Child._fixed_length_attrs  # pylint: disable=protected-access
        )
        assert (
            "other_data"
            not in Parent._fixed_length_attrs  # pylint: disable=protected-access
        )


class TestPosDataBase:
    """Tests for the PosDataBase class."""

    def test_basic_creation(self):
        """Test basic PosDataBase subclass creation."""
        obj = SimplePosData()

        assert obj.units == "cm"
        assert np.array_equal(obj.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_units_validation(self):
        """Test that invalid units raise error."""
        with pytest.raises(ValueError, match="must be either"):
            SimplePosData(units="meters")

    def test_field_units_instance_resolution(self):
        """Test field_units resolves 'instance' to current units."""
        obj = SimplePosData(units="cm")

        units = obj.field_units

        assert units["position"] == "cm"  # Resolved from 'instance'
        assert units["length"] == "cm"  # Resolved from 'instance'
        assert units["energy"] == "MeV"  # Fixed unit

    def test_field_units_changes_with_conversion(self):
        """Test field_units updates when units change."""

        # Create a mock Meta object
        class MockMeta:
            size = np.array([0.1, 0.1, 0.1])

            def to_px(self, coords):
                return coords / self.size

            def to_cm(self, coords):
                return coords * self.size

        meta = MockMeta()
        obj = SimplePosData(units="cm")

        assert obj.field_units["position"] == "cm"

        obj.to_px(meta)  # type: ignore

        assert obj.field_units["position"] == "px"

    def test_to_px_conversion(self):
        """Test to_px method converts coordinates correctly."""

        class MockMeta:
            size = np.array([2.0, 2.0, 2.0])

            def to_px(self, coords):
                return coords / self.size

        meta = MockMeta()
        obj = SimplePosData(
            units="cm", position=np.array([10.0, 20.0, 30.0], dtype=np.float32)
        )

        obj.to_px(meta)  # type: ignore

        assert obj.units == "px"
        assert np.allclose(obj.position, [5.0, 10.0, 15.0])

    def test_to_px_already_px(self):
        """Test to_px raises error if already in pixels."""

        class MockMeta:
            size = np.array([2.0, 2.0, 2.0])

        meta = MockMeta()
        obj = SimplePosData(units="px")

        with pytest.raises(ValueError, match="already expressed in pixels"):
            obj.to_px(meta)  # type: ignore

    def test_to_cm_conversion(self):
        """Test to_cm method converts coordinates correctly."""

        class MockMeta:
            size = np.array([2.0, 2.0, 2.0])

            def to_cm(self, coords):
                return coords * self.size

        meta = MockMeta()
        obj = SimplePosData(
            units="px", position=np.array([5.0, 10.0, 15.0], dtype=np.float32)
        )

        obj.to_cm(meta)  # type: ignore

        assert obj.units == "cm"
        assert np.allclose(obj.position, [10.0, 20.0, 30.0])

    def test_to_cm_already_cm(self):
        """Test to_cm raises error if already in centimeters."""

        class MockMeta:
            size = np.array([2.0, 2.0, 2.0])

        meta = MockMeta()
        obj = SimplePosData(units="cm")

        with pytest.raises(ValueError, match="already expressed in centimeters"):
            obj.to_cm(meta)  # type: ignore

    def test_normed_vector_conversion(self):
        """Test that normed vector attributes are converted correctly."""

        class MockMeta:
            lower = np.array([10.0, 10.0, 10.0])
            size = np.array([2.0, 2.0, 2.0])

            def to_px(self, coords):
                return coords / self.size

        # Test with a normed vector (units="instance") that should be scaled but not shifted
        meta = MockMeta()
        obj = SimpleNormedVectorData(
            units="cm", vector=np.array([0.0, 1.0, 2.0], dtype=np.float64)
        )

        obj.to_px(meta)  # type: ignore

        assert obj.units == "px"
        assert np.allclose(
            obj.vector, [0.0, 0.5, 1.0]
        )  # Should be scaled by 1/voxel size, but not shifted by lower bound since it's a normed vector

        # Same but from pixels to centimeters
        meta = MockMeta()
        obj = SimpleNormedVectorData(
            units="px", vector=np.array([0.0, 0.5, 1.0], dtype=np.float64)
        )
        obj.to_cm(meta)  # type: ignore

        assert obj.units == "cm"
        assert np.allclose(
            obj.vector, [0.0, 1.0, 2.0]
        )  # Should be scaled by voxel size, but not shifted by lower bound since it's

        # Make sure normalized vector (unitless) is not affected by conversion
        obj_normalized = SimpleNormalizedVectorData(
            units="cm", vector=np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )

        obj_normalized.to_px(meta)  # type: ignore

        assert obj_normalized.units == "px"
        assert np.allclose(
            obj_normalized.vector, [0.0, 1.0, 0.0]
        )  # Should be unchanged since it's a normalized vector

    def test_scalar_length_conversion(self):
        """Test that scalar length attributes are converted."""

        class MockMeta:
            size = np.array([2.0, 2.0, 2.0])

            def to_px(self, coords):
                return coords / self.size

        meta = MockMeta()
        obj = SimplePosData(units="cm", length=10.0)

        obj.to_px(meta)  # type: ignore

        assert obj.length == 5.0  # 10.0 / 2.0
