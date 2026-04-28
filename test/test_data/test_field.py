"""Tests for FieldMetadata class."""

import dataclasses

import numpy as np
import pytest

from spine.data.field import FieldMetadata


class TestFieldMetadata:
    """Tests for the FieldMetadata class."""

    def test_basic_creation(self):
        """Test basic FieldMetadata creation."""
        meta = FieldMetadata(units="MeV")
        assert meta.units == "MeV"
        assert meta.index is False
        assert meta.length is None

    def test_multiple_fields(self):
        """Test FieldMetadata with multiple fields."""
        meta = FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="MeV",
        )
        assert meta.length == 3
        assert meta.dtype == np.float32
        assert meta.position is True
        assert meta.units == "MeV"

    def test_flags(self):
        """Test boolean flags."""
        meta = FieldMetadata(index=True, skip=True)
        assert meta.index is True
        assert meta.skip is True
        assert meta.lite_skip is False
        assert meta.cat is False

    def test_enum(self):
        """Test enumeration metadata."""
        enum_map = {0: "electron", 1: "muon", 2: "pion"}
        meta = FieldMetadata(enum=enum_map)
        assert meta.enum == enum_map

    def test_enum_type_validation(self):
        """Test that enum must be a dict."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            FieldMetadata(enum=["electron", "muon"])  # type: ignore

    def test_immutable(self):
        """Test that FieldMetadata is immutable (frozen dataclass)."""
        meta = FieldMetadata(units="MeV")
        assert meta.units == "MeV"

        # FieldMetadata is a frozen dataclass, so it's immutable
        with pytest.raises(TypeError):
            meta["units"] = "GeV"  # type: ignore

        # Also can't modify attributes directly
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            meta.units = "GeV"  # type: ignore

    def test_getitem(self):
        """Test __getitem__ method."""
        meta = FieldMetadata(units="MeV", index=True)
        assert meta["units"] == "MeV"
        assert meta["index"] is True
        with pytest.raises(KeyError):
            _ = meta["nonexistent"]

    def test_len(self):
        """Test __len__ method."""
        meta = FieldMetadata(units="MeV", index=True, skip=True)
        assert len(meta) == 3  # three fields set: units, index, skip

        meta2 = FieldMetadata()
        assert len(meta2) == 0  # No fields set

    def test_iter(self):
        """Test __iter__ method."""
        meta = FieldMetadata(units="MeV", index=True, skip=True)
        keys = list(meta)
        assert "units" in keys
        assert "index" in keys
        assert "skip" in keys  # skip is not None, so it should be included

        meta2 = FieldMetadata()
        keys2 = list(meta2)
        assert len(keys2) == 0  # No fields set, so no keys should be returned

    def test_as_dict(self):
        """Test as_dict method."""
        meta = FieldMetadata(units="MeV", index=True, length=3)
        meta_dict = meta.as_dict()
        assert meta_dict["units"] == "MeV"
        assert meta_dict["index"] is True
        assert meta_dict["length"] == 3

        # Fields that are None or False should not be included
        assert "dtype" not in meta_dict
        assert "position" not in meta_dict
