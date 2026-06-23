"""Tests for construct builder base behavior."""

import numpy as np
import pytest

from spine.construct.fragment import FragmentBuilder
from spine.construct.utils import get_batch_size, is_single_index
from spine.data.out import RecoFragment


def test_construct_reports_missing_required_key():
    """Missing required products should fail with the product name."""
    builder = FragmentBuilder(mode="reco", units="px")

    with pytest.raises(KeyError, match="fragment_clusts"):
        builder.construct("build_reco", {})


def test_builder_validates_mode_and_units():
    """Direct builder construction should validate runtime string values."""
    with pytest.raises(ValueError, match="Run mode"):
        FragmentBuilder(mode="invalid", units="px")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Units"):
        FragmentBuilder(mode="reco", units="mm")  # type: ignore[arg-type]


def test_index_helpers_handle_scalar_and_batched_inputs():
    """Index helper utilities should distinguish scalar and batched forms."""
    assert is_single_index(0) is True
    assert is_single_index(np.int64(0)) is True
    assert is_single_index(np.array(0)) is True
    assert is_single_index([0, 1]) is False
    assert get_batch_size([0, 1]) == 2
    assert get_batch_size(np.array([0, 1])) == 2

    with pytest.raises(TypeError, match="received a scalar value"):
        get_batch_size(0)

    with pytest.raises(TypeError, match="sized batch container"):
        get_batch_size(object())


def test_builder_call_builds_single_reco_entry(points, depositions):
    """Builder __call__ should populate the expected output key for one entry."""
    builder = FragmentBuilder(mode="reco", units="px")
    data = {
        "index": 0,
        "points": points,
        "depositions": depositions,
        "fragment_clusts": [np.array([0, 1], dtype=np.int32)],
        "fragment_shapes": np.array([0], dtype=np.int32),
    }

    builder(data)

    assert "reco_fragments" in data
    assert len(data["reco_fragments"]) == 1


def test_check_units_requires_meta_for_conversion():
    """Loading objects with mismatched units should require metadata."""
    builder = FragmentBuilder(mode="reco", units="cm")
    data = {"reco_fragments": [RecoFragment(id=0, units="px")]}

    with pytest.raises(KeyError, match="metadata"):
        builder.check_units(data, "reco_fragments")


def test_batched_load_unit_check_uses_selected_entry(meta_cm):
    """Unit checks for batched loads should inspect only the requested entry."""
    builder = FragmentBuilder(mode="reco", units="cm")
    entry0 = [RecoFragment(id=0, units="px")]
    entry1 = [RecoFragment(id=0, units="px", start_point=np.array([1, 1, 1]))]
    data = {
        "index": np.array([0, 1]),
        "reco_fragments": [entry0, entry1],
        "meta": [meta_cm, meta_cm],
    }

    result = builder.process(data, "reco", entry=1)

    assert result[0].units == "cm"
    assert entry0[0].units == "px"
    np.testing.assert_allclose(result[0].start_point, [2.0, 2.0, 2.0])
