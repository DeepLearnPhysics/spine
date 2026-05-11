"""Test that the miscellaneous data parsers work as intended."""

from dataclasses import asdict
from types import SimpleNamespace
from warnings import warn

import numpy as np
import pytest

from spine.data.larcv import CRTHit, Flash, RunInfo, Trigger
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.geo import GeoManager
from spine.io.parse.larcv.misc import *
from spine.utils.conditional import LARCV_AVAILABLE

pytestmark = pytest.mark.skipif(
    not LARCV_AVAILABLE, reason="LArCV is required to generate parser fixtures."
)


@pytest.mark.parametrize("projection_id", [0, 1, 2])
def test_parse_meta2d(sparse2d_event, projection_id):
    """Tests the parsing of metadata for 2D sparse events."""
    # Initialize the parser
    parser = MetaParser(
        dtype="float32", sparse_event=sparse2d_event, projection_id=projection_id
    )

    # Parse the data
    result = parser.process(sparse_event=sparse2d_event)

    # Do a few basic checks
    # - The object returned should be a Meta object
    # - Each of its attributes should be of length 2
    assert isinstance(result, ImageMeta2D)
    for v in asdict(result).values():
        assert len(v) == 2


def test_parse_meta3d(sparse3d_event):
    """Tests the parsing of metadata for 3D sparse events."""
    # Initialize the parser
    parser = MetaParser(dtype="float32", sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # Do a few basic checks
    # - The object returned should be a Meta object
    # - Each of its attributes should be of length 3
    assert isinstance(result, ImageMeta3D)
    for v in asdict(result).values():
        assert len(v) == 3


def test_parse_meta_requires_exactly_one_source():
    """Metadata parsing should reject ambiguous or missing sources."""
    parser = MetaParser(dtype="float32")

    with pytest.raises(ValueError, match="exactly one"):
        parser.process()


def test_parse_run_info(sparse3d_event):
    """Tests the parsing of the run info of 3D sparse events."""
    # Initialize the parser
    parser = RunInfoParser(dtype="float32", sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # Do a few basic checks
    # - The object returned should be a RunInfo object
    assert isinstance(result, RunInfo)


@pytest.mark.parametrize("flash_event", [0, 1, 10], indirect=True)
def test_parse_flashes(flash_event):
    """Tests the parsing of a list of optical flashes."""
    # Must initialize the geomtry singleton before parsing flashes
    GeoManager.initialize_or_get(detector="icarus")

    # Initialize the parser
    parser = FlashParser(dtype="float32", flash_event=flash_event)

    # Parse the data
    result = parser.process(flash_event=flash_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == flash_event.as_vector().size()
    if len(result) > 0:
        assert isinstance(result[0], Flash)
    assert isinstance(result.default, Flash)


@pytest.mark.parametrize("flash_event_list", [1, 2], indirect=True)
def test_parse_flash_lists(flash_event_list):
    """Tests the parsing of a list of list of optical flashes."""
    # Initialize the parser
    parser = FlashParser(dtype="float32", flash_event_list=flash_event_list)

    # Parse the data
    result = parser.process(flash_event_list=flash_event_list)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    assert len(result) == np.sum(
        [event.as_vector().size() for event in flash_event_list]
    )
    if len(result) > 0:
        assert isinstance(result[0], Flash)
    assert isinstance(result.default, Flash)


@pytest.mark.parametrize("crthit_event", [0, 1, 10], indirect=True)
def test_parse_crthits(crthit_event):
    """Tests the parsing of a list of CRT hits."""
    # Initialize the parser
    parser = CRTHitParser(dtype="float32", crthit_event=crthit_event)

    # Parse the data
    result = parser.process(crthit_event=crthit_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == crthit_event.as_vector().size()
    if len(result) > 0:
        assert isinstance(result[0], CRTHit)
    assert isinstance(result.default, CRTHit)


def test_parse_trigger(trigger_event):
    """Tests the parsing of trigger information."""
    # Skip this test if larcv does not contain the Trigger object
    if trigger_event is None:
        warn("Cannot test the trigger parser, Trigger object missing from LArCV2.")
        return

    # Initialize the parser
    parser = TriggerParser(dtype="float32", trigger_event=trigger_event)

    # Parse the data
    result = parser.process(trigger_event=trigger_event)

    # Do a few basic checks
    # - The object returned should be a Trigger object
    assert isinstance(result, Trigger)


@pytest.mark.parametrize("crthit_event", [1], indirect=True)
def test_misc_parser_call_paths(sparse3d_event, crthit_event, trigger_event):
    """Wrapper calls should route named inputs through the misc parsers."""
    meta_parser = MetaParser(dtype="float32", sparse_event="sparse")
    assert isinstance(meta_parser({"sparse": sparse3d_event}), ImageMeta3D)

    run_info_parser = RunInfoParser(dtype="float32", sparse_event="sparse")
    assert isinstance(run_info_parser({"sparse": sparse3d_event}), RunInfo)

    crthit_parser = CRTHitParser(dtype="float32", crthit_event="crthit")
    assert isinstance(crthit_parser({"crthit": crthit_event}).default, CRTHit)

    if trigger_event is not None:
        trigger_parser = TriggerParser(dtype="float32", trigger_event="trigger")
        assert isinstance(trigger_parser({"trigger": trigger_event}), Trigger)


def test_flash_parser_special_cases(monkeypatch):
    """Flash parsing should cover merger setup and resize branches."""

    class DummyFlash:
        def __init__(self, pe_per_ch, volume_id=0):
            self.pe_per_ch = np.asarray(pe_per_ch, dtype=np.float32)
            self.volume_id = volume_id
            self.id = 0

    class DummyEvent:
        def __init__(self, flashes):
            self.flashes = flashes

        def as_vector(self):
            return self.flashes

    monkeypatch.setattr(
        "spine.io.parse.larcv.misc.GeoManager.get_instance",
        lambda: SimpleNamespace(optical=SimpleNamespace(num_channels_per_volume=4)),
    )
    monkeypatch.setattr(
        "spine.io.parse.larcv.misc.Flash.from_larcv", lambda flash: flash
    )
    monkeypatch.setattr("spine.io.parse.larcv.misc.larcv.Flash", lambda flash: flash)
    monkeypatch.setattr(
        "spine.io.parse.larcv.misc.FlashMerger",
        lambda **kwargs: (lambda flashes: (list(reversed(flashes)), None)),
    )

    parser = FlashParser(dtype="float32", flash_event="flash", merge={"dt": 1.0})
    result = parser({"flash": DummyEvent([DummyFlash([1, 2, 3, 4])])})
    assert result[0].pe_per_ch.shape == (4,)

    parser = FlashParser(dtype="float32", flash_event_list=["f0", "f1"])
    result = parser(
        {
            "f0": DummyEvent([DummyFlash([1, 2])]),
            "f1": DummyEvent([DummyFlash(np.arange(8, dtype=np.float32))]),
        }
    )
    assert result[0].pe_per_ch.shape == (4,)
    assert result[1].pe_per_ch.shape == (4,)


def test_flash_parser_requires_optical_geometry(monkeypatch):
    """Flash parsing should fail fast when the optical geometry is unavailable."""
    monkeypatch.setattr(
        "spine.io.parse.larcv.misc.GeoManager.get_instance",
        lambda: SimpleNamespace(optical=None),
    )

    with pytest.raises(ValueError, match="Optical geometry"):
        FlashParser(dtype="float32", flash_event="flash")
