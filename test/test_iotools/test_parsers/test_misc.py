"""Test that the miscellaneous data parsers work as intended."""

import pytest
from warnings import warn
from dataclasses import asdict

import numpy as np

from mlreco import Meta, RunInfo, Flash, CRTHit, Trigger
from mlreco.iotools.parsers.misc import *

from test.test_iotools.test_parsers.fixtures import (
        fixture_sparse2d_event, fixture_sparse3d_event,
        fixture_flash_event, fixture_flash_event_list,
        fixture_crthit_event, fixture_trigger_event)


@pytest.mark.parametrize('projection_id', [0, 1, 2])
def test_parse_meta2d(sparse2d_event, projection_id):
    """Tests the parsing of metadata for 2D sparse events."""
    # Initialize the parser
    parser = MetaParser(
            sparse_event=sparse2d_event, projection_id=projection_id)

    # Parse the data
    result = parser.process(sparse_event=sparse2d_event)

    # Do a few basic checks
    # - The object returned should be a Meta object
    # - Each of its attributes should be of length 2
    assert isinstance(result, Meta)
    for v in asdict(result).values():
        assert len(v) == 2


def test_parse_meta3d(sparse3d_event):
    """Tests the parsing of metadata for 3D sparse events."""
    # Initialize the parser
    parser = MetaParser(sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # Do a few basic checks
    # - The object returned should be a Meta object
    # - Each of its attributes should be of length 3
    assert isinstance(result, Meta)
    for v in asdict(result).values():
        assert len(v) == 3


def test_parse_run_info(sparse3d_event):
    """Tests the parsing of the run info of 3D sparse events."""
    # Initialize the parser
    parser = RunInfoParser(sparse_event=sparse3d_event)

    # Parse the data
    result = parser.process(sparse_event=sparse3d_event)

    # Do a few basic checks
    # - The object returned should be a RunInfo object
    assert isinstance(result, RunInfo)


@pytest.mark.parametrize('flash_event', [0, 1, 10], indirect=True)
def test_parse_flashes(flash_event):
    """Tests the parsing of a list of optical flashes."""
    # Initialize the parser
    parser = FlashParser(flash_event=flash_event)

    # Parse the data
    result = parser.process(flash_event=flash_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == flash_event.as_vector().size()
    if len(result):
        assert type(result[0]) == Flash
    assert isinstance(result.default, Flash)


@pytest.mark.parametrize('flash_event_list', [1, 2], indirect=True)
def test_parse_flashes(flash_event_list):
    """Tests the parsing of a list of list of optical flashes."""
    # Initialize the parser
    parser = FlashParser(flash_event_list=flash_event_list)

    # Parse the data
    result = parser.process(flash_event_list=flash_event_list)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    assert len(result) == np.sum(
            [event.as_vector().size() for event in flash_event_list])
    if len(result):
        assert type(result[0]) == Flash
    assert isinstance(result.default, Flash)


@pytest.mark.parametrize('crthit_event', [0, 1, 10], indirect=True)
def test_parse_crthits(crthit_event):
    """Tests the parsing of a list of CRT hits."""
    # Initialize the parser
    parser = CRTHitParser(crthit_event=crthit_event)

    # Parse the data
    result = parser.process(crthit_event=crthit_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == crthit_event.as_vector().size()
    if len(result):
        assert type(result[0]) == CRTHit
    assert isinstance(result.default, CRTHit)


@pytest.mark.parametrize('crthit_event', [0, 1, 10], indirect=True)
def test_parse_crthits(crthit_event):
    """Tests the parsing of a list of CRT hits."""
    # Initialize the parser
    parser = CRTHitParser(crthit_event=crthit_event)

    # Parse the data
    result = parser.process(crthit_event=crthit_event)

    # Do a few basic checks
    # - The list produced is of the expected size
    # - The objects in the list are of the expected type
    # - The list provides a default object, even when it is empty
    assert len(result) == crthit_event.as_vector().size()
    if len(result):
        assert type(result[0]) == CRTHit
    assert isinstance(result.default, CRTHit)


def test_parse_trigger(trigger_event):
    """Tests the parsing of trigger information."""
    # Skip this test if larcv does not contain the Trigger object
    if trigger_event is None:
        warn("Cannot test the parser, Trigger object missing from LArCV2.")
        return

    # Initialize the parser
    parser = TriggerParser(trigger_event=trigger_event)

    # Parse the data
    result = parser.process(trigger_event=trigger_event)

    # Do a few basic checks
    # - The object returned should be a Trigger object
    assert isinstance(result, Trigger)
