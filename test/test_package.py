"""Tests for lightweight top-level SPINE package modules."""

from __future__ import annotations

import sys
import typing
from types import ModuleType

import pytest

import spine
from spine.banner import ascii_logo
from spine.version import __version__


def test_top_level_exports_and_banner():
    """The top-level package should expose version metadata and banner text."""
    assert spine.__version__ == __version__
    assert "██████████" in ascii_logo
    assert "Driver" in spine.__all__


def test_lazy_driver_import(monkeypatch):
    """Driver should be imported lazily through __getattr__."""
    fake_driver_module = ModuleType("spine.driver")

    class DummyDriver:
        pass

    fake_driver_module.Driver = DummyDriver
    monkeypatch.setitem(sys.modules, "spine.driver", fake_driver_module)
    monkeypatch.delitem(spine.__dict__, "Driver", raising=False)

    driver_cls = spine.__getattr__("Driver")

    assert driver_cls is DummyDriver
    assert spine.Driver is DummyDriver
    with pytest.raises(AttributeError, match="has no attribute"):
        spine.__getattr__("MissingThing")


def test_type_checking_import_path(monkeypatch):
    """The TYPE_CHECKING import path should remain safe under reload."""
    fake_driver_module = ModuleType("spine.driver")

    class DummyDriver:
        pass

    fake_driver_module.Driver = DummyDriver
    monkeypatch.setitem(sys.modules, "spine.driver", fake_driver_module)
    monkeypatch.setattr(typing, "TYPE_CHECKING", True)

    import importlib

    reloaded = importlib.reload(spine)

    assert reloaded.__dict__["Driver"] is DummyDriver
