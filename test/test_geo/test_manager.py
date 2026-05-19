"""Tests for the geometry singleton manager."""

from types import SimpleNamespace

import pytest

from spine.geo.manager import GeoManager


@pytest.fixture(autouse=True)
def reset_geo_manager():
    """Reset the singleton before and after each test."""
    GeoManager.reset()
    yield
    GeoManager.reset()


def test_initialize_rejects_existing_instance(monkeypatch):
    """Explicit initialization should fail if the singleton already exists."""
    monkeypatch.setattr(
        "spine.geo.manager.geo_factory",
        lambda detector, tag=None, version=None: SimpleNamespace(
            name=detector, tag=tag, version=version
        ),
    )

    GeoManager.initialize("demo")

    with pytest.raises(ValueError, match="already initialized"):
        GeoManager.initialize("demo")


def test_initialize_or_get_reuses_existing_when_options_omitted(monkeypatch):
    """Omitted tag/version should not force reinitialization for the same detector."""
    calls = []

    def factory(detector, tag=None, version=None):
        calls.append((detector, tag, version))
        return SimpleNamespace(name=detector, tag="latest", version="10.0")

    monkeypatch.setattr("spine.geo.manager.geo_factory", factory)

    first = GeoManager.initialize_or_get("demo")
    second = GeoManager.initialize_or_get("DEMO")

    assert first is second
    assert calls == [("demo", None, None)]


def test_initialize_or_get_reinitializes_on_specific_mismatch(monkeypatch):
    """Specific tag/version requests should replace incompatible instances."""
    calls = []

    def factory(detector, tag=None, version=None):
        calls.append((detector, tag, version))
        return SimpleNamespace(name=detector, tag=tag or "latest", version="2.0")

    monkeypatch.setattr("spine.geo.manager.geo_factory", factory)

    first = GeoManager.initialize_or_get("demo", tag="v1", version=1)
    second = GeoManager.initialize_or_get("demo", tag="v2", version=2)

    assert first is not second
    assert calls == [("demo", "v1", 1), ("demo", "v2", 2)]


def test_initialize_or_get_reuses_existing_major_version(monkeypatch):
    """Major-only version requests should match initialized minor versions."""
    calls = []

    def factory(detector, tag=None, version=None):
        calls.append((detector, tag, version))
        return SimpleNamespace(name=detector, tag=tag or "latest", version="6.5")

    monkeypatch.setattr("spine.geo.manager.geo_factory", factory)

    first = GeoManager.initialize_or_get("demo")
    second = GeoManager.initialize_or_get("demo", version="6")

    assert first is second
    assert calls == [("demo", None, None)]


def test_get_instance_reports_uninitialized_access():
    """Accessing an uninitialized singleton should fail with caller context."""
    with pytest.raises(ValueError, match="Geometry singleton instance"):
        GeoManager.get_instance()


def test_get_instance_returns_initialized_instance():
    """Strict singleton access should return the current geometry."""
    GeoManager._instance = SimpleNamespace(name="demo", tag=None, version=None)

    assert GeoManager.is_initialized() is True
    assert GeoManager.get_instance() is GeoManager._instance


def test_get_instance_if_initialized():
    """Optional singleton access should return None before initialization."""
    assert GeoManager.is_initialized() is False
    assert GeoManager.get_instance_if_initialized() is None
    GeoManager._instance = SimpleNamespace(name="demo", tag=None, version=None)
    assert GeoManager.get_instance_if_initialized() is GeoManager._instance
