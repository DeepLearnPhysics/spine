"""Shared fixtures for geometry tests."""

from pathlib import Path

import pytest


def write_geometry_config(
    root: Path,
    detector: str,
    tag: str,
    version: str | int | float,
) -> Path:
    """Write a minimal two-TPC geometry config."""
    config_dir = root / detector
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{detector}_{tag}_geometry.yaml"
    path.write_text(
        f"""
name: {detector}
tag: {tag}
version: {version}
tpc:
  dimensions: [10.0, 20.0, 30.0]
  positions:
    - [-6.0, 0.0, 0.0]
    - [6.0, 0.0, 0.0]
  module_ids: [0, 0]
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def geo_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated geometry config directory."""
    from spine.geo import factories

    monkeypatch.setattr(factories, "GEO_CONFIG_DIR", tmp_path)
    return tmp_path
