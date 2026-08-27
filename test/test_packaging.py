"""Tests for source-distribution and runtime packaging contracts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _optional_dependency_names(pyproject: str) -> set[str]:
    """Extract optional-dependency names without requiring Python 3.11 TOML."""
    section = pyproject.split("[project.optional-dependencies]", maxsplit=1)[1]
    section = section.split("\n[", maxsplit=1)[0]
    return set(re.findall(r"(?m)^([A-Za-z0-9_-]+)\s*=\s*\[", section))


def test_dockerfiles_reference_declared_optional_dependencies():
    """Every package extra requested by a Dockerfile must be declared."""
    pyproject = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    declared = _optional_dependency_names(pyproject)

    # Collect comma-separated extras from local project specifications such as
    # ``.[viz]`` while ignoring ordinary bracket syntax in Docker commands.
    referenced = set()
    for path in (ROOT_DIR / "docker").rglob("Dockerfile"):
        contents = path.read_text(encoding="utf-8")
        for group in re.findall(r"\.\[([A-Za-z0-9_, -]+)\]", contents):
            referenced.update(name.strip() for name in group.split(","))

    assert referenced <= declared, (
        "Dockerfiles reference undefined package extras: "
        f"{sorted(referenced - declared)}"
    )


def test_spine_image_installs_and_verifies_tensorboard():
    """The full training image should include its TensorBoard integration."""
    dockerfile = (ROOT_DIR / "docker" / "spine" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert '"tensorboard>=2.10.0"' in dockerfile
    assert "from torch.utils.tensorboard import SummaryWriter" in dockerfile
