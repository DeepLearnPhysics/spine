"""Tests for modifier conflict validation during configuration composition."""

import pytest
import yaml

from spine.config import load_config_file
from spine.config.api import META_CONFLICTS_WITH, META_KEY
from spine.config.errors import ConfigValidationError


def _write_modifier(path, name, conflicts=None):
    metadata = {"kind": "mod", "name": name, "description": name}
    if conflicts is not None:
        metadata[META_CONFLICTS_WITH] = conflicts
    path.write_text(yaml.safe_dump({META_KEY: metadata, name: True}))


@pytest.mark.parametrize(
    ("forward_conflicts", "backward_conflicts", "include_order"),
    [
        ("sce_backward", None, ["forward.yaml", "backward.yaml"]),
        (None, ["sce_forward"], ["forward.yaml", "backward.yaml"]),
        (["sce_backward"], ["sce_forward"], ["forward.yaml", "backward.yaml"]),
        ("sce_backward", None, ["backward.yaml", "forward.yaml"]),
    ],
)
def test_modifier_conflicts_are_order_independent(
    tmp_path,
    forward_conflicts,
    backward_conflicts,
    include_order,
):
    """One-sided, symmetric, string, list, and reversed conflicts all fail."""
    _write_modifier(tmp_path / "forward.yaml", "sce_forward", forward_conflicts)
    _write_modifier(tmp_path / "backward.yaml", "sce_backward", backward_conflicts)
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(yaml.safe_dump({"include": include_order}))

    with pytest.raises(ConfigValidationError) as exc_info:
        load_config_file(str(bundle))

    message = str(exc_info.value)
    assert "sce_forward" in message
    assert "sce_backward" in message
    assert str(tmp_path / "forward.yaml") in message
    assert str(tmp_path / "backward.yaml") in message


def test_unrelated_modifiers_compose(tmp_path):
    """Modifiers without a declared relationship still compose."""
    _write_modifier(tmp_path / "first.yaml", "first", "third")
    _write_modifier(tmp_path / "second.yaml", "second", [])
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text("include: [first.yaml, second.yaml]\n")

    assert load_config_file(str(bundle)) == {"first": True, "second": True}


def test_modifier_conflict_is_detected_through_nested_include(tmp_path):
    """Modifier provenance propagates out of nested bundles."""
    _write_modifier(tmp_path / "forward.yaml", "sce_forward", "sce_backward")
    _write_modifier(tmp_path / "backward.yaml", "sce_backward")
    nested = tmp_path / "nested.yaml"
    nested.write_text("include: backward.yaml\n")
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text("include: [nested.yaml, forward.yaml]\n")

    with pytest.raises(ConfigValidationError) as exc_info:
        load_config_file(str(bundle))

    message = str(exc_info.value)
    assert "sce_forward" in message
    assert "sce_backward" in message
    assert str(tmp_path / "forward.yaml") in message
    assert str(tmp_path / "backward.yaml") in message
