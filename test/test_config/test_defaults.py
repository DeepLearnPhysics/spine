"""Contracts for the configuration examples shipped with SPINE."""

from pathlib import Path

import yaml

from spine.config import load_config, load_config_file
from spine.config.factory import parse_module_config

CONFIG_DIR = Path(__file__).parents[2] / "config"


def test_defaults_use_canonical_descriptors():
    """Implementation selectors are providers; names identify ordered entries."""

    def check(node, path=()):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                assert key.value not in {"parser", "args", "kwargs"}, path
                if key.value == "name":
                    assert path[-2:] == ("stages", "[]") or any(
                        part.endswith("~") for part in path
                    ), path
                check(value, (*path, key.value))
        elif isinstance(node, yaml.SequenceNode):
            for value in node.value:
                check(value, (*path, "[]"))

    for path in CONFIG_DIR.rglob("*.yaml"):
        check(yaml.compose(path.read_text()), (str(path),))


def test_default_analyzers_preserve_base_post_processors():
    """Named edits preserve processor order and alter only matching settings."""
    base = load_config_file(
        str(CONFIG_DIR / "full_chain/full_chain_regression.yaml"), download=False
    )
    combined = load_config(
        "include: [full_chain/full_chain_regression.yaml, test/analyzers.yaml]",
        root_dir=str(CONFIG_DIR),
        download=False,
    )
    original = parse_module_config(base["post"], stages_key="stages")
    edited = parse_module_config(combined["post"], stages_key="stages")
    assert list(edited) == [*original, "ppn"]
    for name, spec in original.items():
        if name == "match":
            assert edited[name]["cfg"] == {
                **spec["cfg"],
                "truth_point_mode": "points_adapt",
                "fragment": True,
            }
        else:
            assert edited[name] == spec
