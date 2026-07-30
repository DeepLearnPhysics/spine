"""Enforce explicit exception handling throughout the model package."""

import ast
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "spine" / "model"


def test_model_package_has_no_executable_assertions():
    """Assertions must not be used for configuration or runtime validation."""
    assertions = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        assertions.extend(
            f"{path.relative_to(MODEL_DIR)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )

    if assertions:
        locations = "\n".join(f"  - {location}" for location in assertions)
        raise AssertionError(
            "Use an explicit exception instead of assert in spine.model:\n"
            f"{locations}"
        )
