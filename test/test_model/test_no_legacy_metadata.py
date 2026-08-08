"""Prevent obsolete model metadata from returning."""

import ast
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "spine" / "model"
LEGACY_ATTRIBUTES = {"INPUT_SCHEMA", "MODULES"}


def test_model_classes_have_no_legacy_metadata():
    """Model inputs and configuration blocks are defined by current APIs."""
    assignments = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = {
                    target.id for target in node.targets if isinstance(target, ast.Name)
                }
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = {node.target.id}
            else:
                continue

            for name in names & LEGACY_ATTRIBUTES:
                assignments.append(
                    f"{path.relative_to(MODEL_DIR)}:{node.lineno} ({name})"
                )

    if assignments:
        locations = "\n".join(f"  - {location}" for location in assignments)
        raise AssertionError(
            "Remove obsolete model class metadata assignments:\n" f"{locations}"
        )
