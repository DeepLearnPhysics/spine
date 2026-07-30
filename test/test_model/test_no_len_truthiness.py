"""Prevent implicit length truth-testing throughout the model package."""

import ast
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "spine" / "model"


def test_model_package_compares_lengths_explicitly():
    """Length checks must state the intended zero or nonzero comparison."""
    violations = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
            ):
                continue

            parent = parents.get(node)
            is_implicit = isinstance(
                parent, (ast.If, ast.IfExp, ast.While, ast.BoolOp, ast.comprehension)
            ) or (isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not))
            if is_implicit:
                violations.append(f"{path.relative_to(MODEL_DIR)}:{node.lineno}")

    if violations:
        locations = "\n".join(f"  - {location}" for location in violations)
        raise AssertionError(
            "Compare len(...) explicitly in spine.model:\n" f"{locations}"
        )
