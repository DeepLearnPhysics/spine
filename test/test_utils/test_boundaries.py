"""Architectural guardrails for the deliberately narrow utility package."""

import ast
from pathlib import Path

import spine.utils


def test_utils_contains_only_approved_shared_modules():
    """Domain and implementation-owned code must not drift back into utils."""
    root = Path(spine.utils.__file__).parent
    modules = {path.stem for path in root.glob("*.py")}
    assert modules == {
        "__init__",
        "conditional",
        "docstring",
        "ghost",
        "jit",
        "manager",
        "optical",
        "ppn",
        "stopwatch",
    }

    packages = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert packages == {"torch"}


def test_utils_does_not_depend_on_implementation_layers():
    """Shared utilities may not import their IO, model, or workflow consumers."""
    root = Path(spine.utils.__file__).parent
    forbidden = ("spine.ana", "spine.io", "spine.model", "spine.post")
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        assert not any(
            module.startswith(prefix) for module in imports for prefix in forbidden
        ), f"{path.relative_to(root)} imports an implementation layer"
