"""Enforce a consistent PyTorch namespace in the model package."""

import ast
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "spine" / "model"


def test_model_package_does_not_alias_torch_nn():
    """Model code accesses neural-network modules through ``torch.nn``."""
    violations = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            aliases_nn = isinstance(node, ast.Import) and any(
                name.name == "torch.nn" and name.asname == "nn" for name in node.names
            )
            imports_nn = (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch"
                and any(name.name == "nn" for name in node.names)
            )
            if aliases_nn or imports_nn:
                violations.append(f"{path.relative_to(MODEL_DIR)}:{node.lineno}")

    if violations:
        locations = "\n".join(f"  - {location}" for location in violations)
        raise AssertionError(
            "Use the torch.nn namespace directly in spine.model:\n" f"{locations}"
        )


def test_model_losses_do_not_inherit_from_protected_torch_base():
    """Custom losses inherit from the public ``torch.nn.Module`` API."""
    violations = []
    for path in MODEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if ast.unparse(base) == "torch.nn.modules.loss._Loss":
                    violations.append(f"{path.relative_to(MODEL_DIR)}:{node.lineno}")

    if violations:
        locations = "\n".join(f"  - {location}" for location in violations)
        raise AssertionError(
            "Use torch.nn.Module instead of PyTorch's protected _Loss base:\n"
            f"{locations}"
        )
