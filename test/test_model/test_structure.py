"""Guard the ownership-oriented model package structure."""

import ast
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "spine" / "model"


def test_model_package_has_no_generic_layer_namespace():
    """Shared and model-owned components live in descriptive packages."""
    assert not (MODEL_DIR / "layer").exists()


def test_complex_models_own_their_implementations():
    """Complex maintained model families are represented by packages."""
    for name in (
        "full_chain",
        "graph_spice",
        "grappa",
        "spice",
        "uresnet",
    ):
        model_package = MODEL_DIR / name
        assert model_package.is_dir()
        assert (model_package / "__init__.py").is_file()
        assert not (MODEL_DIR / f"{name}.py").exists()


def test_uresnet_variants_share_one_owner_package():
    """UResNet variants must not reappear as top-level model modules."""
    uresnet_package = MODEL_DIR / "uresnet"

    assert (uresnet_package / "bayes.py").is_file()
    assert (uresnet_package / "ppn").is_dir()
    assert not (uresnet_package / "duq.py").exists()
    assert not (MODEL_DIR / "bayes_uresnet.py").exists()
    assert not (MODEL_DIR / "uresnet_ppn").exists()


def test_model_package_has_no_unowned_vertex_prototypes():
    """Vertex models must live with the architecture that implements them."""
    assert not (MODEL_DIR / "vertex.py").exists()


def test_shared_packages_do_not_depend_on_model_families():
    """Reusable infrastructure must remain independent of concrete models."""
    shared_packages = ("cnn", "common", "pointcloud", "sparse")
    model_families = (
        "full_chain",
        "graph_spice",
        "grappa",
        "spice",
        "uresnet",
    )
    forbidden_prefixes = tuple(f"spine.model.{family}" for family in model_families)
    violations = []

    for package in shared_packages:
        for path in (MODEL_DIR / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.startswith(forbidden_prefixes)
                ):
                    violations.append(f"{path.relative_to(MODEL_DIR)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.startswith(forbidden_prefixes):
                            violations.append(
                                f"{path.relative_to(MODEL_DIR)}:{node.lineno}"
                            )

    assert (
        not violations
    ), "Shared model infrastructure imports concrete model families:\n" + "\n".join(
        f"  - {location}" for location in violations
    )
