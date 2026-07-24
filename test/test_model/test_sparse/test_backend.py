"""Tests for sparse backend selection and isolation."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from spine.model import sparse
from spine.model.sparse import backend


def test_sparse_api_does_not_expose_backend_names():
    """Models consume semantic operations, not native backend class names."""
    assert not any(name.startswith("Minkowski") for name in dir(sparse))


def test_backend_import_is_isolated_to_adapter():
    """The selected engine may only be imported by its backend adapter."""
    repository = Path(__file__).resolve().parents[3]
    model_root = repository / "src" / "spine" / "model"
    adapter = model_root / "sparse" / "backends" / "minkowski.py"

    for path in model_root.rglob("*.py"):
        if path == adapter:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module or ""}
            else:
                continue
            assert "MinkowskiEngine" not in names, path


def test_default_backend_implements_public_operations():
    """The default adapter resolves the semantic operation contract."""
    assert backend.name() == "minkowski"
    assert backend.adapter().__name__.endswith(".backends.minkowski")
    assert backend.module("Convolution") is not None


def test_backend_rejects_an_unknown_adapter(monkeypatch):
    """An unsupported configured backend produces a useful public error."""
    monkeypatch.setattr(backend, "_BACKEND_NAME", "does_not_exist")
    monkeypatch.setattr(backend, "_ADAPTER", None)

    with pytest.raises(ValueError, match="Unsupported sparse backend"):
        backend.adapter()


def test_backend_preserves_nested_import_errors(monkeypatch):
    """A backend's missing dependency is not misreported as a bad backend."""

    def fail_import(_):
        raise ModuleNotFoundError("missing dependency", name="dependency")

    monkeypatch.setattr(backend, "_BACKEND_NAME", "broken")
    monkeypatch.setattr(backend, "_ADAPTER", None)
    monkeypatch.setattr(backend.importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        backend.adapter()


def test_minkowski_adapter_rejects_an_unknown_operation():
    """The selected adapter reports unsupported semantic operations."""
    with pytest.raises(ValueError, match="does not implement"):
        backend.adapter().module("UnknownOperation")


def test_minkowski_adapter_reports_a_missing_engine(monkeypatch):
    """Importing the adapter without MinkowskiEngine gives a clear error."""
    adapter_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spine"
        / "model"
        / "sparse"
        / "backends"
        / "minkowski.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_missing_minkowski",
        adapter_path,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "MinkowskiEngine", None)

    with pytest.raises(ImportError, match="MinkowskiEngine is required"):
        spec.loader.exec_module(module)
