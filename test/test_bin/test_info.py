"""Tests for lightweight CLI environment reporting."""

import builtins
from types import ModuleType

from spine.bin import info as info_module


def test_get_version_show_info_and_dependency_checks(monkeypatch, capsys):
    """Version and info helpers should handle unavailable dependencies."""
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "spine.version":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert info_module.get_version() == "unknown"

    monkeypatch.setattr(info_module, "get_version", lambda: "1.2.3")
    original_check_dependencies = info_module.check_dependencies
    monkeypatch.setattr(
        info_module,
        "check_dependencies",
        lambda: {
            "torch": None,
            "matplotlib": "3.8.0",
            "plotly": None,
            "seaborn": "0.13.0",
            "torch-geometric": None,
            "torch-scatter": None,
            "torch-cluster": None,
            "MinkowskiEngine": None,
        },
    )
    info_module.show_info()
    output = capsys.readouterr().out
    assert "SPINE (Scalable Particle Imaging with Neural Embeddings) v1.2.3" in output
    assert "PyTorch not found" in output
    assert "Visualization: Not available" in output

    monkeypatch.setattr(info_module, "check_dependencies", original_check_dependencies)

    def fake_missing_dep_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "matplotlib", "plotly", "seaborn", "MinkowskiEngine"}:
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_missing_dep_import)
    monkeypatch.setattr(
        info_module,
        "package_version",
        lambda name: (_ for _ in ()).throw(info_module.PackageNotFoundError(name)),
    )
    deps = info_module.check_dependencies()
    assert deps["torch"] is None
    assert deps["MinkowskiEngine"] is None
    assert set(deps) == {
        "torch",
        "matplotlib",
        "plotly",
        "seaborn",
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "MinkowskiEngine",
    }


def test_get_version_and_dependency_checks_success(monkeypatch):
    """Version lookup and dependency probes should report installed modules."""
    from spine.version import __version__

    original_import = builtins.__import__

    def fake_dep_import(name, globals=None, locals=None, fromlist=(), level=0):
        versions = {
            "torch": "2.0.0",
            "matplotlib": "3.8.0",
            "plotly": "5.0.0",
            "seaborn": "0.13.0",
            "MinkowskiEngine": "0.5.4",
        }
        if name in versions:
            module = ModuleType(name)
            module.__version__ = versions[name]
            return module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_dep_import)
    monkeypatch.setattr(
        info_module,
        "package_version",
        lambda name: {
            "torch-geometric": "2.6.0",
            "torch-scatter": "2.1.2",
            "torch-cluster": "1.6.3",
            "MinkowskiEngine": "0.5.4",
        }[name],
    )

    assert info_module.get_version() == __version__
    deps = info_module.check_dependencies()
    assert deps["torch"] == "2.0.0"
    assert deps["matplotlib"] == "3.8.0"
    assert deps["plotly"] == "5.0.0"
    assert deps["seaborn"] == "0.13.0"
    assert deps["torch-geometric"] == "2.6.0"
    assert deps["torch-scatter"] == "2.1.2"
    assert deps["torch-cluster"] == "1.6.3"
    assert deps["MinkowskiEngine"] == "0.5.4"


def test_show_info_reports_available_optional_features(monkeypatch, capsys):
    """Info output should report available model and visualization extras."""
    monkeypatch.setattr(info_module, "get_version", lambda: "1.2.3")
    monkeypatch.setattr(
        info_module,
        "check_dependencies",
        lambda: {
            "torch": "2.0.0",
            "matplotlib": "3.8.0",
            "plotly": "5.0.0",
            "seaborn": None,
            "torch-geometric": "2.6.0",
            "torch-scatter": "2.1.2",
            "torch-cluster": "1.6.3",
            "MinkowskiEngine": "0.5.4",
        },
    )

    info_module.show_info()

    output = capsys.readouterr().out
    assert "Model stack: Available" in output
    assert "Visualization: Available (Plotly 5.0.0)" in output
