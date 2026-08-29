"""Provide lightweight version and dependency information for the CLI.

The ``--version`` and ``--info`` paths must remain usable when the optional
machine-learning stack is unavailable. This module therefore avoids importing
heavy dependencies at module load time and probes them only when requested.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

__all__ = ["check_dependencies", "get_version", "show_info"]


def get_version() -> str:
    """Return the installed SPINE version without heavy imports.

    Returns
    -------
    str
        Package version, or ``"unknown"`` when version metadata cannot be
        imported.
    """
    try:
        from spine.version import __version__

        return __version__
    except ImportError:
        return "unknown"


def show_info() -> None:
    """Print package, runtime and optional-feature availability.

    Dependency status is collected lazily so this report remains available in
    lightweight installations that do not include PyTorch or visualization
    packages.
    """
    print(f"SPINE (Scalable Particle Imaging with Neural Embeddings) v{get_version()}")
    print("https://github.com/DeepLearnPhysics/spine")
    print()

    # Present the raw dependency versions first for easy environment diagnosis.
    deps = check_dependencies()
    print("Dependency Status:")
    print("-" * 40)
    for name, version in deps.items():
        status = f"✓ {version}" if version else "✗ Not available"
        print(f"{name:15}: {status}")

    print(f"\nPython: {sys.version}")
    print()

    # Summarize which user-facing feature groups are complete.
    print("Available functionality:")
    print("  Core: Mathematical operations, data handling, I/O")
    model_deps = (
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "MinkowskiEngine",
    )
    missing_model_deps = [name for name in model_deps if not deps[name]]
    if not missing_model_deps:
        print("  Model stack: Available")
    else:
        print(f"  Model stack: Incomplete (missing: {', '.join(missing_model_deps)})")

    if deps["plotly"]:
        print(f"  Visualization: Available (Plotly {deps['plotly']})")
    else:
        print("  Visualization: Not available (install with: pip install spine[viz])")

    if deps["torch"] is None:
        print("\n" + "=" * 50)
        print("NOTICE: PyTorch not found!")
        print("For full ML functionality, use the released SPINE container")
        print("or install the compatible ML ecosystem manually.")
        print("=" * 50)


def check_dependencies() -> dict[str, str | None]:
    """Collect versions of optional SPINE dependencies.

    Python-only packages are imported lazily to read their versions. Compiled
    model extensions are queried through distribution metadata instead, which
    avoids initializing CUDA or loading binary extensions during ``--info``.

    Returns
    -------
    dict[str, str or None]
        Dependency names mapped to installed versions, or ``None`` when a
        dependency is unavailable.
    """
    deps: dict[str, str | None] = {}

    # Probe packages whose imports do not initialize compiled model extensions.
    for name in ("torch", "matplotlib", "plotly", "seaborn"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            deps[name] = str(version) if version is not None else None
        except ImportError:
            deps[name] = None

    # Inspect compiled packages by metadata to avoid CUDA/extension startup.
    for distribution in (
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "MinkowskiEngine",
    ):
        try:
            deps[distribution] = package_version(distribution)
        except PackageNotFoundError:
            deps[distribution] = None

    return deps
